"""OFFLINE CPU inference. Includes CPU image loading, preprocessing, all models, and CSV writing."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import get_model, get_transform, parameter_count


class Images(Dataset):
    def __init__(self, root, size):
        self.files = sorted(Path(root).glob('*.jpg'))
        if len(self.files) != 500:
            raise ValueError(f'Expected 500 JPGs, found {len(self.files)} in {root}')
        self.transform = get_transform(size, train=False)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(self.files[idx]) as im:
            image = self.transform(im.convert('RGB'))
        return image, self.files[idx].stem


def run(test_dir, output, config_file, batch_size=16):
    start = time.perf_counter()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    root = Path(config_file).resolve().parent
    conf = json.loads(Path(config_file).read_text(encoding='utf-8'))
    models = conf['models']
    assert models and len(models) <= 4
    total_params, weights_sum = 0, 0
    accumulated = None
    expected_ids = None
    for spec in models:
        path = root / spec['checkpoint']
        assert path.is_file(), path
        model = get_model(spec['arch'], pretrained=False)  # no download
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        nparams = parameter_count(model)
        total_params += nparams
        assert total_params <= 20_000_000, '20M parameter limit exceeded'
        model.eval().to('cpu')
        temp = float(spec['temperature'])
        weight = float(spec['weight'])
        assert .01 <= temp <= 10 and weight > 0
        weights_sum += weight
        loader = DataLoader(Images(test_dir, int(spec['size'])), batch_size=batch_size,
                            shuffle=False, num_workers=0)
        ids, predictions = [], []
        with torch.inference_mode():
            for image, batch_ids in loader:
                original = torch.softmax(model(image) / temp, 1)[:, 1]
                if spec['tta']:
                    flipped = torch.softmax(model(torch.flip(image, dims=[3])) / temp, 1)[:, 1]
                    original = (original + flipped) / 2
                ids.extend(batch_ids)
                predictions.extend(original.numpy().tolist())
        if expected_ids is None: expected_ids = ids
        assert ids == expected_ids
        arr = np.asarray(predictions, dtype=np.float64)
        accumulated = arr * weight if accumulated is None else accumulated + arr * weight
        print(f'finished {spec["arch"]} size={spec["size"]} tta={spec["tta"]} '
              f'params={nparams:,} elapsed={time.perf_counter()-start:.1f}s', flush=True)
        del model
    probs = accumulated / weights_sum
    result = pd.DataFrame({'id': expected_ids, 'prob': probs})
    assert len(result) == 500 and result['id'].is_unique
    assert np.isfinite(result['prob'].values).all() and result['prob'].between(0, 1).all()
    result.to_csv(output, index=False)
    duration = time.perf_counter() - start
    print(f'CPU end-to-end duration={duration:.1f}s / 600s limit; combined params={total_params:,}')
    print(f'CSV={output} (500 records)')
    if duration > 600:
        raise RuntimeError('CPU inference exceeds 10 minutes. Simplify TTA or models.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('submission.csv'))
    ap.add_argument('--config', type=Path, default=Path(__file__).parent / 'submission_config.json')
    ap.add_argument('--batch_size', type=int, default=16)
    args = ap.parse_args()
    run(args.test_dir, args.output, args.config, args.batch_size)
