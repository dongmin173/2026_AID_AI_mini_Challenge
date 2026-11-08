"""Build a fully offline, self-contained competition ZIP and reproduce its CSV on CPU."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model import get_model, parameter_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', type=Path, required=True, help='candidates/*.json from compare.py')
    ap.add_argument('--test-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='Output ZIP')
    ap.add_argument('--csv', type=Path, required=True, help='Output CSV')
    ap.add_argument('--notebook', type=Path, help='Original B0 training notebook; optional')
    args = ap.parse_args()
    spec = json.loads(args.candidate.read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parent
    total = 0
    with tempfile.TemporaryDirectory(prefix='mini_challenge_') as tmp:
        folder = Path(tmp) / 'package'
        folder.mkdir()
        (folder / 'models').mkdir()
        for f in ['model.py', 'predict.py', 'train.py']:
            shutil.copy2(root / f, folder / f)
        if args.notebook:
            if not args.notebook.is_file(): ap.error(f'Notebook not found: {args.notebook}')
            shutil.copy2(args.notebook, folder / 'original_training_notebook.ipynb')
        for i, m in enumerate(spec['models']):
            src = Path(m['checkpoint_source'])
            if not src.is_file(): ap.error(f'Weights not found: {src}')
            arch = m['arch']
            model = get_model(arch, pretrained=False)
            model.load_state_dict(torch.load(src, map_location='cpu', weights_only=True))
            total += parameter_count(model)
            if total > 20_000_000: ap.error(f'Parameter limit exceeded: {total:,}')
            rel = f'models/{i}_{arch}.pt'
            shutil.copy2(src, folder / rel)
            m['checkpoint'] = rel
            del m['checkpoint_source']
        (folder / 'submission_config.json').write_text(
            json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        (folder / 'README.txt').write_text(
            'Offline CPU reproduction:\n'
            'python predict.py --test_dir PATH_TO_TEST --output submission.csv\n\n'
            'Model architecture is in model.py, all weights are in models/,\n'
            'fine-tuning script is train.py. If provided, original training\n'
            'notebook is included. Requires torch, torchvision, numpy, pandas, PIL.\n'
            'No downloads during inference. Class 6e=0, it=1.\n', encoding='utf-8')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for f in folder.rglob('*'):
                if f.is_file(): z.write(f, f.relative_to(folder))
        print(f'Created {args.output} size={args.output.stat().st_size/1024**2:.1f} MiB; params={total:,}')
        assert args.output.stat().st_size <= 200 * 1024**2
        # Always run from freshly extracted archive (not developer working directory).
        verify = Path(tmp) / 'verify'
        with zipfile.ZipFile(args.output) as z: z.extractall(verify)
        tmp_csv = Path(tmp) / 'reproduced.csv'
        subprocess.run([sys.executable, str(verify / 'predict.py'),
                        '--test_dir', str(args.test_dir.resolve()),
                        '--output', str(tmp_csv)], check=True)
        df = pd.read_csv(tmp_csv, dtype={'id': str})
        assert df.shape == (500, 2) and df.columns.tolist() == ['id', 'prob']
        assert df.id.is_unique and df.prob.notna().all() and df.prob.between(0, 1).all()
        testids = {x.stem for x in args.test_dir.glob('*.jpg')}
        assert set(df.id.astype(str)) == testids
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_csv, args.csv)
        print('Verified independent ZIP extraction -> CPU prediction -> CSV:', args.csv)
        print('IMPORTANT: Also measure 500-image CPU time on Google Colab, the official environment.')


if __name__ == '__main__':
    main()
