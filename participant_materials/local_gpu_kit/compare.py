"""Compare models and fixed-weight ensembles on the *same* held-out split.
Validation is separate from training, but multiple hyperparameter attempts can still overfit it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from model import get_model, get_transform, parameter_count


def infer_validation(train_dir, arch, size, checkpoint, batch_size, device):
    ds = ImageFolder(str(train_dir), transform=get_transform(size, train=False))
    assert ds.class_to_idx == {'6e': 0, 'it': 1} and len(ds) == 2906
    idx = np.arange(len(ds))
    y = np.array(ds.targets)
    _, val_idx = train_test_split(idx, test_size=.2, random_state=42, stratify=y)
    val = Subset(ds, val_idx)
    loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    model = get_model(arch, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    nparam = parameter_count(model)
    model.eval().to(device)
    originals, flips, labels = [], [], []
    with torch.inference_mode():
        for image, label in loader:
            image = image.to(device)
            logits = model(image)
            logits_flip = model(torch.flip(image, dims=[3]))
            originals.append(logits.cpu())
            flips.append(logits_flip.cpu())
            labels.append(label)
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    return torch.cat(originals), torch.cat(flips), torch.cat(labels), nparam


def fit_temperature(logits, labels):
    from torch.nn import functional as F
    def cross_entropy(temp):
        return F.cross_entropy(logits / float(temp), labels).item()
    result = minimize_scalar(cross_entropy, bounds=(0.5, 5.0), method='bounded')
    return float(result.x) if result.fun < cross_entropy(1.0) else 1.0


def probs(logits, flip_logits, temperature):
    p = torch.softmax(logits / temperature, 1)[:, 1].numpy()
    pf = torch.softmax(flip_logits / temperature, 1)[:, 1].numpy()
    return p.astype('float64'), ((p + pf) / 2).astype('float64')


def score(name, p, labels, models, registry, out):
    from sklearn.metrics import log_loss
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = log_loss(labels, p, labels=[0, 1])
    acc = ((p >= .5) == labels).mean()
    count = sum(registry[x]['params'] for x, w in models if w > 0)
    if count > 20_000_000:
        print(f'SKIP {name}: too many parameters {count:,}')
        return
    print(f'{name:<25} val_logloss={ll:.5f} val_acc={acc:.2%} params={count:,}')
    out.append(dict(name=name, val_logloss=ll, val_accuracy=acc, params=count))
    spec = {'models': [{
        'arch': registry[key]['arch'],
        'size': registry[key]['size'],
        'checkpoint_source': str(registry[key]['path'].resolve()),
        'temperature': registry[key]['temp'],
        'tta': registry[key]['tta'],
        'weight': weight
    } for key, weight in models]}
    (OUTPUT / f'{name}.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('..'))
    ap.add_argument('--baseline', type=Path, required=True, help='Original Colab model.pt')
    ap.add_argument('--b0-288', type=Path, help='runs/b0_288/best_model.pt')
    ap.add_argument('--b2', type=Path, help='runs/b2_288/best_model.pt')
    ap.add_argument('--output', type=Path, default=Path('candidates'))
    ap.add_argument('--batch-size', type=int, default=12)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = ap.parse_args()
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                       'cpu' if args.device == 'auto' else args.device)
    paths = {'baseline': (args.baseline, 'b0', 224),
             'b0_288': (args.b0_288, 'b0', 288),
             'b2_288': (args.b2, 'b2', 288)}
    registry = {}
    records = []
    label_vector = None
    for key, (path, arch, size) in paths.items():
        if path is None:
            continue
        if not path.is_file():
            ap.error(f'Missing checkpoint: {path}')
        logits, flips, labels, params = infer_validation(
            args.data / 'train', arch, size, path, args.batch_size, dev)
        ys = labels.numpy()
        if label_vector is None:
            label_vector = ys
        else:
            assert np.array_equal(label_vector, ys), 'Validation splits differ!'
        temp = 1.032436635407596 if key == 'baseline' else fit_temperature(logits, labels)
        original, tta = probs(logits, flips, temp)
        registry[key] = dict(path=path, arch=arch, size=size, temp=temp,
                             params=params, tta=False, original=original, flipped=tta)
        print(f'{key} temperature={temp:.5f}')
        score(key + '_original', original, ys, [(key, 1.)], registry, records)
        registry[key]['tta'] = True
        score(key + '_tta', tta, ys, [(key, 1.)], registry, records)
    # Fixed 50/50 combinations to avoid tuning dozens of weights against the small val set.
    for a, b in [('baseline', 'b0_288'), ('baseline', 'b2_288'), ('b0_288', 'b2_288')]:
        if a in registry and b in registry:
            rega, regb = registry[a], registry[b]
            pa, pb = rega['flipped'], regb['flipped']
            rega['tta'] = True
            regb['tta'] = True
            score(f'ensemble_{a}_{b}', (pa + pb) / 2, label_vector,
                  [(a, .5), (b, .5)], registry, records)
            # Cheaper CPU option: TTA on the first model, original on the second.
            regb['tta'] = False
            score(f'ensemble_{a}_tta_{b}_orig', (pa + regb['original']) / 2,
                  label_vector, [(a, .5), (b, .5)], registry, records)
            regb['tta'] = True
    table = pd.DataFrame(records).sort_values('val_logloss')
    table.to_csv(OUTPUT / 'validation_comparison.csv', index=False)
    print('\nBest candidates:\n', table.to_string(index=False))
    print('\nConfig files saved to:', OUTPUT.resolve())
    print('CAUTION: the same val split was used for checkpoint choice and temperature calibration.')
