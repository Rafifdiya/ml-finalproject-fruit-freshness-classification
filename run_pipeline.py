"""Run full ML pipeline: feature extraction (with augmentation) → training."""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from features import extract_features, extract_features_augmented

DATA_DIR      = Path(__file__).parent / 'data' / 'raw' / 'Dataset' / 'archive' / 'Fruits_Vegetables_Dataset(12000)'
PROCESSED_DIR = Path(__file__).parent / 'data' / 'processed'
SPLIT_CONFIG  = Path(__file__).parent / 'data' / 'split' / 'split_config.json'
PROCESSED_DIR.mkdir(exist_ok=True)

SELECTED_ITEMS = ['Apple', 'Banana', 'Orange', 'Carrot', 'Cucumber', 'Potato', 'Tomato']


def load_split_config():
    with open(SPLIT_CONFIG) as f:
        return json.load(f)


def collect_records():
    records = []
    for category in ['Fruits', 'Vegetables']:
        cat_dir = DATA_DIR / category
        if not cat_dir.exists():
            print(f'WARNING: {cat_dir} not found, skip')
            continue
        for class_folder in sorted(cat_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            name = class_folder.name
            if name.startswith('Fresh'):
                freshness, item = 'Fresh', name[5:]
            elif name.startswith('Rotten'):
                freshness, item = 'Rotten', name[6:]
            else:
                continue
            if item not in SELECTED_ITEMS:
                continue
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    records.append({
                        'path':  str(img_path),
                        'label': 0 if freshness == 'Fresh' else 1,
                    })
    return pd.DataFrame(records)


def extract_split(records, augment, label):
    X, y, failed = [], [], []
    t0 = time.perf_counter()
    total = len(records)
    for i, (_, row) in enumerate(records.iterrows()):
        try:
            if augment:
                feats = extract_features_augmented(row['path'])
            else:
                feats = [extract_features(row['path'])]
            for f in feats:
                X.append(f)
                y.append(row['label'])
        except Exception as e:
            failed.append(row['path'])
        done = i + 1
        if done % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(f'  [{label}] {done}/{total} gambar | {elapsed:.1f}s')
    elapsed = time.perf_counter() - t0
    print(f'  [{label}] selesai {total} gambar → {len(X)} vektor fitur | {elapsed:.1f}s | failed: {len(failed)}')
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def run_extraction():
    print('=== FASE 3: Feature Extraction + Augmentation ===')
    if not DATA_DIR.exists():
        print(f'ERROR: Dataset not found at {DATA_DIR}')
        sys.exit(1)

    df = collect_records()
    print(f'Total gambar: {len(df)} | Fresh: {(df["label"]==0).sum()} | Rotten: {(df["label"]==1).sum()}')

    cfg = load_split_config()
    train_df, test_df = train_test_split(
        df,
        test_size=cfg['test_size'],
        random_state=cfg['random_state'],
        stratify=df['label'] if cfg['stratified'] else None,
    )
    print(f'Split: train={len(train_df)} | test={len(test_df)}')

    # Train: original + 5 augmentations per gambar = 6× data
    print(f'\nExtracting TRAIN dengan augmentasi (6× = ~{len(train_df)*6} vektor)...')
    X_train, y_train = extract_split(train_df, augment=True, label='TRAIN')

    # Test: original only — no augmentation to avoid leakage
    print(f'\nExtracting TEST tanpa augmentasi ({len(test_df)} vektor)...')
    X_test, y_test = extract_split(test_df, augment=False, label='TEST')

    np.save(PROCESSED_DIR / 'X_train.npy', X_train)
    np.save(PROCESSED_DIR / 'y_train.npy', y_train)
    np.save(PROCESSED_DIR / 'X_test.npy',  X_test)
    np.save(PROCESSED_DIR / 'y_test.npy',  y_test)

    print(f'\nX_train: {X_train.shape} ({X_train.nbytes/1024/1024:.1f} MB)')
    print(f'X_test:  {X_test.shape}  ({X_test.nbytes/1024/1024:.1f} MB)')
    print('Saved X_train/y_train/X_test/y_test ke data/processed/')
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = run_extraction()

    print('\n=== FASE 4: Training ===')
    from train import train
    results = train(X_train, y_train, X_test, y_test)
    print('\nDone. Models saved to models/')
    for name, r in results.items():
        print(f"{name:<20} acc={r['metrics']['accuracy']*100:.1f}%  lat={r['latency_ms']:.2f}ms")
