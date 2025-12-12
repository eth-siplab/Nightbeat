import numpy as np
import os
import polars as pl
import glob
from pathlib import Path

path_dict = {
    'wristbcg': 0,
    'aw': 1
    }

def get_repo_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_base_paths(raw=False, transformed=False, results=False):
    # Use pathlib to get the repo root (2 levels up from helpers/data_loader.py)
    repo_root = Path(__file__).resolve().parent.parent
    data_root = repo_root / 'data'

    if raw:
        base_dir = data_root / 'raw'
    elif transformed:
        base_dir = data_root / 'transformed'
    elif results:
        base_dir = data_root / 'results'
    else:
        base_dir = data_root / 'aligned'

    nightbeatdb_path = base_dir / 'nightbeatdb'
    oliviawalch_path = base_dir / 'aw'

    # Ensure directories exist
    os.makedirs(nightbeatdb_path, exist_ok=True)
    os.makedirs(oliviawalch_path, exist_ok=True)

    return str(nightbeatdb_path), str(oliviawalch_path)

wristbcg_path, oliviawalch_path = get_base_paths()

def load_test_data_nightbeatdb(subject_id:int = 0, base_folder:str = wristbcg_path, windows:int = 5):
    df = pl.read_parquet(os.path.join(base_folder, f'{int(subject_id):02d}.parquet'))
    df = df[0:windows]
    return df

def load_test_data_oliviawalch(subject_id:int = 1360686, base_folder:str = oliviawalch_path, windows:int = 5):
    df = pl.read_parquet(os.path.join(base_folder, f'{int(subject_id):02d}.parquet'))
    df = df[0:windows]
    return df

def load_transformed_data(subject_id=0, dataset='nightbeatdb', method='bioinsights', length = ''):
    assert dataset in ['nightbeatdb', 'aw'], 'Dataset not recognized'

    if isinstance(subject_id, str) and 'min_' in subject_id:
        subject_id = subject_id.split('_')[1]

    # Standardize naming convention logic
    prefix = method
    
    if dataset == 'nightbeatdb':
        load_path = os.path.join(get_base_paths(transformed=True)[0], f'{prefix}_{dataset}_{int(subject_id):02d}.parquet')
    elif dataset == 'aw':
        load_path = os.path.join(get_base_paths(transformed=True)[1], f'{prefix}_{dataset}_{int(subject_id):02d}.parquet')
        
    print(f'Loading {load_path}')
    df = pl.read_parquet(load_path)
    return df

def get_aw_subjects(processed_path=None):
    if processed_path is None:
        processed_path = get_base_paths()[1]
    subjects_aw = glob.glob(os.path.join(processed_path, '*.parquet'))
    subjects_aw = [os.path.basename(subject).split('.')[0] for subject in subjects_aw]
    return subjects_aw

def get_wbcg_subjects(processed_path=None):
    if processed_path is None:
        processed_path = get_base_paths()[0]
    subjects_wbcg = glob.glob(os.path.join(processed_path, '*.parquet'))
    subjects_wbcg = [os.path.basename(subject).split('.')[0] for subject in subjects_wbcg]
    return subjects_wbcg