import argparse
import multiprocessing
from functools import partial

from helpers.data_loader import get_aw_subjects, get_wbcg_subjects
from BCGAlgorithms.pwr import pwr_load_and_predict
from BCGAlgorithms.jerks import jerks_load_and_predict
from BCGAlgorithms.bioinsights import bioinsights_load_and_predict
from BCGAlgorithms.nightbeat import nightbeat_load_and_predict

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def validate_datasets(datasets):
    allowed = {'aw', 'nightbeatdb'}
    invalid = [d for d in datasets if d not in allowed]
    assert not invalid, f"Invalid datasets: {invalid}"


def validate_algorithm(algorithm):
    allowed = {'nightbeat', 'pwr', 'jerks', 'bioinsights', 'all'}
    assert algorithm in allowed, f"Invalid algorithm: {algorithm}"


def load_subjects(dataset):
    if dataset == 'aw':
        return get_aw_subjects()
    if dataset == 'nightbeatdb':
        return get_wbcg_subjects()
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_nightbeat_predict(datasets, workers, win_size):
    print("Nightbeat predict")
    for dataset in datasets:
        subjects = load_subjects(dataset)
        n = len(subjects)
        if n == 0:
            continue

        func = partial(
            nightbeat_load_and_predict,
            dataset=dataset,
            all_subjects=subjects,
            win_size=win_size,
            curve_tracing=True,
            motion_artifacts=True,
            peak_detection=True
        )

        if workers == 1:
            for i in range(n):
                func(i)
        else:
            with multiprocessing.Pool(workers) as p:
                p.map(func, range(n))


def run_pwr_predict(datasets, workers, win_size):
    print("PWR predict")
    for dataset in datasets:
        subjects = load_subjects(dataset)
        n = len(subjects)
        if n == 0:
            continue

        func = partial(
            pwr_load_and_predict,
            dataset=dataset,
            all_subjects=subjects,
            win_size=win_size,
        )

        if workers == 1:
            for i in range(n):
                func(i)
        else:
            with multiprocessing.Pool(workers) as p:
                p.map(func, range(n))


def run_jerks_predict(datasets, workers, win_size):
    print("JERKS predict")
    for dataset in datasets:
        subjects = load_subjects(dataset)
        n = len(subjects)
        if n == 0:
            continue

        func = partial(
            jerks_load_and_predict,
            dataset=dataset,
            all_subjects=subjects,
            win_size=win_size,
        )

        if workers == 1:
            for i in range(n):
                func(i)
        else:
            with multiprocessing.Pool(workers) as p:
                p.map(func, range(n))


def run_bioinsights_predict(datasets, workers, win_size):
    print("BIOINSIGHTS predict")
    for dataset in datasets:
        subjects = load_subjects(dataset)
        n = len(subjects)
        if n == 0:
            continue

        func = partial(
            bioinsights_load_and_predict,
            dataset=dataset,
            all_subjects=subjects,
            win_size=win_size,
        )

        if workers == 1:
            for i in range(n):
                func(i)
        else:
            with multiprocessing.Pool(workers) as p:
                p.map(func, range(n))


def main(datasets=None, algorithm='nightbeat', workers=1, win_size=20):
    if datasets is None:
        datasets = ['aw', 'nightbeatdb']
    validate_datasets(datasets)
    validate_algorithm(algorithm)
    assert workers >= 1, "workers must be >= 1"

    if algorithm in ('nightbeat', 'all'):
        run_nightbeat_predict(datasets, workers, win_size)
    if algorithm in ('pwr', 'all'):
        run_pwr_predict(datasets, workers, win_size)
    if algorithm in ('jerks', 'all'):
        run_jerks_predict(datasets, workers, win_size)
    if algorithm in ('bioinsights', 'all'):
        run_bioinsights_predict(datasets, workers, win_size)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Run predictions on transformed datasets.')
    argparser.add_argument('--datasets', nargs='+', default=['aw', 'nightbeatdb'],
                           help='Datasets to predict on (aw, nightbeatdb)')
    argparser.add_argument('--algorithm', default='nightbeat',
                           help='Algorithm: nightbeat, pwr, jerks, bioinsights, all')
    argparser.add_argument('--workers', default=1, type=int,
                           help='Number of worker processes (1 = no multiprocessing)')
    argparser.add_argument('--win_size', default=20, type=int,
                           help='Prediction window size')
    args = argparser.parse_args()
    main(datasets=args.datasets, algorithm=args.algorithm, workers=args.workers, win_size=args.win_size)
