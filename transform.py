import argparse
import multiprocessing
from functools import partial

from helpers.data_loader import get_aw_subjects, get_wbcg_subjects
from BCGAlgorithms.pwr import pwr_load_and_process
from BCGAlgorithms.jerks import jerks_load_and_process
from BCGAlgorithms.bioinsights import bioinsights_load_and_process
from BCGAlgorithms.nightbeat import nightbeat_load_and_process

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# you'll have to adapt this to your system; if you have no GPU, leave it as an empty list
# if you have GPUs, e.g.: ['cuda:0', 'cuda:1', ...]
possible_gpus = ['cuda:0']


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


def run_nightbeat_transform(datasets, workers, rows):
    print("Nightbeat transform")
    for dataset in datasets:
        subjects = load_subjects(dataset)
        n = len(subjects)
        if n == 0:
            continue

        if workers == 1:
            for i in range(n):
                nightbeat_load_and_process(
                    i,
                    dataset=dataset,
                    rows=rows,
                    all_subjects=subjects,
                )
        else:
            with multiprocessing.Pool(workers) as p:
                p.map(
                    partial(
                        nightbeat_load_and_process,
                        dataset=dataset,
                        rows=rows,
                        all_subjects=subjects,
                    ),
                    range(n),
                )


def run_pwr_transform(datasets, workers, rows):
    print("PWR transform")
    for dataset in datasets:
        subjects = load_subjects(dataset)
        n = len(subjects)
        if n == 0:
            continue

        if workers == 1:
            for i in range(n):
                pwr_load_and_process(
                    i,
                    dataset=dataset,
                    rows=rows,
                    all_subjects=subjects,
                    available_devices=possible_gpus,
                )
        else:
            with multiprocessing.Pool(workers) as p:
                p.map(
                    partial(
                        pwr_load_and_process,
                        dataset=dataset,
                        rows=rows,
                        all_subjects=subjects,
                        available_devices=possible_gpus,
                    ),
                    range(n),
                )


def run_jerks_transform(datasets, workers, rows):
    print("JERKS transform")
    for dataset in datasets:
        subjects = load_subjects(dataset)
        n = len(subjects)
        if n == 0:
            continue

        if workers == 1:
            for i in range(n):
                jerks_load_and_process(
                    i,
                    dataset=dataset,
                    rows=rows,
                    all_subjects=subjects,
                )
        else:
            with multiprocessing.Pool(workers) as p:
                p.map(
                    partial(
                        jerks_load_and_process,
                        dataset=dataset,
                        rows=rows,
                        all_subjects=subjects,
                    ),
                    range(n),
                )


def run_bioinsights_transform(datasets, workers, rows):
    print("BIOINSIGHTS transform")
    for dataset in datasets:
        subjects = load_subjects(dataset)
        n = len(subjects)
        if n == 0:
            continue

        if workers == 1:
            for i in range(n):
                bioinsights_load_and_process(
                    i,
                    dataset=dataset,
                    rows=rows,
                    all_subjects=subjects,
                )
        else:
            with multiprocessing.Pool(workers) as p:
                p.map(
                    partial(
                        bioinsights_load_and_process,
                        dataset=dataset,
                        rows=rows,
                        all_subjects=subjects,
                    ),
                    range(n),
                )


def main(datasets=None, algorithm='nightbeat', workers=1, rows=1_000_000):
    if datasets is None:
        datasets = ['aw', 'nightbeatdb']

    validate_datasets(datasets)
    validate_algorithm(algorithm)
    assert workers >= 1, "workers must be >= 1"

    if algorithm in ('nightbeat', 'all'):
        run_nightbeat_transform(datasets, workers, rows)
    if algorithm in ('pwr', 'all'):
        run_pwr_transform(datasets, workers, rows)
    if algorithm in ('jerks', 'all'):
        run_jerks_transform(datasets, workers, rows)
    if algorithm in ('bioinsights', 'all'):
        run_bioinsights_transform(datasets, workers, rows)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Transform datasets (feature extraction / preprocessing).')
    argparser.add_argument('--datasets', nargs='+', default=['nightbeatdb', 'aw'],
                           help='Datasets to transform (aw, nightbeatdb)')
    argparser.add_argument('--algorithm', default='nightbeat',
                           help='Algorithm: nightbeat, pwr, jerks, bioinsights, all')
    argparser.add_argument('--workers', default=1, type=int,
                           help='Number of worker processes (1 = no multiprocessing)')
    argparser.add_argument('--rows', default=1_000_000, type=int,
                           help='Max number of rows to process')
    args = argparser.parse_args()

    main(
        datasets=args.datasets,
        algorithm=args.algorithm,
        workers=args.workers,
        rows=args.rows,
    )
