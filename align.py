import argparse
from helpers.ojwalch_prep import process_raw_aw_dataset
from helpers.nightbeatdb_helpers import process_raw_nightbeatdb_dataset

def main(datasets = ['aw', 'nightbeatdb']):

    assert all(d in ['aw', 'nightbeatdb'] for d in datasets), 'Datasets must be from the list: aw, nightbeatdb'

    if 'nightbeatdb' in datasets:
        process_raw_nightbeatdb_dataset()
        
    if 'aw' in datasets:
        process_raw_aw_dataset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process raw datasets into aligned format.')
    parser.add_argument('--datasets', nargs='+', default=['aw', 'nightbeatdb'], help='Datasets to process (options: aw, nightbeatdb)')
    args = parser.parse_args()

    main(datasets=args.datasets)