# EXECUTING THIS FILE WILL DOWNLOAD THE DATASET CORRESPONDING TO THE FOLLOW PUBLICATION
# TAKEN FROM SLEEP:
# @article{10.1093/sleep/zsz180,
#     author = {Walch, Olivia and Huang, Yitong and Forger, Daniel and Goldstein, Cathy},
#     title = {Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device},
#     journal = {Sleep},
#     volume = {42},
#     number = {12},
#     pages = {zsz180},
#     year = {2019},
#     month = {08},
#     abstract = {Wearable, multisensor, consumer devices that estimate sleep are now commonplace, but the algorithms used by these devices to score sleep are not open source, and the raw sensor data is rarely accessible for external use. As a result, these devices are limited in their usefulness for clinical and research applications, despite holding much promise. We used a mobile application of our own creation to collect raw acceleration data and heart rate from the Apple Watch worn by participants undergoing polysomnography, as well as during the ambulatory period preceding in lab testing. Using this data, we compared the contributions of multiple features (motion, local standard deviation in heart rate, and “clock proxy”) to performance across several classifiers. Best performance was achieved using neural nets, though the differences across classifiers were generally small. For sleep-wake classification, our method scored 90\% of epochs correctly, with 59.6\% of true wake epochs (specificity) and 93\% of true sleep epochs (sensitivity) scored correctly. Accuracy for differentiating wake, NREM sleep, and REM sleep was approximately 72\% when all features were used. We generalized our results by testing the models trained on Apple Watch data using data from the Multi-ethnic Study of Atherosclerosis (MESA), and found that we were able to predict sleep with performance comparable to testing on our own dataset. This study demonstrates, for the first time, the ability to analyze raw acceleration and heart rate data from a ubiquitous wearable device with accepted, disclosed mathematical methods to improve accuracy of sleep and sleep stage prediction.},
#     issn = {0161-8105},
#     doi = {10.1093/sleep/zsz180},
#     url = {https://doi.org/10.1093/sleep/zsz180},
#     eprint = {https://academic.oup.com/sleep/article-pdf/42/12/zsz180/31613664/zsz180.pdf},
# }

# DOWNLOADING EVERYTHING FROM PHYSIONET WILL TAKE A WHILE (MIGHT BE AN HOUR OR TWO)

import polars as pl
import numpy as np
from scipy import interpolate
from helpers.data_loader import get_base_paths

import glob
import os
import subprocess
import shutil


def get_aw_subjects_raw():
    wristbcg_path, oliviawalch_path = get_base_paths(raw=True)

    subjects = [p.split('heart_rate')[1].split('_')[0] for p in glob.glob(os.path.join(oliviawalch_path, 'heart_rate', '*.txt'))]
    subjects = [p.split('\\')[-1] for p in subjects]
    subjects = [p.split('/')[-1] for p in subjects]

    return subjects

def process_aw_subject_raw(subject_id, oliviawalch_path, window_length=180, overlap=120, sample_window_lenth_factor=0.99, max_acc_sampling_gap = 2.5/50, max_hr_sampling_gap = 10):
    
    hr = pl.read_csv(os.path.join(oliviawalch_path, 'heart_rate', f'{subject_id}_heartrate.txt'), has_header=False, separator=',', new_columns=['time', 'hr'], schema = {'time':pl.Float64, 'hr':pl.Float64})
    acc = pl.read_csv(os.path.join(oliviawalch_path, 'motion', f'{subject_id}_acceleration.txt'), has_header=False, separator=' ', new_columns=['time', 'x', 'y', 'z'], schema = {'time':pl.Float64, 'x':pl.Float64, 'y':pl.Float64, 'z':pl.Float64})

    hr = hr.filter((pl.col('time') > min(acc['time'])) & (pl.col('time') < max(acc['time'])))
    acc = acc.filter((pl.col('time') > min(hr['time'])) & (pl.col('time') < max(hr['time'])))
    
    start_time = max(min(hr['time']), min(acc['time']))
    
    acc_x = []
    acc_y = []
    acc_z = []
    acc_mag = []
    hr_vals = []
    hr_times = []
    start_times = []

    while start_time + window_length < max(hr['time']):
        hr_window = hr.filter((pl.col('time') > start_time) & (pl.col('time') < start_time + window_length))
        acc_window = acc.filter((pl.col('time') > start_time) & (pl.col('time') < start_time + window_length))

        len_acc_window = len(acc_window)
        len_hr_window = len(hr_window)

        acc_window_diffs = np.array(acc_window['time'].diff().to_numpy())
        hr_window_diffs = np.array(hr_window['time'].diff().to_numpy())

        if np.sum((acc_window_diffs > max_acc_sampling_gap)) == 0 and np.sum((hr_window_diffs > max_hr_sampling_gap)) == 0 and len_acc_window > window_length * 50 * sample_window_lenth_factor:

            t = acc_window['time'].to_numpy()
            
            t = t - t[0]
            # sample to 100Hz
            t_new = np.arange(t[0], t[-1], 0.01)
            
            x = acc_window['x'].to_numpy().reshape(-1)
            y = acc_window['y'].to_numpy().reshape(-1)
            z = acc_window['z'].to_numpy().reshape(-1)
            
            x = interpolate.interp1d(t, x)
            x = x(t_new)
            y = interpolate.interp1d(t, y)
            y = y(t_new)
            z = interpolate.interp1d(t, z)
            z = z(t_new)
            
            mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            
            start_times.append(start_time)
            acc_x.append(x)
            acc_y.append(y)
            acc_z.append(z)
            acc_mag.append(mag)
            hr_times.append(hr_window['time'].to_numpy()-start_time)
            hr_vals.append(hr_window['hr'].to_numpy())
            
        start_time += window_length - overlap
    
    print(f'Checked {len(start_times)} windows')

    df = pl.DataFrame({
        'pid': subject_id,
        'start': start_times,
        'acc_a_x': acc_x,
        'acc_a_y': acc_y,
        'acc_a_z': acc_z,
        'acc_a_mag': acc_mag,
        'hr_times': hr_times,
        'hr_vals': hr_vals
    }, schema = {
        'pid': pl.Int64,
        'start': pl.Float64,
        'acc_a_x': pl.List(pl.Float64),
        'acc_a_y': pl.List(pl.Float64),
        'acc_a_z': pl.List(pl.Float64),
        'acc_a_mag': pl.List(pl.Float64),
        'hr_times': pl.List(pl.Int64),
        'hr_vals': pl.List(pl.Int64)
    })

    if len(df) > 0:
        df.write_parquet(os.path.join(get_base_paths()[1], f'{subject_id}.parquet'))
    else:
        print(f'No data for {subject_id}')


def download_ojwalch_dataset():

    # check that the aw/motion/ and aw/heart_rate/ folders have the necessary number of files

    if len(glob.glob(os.path.join(get_base_paths(raw=True)[1], 'motion', '*.txt'))) == 31 and len(glob.glob(os.path.join(get_base_paths(raw=True)[1], 'heart_rate', '*.txt'))) == 31:

        return f"dataset has already been downloaded, if you want to download it again rename the folder {os.path.join(get_base_paths()[1], 'physionet')}"

    else:
        url = "https://physionet.org/files/sleep-accel/1.0.0/"
        
        # see https://physionet.org/content/sleep-accel/1.0.0/ for instructions by Phsyionet if you want to download manually
        result = subprocess.run(['wget', '-r', '-N', '-c', url, f'--directory-prefix={get_base_paths(raw=True)[1]}'])
        print(f"Donloaded into: {result}")

        heart_rate_files = glob.glob(os.path.join(get_base_paths(raw=True)[1], 'physionet.org', 'files', 'sleep-accel', '1.0.0', 'heart_rate', '*'))
        motion_files = glob.glob(os.path.join(get_base_paths(raw=True)[1], 'physionet.org', 'files', 'sleep-accel', '1.0.0', 'motion', '*'))

        os.mkdir(os.path.join(get_base_paths(raw=True)[1], 'heart_rate'))
        os.mkdir(os.path.join(get_base_paths(raw=True)[1], 'motion'))

        for p in heart_rate_files:
            new_p = os.path.join(get_base_paths(raw=True)[1], 'heart_rate' ,os.path.basename(p))
            os.rename(p, new_p)

        for p in motion_files:
            new_p = os.path.join(get_base_paths(raw=True)[1], 'motion' ,os.path.basename(p))
            os.rename(p, new_p)

        return 'downloaded the AW dataset from physionet.org'

def moving_test():
    heart_rate_files = glob.glob(os.path.join(get_base_paths(raw=True)[1], 'physionet.org', 'files', 'sleep-accel', '1.0.0', 'heart_rate', '*'))
    motion_files = glob.glob(os.path.join(get_base_paths(raw=True)[1], 'physionet.org', 'files', 'sleep-accel', '1.0.0', 'motion', '*'))

    os.mkdir(os.path.join(get_base_paths(raw=True)[1], 'heart_rate'))
    os.mkdir(os.path.join(get_base_paths(raw=True)[1], 'motion'))

    for p in heart_rate_files:
        new_p = os.path.join(get_base_paths(raw=True)[1], 'heart_rate' ,os.path.basename(p))
        os.rename(p, new_p)

    for p in motion_files:
        new_p = os.path.join(get_base_paths(raw=True)[1], 'motion' ,os.path.basename(p))
        os.rename(p, new_p)

    return 'moved all files'

def process_raw_aw_dataset():

    print(download_ojwalch_dataset())

    subjects = get_aw_subjects_raw()

    print(subjects)

    ojwalch_path = get_base_paths(raw=True)[1]

    for s in subjects:
        print(f'processing aw subject {s}...', end = ' ')
        process_aw_subject_raw(s, ojwalch_path)

    print(f"YOU CAN DELETE THE DIRECTORIES {os.path.join(get_base_paths(raw=True)[1], 'motion')} and {os.path.join(get_base_paths(raw=True)[1], 'heart_rate')} now if you want")

    return f'processed all AW files, see directory {get_base_paths()[1]}'

if __name__ == '__main__':

    process_raw_aw_dataset()