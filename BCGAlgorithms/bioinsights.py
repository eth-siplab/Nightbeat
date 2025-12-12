# Here we implement the Bioinsights algorithm as described in:
'''
Hernandez, Javier, Daniel J. McDuff, and Rosalind W. Picard.
"BioInsights: Extracting personal data from “Still” wearable motion sensors."
2015 IEEE 12th International Conference on Wearable and Implantable Body Sensor Networks (BSN). IEEE, 2015.
'''
# This is a Python translation based on the approach described in the paper.

import polars as pl
import numpy as np
import os
from scipy.signal import convolve, sosfiltfilt

from BCGAlgorithms.shared_functionalities import STFT_from_window, get_hr_label, predict_from_stft, get_butter_sos, STFT_FLOAT_TYPE
from BCGAlgorithms.processor import BCGProcessor
from helpers.data_loader import load_test_data_nightbeatdb, load_test_data_oliviawalch, get_repo_path, load_transformed_data, get_base_paths

def bioinsights_detrend(signal, fs, detrend_window_size=None):
    if not detrend_window_size:
        detrend_window_size = int(3 * fs)

    signal_detrend = signal - convolve(signal, np.ones(detrend_window_size) / detrend_window_size, 'same')
    return signal_detrend

def bioinsights_process_window(row, fs=100, nperseg=2**10, noverlap=2**10-20, padding_factor=10, sos_bp=None, sos_mag=None):
    acc_x = np.array(row['acc_a_x'])
    acc_y = np.array(row['acc_a_y'])
    acc_z = np.array(row['acc_a_z'])

    length = len(acc_x) - len(acc_x) % fs

    acc_x = acc_x[:length]
    acc_y = acc_y[:length]
    acc_z = acc_z[:length]
    
    acc_x = bioinsights_detrend(acc_x, fs)
    acc_y = bioinsights_detrend(acc_y, fs)
    acc_z = bioinsights_detrend(acc_z, fs)

    # Use pre-calculated SOS filters for Bandpass 10-13 Hz
    acc_x = sosfiltfilt(sos_bp, acc_x)
    acc_y = sosfiltfilt(sos_bp, acc_y)
    acc_z = sosfiltfilt(sos_bp, acc_z)

    mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Use pre-calculated SOS filters for Magnitude Bandpass (0.5-3.0 Hz)
    mag = sosfiltfilt(sos_mag, mag)

    magnitude, extent = STFT_from_window(mag, fs, nperseg=nperseg, noverlap=noverlap, padding_factor=padding_factor)

    return {
        'magnitude': magnitude,
        'extent': extent,
        'acc_mag': mag
    }

def bioinsights_load_and_process(subject, all_subjects=[], dataset='nightbeatdb', rows=1000):
    fs = 100
    
    # Pre-calculate filters
    # 1. Bandpass 10-13 Hz
    sos_bp = get_butter_sos(10, 13, fs, order=4)
    # 2. Bandpass 0.5-3.0 Hz
    sos_mag = get_butter_sos(0.5, 3.0, fs, order=4)

    output_schema = {
        'row_idx': pl.Int64,
        'time': pl.Float64,
        'magnitude': pl.Array(pl.Array(STFT_FLOAT_TYPE, 946), 359),
        'extent': pl.List(pl.Float32),
        'acc_mag': pl.List(pl.Float64)
    }

    processor = BCGProcessor(
        algo_name='bioinsights',
        process_window_func=bioinsights_process_window,
        output_schema=output_schema,
        fs=fs
    )

    return processor.load_and_process(
        subject, all_subjects, dataset, rows,
        sos_bp=sos_bp, sos_mag=sos_mag
    )

def bioinsights_predict_whole_window(row, fs, win_size, overlap=0.5, debug = False):

    assert win_size in [20, 30, 60], 'Window size not recognized'

    times_l = []
    preds_l = []
    labels_l = []

    start_sec = 60

    while start_sec < 120:

        end_sec = start_sec + win_size

        if debug:
            print(f"Predicting window {start_sec} to {end_sec}")

        magnitude = np.array(row['magnitude'][0])
        extent = row['extent'][0]
        t = np.linspace(extent[0], extent[1], magnitude.shape[1])
        f = np.linspace(extent[2], extent[3], magnitude.shape[0])

        t_index = np.where((t >= start_sec) & (t < end_sec))[0]

        label = get_hr_label(row, start_sec, end_sec)

        pred = predict_from_stft(magnitude[:, t_index], f, fs, debug, specific_algorithm='bioinsights')

        time = row['time'][0] + start_sec

        times_l.append(time)
        preds_l.append(pred)
        labels_l.append(label)

        start_sec += int(win_size * (1 - overlap))

    return times_l, preds_l, labels_l

def bioinsights_predict_df(df, fs=100, win_size=20, overlap=0.5):

    row_idxs_l = []
    times_l = []
    preds_l = []
    labels_l = []

    for i in range(len(df)):

        row = df[i]
        row_idxs = row['row_idx'][0]

        times, preds, labels = bioinsights_predict_whole_window(row, fs, win_size, overlap)

        row_idxs_l.extend([row_idxs] * len(times))
        times_l.extend(times)
        preds_l.extend(preds)
        labels_l.extend(labels)

    df = pl.DataFrame({
        'row_idx': row_idxs_l,
        'time': times_l,
        'pred': preds_l,
        'label': labels_l
    }, schema={
        'row_idx': pl.Int64,
        'time': pl.Float64,
        'pred': pl.Float64,
        'label': pl.Float64
    })

    return df

def bioinsights_load_and_predict(subject, all_subjects = [], dataset='nightbeatdb', win_size = 30):

    df = load_transformed_data(subject_id=all_subjects[subject], method='bioinsights', dataset=dataset)

    if dataset == 'nightbeatdb':
        df_orig = load_test_data_nightbeatdb(subject_id=all_subjects[subject], windows=1000)
    elif dataset == 'aw':
        df_orig = load_test_data_oliviawalch(subject_id=all_subjects[subject], windows=1000)

    df_orig = df_orig.with_row_index(name='row_idx').with_columns(pl.col('row_idx').cast(pl.Int64).alias('row_idx'))

    if 'start_100Hz' in df_orig.columns:
        df_orig = df_orig.with_columns(pl.col('start_100Hz').cast(pl.Float64).alias('time'))
    elif 'start' in df_orig.columns:
        df_orig = df_orig.with_columns(pl.col('start').cast(pl.Float64).alias('time'))

    df = df.join(df_orig, on=['row_idx', 'time'], how='inner', coalesce=True)

    if 'start_100Hz' in df.columns:
        df = df.filter(pl.col('hrv_is_valid').arr.get(1024 * 90) == 1)

    print(f"Predicting subject {all_subjects[subject]} in dataset {dataset}")

    df = bioinsights_predict_df(df, win_size=win_size)

    base_path = get_base_paths(results=True)[0 if dataset == 'nightbeatdb' else 1]
    out_name = f'bioinsights_{dataset}_{int(all_subjects[subject]):02d}_preds{win_size}.parquet'
    
    print(f"Saving in {os.path.join(base_path, out_name)}")
    df.write_parquet(os.path.join(base_path, out_name))

    return 'saved'

if __name__ == '__main__':
    bioinsights_load_and_predict(0, all_subjects=range(31), dataset='wristbcg', win_size=30)
    bioinsights_load_and_predict(0, all_subjects=range(31), dataset='wristbcg', win_size=20)
    bioinsights_load_and_predict(0, all_subjects=range(31), dataset='wristbcg', win_size=60)