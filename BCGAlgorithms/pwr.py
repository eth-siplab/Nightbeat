# This file implements the "pwr" algorithm as described in:
'''
Zschocke, Johannes, et al.
"Reconstruction of pulse wave and respiration from wrist accelerometer during sleep."
IEEE Transactions on Biomedical Engineering 69.2 (2021): 830-839.
'''
# A matlab implementation was provided by the authors, this is a Python translation based on the app

import polars as pl
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, find_peaks
import torch
from datetime import datetime
import os

from BCGAlgorithms.shared_functionalities import STFT_from_window, autocorrelation_torch, predict_from_stft, get_hr_label, rolling_window, STFT_FLOAT_TYPE
from BCGAlgorithms.processor import BCGProcessor
from helpers.data_loader import load_test_data_nightbeatdb, load_test_data_oliviawalch, get_base_paths, load_transformed_data

def bandpass_filter_optimized(data, b, a):
    return filtfilt(b, a, data)

def process_acceleration_data_opt(acceleration_data, b, a, fs=100):
    if not isinstance(acceleration_data, np.ndarray):
        acceleration_data = np.array(acceleration_data)

    # Optimized Rolling Mean using Convolution
    win_size = fs
    kernel = np.ones(win_size) / win_size
    # mode='same' handles padding automatically compared to manual approach
    rolling_mean = np.convolve(acceleration_data, kernel, mode='same')
    acceleration_data = acceleration_data - rolling_mean

    # Bandpass using precomputed coeffs
    filtered_data = bandpass_filter_optimized(acceleration_data, b, a)

    # Hilbert Transform
    analytic_signal = hilbert(filtered_data)
    amplitude_envelope = np.abs(analytic_signal)

    return filtered_data, amplitude_envelope

def compute_accelerometer_roll(axis, z_axis):
    angle = np.arctan2(axis, z_axis)
    return angle

def compute_accelerometer_pitch(axis, magnitude):
    angle = np.arctan2(axis, magnitude)
    return angle

def detect_peaks(axis, min_distance=0.5, threshold=0.5, fs=100, smoothing=1, plot=False, use_envelope=False):
    if use_envelope:
        # get rolling maximum of axis
        envelope = np.max(rolling_window(axis, int(smoothing * fs)), axis=1)
        pad_diff = len(axis) - len(envelope)
        envelope = np.pad(envelope, (pad_diff // 2, pad_diff // 2 + pad_diff % 2), 'edge')

        # get rolling mean of envelope
        envelope = np.mean(rolling_window(envelope, int(smoothing / 3 * 2 * fs)), axis=1)
        pad_diff = len(axis) - len(envelope)
        envelope = np.pad(envelope, (pad_diff // 2, pad_diff // 2 + pad_diff % 2), 'edge')

        threshold = threshold * envelope
    else:
        threshold = np.percentile(axis, 100 * threshold) # method='inverted_cdf' not strictly needed
        envelope = np.repeat(threshold, len(axis))

    peaks, _ = find_peaks(axis, distance=min_distance*fs, height=threshold)
    return peaks

def select_best_axis_torch(axes_dict, fs=100, min_hz=0.5, max_hz=3.5, device='cpu'):
    # Efficiently select the axis with highest autocorrelation in range
    max_idx = int(1 / max_hz * fs)
    min_idx = int(1 / min_hz * fs)
    
    best_corr = -1
    best_key = None
    
    # Can process sequentially or batch if tensor construction overhead is low
    for key, data in axes_dict.items():
        ac = autocorrelation_torch(data, device=device)
        # Check max in range
        local_max = ac[max_idx:min_idx].max().item()
        
        if local_max > best_corr:
            best_corr = local_max
            best_key = key
            
    return axes_dict[best_key], best_key

def calculate_pwi(peaks, fs=100):
    pwi = np.diff(peaks) / fs
    return pwi

def pwr_process_window(row, fs, nperseg=2**10, noverlap=2**10-20, padding_factor=10, 
                       min_distance=0.5, threshold=0.5, debug=False, device='cpu', 
                       b_acc=None, a_acc=None, b_main=None, a_main=None):

    acc_x = np.array(row['acc_a_x'])
    acc_y = np.array(row['acc_a_y'])
    acc_z = np.array(row['acc_a_z'])

    # Ensure length is multiple of fs
    trim = len(acc_x) - len(acc_x) % fs
    acc_x, acc_y, acc_z = acc_x[:trim], acc_y[:trim], acc_z[:trim]

    mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

    # Process axes using cached filter coefficients
    _, env_x = process_acceleration_data_opt(acc_x, b_acc, a_acc, fs)
    _, env_y = process_acceleration_data_opt(acc_y, b_acc, a_acc, fs)
    _, env_z = process_acceleration_data_opt(acc_z, b_acc, a_acc, fs)

    roll_x = compute_accelerometer_roll(acc_x, acc_z)
    roll_y = compute_accelerometer_roll(acc_y, acc_z)
    pitch_x = compute_accelerometer_pitch(acc_x, mag)
    pitch_y = compute_accelerometer_pitch(acc_y, mag)

    _, env_rx = process_acceleration_data_opt(roll_x, b_acc, a_acc, fs)
    _, env_ry = process_acceleration_data_opt(roll_y, b_acc, a_acc, fs)
    _, env_px = process_acceleration_data_opt(pitch_x, b_acc, a_acc, fs)
    _, env_py = process_acceleration_data_opt(pitch_y, b_acc, a_acc, fs)

    axes_map = {
        'x': env_x, 'y': env_y, 'z': env_z,
        'roll_x': env_rx, 'roll_y': env_ry,
        'pitch_x': env_px, 'pitch_y': env_py
    }

    # Select best axis
    best_axis_data, axis_name = select_best_axis_torch(axes_map, fs=fs, device=device)

    best_peaks = detect_peaks(best_axis_data, min_distance, threshold, fs=fs)

    # Filter best axis for STFT using main filter coefficients
    best_axis_filtered = bandpass_filter_optimized(best_axis_data, b_main, a_main)

    magnitude, extent = STFT_from_window(best_axis_filtered, fs, nperseg=nperseg, noverlap=noverlap, padding_factor=padding_factor)

    return {
        'magnitude': magnitude,
        'extent': extent,
        'best_axis_filtered': best_axis_filtered,
        'best_peaks': best_peaks,
        'axis': axis_name
    }

def pwr_load_and_process(subject, all_subjects = [], dataset='nightbeatdb', rows=1000, debug=False, available_devices = ['cpu']):
    fs = 100
    nyq = 0.5 * fs
    
    # Pre-calculate filters
    # 1. Accel Bandpass: 5.0 - 14.0 Hz
    b_acc, a_acc = butter(4, [5.0/nyq, 14.0/nyq], btype='band')
    
    # 2. Main Signal Bandpass: 0.5 - 3.5 Hz
    b_main, a_main = butter(4, [0.5/nyq, 3.5/nyq], btype='band')

    device = available_devices[subject % len(available_devices)]

    output_schema = {
        'row_idx': pl.Int64,
        'time': pl.Float64,
        'magnitude': pl.Array(pl.Array(STFT_FLOAT_TYPE, 946), 359),
        'extent': pl.List(pl.Float32),
        'best_axis_filtered': pl.List(pl.Float64),
        'best_peaks': pl.List(pl.Int64),
        'axis': pl.Utf8
    }

    processor = BCGProcessor(
        algo_name='pwr',
        process_window_func=pwr_process_window,
        output_schema=output_schema,
        fs=fs
    )
    
    return processor.load_and_process(
        subject, all_subjects, dataset, rows,
        b_acc=b_acc, a_acc=a_acc, 
        b_main=b_main, a_main=a_main,
        device=device, debug=debug
    )

def pwr_pred_from_peaks(peaks, start_sec, end_sec, fs, debug=False):
    peaks = peaks[(peaks >= start_sec * fs) & (peaks < end_sec * fs)]
    pwi = calculate_pwi(peaks, fs)
    if len(pwi) == 0: return 0
    return 60 / np.mean(pwi)

def pwr_data_quality_metric(acc_mag, rolling_size=100, fs = 100, t=[]):
    # get MAD of the magnitude
    mad = rolling_window(acc_mag, rolling_size)
    mad = np.mean(mad, axis=1)
    length_diff = len(acc_mag) - len(mad)
    if length_diff < 0:
        mad = mad[:length_diff]
        length_diff = 0
    mad = np.pad(mad, (length_diff // 2, length_diff // 2 + length_diff % 2), 'edge')
    return np.abs(acc_mag - mad)

def pwr_predict_whole_window(row, fs, win_size, overlap=0.5, peak_predict = True, debug = False):
    assert win_size in [20, 30, 60], 'Window size not recognized'
    times_l = []
    preds_l = []
    labels_l = []
    quality_l = []

    start_sec = 60
    magnitude = np.array(row['magnitude'][0])
    extent = row['extent'][0]
    t = np.linspace(extent[0], extent[1], magnitude.shape[1])
    f = np.linspace(extent[2], extent[3], magnitude.shape[0])

    sig_length = t[0] + t[-1]

    if sig_length < 170:
        return times_l, preds_l, labels_l, quality_l

    data_quality = pwr_data_quality_metric(np.array(row['acc_a_mag'][0] if 'acc_a_mag' in row else np.zeros_like(t)), int(2*(sig_length - 120 - win_size/2)), t=t)

    while start_sec < 120:
        end_sec = start_sec + win_size
        if debug:
            print(f"Predicting window {start_sec} to {end_sec}")

        t_index = np.where((t >= start_sec) & (t < end_sec))[0]
        label = get_hr_label(row, start_sec, end_sec)

        if peak_predict:
            peaks = np.array(row['best_peaks'][0])
            pred = pwr_pred_from_peaks(peaks, start_sec, end_sec, fs, debug)
        else:
            pred = predict_from_stft(magnitude[:, t_index], f, fs, debug, specific_algorithm='bioinsights')

        win_quality = np.mean(data_quality[start_sec*100:end_sec*100]) / (end_sec-start_sec) if len(data_quality) > end_sec*100 else 0

        times_l.append(row['time'][0] + start_sec)
        preds_l.append(pred)
        labels_l.append(label)
        quality_l.append(win_quality)

        start_sec += int(win_size * (1 - overlap))

    return times_l, preds_l, labels_l, quality_l

def pwr_predict_df(df, fs=100, win_size=20, overlap=0.5, peak_predict=True, debug=False):
    row_idxs_l = []
    times_l = []
    preds_l = []
    labels_l = []
    quality_l = []

    for i in range(len(df)):
        row = df[i]
        row_idxs = row['row_idx'][0]
        times, preds, labels, quality = pwr_predict_whole_window(row, fs, win_size, overlap, peak_predict=peak_predict, debug=debug)

        row_idxs_l.extend([row_idxs] * len(times))
        times_l.extend(times)
        preds_l.extend(preds)
        labels_l.extend(labels)
        quality_l.extend(quality)

    df = pl.DataFrame({
        'row_idx': row_idxs_l,
        'time': times_l,
        'pred': preds_l,
        'label': labels_l,
        'quality': quality_l
    }, schema={
        'row_idx': pl.Int64,
        'time': pl.Float64,
        'pred': pl.Float64,
        'label': pl.Float64,
        'quality': pl.Float64
    })
    return df

def pwr_load_and_predict(subject, all_subjects = [], dataset='nightbeatdb', win_size = 30, peak_predict=True):
    df = load_transformed_data(subject_id=all_subjects[subject], method='pwr', dataset=dataset)

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

    df = pwr_predict_df(df, win_size=win_size, peak_predict=peak_predict)

    base_path = get_base_paths(results=True)[0 if dataset == 'nightbeatdb' else 1]
    out_name = f'pwr_{dataset}_{int(all_subjects[subject]):02d}_preds{win_size}.parquet'
    
    print(f"Saving in {os.path.join(base_path, out_name)}")
    df.write_parquet(os.path.join(base_path, out_name))

    return 'saved'

if __name__ == '__main__':
    pwr_load_and_process(0, dataset='wristbcg', rows=5)
    pwr_load_and_predict(0, [759667], dataset='aw', win_size=60)