import polars as pl
import numpy as np
import os
from scipy.stats import iqr

from BCGAlgorithms.shared_functionalities import (
    filter_signal_butter, 
    STFT_from_window, 
    get_hr_label, 
    predict_from_stft,
    get_butter_sos,
    STFT_FLOAT_TYPE
)
from BCGAlgorithms.curve_tracing import get_curves, refine_curves
from BCGAlgorithms.peak_detection import get_narrow_peaks
from BCGAlgorithms.processor import BCGProcessor

from helpers.data_loader import (
    load_test_data_nightbeatdb, 
    load_test_data_oliviawalch, 
    get_base_paths, 
    load_transformed_data
)

import matplotlib.pyplot as plt


def nightbeat_process_window(row, fs=100, nperseg=2 ** 10, noverlap=2 ** 10 - 20, padding_factor=10, sos_main=None, sos_bcg=None):
    """
    Optimized window processing using cached SOS filters.
    """
    acc_x = np.array(row['acc_a_x'])
    acc_y = np.array(row['acc_a_y'])
    acc_z = np.array(row['acc_a_z'])

    length = len(acc_x) - len(acc_x) % fs

    acc_x = acc_x[:length]
    acc_y = acc_y[:length]
    acc_z = acc_z[:length]

    # Use pre-calculated SOS filters passed via kwargs
    acc_x = filter_signal_butter(acc_x, sos=sos_bcg)
    acc_y = filter_signal_butter(acc_y, sos=sos_bcg)
    acc_z = filter_signal_butter(acc_z, sos=sos_bcg)

    acc_a_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    input_signal = filter_signal_butter(acc_a_mag, sos=sos_main)

    magnitude, extent = STFT_from_window(input_signal, fs, nperseg=nperseg, noverlap=noverlap, padding_factor=padding_factor)
    
    return {
        'magnitude': magnitude,
        'extent': extent,
        'acc_mag': input_signal
    }


def nightbeat_load_and_process(subject, all_subjects=[], dataset='nightbeatdb', rows=1000):
    fs = 100
    
    # Pre-calculate filters once per dataset processing
    # Main signal: 0.5 - 3.5 Hz
    sos_main = get_butter_sos(0.5, 3.5, fs, order=4)
    # BCG signal: 5 - 14 Hz
    sos_bcg = get_butter_sos(5, 14, fs, order=5)
    
    # Define the expected output schema
    output_schema = {
        'row_idx': pl.Int64,
        'time': pl.Float64,
        'magnitude': pl.Array(pl.Array(STFT_FLOAT_TYPE, 946), 359), # Dimensions of STFT settings
        'extent': pl.List(pl.Float32),
        'acc_mag': pl.List(pl.Float64)
    }

    # Initialize generic processor
    processor = BCGProcessor(
        algo_name='nightbeat', 
        process_window_func=nightbeat_process_window, 
        output_schema=output_schema,
        fs=fs
    )

    # Run processing, passing filter coefficients as kwargs
    return processor.load_and_process(
        subject, all_subjects, dataset, rows, 
        sos_main=sos_main, sos_bcg=sos_bcg
    )


def nightbeat_motion_artifacts(row):
    magnitude_over_time = np.sum(np.array(row['magnitude'][0]), axis=0)
    median_mag = np.median(magnitude_over_time)
    iqr_mag = iqr(magnitude_over_time)
    
    # Avoid division by zero
    if iqr_mag == 0:
        return np.zeros_like(magnitude_over_time)
        
    magnitude_over_time = np.maximum(0, magnitude_over_time - median_mag)
    return magnitude_over_time / iqr_mag


def nightbeat_predict_whole_window(row, fs, win_size, overlap=0.5, debug=False, quality_thresh=6, motion_artifacts=True, curve_tracing=False, peak_detection=False):
    assert win_size in [20, 30, 60], 'Window size not recognized'

    times_l = []
    preds_l = []
    labels_l = []
    quality_l = []
    preds_peaks_l = []

    start_sec = 60

    magnitude = np.array(row['magnitude'][0])
    extent = row['extent'][0]
    t = np.linspace(extent[0], extent[1], magnitude.shape[1])
    f = np.linspace(extent[2], extent[3], magnitude.shape[0])

    f_index = (f >= 0.5) & (f <= 3.5)

    if motion_artifacts:
        data_quality = nightbeat_motion_artifacts(row)
        assert magnitude.shape[1] == len(data_quality), 'Magnitude and data_quality do not match'

    if curve_tracing:
        curve_idxs, curve_lens, points_curves, stft_peaks = get_curves(magnitude[f_index, :], threshold=0.5, height=0.5, width=1, mid_point=False, double_threshold=0.05, search_radius_px=5, maximum_gap=5, min_length=100)
        curve_idxs, curve_lens, points_curves = refine_curves(stft_peaks, curve_lens, points_curves, f, std_1=3, std_2=5, debug=debug)

        if len(curve_lens) == 0:
            more_peaks_thresh = 0.05
            while len(curve_lens) == 0 and more_peaks_thresh <= 0.2:
                curve_idxs, curve_lens, points_curves, stft_peaks = get_curves(magnitude[f_index, :], threshold=0.3-more_peaks_thresh, height=0.3-more_peaks_thresh, width=1, mid_point=False, double_threshold=0.05, search_radius_px=5, maximum_gap=5, min_length=100)
                curve_idxs, curve_lens, points_curves = refine_curves(stft_peaks, curve_lens, points_curves, f, std_1=3, std_2=5, debug=debug)
                more_peaks_thresh += 0.05

    while start_sec < 120:
        end_sec = start_sec + win_size

        if debug:
            print(f"Predicting window {start_sec} to {end_sec}")

        t_index = np.where((t >= start_sec) & (t < end_sec))[0]

        label = get_hr_label(row, start_sec, end_sec)

        if curve_tracing:
            f_s = f[f_index][np.where(curve_idxs[:, t_index] > 0)[0]]
            if len(f_s) == 0:
                start_sec += int(win_size * (1 - overlap))
                continue
            pred = np.mean(f_s) * 60
        else:
            pred = predict_from_stft(magnitude[:, t_index], f, fs, debug, specific_algorithm='bioinsights')

        if peak_detection:
            pred_peak, narrowly_filtered_signal, detected_peaks = get_narrow_peaks(row['acc_mag'][0][start_sec*fs:end_sec*fs], pred, filter_width=0.1, kaiser_beta=2, filter_order=10, fs=fs, robust_ibis=False, debug=debug)
            preds_peaks_l.append(pred_peak)
        else:
            preds_peaks_l.append(-1)

        if motion_artifacts:
            win_quality = np.mean(data_quality[t_index] > quality_thresh)

        time = row['time'][0] + start_sec
        times_l.append(time)
        preds_l.append(pred)
        labels_l.append(label)
        if motion_artifacts:
            quality_l.append(win_quality)

        start_sec += int(win_size * (1 - overlap))

    if motion_artifacts:
        return times_l, preds_l, labels_l, quality_l, preds_peaks_l
    else:
        return times_l, preds_l, labels_l, preds_peaks_l


def nightbeat_predict_df(df, fs=100, win_size=20, overlap=0.5, motion_artifacts=False, curve_tracing=False, peak_detection=False, debug=False):
    row_idxs_l = []
    times_l = []
    preds_l = []
    labels_l = []
    quality_l = []
    preds_peaks_l = []

    for i in range(len(df)):
        if debug:
            print(f"Predicting row {i} out of {len(df)}")
        row = df[i]
        row_idxs = row['row_idx'][0]
        
        if motion_artifacts:
            times, preds, labels, quality, preds_peaks = nightbeat_predict_whole_window(row, fs, win_size, overlap, motion_artifacts=motion_artifacts, curve_tracing=curve_tracing, peak_detection=peak_detection, debug=debug)
            quality_l.extend(quality)
        else:
            times, preds, labels, preds_peaks = nightbeat_predict_whole_window(row, fs, win_size, overlap, motion_artifacts=motion_artifacts, curve_tracing=curve_tracing, peak_detection=peak_detection, debug=debug)
            quality_l.extend([1] * len(times))

        row_idxs_l.extend([row_idxs] * len(times))
        times_l.extend(times)
        preds_l.extend(preds)
        labels_l.extend(labels)
        preds_peaks_l.extend(preds_peaks)

    df = pl.DataFrame({
        'row_idx': row_idxs_l,
        'time': times_l,
        'pred': preds_l,
        'pred_peaks': preds_peaks_l,
        'label': labels_l,
        'quality': quality_l
    }, schema={
        'row_idx': pl.Int64,
        'time': pl.Float64,
        'pred': pl.Float64,
        'pred_peaks': pl.Float64,
        'label': pl.Float64,
        'quality': pl.Float64
    })

    return df


def median_smoother(df, win_seconds, column='pred'):
    median_obs = np.zeros(len(df))
    # Optimized: If possible, this should be done using polars rolling expressions, 
    # but maintaining loop for exact logic replication unless strictly needed.
    # Note: A pure Polars rolling window would be much faster here.
    for i in range(len(df)):
        median_obs[i] = df.filter(pl.col('time').is_between(df['time'][i] - win_seconds / 2, df['time'][i] + win_seconds / 2))[column].median()
    df = df.with_columns(pl.Series(name=f"{column}_median", values=median_obs))
    return df


def nightbeat_load_and_predict(subject, all_subjects=[], dataset='nightbeatdb', curve_tracing=False, peak_detection=False, motion_artifacts=False, win_size=20):

    df = load_transformed_data(subject_id=all_subjects[subject], method='nightbeat', dataset=dataset)

    if dataset == 'nightbeatdb':
        df_orig = load_test_data_nightbeatdb(subject_id=all_subjects[subject], windows=1000)
    elif dataset == 'aw':
        df_orig = load_test_data_oliviawalch(subject_id=all_subjects[subject], windows=1000)
    else:
        raise ValueError('Dataset not recognized')

    df_orig = df_orig.with_row_index(name='row_idx').with_columns(pl.col('row_idx').cast(pl.Int64).alias('row_idx'))

    if 'start_100Hz' in df_orig.columns:
        df_orig = df_orig.with_columns(pl.col('start_100Hz').cast(pl.Float64).alias('time'))
    elif 'start' in df_orig.columns:
        df_orig = df_orig.with_columns(pl.col('start').cast(pl.Float64).alias('time'))

    df = df.join(df_orig, on=['row_idx', 'time'], how='inner', coalesce=True)

    if 'start_100Hz' in df.columns:
        df = df.filter(pl.col('hrv_is_valid').arr.get(1024 * 90) == 1)

    print(f"Predicting subject {all_subjects[subject]} in dataset {dataset}")

    df = nightbeat_predict_df(df, win_size=win_size, curve_tracing=curve_tracing, peak_detection=peak_detection, motion_artifacts=motion_artifacts)

    dataset_idx = 0
    if dataset == 'aw':
        dataset_idx = 1

    suffix = f'_{win_size}'
    
    if curve_tracing:
        filename = f'nightbeat_ct_{dataset}_{int(all_subjects[subject]):02d}_preds{win_size}.parquet'
    else:
        filename = f'nightbeat_{dataset}_{int(all_subjects[subject]):02d}_preds{win_size}.parquet'
        
    out_path = os.path.join(get_base_paths(results=True)[dataset_idx], filename)
    print(f"Saving in {out_path}")
    df.write_parquet(out_path)

    return 'saved'


if __name__ == '__main__':
    nightbeat_load_and_predict(11, all_subjects=range(31), dataset='wristbcg', win_size=30, curve_tracing=True, peak_detection=True, motion_artifacts=True)