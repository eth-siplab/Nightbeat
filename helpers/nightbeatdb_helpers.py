import polars as pl
import numpy as np
import polars as pl
import os

from helpers.data_loader import get_base_paths

def chunk_trace(ecg_m, acc_a, acc_m, hrv_isvalid, hr_bxb, hr_bxb_rate, chunksize=180, overlap=120):
    # chunk size and overlap are in seconds
    # ecg_m, hr_bxb, and hrv_isvalid are 1024 Hz
    # acc_a and acc_m are 100 Hz

    # LOGIC: each window is chunksize seconds long, including overlap/2 on each side: each window of interest is chunksize - overlap seconds long
    # since HRV_is_valid is once per minute, we'll use a buffer of 1-min on each side and the

    # print(hr_bxb_rate)

    if isinstance(hr_bxb, np.ndarray) != True:
        hr_bxb = np.array(hr_bxb)

    if isinstance(hr_bxb_rate, np.ndarray) != True:

        hr_bxb_rate = np.array(hr_bxb_rate)

    for win in range(len(ecg_m) // 1024 // (chunksize - overlap)):
        start_1024 = win * 1024 * (chunksize - overlap)
        end_1024 = start_1024 + 1024 * chunksize

        start_100 = win * 100 * (chunksize - overlap)
        end_100 = start_100 + 100 * chunksize

        hr_bxb_idxs = (hr_bxb < end_1024) & (hr_bxb > start_1024)

        yield ecg_m[start_1024:end_1024], acc_a[:, start_100:end_100], acc_m[:, start_100:end_100], hrv_isvalid[start_1024:end_1024], hr_bxb[hr_bxb_idxs] - start_1024, hr_bxb_rate[hr_bxb_idxs], start_100, start_1024

def process_npy_files(pid, load_path, save_path, chunksize=180, overlap=120):

    acc_a       = np.load(os.path.join(load_path, f"{pid:02d}_acc_a.npz"))['arr_0']
    acc_m       = np.load(os.path.join(load_path, f"{pid:02d}_acc_m.npz"))['arr_0']
    ecg_m       = np.load(os.path.join(load_path, f"{pid:02d}_ecg_m.npz"))['arr_0']
    hrv_isvalid = np.load(os.path.join(load_path, f"{pid:02d}_hrv_isvalid.npz"))['arr_0'].astype(np.int64)
    hr_bxb      = np.load(os.path.join(load_path, f"{pid:02d}_hr_bxb.npz"))['arr_0']
    hr_bxb_rate = np.load(os.path.join(load_path, f"{pid:02d}_hr_bxb_rate.npz"))['arr_0']

    pids = []
    start_100Hz = []
    start_1024Hz = []
    ecg = []
    acc_a_x = []
    acc_a_y = []
    acc_a_z = []
    acc_a_mag = []
    acc_m_x = []
    acc_m_y = []
    acc_m_z = []
    acc_m_mag = []
    hrv_is_valid = []
    beats_m = []
    beats_m_hr = []

    for i, win in enumerate(chunk_trace(ecg_m, acc_a, acc_m, hrv_isvalid, hr_bxb, hr_bxb_rate, chunksize=chunksize, overlap=overlap)):

        if len(win[0]) != chunksize * 1024:
            continue

        pids.append(int(pid))

        start_100Hz.append(win[6])
        start_1024Hz.append(win[7])

        ecg.append(win[0].tolist())

        acc_a_x.append(win[1][1, :].tolist())
        acc_a_y.append(win[1][2, :].tolist())
        acc_a_z.append(win[1][3, :].tolist())
        acc_a_mag.append(win[1][4, :].tolist())

        acc_m_x.append(win[2][0, :].tolist())
        acc_m_y.append(win[2][1, :].tolist())
        acc_m_z.append(win[2][2, :].tolist())
        acc_m_mag.append(win[2][3, :].tolist())

        hrv_is_valid.append(win[3].tolist())

        beats_m.append(win[4].tolist())
        beats_m_hr.append(win[5])

    print(f"checked {i} windows")

    df = pl.DataFrame({
        'pid': pids,
        'start_100Hz': start_100Hz,
        'start_1024Hz': start_1024Hz,
        'ecg': ecg,
        'acc_a_x': acc_a_x,
        'acc_a_y': acc_a_y,
        'acc_a_z': acc_a_z,
        'acc_a_mag': acc_a_mag,
        'acc_m_x': acc_m_x,
        'acc_m_y': acc_m_y,
        'acc_m_z': acc_m_z,
        'acc_m_mag': acc_m_mag,
        'hrv_is_valid': hrv_is_valid,
        'beats_m': beats_m,
        'hr_beats': beats_m_hr
    }, schema={
        'pid': pl.Int64,
        'start_100Hz': pl.Int64,
        'start_1024Hz': pl.Int64,
        'ecg': pl.Array(pl.Float64, 1024 * chunksize),
        'acc_a_x': pl.Array(pl.Float64, 100 * chunksize),
        'acc_a_y': pl.Array(pl.Float64, 100 * chunksize),
        'acc_a_z': pl.Array(pl.Float64, 100 * chunksize),
        'acc_a_mag': pl.Array(pl.Float64, 100 * chunksize),
        'acc_m_x': pl.Array(pl.Float64, 100 * chunksize),
        'acc_m_y': pl.Array(pl.Float64, 100 * chunksize),
        'acc_m_z': pl.Array(pl.Float64, 100 * chunksize),
        'acc_m_mag': pl.Array(pl.Float64, 100 * chunksize),
        'hrv_is_valid': pl.Array(pl.Int64, 1024 * chunksize),
        'beats_m': pl.List(pl.Int64),
        'hr_beats': pl.List(pl.Float64)
    })

    processed_path = os.path.join(save_path)

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    df.write_parquet(os.path.join(processed_path, f"{pid:02d}.parquet"))

    return f'processed {pid:02d}'

def process_raw_nightbeatdb_dataset():

    load_path = get_base_paths(raw=True)[0]
    save_path = get_base_paths()[0]

    pids = os.listdir(load_path)
    pids = [int(f.split('_')[0]) for f in pids if f.endswith('_acc_a.npz')]
    pids = sorted(pids)
    
    for pid in pids:
        print(f'Processing nightbeatdb subject {pid:02d}...', end=' ')
        process_npy_files(pid, load_path=load_path, save_path=save_path)

    return None

if __name__ == '__main__':

    process_raw_nightbeatdb_dataset()