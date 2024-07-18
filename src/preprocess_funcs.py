import numpy as np
import mne


mne.set_log_level("WARNING")


def notch_filter(data, sfreq, freqs):
    freqs = freqs[freqs < sfreq / 2]
    info = mne.create_info(ch_names=[str(i) for i in range(data.shape[0])], sfreq=sfreq)
    raw = mne.io.RawArray(data, info)
    raw.notch_filter(freqs=freqs, filter_length="auto", phase="zero", picks="all")

    return raw.get_data().astype(np.float32)


def resample(data, sfreq, new_sfreq):
    info = mne.create_info(ch_names=[str(i) for i in range(data.shape[0])], sfreq=sfreq)
    raw = mne.io.RawArray(data, info)
    raw.resample(sfreq=new_sfreq)

    return raw.get_data().astype(np.float32)


def scale_data(data):
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_std = np.std(data, axis=1, keepdims=True)
    scaled_data = (data - data_mean) / data_std
    return scaled_data


def normalize_data(data):
    """
    データをL2ノルムで正規化します。

    Parameters:
    data (np.ndarray): 入力データ（チャネル x 時間）

    Returns:
    normalized_data (np.ndarray): 正規化されたデータ
    """
    l2_norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized_data = data / l2_norm
    return normalized_data


def decimate_data(data, sfreq, decim_factor):
    """
    データをダウンサンプリング（デシメーション）します。

    Parameters:
    data (np.ndarray): 入力データ（チャネル x 時間）
    sfreq (float): サンプリング周波数
    decim_factor (int): ダウンサンプリング因子

    Returns:
    decimated_data (np.ndarray): ダウンサンプリングされたデータ
    """
    info = mne.create_info(ch_names=[str(i) for i in range(data.shape[0])], sfreq=sfreq)
    raw = mne.io.RawArray(data, info)

    # ローパスフィルタを適用
    raw.filter(l_freq=None, h_freq=sfreq / (2 * decim_factor), picks="all")

    # デシメーション
    raw.resample(sfreq / decim_factor)
    decimated_data = raw.get_data().astype(np.float32)
    return decimated_data
