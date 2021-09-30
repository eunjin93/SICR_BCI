import numpy as np
import scipy.io
import Setting as st

from mne.filter import filter_data

def raw_eeg_to_npy(sbj, fs=512, rs=2, mi=3, channel=64, npy_save=False):
    """
    :param sbj: subject index
    :param fs: sampling frequency
    :param rs: Resting state time (sec.)
    :param mi: Motor imagery task time (sec.)
    :param npy_save:
    :return:
    """

    eeg = scipy.io.loadmat(st.raw_data_path + "s%02d.mat" % sbj)["eeg"]
    n_trials = np.array(eeg["n_imagery_trials"][0, 0][0][0])
    left_mi, right_mi = np.array(eeg["imagery_left"][0,0])[:channel,:], np.array(eeg["imagery_right"][0,0])[:channel,:]
    onset = np.array(eeg["imagery_event"][0,0])[-1]

    temp_left = np.zeros(shape=[channel, fs*(rs+mi), n_trials])
    temp_right = np.zeros(shape=[channel, fs*(rs+mi), n_trials])

    j = 0
    for i in range(onset.shape[0]):
        if onset[i] == 1:
            temp_left[:, :, j], temp_right[:, :, j] = left_mi[:, (i-fs*rs+1):(i+fs*mi+1)], right_mi[:, (i-fs*rs+1):(i+fs*mi+1)]
            j += 1

    data = np.concatenate((temp_left, temp_right), -1) #[channel, timepoint (resting+task), trial]

    if npy_save:
        np.save(st.raw_data_path + "npy/s%02d.npy" %(sbj), data)

    return data


def laplacian_filterting(eeg, filter_type = "large"):
    if filter_type == "large":
        NNs = [[4, 32], [8, 36], [9, 35], [32, 5, 38, 11], [0, 37, 6, 12], [0, 3, 13], [0, 4, 14], [1, 9, 15], [1, 10, 16],
               [2, 7, 46, 17], [36, 8, 45, 18], [3, 13, 48, 19], [4, 14, 47, 20], [5, 11, 21], [6, 12, 22], [7, 17, 24],
               [8, 18, 24], [9, 15, 31, 21], [10, 16, 55, 25], [11, 21, 56, 26], [12, 22, 30, 26], [13, 23, 19],
               [14, 20], [14, 21], [17, 29], [18, 23, 62], [19, 22, 63], [29], [30, 24, 61], [31, 24, 61], [47, 20, 57, 28],
               [46, 17, 54, 29], [1, 34, 37], [0, 38, 41], [32, 42, 44], [2, 41, 45], [1, 34, 46], [32, 4, 39, 47], [33, 3, 40, 48],
               [33, 37, 49, 41], [38, 49], [39, 51], [44, 52], [34, 45, 53], [34, 46, 42, 54], [35, 10, 43, 55], [36, 9, 44, 31],
               [37, 12, 49, 30], [38, 11, 50, 56], [39, 47, 51, 57], [41, 48, 58], [49, 59], [42, 54, 61], [44, 55, 61],
               [44, 31, 62, 52], [45, 18, 53, 29], [48, 19, 58, 28], [49, 30, 59, 63], [50, 56, 60, 63], [51, 57],
               [58], [53, 29], [54, 25, 60], [57, 26]]
    else: raise NameError("We have Large Laplacian filtering only.")

    LF_eeg = np.zeros(shape=eeg.shape)
    for channel in range(LF_eeg.shape[0]):
        target = 0
        for NN in range(len(NNs[channel])):
            target += eeg[NNs[channel][NN], :, :]
        target = target / len(NNs[channel])
        LF_eeg[channel, :, :] = eeg[channel, :, :] - target
    return LF_eeg


def baseline_correction(eeg, select_baseline=2, removing=0.5, sfreq=512):
    baseline = np.expand_dims(np.mean(eeg[:, :select_baseline*sfreq, :], axis=1), 1)
    baseline = np.tile(baseline, reps=(1, eeg.shape[1], 1))

    eeg -= baseline
    eeg = eeg[:, select_baseline*sfreq:, :]
    eeg = eeg[:, int(removing*sfreq):int(eeg.shape[1]-removing*sfreq), :]
    return eeg # (64 channels, 1024 timepoints, 200 trials (LH 100, RH 100))


def divide_tr_vl_ts(eeg, seed1=951014, seed2=5930, tr_ratio=0.7, vl_ratio=0.1, ts_ratio=0.2):

    n_trials = int(eeg.shape[-1]/2)
    left_mi = eeg[:, :, :n_trials]
    right_mi = eeg[:, :, n_trials:]

    # Randomize the dataset
    np.random.seed(seed=seed1)
    rnd_idx1 = np.random.permutation(n_trials)
    left_mi = left_mi[:, :, rnd_idx1]

    np.random.seed(seed=seed2)
    rnd_idx2 = np.random.permutation(n_trials)
    right_mi = right_mi[:, :, rnd_idx2]

    tr_mi = np.concatenate((left_mi[:, :, :int(n_trials*tr_ratio)], right_mi[:, :, :int(n_trials*tr_ratio)]), axis=-1)
    vl_mi = np.concatenate((left_mi[:, :, int(n_trials*tr_ratio):(int(n_trials*tr_ratio) + int(n_trials*vl_ratio))],
                            right_mi[:, :, int(n_trials*tr_ratio):(int(n_trials*tr_ratio + int(n_trials*vl_ratio)))]), axis=-1)
    ts_mi = np.concatenate((left_mi[:, :, -int(n_trials*ts_ratio):], right_mi[:, :, -int(n_trials*ts_ratio):]), axis=-1)
    return tr_mi, vl_mi, ts_mi

def gaussian_normalization(tr_eeg, vl_eeg, ts_eeg):
    avg, std = np.mean(np.mean(tr_eeg, axis=-1), axis=-1), np.std(np.std(tr_eeg, axis=-1), axis=-1)

    for channel in range(tr_eeg.shape[0]):
        tr_eeg[channel, :, :] = (tr_eeg[channel, :, :] - avg[channel]/std[channel])
        vl_eeg[channel, :, :] = (vl_eeg[channel, :, :] - avg[channel]/std[channel])
        ts_eeg[channel, :, :] = (ts_eeg[channel, :, :] - avg[channel]/std[channel])

    return tr_eeg, vl_eeg, ts_eeg


def full_preprocessing(sbj, bpf=True, fs=512, low=4, high=40, verbose=True, save=False):

    data = raw_eeg_to_npy(sbj, fs=fs, rs=2, mi=3, channel=64, npy_save=False)
    data = laplacian_filterting(data)
    data = baseline_correction(data) #[channel, timepoint, trial]

    if bpf:
        data = np.swapaxes(data, -2, -1)
        data = filter_data(data, sfreq=fs, l_freq=low, h_freq=high, verbose=False)
        data = np.swapaxes(data, -2, -1)

    tr_data, vl_data, ts_data = divide_tr_vl_ts(data)
    tr_data, vl_data, ts_data = gaussian_normalization(tr_data, vl_data, ts_data)

    if verbose:
        print("The data has the form [64 channel, 1024 timepoints (2 sec), trials].")
        print("For each data bin, first half trials are left-hand MI and the others are right-hand MI.")

    if save:
        np.save(st.preprocessed_data_path + "Preprocessed_s%02d_train.npy" % (sbj), tr_data)
        np.save(st.preprocessed_data_path + "Preprocessed_s%02d_valid.npy" % (sbj), vl_data)
        np.save(st.preprocessed_data_path + "Preprocessed_s%02d_test.npy" % (sbj), ts_data)

    return


sources = np.arange(1, 53)
for sbj in sources:
    full_preprocessing(sbj, save=True)