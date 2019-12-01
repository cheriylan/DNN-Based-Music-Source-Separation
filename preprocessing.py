import numpy as np
import os
import soundfile as sf

from consts import *
from scipy.signal import stft, istft
from os import path as op
from math import pi

def diffRescale(phase):
    """ Make data from audio recoding
        args:
        audio_track numpy array stereo mixture shape (num_samples, 2)
        Returns:
            amplitude, phase and processed phase derivatives
        """

    dt = np.diff(phase)
    dt = np.concatenate([np.zeros((phase.shape[0], phase.shape[1], 1)), dt], axis=-1)
    time_shift = 2 * pi * HOP_SIZE / WINDOW_SIZE * np.arange(phase.shape[0])
    dt = (dt.T + time_shift).T
    dt = (dt + pi) % (2 * pi) - pi

    df = np.diff(phase, axis=0)
    df = np.concatenate([np.zeros((1, phase.shape[1], phase.shape[2])), df], axis=0)
    df = df + pi
    df = (df + pi) % (2 * pi) - pi

    res = np.concatenate((dt, df), axis=1)

    return res


def processInput(tracks, context=CONTEXT_SIZE):

    feature_list = list()
    for track in tracks:
        f, t, track_stft = stft(track.audio, fs=RATE, nperseg=WINDOW_SIZE, noverlap=OVERLAP, axis=0)
        amplitude = np.absolute(track_stft)
        phase = np.angle(track_stft)
        feature_list.append((amplitude, diffRescale(phase)))

    time_frames_list = list()
    padded_features = list()
    for features in feature_list:
        time_frames = features[0].shape[-1]
        time_frames_list.append(time_frames)
        temp = list()
        for feature in features:
            temp.append(np.concatenate([np.zeros(feature.shape[:2] + (context, )), feature, np.zeros(feature.shape[:2] + (context,))], axis=-1))
        padded_features.append(temp)

    amplitude_out = list()
    phase_out = list()
    for time, feats in zip(time_frames_list, padded_features):
        amp, p = feats
        for i in range(time):
            amplitude_out.append(amp[:, :, i: i + 2 * context + 1])
            phase_out.append(p[:, :, i: i + 2 * context + 1])
    return [amplitude_out, phase_out]


def processTarget(tracks, target):
    amplitude_out = None
    empty = True
    for track in tracks:

        f, t, track_stft = stft(track.targets[target].audio, fs=RATE, nperseg=WINDOW_SIZE, noverlap=OVERLAP, axis=0)
        amplitude = np.absolute(track_stft)

        if empty:
            amplitude_out = amplitude
            empty = False
        else:
            amplitude_out = np.concatenate([amplitude_out, amplitude], axis=-1)

    amplitude_out = amplitude_out.swapaxes(0, 2)
    amplitude_out = amplitude_out.swapaxes(1, 2)
    amplitude_out = (amplitude_out - AMPLITUDE_MEAN) / AMPLITUDE_STD
    
    return amplitude_out


def makePredictions(model, track, target, context=CONTEXT_SIZE):

    _input = processInput([track.targets[target]], context=context)
    predictions = model.predict(_input)

    f, t, track_stft = stft(track.targets[target].audio, fs=RATE, nperseg=WINDOW_SIZE, noverlap=OVERLAP, axis=0)
    phase = np.angle(track_stft)

    amplitude = predictions.swapaxes(0, 2)
    amplitude = amplitude.swapaxes(1, 0)
    amplitude = amplitude * AMPLITUDE_STD - AMPLITUDE_MEAN
    _out = istft(amplitude * np.exp(1j * phase), fs=RATE, nperseg=WINDOW_SIZE, noverlap=OVERLAP, freq_axis=0) [1]
    _out = _out.swapaxes(0, 1)

    return _out


def saveEstimates(user_estimates, track, estimates_dir):
    track_estimate_dir = op.join(estimates_dir, track.subset, track.filename)
    if not os.path.exists(track_estimate_dir):
        os.makedirs(track_estimate_dir)

    for target, estimate in list(user_estimates.items()):
        target_path = op.join(track_estimate_dir, target + '.wav')
        sf.write(target_path, estimate, track.rate)
    pass
