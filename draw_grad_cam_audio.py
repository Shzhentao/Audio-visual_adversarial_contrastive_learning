import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import argparse
import torch
import librosa
import librosa.display


parser = argparse.ArgumentParser(description='Evaluation on ESC Sound Classification')
parser.add_argument('--index', default='2')

args = parser.parse_args()
# data_address = './data/esc'
data_address = './data/kinetics/audio'
audio_origin = torch.load("{}/audio-origin-{:02d}-{:05d}.pt".format(data_address, 0, 0)).numpy()
audio = torch.load("{}/audio{:02d}-{:05d}.pt".format(data_address, 0, 0)).numpy()
audio_fps = 24000
n_fft = 512
if n_fft == 512 and audio_fps == 24000:
    stats = np.load('datasets/assets/audio-spectDB-24k-513-norm-stats.npz')
elif n_fft == 256 and audio_fps == 24000:
    stats = np.load('datasets/assets/audio-spectDB-24k-257-norm-stats.npz')

# batch_index = int(args.index)
batch_size = audio.shape[0]
# print(batch_size)
for batch_index in range(batch_size):
    audio_origin_one = audio_origin[batch_index][0]
    audio_one = audio[batch_index][0]
    # print(audio_origin_one.dtype)
    # print(audio_one.dtype)  # float32
    # print(audio_one.shape)  # (200, 257)
    # print(type(audio_one))  # <class 'numpy.ndarray'>
    audio_one = np.transpose(audio_one, (1, 0))
    # print(audio_one.shape)  # (257, 200)

    std_mean, std_std = stats['mean'], stats['std']  # nfft 512 (257,)  # # nfft 256 (129,)
    audio_one = audio_one * (std_std[:, np.newaxis] + 1e-5) + std_mean[:, np.newaxis]

    # draw wave figure
    Time = np.linspace(0, len(audio_origin_one) / audio_fps, num=len(audio_origin_one))
    plt.figure(1)
    plt.title("Signal Wave")
    plt.plot(Time, audio_origin_one)

    # draw audio spectrogram figure
    # plt.figure(figsize=(14, 5))
    # librosa.display.specshow(audio_one, sr = audio_fps, x_axis = 'time', y_axis = 'hz')
    # plt.colorbar()
    hop_size = 1. / 100
    sr = 24000
    hop_length = int(hop_size * sr)
    fig, ax = plt.subplots()
    # S_db_hr = librosa.amplitude_to_db(np.abs(D_highres), ref=np.max)
    img = librosa.display.specshow(audio_one, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='Higher time and frequency resolution')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    plt.show(block=False)
    # plt.pause(10)
    b = input()
    if b == 'g':
        plt.close('all')
    else:
        plt.close('all')
    

# # print(audio_origin.shape)  # (32, 1, 48000)
# # print(audio.shape)  # (32, 1, 200, 257)
# # print(type(audio_origin))  # <class 'numpy.ndarray'>  # <class 'torch.Tensor'>
# # print(type(audio))  # <class 'numpy.ndarray'>  # <class 'torch.Tensor'>
# # print(audio_origin.dtype)  # torch.float64  # float64
# # print(audio.dtype)  # torch.float32  # float32


# audio_address = '/mnt/Zhentaob/datasets/ESC50/data/airplane/1-11687-A-47.wav'
# spf = wave.open(audio_address, "r")
#
# # Extract Raw Audio from Wav File
# signal = spf.readframes(-1)
# signal = np.fromstring(signal, "Int16")
# # print(signal)  # [-801 -847 1048 ... 1425 1481 1545]
# # print(signal.shape)  # (220500,)
# # print(type(signal))  # <class 'numpy.ndarray'>
# # print(signal.dtype)  # int16
# fs = spf.getframerate()
# # print(fs)  # 44100  # <class 'int'>
#
# # If Stereo
# if spf.getnchannels() == 2:
#     print("Just mono files")
#     sys.exit(0)
#
# Time = np.linspace(0, len(signal) / fs, num=len(signal))
#
# plt.figure(1)
# plt.title("Signal Wave...")
# plt.plot(Time, signal)
# plt.show(block=False)
# plt.pause(10)
# plt.close('all')
