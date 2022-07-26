from torch.nn.functional import mse_loss
import torch
import os
import numpy as np
import soundfile
import librosa
import ray
import sys

num_cpus = int(sys.argv[1])
ray.init(num_cpus=num_cpus)

@ray.remote
def MSECal(paths):

    denoised_path, clean_path = paths
        
    denoised_waveform, _ = soundfile.read(denoised_path)
    clean_waveform, _ = soundfile.read(clean_path)

    denoised_mel = 1 + np.log(1.e-12 + librosa.feature.melspectrogram(denoised_waveform, sr=16000, n_fft=1024, hop_length=256, fmin=20, fmax=8000, n_mels=80)).T / 10.
    clean_mel = 1 + np.log(1.e-12 + librosa.feature.melspectrogram(clean_waveform, sr=16000, n_fft=1024, hop_length=256, fmin=20, fmax=8000, n_mels=80)).T / 10.

    denoised_mel = torch.from_numpy(denoised_mel)
    clean_mel = torch.from_numpy(clean_mel)
    # print(denoised_mel.shape)
    # print(torch.from_numpy(denoised_mel).size())
    return mse_loss(denoised_mel, clean_mel)

def Calculate(clean_dir, denoised_dir):

    folders = os.listdir(denoised_dir)

    all_paths = []
    for folder in folders:
        denoised_folder = os.path.join(denoised_dir, folder)
        clean_folder = os.path.join(clean_dir, folder)

        file_names = os.listdir(denoised_folder)

        for file_name in file_names:
            denoised_path = os.path.join(denoised_folder, file_name)
            clean_path = os.path.join(clean_folder, file_name)

            all_paths.append([denoised_path, clean_path])

    length = len(all_paths)
    futures = []
    for i, path_pair in enumerate(all_paths):
        print('MSE calucated : {} of {}'.format(i, length), end='\r')
        futures.append(MSECal.remote(path_pair))
    # for i, path_pair in enumerate(all_paths):
    #     print('MSE calucated : {} of {}'.format(i, length), end='\r')
    #     futures.append(MSECal(path_pair))

    output = ray.get(futures)
    print('results : ', len(output), sum(output) / len(output))


if __name__ == '__main__':
    clean_dir = '/datasets/wav/val/clean'
    denoised_dir = 'denoised'
    Calculate(clean_dir, denoised_dir)