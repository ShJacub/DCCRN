import os
import numpy as np
import soundfile
import librosa
import ray
# from ray.util.multiprocessing import Pool
import sys

num_cpus = int(sys.argv[1])
ray.init(num_cpus=num_cpus)


def MakeDir(direc, folder_name, folder_path):
    if not folder_name in set(os.listdir(direc)):
        os.mkdir(folder_path)

@ray.remote
def MelToWavOne(paths):

    file_path_mel, file_path_wav = paths
    mel = np.load(file_path_mel)
    mel = mel.astype(np.float64)
    mel -= 1
    mel *= 10
    mel = mel.T
    mel = np.exp(mel)
    mel -= 1e-12
    audio = librosa.feature.inverse.mel_to_audio(mel, sr=16000, n_fft=1024, hop_length=256, fmin=20, fmax=8000, n_iter=32)
    soundfile.write(file_path_wav, audio, 16000)

def MelToWav(data_dir='datasets/mel/val', save_dir='datasets/wav/val'):

    paths = []

    for ClN_folder in os.listdir(data_dir):
        ClN_path_mel = os.path.join(data_dir, ClN_folder)
        ClN_path_wav = os.path.join(save_dir, ClN_folder)

        MakeDir(save_dir, ClN_folder, ClN_path_wav)

        for folder in os.listdir(ClN_path_mel):
            folder_path_mel = os.path.join(ClN_path_mel, folder)
            folder_path_wav = os.path.join(ClN_path_wav, folder)

            print('Folder Path : ', folder_path_mel)

            MakeDir(ClN_path_wav, folder, folder_path_wav)

            for i, file_name in enumerate(os.listdir(folder_path_mel)):

                print('files : {} of {}'.format(i, len(set(os.listdir(folder_path_mel)))), end='\r')
                file_path_mel = os.path.join(folder_path_mel, file_name)
                file_name_wav = file_name.strip('.npy') + '.wav'
                file_path_wav = os.path.join(folder_path_wav, file_name_wav)

                if file_name_wav in set(os.listdir(folder_path_wav)):
                    continue

                paths.append([file_path_mel, file_path_wav])

                # MelToWavOne(file_path_mel, file_path_wav)

    if len(paths) == 0:
        return
    # print('\n', len(paths))
    # print(paths[:10])
    # first = 0
    # length = len(paths)
    # while first < length:
    #     print('Converted : {} of {}'.format(first, length), end='\r')
    #     last = first + num_cpus
    #     last = last if last <= length else length
    #     pool.map(MelToWavOne, paths[first:last])

    length = len(paths)
    futures = []
    for i, path_pair in enumerate(paths):
        print('Converted : {} of {}'.format(i, length), end='\r')
        futures.append(MelToWavOne.remote(path_pair))

    output = ray.get(futures)


if __name__ == '__main__':

    if not os.path.exists('/datasets/wav'):
        os.mkdir('/datasets/wav')
    if not os.path.exists('/datasets/wav/val'):
        os.mkdir('/datasets/wav/val')
    if not os.path.exists('/datasets/wav/train'):
        os.mkdir('/datasets/wav/train')

    MelToWav(data_dir='/datasets/mel/val', save_dir='/datasets/wav/val')
    MelToWav(data_dir='/datasets/mel/train', save_dir='/datasets/wav/train')