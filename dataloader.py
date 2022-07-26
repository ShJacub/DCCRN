import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config as cfg
import soundfile as sf
from torchvision.transforms import RandomCrop
import os


# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def create_dataloader(mode):
    if mode == 'train':
        return DataLoader(
            dataset=Wave_Dataset(mode),
            batch_size=cfg.batch,  # max 3696 * snr types
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
    elif mode == 'valid':
        return DataLoader(
            dataset=Wave_Dataset(mode),
            batch_size=cfg.batch, shuffle=False, num_workers=cfg.num_workers
        )    # max 1152


def create_dataloader_for_test(mode):
    if mode == 'test':
        return DataLoader(
            dataset=Wave_Dataset_for_test(mode),
            batch_size=1, shuffle=False, num_workers=4
        )    # max 192


class Wave_Dataset(Dataset):
    def __init__(self, mode):
        # load data
        if mode == 'train':
            print('<Training dataset>')
            print('Load the data...')
            self.input_path = './input/train_dataset.npy'
            self.noisy_dir = '/datasets/wav/train/noisy'
            self.clean_dir = '/datasets/wav/train/clean'
        elif mode == 'valid':
            self.noisy_dir = '/datasets/wav/val/noisy'
            self.clean_dir = '/datasets/wav/val/clean'
            print('<Validation dataset>')
            print('Load the data...')
            self.input_path = './input/validation_dataset.npy'

        # self.input = np.load(self.input_path)
        self.file_names = self.GetFilenames(self.noisy_dir)
        self.crop = RandomCrop((2, 48000), pad_if_needed=True)

    def GetFilenames(self, direc):

        folders = os.listdir(direc)
        folders_path = [os.path.join(direc, x) for x in folders]
        file_names = [[os.path.join(one_folder, file_name) for file_name in os.listdir(one_folder_path)]\
                        for one_folder, one_folder_path in zip(folders, folders_path)]
        new_file_names = []
        for file_name in file_names:
            new_file_names += file_name
        return new_file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        file_name = self.file_names[idx]
        noisy_path = os.path.join(self.noisy_dir, file_name)
        clean_path = os.path.join(self.clean_dir, file_name)

        inputs, _ = sf.read(noisy_path)
        labels, _ = sf.read(clean_path)


        # inputs = self.input[idx][0]
        # labels = self.input[idx][1]
        # transform to torch from numpy
        inputs = torch.from_numpy(inputs).unsqueeze(0)
        labels = torch.from_numpy(labels).unsqueeze(0)
        # print(inputs.shape, inputs.dtype)

        sounds = self.crop(torch.cat([inputs, labels], dim=0))
        inputs, labels = sounds[:1], sounds[1:]

        return inputs, labels


class Wave_Dataset_for_test(Dataset):
    def __init__(self, mode):
        # load data
        if mode == 'test':
            print('<Test dataset>')
            print('Load the data...')
            self.input_path = './input/recon_test_dataset.npy'

        self.noisy_dir = '/datasets/wav/val/noisy'
        self.clean_dir = '/datasets/wav/val/clean'
        self.file_names = self.GetFilenames(self.noisy_dir)
        print(self.file_names[:5])

        # self.input = np.load(self.input_path)

    def GetFilenames(self, direc):

        folders = os.listdir(direc)
        folders_path = [os.path.join(direc, x) for x in folders]
        file_names = [[os.path.join(one_folder, file_name) for file_name in os.listdir(one_folder_path)]\
                        for one_folder, one_folder_path in zip(folders, folders_path)]
        new_file_names = []
        for file_name in file_names:
            new_file_names += file_name
        return new_file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        file_name = self.file_names[idx]
        noisy_path = os.path.join(self.noisy_dir, file_name)
        clean_path = os.path.join(self.clean_dir, file_name)

        inputs, _ = sf.read(noisy_path)
        labels, _ = sf.read(clean_path)


        # inputs = self.input[idx][0]
        # labels = self.input[idx][1]
        # transform to torch from numpy
        inputs = torch.from_numpy(inputs).unsqueeze(0)
        labels = torch.from_numpy(labels).unsqueeze(0)
        # print(inputs.shape, inputs.dtype)

        return inputs, labels, file_name
