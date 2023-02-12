"""
Data loading Pipeline, Dataset, and Dataloader
"""
import glob
import os

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchaudio.transforms import Resample, Spectrogram, TimeMasking, TimeStretch, FrequencyMasking, MelScale, InverseMelScale, InverseSpectrogram


class WavToSpectrogram(nn.Module):
    def __init__(
            self,
            resample_freq=24_000,
            n_fft=1024,
            n_mel=256,
            stretch_factor=0.8,
    ):
        super().__init__()

        self.resample_freq = resample_freq

        self.spec = Spectrogram(n_fft=n_fft, power=None)

    def forward(self, waveform: torch.Tensor, input_freq) -> torch.Tensor:
        # Resample the input
        resampled = Resample(orig_freq=input_freq, new_freq=self.resample_freq)(waveform)

        # Convert to power spectrogram
        spec = self.spec(resampled)

        return spec, resampled


class PathException(Exception):
    def __init__(self, string):
        super(PathException, self).__init__(string)


class MuseDB_HQ(Dataset):

    def check_dataset_folder(self):
        if not os.path.isdir(self.path_images):
            raise PathException(f"The given path is not a valid: {self.path_images}")

    def __init__(self, path: str, pipeline: WavToSpectrogram = None, max_num_images=2):
        # print('init custom image-sketch dataset')
        self.path_images = path

        self.check_dataset_folder()

        SUPPORTED_AUDIO_FORMATS = ['wav', 'mp3', 'flac']

        # get images
        # print('scanning the images')
        self.wav_paths = []
        for audio_format in SUPPORTED_AUDIO_FORMATS:
            self.wav_paths.extend(glob.glob(os.path.join(path, f'*.{audio_format}')))
            self.wav_paths.extend(glob.glob(os.path.join(path, f'**/*.{audio_format}')))

        # print('sorting the images and sketches')  # in order to be deterministic
        self.wav_paths.sort()

        if max_num_images > 0:
            self.wav_paths = self.wav_paths[:max_num_images]

        self.size = len(self.wav_paths)

        print('Found', self.size, 'images')

        if pipeline is None:
            self.pipeline = WavToSpectrogram()
        else:
            self.pipeline = pipeline

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # path
        path_wav = self.wav_paths[idx]

        # load
        wav, freq = torchaudio.load(path_wav)

        # convert to spectogram
        spec, wav = self.pipeline.forward(wav, freq)

        spec = torch.view_as_real(spec).reshape([4, 513, -1])

        # left (or right, idk) channel:
        # left = torch.view_as_complex(spec[[0, 1], :, :].view(513, -1, 2))

        return {'spectrogram': spec, 'wav': wav}
