import os
import glob
import math 
import random
import pickle 
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.audio_utils import read_audio, add_noise
import torchaudio.transforms as T
from typing import Optional, List
import librosa

class LibriSpeechDataset(Dataset):
    def __init__(self, alignments_path: str,
                audio_dir: str,
                feats_type: str,
                n_mels: Optional[int],
                fs: int,
                rf_size: int,
                stride: int,
                inp_len:int,
                add_augment: bool,
                noise_dir_path: List,
                snr_range: List)-> None:
        
        super().__init__()
        self.data = pickle.load(open(alignments_path, "rb"))
        # del self.data['']
        # del self.data['"']
        self.audio_dir = audio_dir
        self.fs = fs
        self.rf_size = int(rf_size * fs) #in samples
        self.stride = int(stride * fs) #in samples
        self.inp_len = int(inp_len *fs) #in samples
        self.words = list(self.data.keys())
        self.total_frames = int(self.inp_len/self.stride) + 1
        self.snr_range = snr_range
        self.add_augment = add_augment

        if add_augment:
            self.noises = glob.glob(os.path.join(noise_dir_path, "*.wav"), recursive=True)

        if feats_type == "melspectrogram":
            assert n_mels>0, f"number of mels choosen wrong. Specified mels: {n_mels}"
            self.featextract = torch.nn.Sequential(T.MelSpectrogram(sample_rate=self.fs, n_fft=self.rf_size, hop_length=self.stride, n_mels=n_mels, center=True),
                            T.AmplitudeToDB())
        elif feats_type == "spectrogram":
            self.featextract = torch.nn.Sequential(T.Spectrogram(n_fft=self.rf_size-1, hop_length=self.stride, center=True),T.AmplitudeToDB())
        elif feats_type == "raw":
            self.featextract = None
        else:
            raise ValueError("Specified wrong feats_type arg. It must be one of melspectrogram, spectrogram and raw!")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        word = random.choice(self.words)
        # word must have atleast two utterances
        if len(self.data[word]) < 2:
            return self.__getitem__(np.random.choice(1000))
        utt1, utt2, utt1_padlen, utt2_padlen, utt1_len, utt2_len, metadata = self.fetch_utterance_fixed_length(word) # returns None if any of the utterance length is more than the pre-fixed input length
        if utt1 is None:
            return self.__getitem__(np.random.choice(1000))
        if self.add_augment:
            try:
                noise = read_audio(np.random.choice(self.noises))
                snr = np.random.choice(np.arange(self.snr_range[0], self.snr_range[1]))
                noisy_utt2 = add_noise(utt2, noise, snr)
            except:
                return self.__getitem__(np.random.choice(1000))

        utt1_mask = self._get_pad_mask(utt1_padlen)
        utt2_mask = self._get_pad_mask(utt2_padlen)

        # meta.extend([utt1_padlen, utt2_padlen])
        if self.featextract is None:
            return {"utterances": (utt1.unsqueeze_(0), utt2.unsqueeze_(0)),
                    "utterance_idx": (utt1_mask, utt2_mask),
                    "utterance_len":(utt1_len, utt2_len),
                    "utterance_padlen":(utt1_padlen, utt2_padlen),
                    'word': word}
        else:
            if self.add_augment:
                noisy_utt2_feats = self.featextract(noisy_utt2).permute(1,0)
                noisy_utt2 = noisy_utt2.unsqueeze_(0)
            else:
                noisy_utt2_feats = 0
                noisy_utt2 = 0

            return {"utterances": (self.featextract(utt1).permute(1,0), self.featextract(utt2).permute(1,0), noisy_utt2_feats),
                    #"utterances": (self.get_mfcc_feats(utt1).permute(1,0), self.get_mfcc_feats(utt2).permute(1,0), noisy_utt2_feats),
                    "utterance_idx": (utt1_mask, utt2_mask),
                    "utterance_len":(utt1_len, utt2_len),
                    "utterances_raw": (utt1.unsqueeze_(0), utt2.unsqueeze_(0), noisy_utt2),
                    'word': word,
                    'metadata': metadata}
    

    def get_mfcc_feats(self, y, n_mfcc=13, lifter=0):
        mfccs = librosa.feature.mfcc(y=y.numpy(), sr=self.fs, n_mfcc=n_mfcc, lifter=lifter, n_fft=self.rf_size, hop_length=self.stride, center=True)
        mfccs_der1 = librosa.feature.delta(mfccs, order=1)
        mfccs_der2 = librosa.feature.delta(mfccs, order=2)
        mfcc_feats = np.vstack((mfccs, mfccs_der1, mfccs_der2))   
        mfcc_feats = np.pad(mfcc_feats, [(0,1), (0,0)], "constant")
        return torch.from_numpy(mfcc_feats)     

    def fetch_utterance_fixed_length(self, word: str):
        metadata={}
        # fetch metadata of any two random utterances of the word
        # fname1, fname2 = random.sample(list(self.data[word].keys()), 2)
        fname1, fname2 = np.random.choice(list(self.data[word].keys()), 2, replace=False)
        
        utt1_bound, utt2_bound = (np.array(self.data[word][fname1])*self.fs).astype(int), (np.array(self.data[word][fname2])*self.fs).astype(int)
        utt1_len, utt2_len = utt1_bound[1]-utt1_bound[0], utt2_bound[1]-utt2_bound[0]
        
        # make sure the first utterance always has a smaller length 
        if utt1_len > utt2_len:
            fname1, fname2 = fname2, fname1
            utt1_bound, utt2_bound = utt2_bound, utt1_bound
            utt1_len, utt2_len = utt2_len, utt1_len
        metadata["filenames"] = (fname1, fname2)
        metadata["word_boundary"] = (self.data[word][fname1], self.data[word][fname2])

        # # get file names containing word utterance
        # audio1_data = read_audio(os.path.join(self.audio_dir,"/".join(fname1.split("-")[:2]), fname1+".flac"))
        # audio2_data = read_audio(os.path.join(self.audio_dir,"/".join(fname2.split("-")[:2]), fname2+".flac"))

        # get file names containing word utterance
        audio1_data = read_audio(os.path.join(self.audio_dir, fname1))
        audio2_data = read_audio(os.path.join(self.audio_dir, fname2))
        
        # get context padding lengths
        utt1_pad_len, utt2_pad_len = (self.inp_len - utt1_len)/2, (self.inp_len - utt2_len)/2

        # exit if utterance length is more than fixed input length to the model
        if utt1_pad_len <=0 or utt2_pad_len<=0:
            return [None]*7

        # add zero padding if context padding not sufficient
        utt1_zero_pad_l, utt1_zero_pad_r = max(0,math.floor(utt1_pad_len)-utt1_bound[0]), max(0,math.ceil(utt1_pad_len)-(len(audio1_data)-utt1_bound[1]))
        utt1_data = audio1_data[max(0,utt1_bound[0]-math.floor(utt1_pad_len)) : min(len(audio1_data), utt1_bound[1]+math.ceil(utt1_pad_len))]
        utt1_data = F.pad(utt1_data, (utt1_zero_pad_l, utt1_zero_pad_r))

        utt2_zero_pad_l, utt2_zero_pad_r = max(0,math.floor(utt2_pad_len)-utt2_bound[0]), max(0,math.ceil(utt2_pad_len)-(len(audio2_data)-utt2_bound[1]))
        utt2_data = audio2_data[max(0,utt2_bound[0]-math.floor(utt2_pad_len)) : min(len(audio2_data), utt2_bound[1]+math.ceil(utt2_pad_len))]
        utt2_data = F.pad(utt2_data, (utt2_zero_pad_l, utt2_zero_pad_r))

        # testing
        # print(f"utternance1 bound: {utt1_bound}")
        # print(f"utternance2 bound: {utt2_bound}")
        # print(f"padding lengths: {utt1_pad_len, utt2_pad_len}")
        # print(f"zero_padding_lengths: {utt1_zero_pad_l, utt1_zero_pad_r}")
        
        # print(f"utterance1 context start/end: {max(0,utt1_bound[0]-math.floor(utt1_pad_len)), min(len(audio1_data), utt1_bound[1]+math.ceil(utt1_pad_len))}")
        # print(f"utterance2 context start/end: {max(0,utt2_bound[0]-math.floor(utt2_pad_len)), min(len(audio2_data), utt2_bound[1]+math.ceil(utt2_pad_len))}")
        
        # s = int(utt1_pad_len) 
        # e = int(len(utt1_data)-utt1_pad_len)
        # u1 = audio1_data[utt1_bound[0]:utt1_bound[1]]
        # print(len(u1), (e-s), (u1-utt1_data[s:e]).sum())
        # s = int(utt2_pad_len) 
        # e = int(len(utt2_data)-utt2_pad_len)
        # u2 = audio2_data[utt2_bound[0]:utt2_bound[1]]
        # print(len(u2), (e-s), (u2-utt2_data[s:e]).sum())

        assert len(utt2_data) == len(utt1_data) == self.inp_len, f"utterance lengths with added context mismatch {len(utt2_data), len(utt1_data)}. It must be equal to {self.inp_len}"
        return utt1_data, utt2_data, utt1_pad_len, utt2_pad_len, utt1_len, utt2_len, metadata


    def _get_pad_mask(self, pad_len:int):
        pad_frames_len = math.ceil((pad_len/self.stride))
        mask = torch.zeros(self.total_frames, dtype=torch.int8)
        mask[:pad_frames_len] = 1
        mask[-pad_frames_len:] = 1
        return mask
