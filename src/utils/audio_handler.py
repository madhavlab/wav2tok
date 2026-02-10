import os
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional

class Audio:

    sample_rate = None #16000

    def read(self, filepath:str, apply_preemphasis: bool=False, rms_power: Optional[int]=None) -> torch.FloatTensor:
        
        assert isinstance(filepath, str), "filepath must be specified as string"
        assert os.path.exists(filepath), f"{filepath} does not exist."

        try:
            audio_data, sr = torchaudio.load(filepath)

            #mono channel
            if audio_data.dim() == 2:
                audio_data = audio_data[0]#.squeeze_()

            # resample
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate, dtype=audio_data.dtype)
                audio_data = resampler(audio_data)

            # preemphasis
            if apply_preemphasis:
                audio_data = self.__preemphasis(audio_data)
            
            # rms  normalization
            if rms_power is not None:
                audio_data = self.__rms_normalize(audio_data, rms_power=rms_power)
    
            return audio_data
        
        except Exception as e:
            print(e)
            return None
        
    def write(self, audio_data: torch.Tensor, path:str):
        if audio_data.dim() == 1:
            audio_data = audio_data.view(1,-1)
        torchaudio.save(path, audio_data, self.sample_rate)

    @staticmethod
    def __preemphasis(audio_data: torch.FloatTensor, coeff: float=0.97)-> torch.FloatTensor:
        filtered_audio_data = torch.empty_like(audio_data)
        filtered_audio_data[1:] = audio_data[1:] - coeff*audio_data[:-1]
        filtered_audio_data[0] = audio_data[0]
        return filtered_audio_data
    
    @staticmethod
    def __rms_normalize(audio_data: torch.FloatTensor, rms_power: float=-10)-> torch.FloatTensor:
        """
        Parameter:
            audio_data: audio_data, dims: (N,)
            rms_power: rms in dB
        """
        current_rms = torch.pow(torch.mean(torch.pow(audio_data,2)) ,0.5)
        scaling_factor = (10**(rms_power/10))/current_rms
        return audio_data*scaling_factor

        