import torch
import torchaudio.transforms as T


class AudioFeature():
    """ Provides different methods to generate audio features in frequency domain.

    Methods:
    --------
    get_spectrogram: Provides vanilla spectrogram
    get_log_spectrogram: Spectrogram with power expressed in dB
    get_mel_spectrogram: Spectrogram with frequency axis in mel scale
    get_log_mel_spectrogram: Spectrogram with power in dB and frequency axis in mel scale
    """
    def __init__(self, n_fft, hop_length, n_mels, fs):
        """
        Parameters:
        n_fft: int, N points fft to compute DFT. Also, is the window length
        hop_length: int, window shift in number of samples
        n_mels: int, number of mel bands 
        fs: sample rate of an audio
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fs = fs

    def get_spectrogram(self, waveform):
        return T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)(waveform)    

    def get_log_spectrogram(self, waveform):
        return torch.nn.Sequential(T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length),T.AmplitudeToDB())(waveform)       

    def get_mel_spectrogram(self, waveform):
        return T.MelSpectrogram(sample_rate=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)(waveform)

    def get_log_mel_spectrogram(self, waveform):
        return torch.nn.Sequential(T.MelSpectrogram(sample_rate=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels),
                                T.AmplitudeToDB())(waveform)