import os
import torch
import torchaudio
import torchaudio.transforms as T

def read_audio(filepath, fs=16000, normalize=False, preemphasis=False):
    """
    Reads audio file stored at <filepath>
    Parameters:
        filepath (str): audio file path
        fs (int, optional): samping rate
        mono (boolean, optional): return single channel
        normalize(boolean, optional): peak normalization of signal
        preemphasis (boolean, optional): apply pre-emphasis filter
    Returns:
        waveform (tensor): audio signal, dim(N,)
    """
    assert isinstance(filepath, str), "filepath must be specified as string"
    assert os.path.exists(filepath), f"{filepath} does not exist."

    try:
        waveform, sr = torchaudio.load(filepath)
        if waveform.dim() == 2:
            waveform = waveform.squeeze_()

        # preemphasis
        if preemphasis:
            waveform = pre_emphasis(waveform)
        # resample
        if sr != fs:
            resampler = T.Resample(sr, fs, dtype=waveform.dtype)
            waveform = resampler(waveform)
        # normalize
        if normalize:
            waveform = rms_normalize(waveform)
        return waveform
    except Exception as e: 
        return None

def rms_normalize(waveform, r=-10):
    """
    RMS-normalization of  <waveform>
    Parameter:
        waveform (tensor): waveform, dims: (N,)
        rms (float): rms in dB
    """
    current_rms = torch.pow(torch.mean(torch.pow(waveform,2)) ,0.5)
    scaling_factor = (10**(r/10))/current_rms
    return waveform*scaling_factor


def pre_emphasis(waveform, coeff=0.97):
    filtered_sig = torch.empty_like(waveform)
    filtered_sig[1:] = waveform[1:] - coeff*waveform[:-1]
    filtered_sig[0] = waveform[0]
    return filtered_sig



def add_noise(clean, noise, snr):
    """
    Adds background <noise> to <clean> signal at desired <SNR> level
    Parameters:
        clean (tensor): clean waveform, dims: (N,)
        noise (tensor): noise waveform, dims: (M,)
        snr (int): SNR level in dB
    Returns:
        noisy signal (tensor), dims: (N,)
    """
    # make equal lengths for clean and noise signals
    if len(clean) > len(noise):
        reps = torch.ceil(torch.tensor(len(clean)/len(noise))).int()
        noise = torch.tile(noise, (reps,))[:len(clean)]
    else:
        start_idx = torch.randint(len(noise) - len(clean), (1,))
        noise = noise[start_idx:start_idx+len(clean)]        
    assert len(noise) == len(clean), f"noise signal {len(noise)} and clean signal {len(clean)} length mismatch"
    
    # add noise at desired snr
    clean_rms = torch.mean(torch.pow(clean, 2))
    noise_rms = torch.mean(torch.pow(noise, 2))
    factor = torch.pow((clean_rms/noise_rms)/torch.pow(torch.tensor(10), (snr/10)), 0.5)
    noise = factor*noise
    noise_clean = clean + noise
    assert 10*torch.log10_(clean_rms/torch.mean(torch.pow(noise, 2))) - snr < 1e-4, f"snr mismatch"
    return noise_clean