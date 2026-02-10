from utils.dataset import LibriSpeechDataset
from utils.ctc_loss import ctc_loss
from utils.utils import get_fixed_length_utterance_pair, get_pad_mask, get_utterance, MemoryMappedArray, generate_query, extract_query_tokens
from utils.audio_handler import Audio