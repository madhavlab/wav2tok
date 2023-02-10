
import torch

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2ForCTC,Wav2Vec2Processor


class Wav2Vec2Original(nn.Module):
  def __init__(self , mode = 0, device = 'cuda:0'):
      super().__init__()
      
      if mode == 0:
      
         self.feat = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
         
         self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
         
      elif mode == 1:
      
         self.feat = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
         
         self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
         
  def forward(self, x):
  
      x = self.feat(x,sampling_rate = 16000, return_tensors = 'pt')

      with torch.no_grad():
           outs = self.model(**x)
           
      t = outs.logits.argmax(-1)
      
      z = outs.last_hidden_state.numpy()
      
      
      return z, t
  

  def get_tokens(self,x):

      z,t = self.forward(x)
      
      
      return z , t
