import os
import yaml
import pickle
import argparse

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichModelSummary, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger

from train import AudioTokenizer
from utils import LibriSpeechDataset, MyCallBack


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
torch.set_float32_matmul_precision('medium')


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=False, action="store", default=os.path.dirname(os.getcwd()), help="parent working directory")
parser.add_argument('--ckpt_dir', type=str, required=True, action="store", help="checkpoint directory")
parser.add_argument('--config', type=str, required=False, action="store", default="main.yaml", help="config file name")
parser.add_argument('-c', '--train_checkpoint', type=str, required=False, action="store", help="checkpoint file(.ckpt) path")
parser.add_argument('--device', type=int, required=False, action="store", nargs='+', default=[0], help="gpu device")
args = parser.parse_args()

# read training parameters
if args.train_checkpoint is not None:
    print("Loading saved hyperparams...")
    cfg = pickle.load(open(os.path.join(args.root_dir, "checkpoints", args.train_checkpoint.split("checkpoints")[1].split("/")[1], "params.pkl"), "rb"))
    cfg['checkpoint'] = args.train_checkpoint
    cfg['seed'] = 13
    cfg['train']['audio_dir_path'] = "/DATA/datasets/LibriSpeech/"
    cfg['valid']['audio_dir_path'] = "/DATA/datasets/LibriSpeech/"
    cfg['train']['word_alignments_path'] = "/DATA/datasets/LibriSpeech/LibriSpeech_alignment_word/words_alignment_train-clean-360.pkl" 
    cfg['valid']['word_alignments_path'] = "/DATA/datasets/LibriSpeech/LibriSpeech_alignment_word/words_alignment_test-clean.pkl" 
else:
    with open(os.path.join(args.root_dir, "config", args.config)) as f:
        print(f"Reading config file: {args.config}...")
        cfg = yaml.load(f, Loader=yaml.FullLoader)

seed_everything(cfg['seed'], workers=True)

# save training parameters
dic_path = os.path.join(args.root_dir, "checkpoints", args.ckpt_dir)
if os.path.isdir(dic_path) is False:
    os.makedirs(dic_path)
pickle.dump(cfg, open(os.path.join(dic_path, "params.pkl"), "wb"))


#########################################################################################################################################
# PREPARE DATASET
#########################################################################################################################################


train_dataset = LibriSpeechDataset(alignments_path=cfg['train']['word_alignments_path'],
                        audio_dir=cfg['train']['audio_dir_path'],
                        feats_type=cfg['feats_type'],
                        n_mels=cfg['n_mels'],
                        fs=cfg['fs'],
                        rf_size=cfg['win_size'], #0.025,
                        stride=cfg['win_hop'], #0.010,
                        inp_len=cfg['seglen'],
                        add_augment=cfg["add_augment"],
                        noise_dir_path=cfg["train"]["noise_dir_path"],
                        snr_range=cfg["snr_range"])#1.0

valid_dataset = LibriSpeechDataset(alignments_path=cfg['valid']['word_alignments_path'],
                        audio_dir=cfg['valid']['audio_dir_path'],
                        feats_type=cfg['feats_type'],
                        n_mels=cfg['n_mels'],
                        fs=cfg['fs'],
                        rf_size=cfg['win_size'], #0.025,
                        stride=cfg['win_hop'], #0.010,
                        inp_len=cfg['seglen'],#1.0
                        add_augment=cfg["add_augment"],
                        noise_dir_path=cfg["valid"]["noise_dir_path"],
                        snr_range=cfg["snr_range"])#1.0

print(f"##########\nTotal unique words -->\nTraining words: {len(train_dataset)}\nValidation words: {len(valid_dataset)}")
train_dataloader = DataLoader(train_dataset, batch_size=int(cfg['batchsize']), shuffle=True, drop_last=True, num_workers=cfg['load_workers'], pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=int(cfg['batchsize']), shuffle=False, drop_last=True, num_workers=cfg['load_workers'], pin_memory=True)


#########################################################################################################################################
# BUILD NECESSARY BLOCKS FOR TRAINING 
#########################################################################################################################################

# pytorch lightning module
model = AudioTokenizer(cfg)


# callbacks
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(filename='{epoch}-{valid_loss:.2f}-{train_loss:.2f}', save_last=True) #monitor="train_loss", save_top_k=3, 


progress_bar = RichProgressBar(refresh_rate=1, theme=RichProgressBarTheme(description='#0066ff', progress_bar='#ccff33', progress_bar_finished='#ccff33', progress_bar_pulse='#ccff33', batch_progress='#ff4d4d', time='#ff4d4d', processing_speed='#ff4d4d', metrics='white'))

#logger
version = "_".join(["bsz:"+str(cfg['batchsize']), "lr:"+str(cfg['lr']), "seg:"+str(cfg['seglen'])])
logger = TensorBoardLogger(save_dir=os.path.join(args.root_dir, "checkpoints"), version=version, name=args.ckpt_dir, default_hp_metric=False) 

# lightning module trainer
trainer = Trainer(devices=args.device,
                  accelerator="gpu",
                  callbacks=[lr_monitor, checkpoint_callback, progress_bar, RichModelSummary()],
                  logger = logger,
                  check_val_every_n_epoch=1,
                  max_epochs=-1,
                  deterministic=False,
                  default_root_dir= os.path.dirname(os.getcwd())+"/checkpoints",
                  )
                

# model fit
if args.train_checkpoint is not None:
    print(f"Training resumes from checkpoint: {cfg['checkpoint']}")
    trainer.fit(model, train_dataloader, valid_dataloader,ckpt_path=cfg['checkpoint'])
else: 
    print("training from scratch begins...")
    trainer.fit(model, train_dataloader, valid_dataloader)


