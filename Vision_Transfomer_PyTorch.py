pip install -q vit-pytorch
pip install -q nnAudio

import os
import time
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
import tensorflow as tf
import tensorflow_datasets as tfds  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nnAudio.Spectrogram import CQT1992v2
from vit_pytorch import ViT
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


SAVEDIR = Path("./")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CFG:
    debug = False
    print_freq = 3000
    num_workers = 4
    scheduler = "CosineAnnealingLR"
    model_name = "vit"
    epochs = 6
    T_max = 3
    lr = 1e-4
    min_lr = 1e-7
    batch_size = 50
    val_batch_size = 100
    weight_decay = 1e-5
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 1127802825
    target_size = 1
    target_col = "target"
    n_fold = 5
    trn_fold = [3]  # [0, 1, 2, 3, 4]
    train = True
    n_hidden = 256
    n_heads = 4
    n_layers = 12
    image_size = (128, 256)
    patch_size = (8, 16)
    cqt_params = {"sr": 2048, 
                  "fmin": 20, "fmax": 530, 
                  "hop_length": 4,
                  'filter_scale':0.5,
                  "bins_per_octave": 27}

def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score


def init_logger(log_file=SAVEDIR / 'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)

gcs_paths = []
for i, j in [(0, 4), (5, 9), (10, 14), (15, 19)]:
    path = f"g2net-waveform-tfrecords-train-{i}-{j}"
    n_trial = 0
    while True:
        try:
            gcs_path = KaggleDatasets().get_gcs_path(path)
            gcs_paths.append(gcs_path)
            print(gcs_path)
            break
        except:
            if n_trial > 10:
                break
            n_trial += 1
            continue
            
all_files = []
for path in gcs_paths:
    all_files.extend(np.sort(np.array(tf.io.gfile.glob(path + "/train*.tfrecords"))))
    
print("train_files: ", len(all_files))
all_files = np.array(all_files)

def count_data_items(fileids, train=True):
    """
    Count the number of samples.
    Each of the TFRecord datasets is designed to contain 28000 samples for train
    22500 for test.
    """
    sizes = 28000 if train else 22500
    return len(fileids) * sizes


AUTO = tf.data.experimental.AUTOTUNE

def prepare_wave(wave):
    wave = tf.reshape(tf.io.decode_raw(wave, tf.float64), (3, 4096))
    normalized_waves = []
    scaling = tf.constant([1.5e-20, 1.5e-20, 0.5e-20], dtype=tf.float64)
    for i in range(3):
#         normalized_wave = wave[i] / tf.math.reduce_max(wave[i])
        normalized_wave = wave[i] / scaling[i]
        normalized_waves.append(normalized_wave)
    wave = tf.stack(normalized_waves, axis=0)
    wave = tf.cast(wave, tf.float32)
    return wave


def read_labeled_tfrecord(example):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return prepare_wave(example["wave"]), tf.reshape(tf.cast(example["target"], tf.float32), [1]), example["wave_id"]


def read_unlabeled_tfrecord(example, return_image_id):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return prepare_wave(example["wave"]), example["wave_id"] if return_image_id else 0


def get_dataset(files, batch_size=16, repeat=False, cache=False, 
                shuffle=False, labeled=True, return_image_ids=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO, compression_type="GZIP")
    if cache:
        ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 2)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_ids), num_parallel_calls=AUTO)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return tfds.as_numpy(ds)

class TFRecordDataLoader:
    def __init__(self, files, batch_size=32, cache=False, train=True, 
                              repeat=False, shuffle=False, labeled=True, 
                              return_image_ids=True):
        self.ds = get_dataset(
            files, 
            batch_size=batch_size,
            cache=cache,
            repeat=repeat,
            shuffle=shuffle,
            labeled=labeled,
            return_image_ids=return_image_ids)
        
        self.num_examples = count_data_items(files, labeled)

        self.batch_size = batch_size
        self.labeled = labeled
        self.return_image_ids = return_image_ids
        self._iterator = None
    
    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self._iterator

    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        return batch

    def __len__(self):
        n_batches = self.num_examples // self.batch_size
        if self.num_examples % self.batch_size == 0:
            return n_batches
        else:
            return n_batches + 1

class CustomViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cqt = CQT1992v2(**cfg.cqt_params)
        self.image_size = cfg.image_size
        self.model = ViT(image_size=cfg.image_size,
                         patch_size=cfg.patch_size,
                         num_classes=1,
                         dim=CFG.n_hidden,
                         depth=CFG.n_layers,
                         heads=CFG.n_heads,
                         mlp_dim=4*CFG.n_hidden,
                         dropout=0.1,
                         emb_dropout=0.1,
                         )

    def forward(self, x):
        sps = []
        for i in range(3):
            s = self.cqt(x[:, i, :])
            s = F.interpolate(s, size=256, mode='linear', align_corners=True)
            sps.append(s)
        x = torch.stack(sps, dim=1)
        output = self.model(x)
        return output

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def max_memory_allocated():
    MB = 1024.0 * 1024.0
    mem = torch.cuda.max_memory_allocated() / MB
    return f"{mem:.0f} MB"

def train_fn(files, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0

    train_loader = TFRecordDataLoader(
        files, batch_size=CFG.batch_size, 
        shuffle=True)
    for step, d in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        x = torch.from_numpy(d[0]).to(device)
        labels = torch.from_numpy(d[1]).to(device)

        batch_size = labels.size(0)
        y_preds = model(x)
        loss = criterion(y_preds.view(-1), labels.view(-1))
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}] '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  'Elapsed: {remain:s} '
                  'Max mem: {mem:s}'
                  .format(
                   epoch+1, CFG.epochs, step, len(train_loader),
                   loss=losses,
                   grad_norm=grad_norm,
                   lr=scheduler.get_last_lr()[0],
                   remain=timeSince(start, float(step + 1) / len(train_loader)),
                   mem=max_memory_allocated()))
    return losses.avg


def valid_fn(files, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    filenames = []
    targets = []
    preds = []
    start = end = time.time()
    valid_loader = TFRecordDataLoader(
        files, batch_size=CFG.batch_size * 2, shuffle=False)
    for step, d in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        targets.extend(d[1].reshape(-1).tolist())
        filenames.extend([f.decode("UTF-8") for f in d[2]])
        x = torch.from_numpy(d[0]).to(device)
        labels = torch.from_numpy(d[1]).to(device)

        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(x)
        loss = criterion(y_preds.view(-1), labels.view(-1))
        losses.update(loss.item(), batch_size)

        preds.append(y_preds.sigmoid().to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0:
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   ))
    predictions = np.concatenate(preds).reshape(-1)
    return losses.avg, predictions, np.array(targets), np.array(filenames)

def train_loop(train_tfrecords: np.ndarray, val_tfrecords: np.ndarray, fold: int):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                             mode='min', 
                                                             factor=CFG.factor, 
                                                             patience=CFG.patience, 
                                                             verbose=True, 
                                                             eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                             T_max=CFG.T_max, 
                                                             eta_min=CFG.min_lr, 
                                                             last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                       T_0=CFG.T_0, 
                                                                       T_mult=1, 
                                                                       eta_min=CFG.min_lr, 
                                                                       last_epoch=-1)
        return scheduler

    model = CustomViT(cfg=CFG)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(optimizer)

    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        print("\n\n")
        start_time = time.time()
        
        # train
        avg_loss = train_fn(train_tfrecords, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds, targets, files = valid_fn(val_tfrecords, model, criterion, device)
        valid_result_df = pd.DataFrame({"target": targets, "preds": preds, "id": files})
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(targets, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        SAVEDIR / f'{CFG.model_name}_fold{fold}_best_score.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        SAVEDIR / f'{CFG.model_name}_fold{fold}_best_loss.pth')
    
    valid_result_df["preds"] = torch.load(SAVEDIR / f"{CFG.model_name}_fold{fold}_best_loss.pth",
                                          map_location="cpu")["preds"]

    return valid_result_df

def get_result(result_df):
    preds = result_df['preds'].values
    labels = result_df[CFG.target_col].values
    score = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}')

if CFG.train:
    # train 
    oof_df = pd.DataFrame()
    kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    folds = list(kf.split(all_files))
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            trn_idx, val_idx = folds[fold]
            train_files = all_files[trn_idx]
            valid_files = all_files[val_idx]
            _oof_df = train_loop(train_files, valid_files, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)
    # CV result
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    # save result
    oof_df.to_csv(SAVEDIR / 'oof_df.csv', index=False)

states = []
for fold  in CFG.trn_fold:
    states.append(torch.load(os.path.join(SAVEDIR, f'{CFG.model_name}_fold{fold}_best_score.pth')))


gcs_paths = []
for i, j in [(0, 4), (5, 9)]:
    path = f"g2net-waveform-tfrecords-test-{i}-{j}"
    n_trial = 0
    while True:
        try:
            gcs_path = KaggleDatasets().get_gcs_path(path)
            gcs_paths.append(gcs_path)
            print(gcs_path)
            break
        except:
            if n_trial > 10:
                break
            n_trial += 1
            continue
            
all_files = []
for path in gcs_paths:
    all_files.extend(np.sort(np.array(tf.io.gfile.glob(path + "/test*.tfrecords"))))
    
print("test_files: ", len(all_files))
all_files = np.array(all_files)


model= CustomViT(cfg=CFG)
model.to(device)

wave_ids = []
probs_all = []

for fold, state in enumerate(states):
    tqdm.write(f"\n\nFold{fold}")
    
    model.load_state_dict(state['model'])
    model.eval()
    probs = []

    test_loader = TFRecordDataLoader(all_files, batch_size=CFG.val_batch_size, 
                                     shuffle=False, labeled=False)

    for i, d in tqdm(enumerate(test_loader), total=len(test_loader)):
        x = torch.from_numpy(d[0]).to(device)

        with torch.no_grad():
            y_preds = model(x)
        preds = y_preds.sigmoid().to('cpu').numpy()
        probs.append(preds)

        if fold==0:
            wave_ids.append(d[1].astype('U13'))

    probs = np.concatenate(probs)
    probs_all.append(probs)

probs_avg = np.asarray(probs_all).mean(axis=0).flatten()
wave_ids = np.concatenate(wave_ids)


test_df = pd.DataFrame({'id': wave_ids, 'target': probs_avg})

folds = '_'.join([str(s) for s in CFG.trn_fold])
test_df.to_csv(f'{CFG.model_name}_folds_{folds}.csv', index = False)

