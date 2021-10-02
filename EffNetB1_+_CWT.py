import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

train = pd.read_csv(r'training_labels.csv')
# test = pd.read_csv(r'sample_submission.csv')

def get_train_file_path(image_id):
    return "g2net_train/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

def get_test_file_path(image_id):
    return "g2net_test/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

train['file_path'] = train['id'].apply(get_train_file_path)


pip install kornia
import kornia.augmentation as K
import torch
import torch.nn as nn
from kornia.geometry.transform import Resize


class RandomRoll(K.base.IntensityAugmentationBase2D):
    def __init__(
        self, dims=[-2, -1], direction=[1, 1], max_shift_pct=[1.0, 1.0], p=0.5, keepdim=True
    ):
        super().__init__(p=p, keepdim=keepdim)
        self.direction = torch.tensor(direction if isinstance(direction, list) else [direction])
        self.max_shift_pct = torch.tensor(
            max_shift_pct if isinstance(max_shift_pct, list) else [max_shift_pct]
        )
        self.dims = tuple(dims if isinstance(dims, list) else [dims])

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, input_shape: torch.Size):
        shifts = torch.rand(input_shape[0], len(self.dims)) * self.max_shift_pct * self.direction
        input_shape = torch.take(torch.tensor(input_shape), torch.tensor(self.dims))[None, :]
        shifts = (shifts * input_shape).int()
        return dict(shifts=shifts)

    def apply_transform(self, input, params, transform=None):
        input = torch.unbind(input, dim=0)
        shifts = torch.unbind(params["shifts"], dim=0)
        input = torch.stack(
            [torch.roll(x, tuple(s.tolist()), dims=self.dims) for x, s in zip(input, shifts)],
            dim=0,
        )
        return input


class RandomRollH(RandomRoll):
    def __init__(self, **kwargs):
        super().__init__(dims=-2, direction=1, max_shift_pct=1.0, **kwargs)


class RandomRollW(RandomRoll):
    def __init__(self, **kwargs):
        super().__init__(dims=-1, direction=1, max_shift_pct=1.0, **kwargs)


# From https://github.com/tomrunia/PyTorchWavelets/blob/master/wavelets_pytorch/wavelets.py

class Morlet(object):
    def __init__(self, w0=6):
        """w0 is the nondimensional frequency constant. If this is
        set too low then the wavelet does not sample very well: a
        value over 5 should be ok.
        """
        self.w0 = w0
        if w0 == 6:
            # value of C_d from TC98
            self.C_d = 0.776

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0, complete=True):
        """
        Complex Morlet wavelet, centred at zero.
        Parameters
        ----------
        t : float
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.
        complete : bool
            Whether to use the complete or the standard version.
        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given time
        See Also
        --------
        scipy.signal.gausspulse
        Notes
        -----
        The standard version::
            pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
        This commonly used wavelet is often referred to simply as the
        Morlet wavelet.  Note that this simplified version can cause
        admissibility problems at low values of `w`.
        The complete version::
            pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))
        The complete version of the Morlet wavelet, with a correction
        term to improve admissibility. For `w` greater than 5, the
        correction term is negligible.
        Note that the energy of the return wavelet is not normalised
        according to `s`.
        The fundamental frequency of this wavelet in Hz is given
        by ``f = 2*s*w*r / M`` where r is the sampling rate.
        """
        w = self.w0

        x = t / s

        output = np.exp(1j * w * x)

        if complete:
            output -= np.exp(-0.5 * (w ** 2))

        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)

        return output

    # Fourier wavelengths
    def fourier_period(self, s):
        """Equivalent Fourier period of Morlet"""
        return 4 * np.pi * s / (self.w0 + (2 + self.w0 ** 2) ** 0.5)

    def scale_from_period(self, period):
        """
        Compute the scale from the fourier period.
        Returns the scale
        """
        # Solve 4 * np.pi * scale / (w0 + (2 + w0 ** 2) ** .5)
        #  for s to obtain this formula
        coeff = np.sqrt(self.w0 * self.w0 + 2)
        return (period * (coeff + self.w0)) / (4.0 * np.pi)

    # Frequency representation
    def frequency(self, w, s=1.0):
        """Frequency representation of Morlet.
        Parameters
        ----------
        w : float
            Angular frequency. If `s` is not specified, i.e. set to 1,
            this can be used as the non-dimensional angular
            frequency w * s.
        s : float
            Scaling factor. Default is 1.
        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given frequency
        """
        x = w * s
        # Heaviside mock
        Hw = np.array(w)
        Hw[w <= 0] = 0
        Hw[w > 0] = 1
        return np.pi ** -0.25 * Hw * np.exp((-((x - self.w0) ** 2)) / 2)

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.
        This can be worked out analytically by solving
            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return 2 ** 0.5 * s


class CWT(nn.Module):
    def __init__(
        self,
        dj=0.0625,
        dt=1 / 2048,
        fmin: int = 20,
        fmax: int = 500,
        output_format="Magnitude",
        trainable=False,
        padding=0,
    ):
        super().__init__()
        self.wavelet = Morlet()

        self.dt = dt
        self.dj = dj
        self.fmin = fmin
        self.fmax = fmax
        self.output_format = output_format
        self.trainable = trainable 
        self.stride = (1, 1)
        self.padding = padding

        self.signal_length = 4096  # x.shape[-1]
        self.channels = 3  # x.shape[-2]
        
        scale_minimum = self.compute_minimum_scale()
        scales = self.compute_optimal_scales(scale_minimum)
        kernel = self.build_wavelet_bank(scales)
    
        if kernel.is_complex():
            self.register_buffer("kernel_real", kernel.real, persistent=False)
            self.register_buffer("kernel_imag", kernel.imag, persistent=False)
        else:
            self.register_buffer("kernel", kernel, persistent=False)

    def compute_optimal_scales(self, scale_minimum):
        """
        Determines the optimal scale distribution.
        :return: np.ndarray, collection of scales
        """
        if self.signal_length is None:
            raise ValueError("Please specify signal_length before computing optimal scales.")
        J = int((1 / self.dj) * np.log2(self.signal_length * self.dt / scale_minimum))
        scales = scale_minimum * 2 ** (self.dj * np.arange(0, J + 1))

        # Remove high and low frequencies
        frequencies = np.array([1 / self.wavelet.fourier_period(s) for s in scales])
        if self.fmin:
            frequencies = frequencies[frequencies >= self.fmin]
            scales = scales[slice(0, len(frequencies))]
        if self.fmax:
            frequencies = frequencies[frequencies <= self.fmax]
            scales = scales[slice(len(scales) - len(frequencies), len(scales))]
        return scales

    def compute_minimum_scale(self):
        """
        Choose s0 so that the equivalent Fourier period is 2 * dt.
        See Torrence & Combo Sections 3f and 3h.
        :return: float, minimum scale level
        """

        def func_to_solve(s):
            return self.wavelet.fourier_period(s) - 2 * self.dt

        return optimize.fsolve(func_to_solve, 1)[0]

    def build_filters(self, scales):
        filters = []
        for scale_idx, scale in enumerate(scales):
            # Number of points needed to capture wavelet
            M = 10 * scale / self.dt
            # Times to use, centred at zero
            t = torch.arange((-M + 1) / 2.0, (M + 1) / 2.0) * self.dt
            if len(t) % 2 == 0:
                t = t[0:-1]  # requires odd filter size
            # Sample wavelet and normalise
            norm = (self.dt / scale) ** 0.5
            filter_ = norm * self.wavelet(t, scale)
            filters.append(torch.conj(torch.flip(filter_, [-1])))

        filters = self.pad_filters(filters)
        return filters

    def pad_filters(self, filters):
        filter_len = filters[-1].shape[0]
        padded_filters = []

        for f in filters:
            pad = (filter_len - f.shape[0]) // 2
            padded_filters.append(nn.functional.pad(f, (pad, pad)))

        return padded_filters

    def build_wavelet_bank(self, scales):
        """This function builds a 2D wavelet filter using wavelets at different scales

        Returns:
            tensor: Tensor of shape (num_widths, 1, channels, filter_len)
        """

        filters = self.build_filters(scales)
        wavelet_bank = torch.stack(filters)
        wavelet_bank = wavelet_bank.view(wavelet_bank.shape[0], 1, 1, wavelet_bank.shape[1])

        return wavelet_bank

    def forward(self, x):
        """Compute CWT arrays from a batch of multi-channel inputs

        Args:
            x (torch.tensor): Tensor of shape (batch_size, channels, time)

        Returns:
            torch.tensor: Tensor of shape (batch_size, channels, widths, time)
        """

        x = x.unsqueeze(1)

        if not hasattr(self, 'kernel'):
            output_real = nn.functional.conv2d(
                x, self.kernel_real, padding=self.padding, stride=self.stride
            )
            output_imag = nn.functional.conv2d(
                x, self.kernel_imag, padding=self.padding, stride=self.stride
            )
            output_real = torch.transpose(output_real, 1, 2)
            output_imag = torch.transpose(output_imag, 1, 2)

            if self.output_format == "Magnitude":
                return torch.sqrt(output_real ** 2 + output_imag ** 2).contiguous()
            else:
                return torch.stack([output_real, output_imag], -1).contiguous()

        else:

            output = nn.functional.conv2d(x, self.kernel, padding=self.padding, stride=self.stride)
            return torch.transpose(output, 1, 2).contiguous()

class CFG:
    num_workers=16
    model_name='tf_efficientnet_b4_ns'
    epochs=8
    lr=1e-4   
    min_lr=1e-6
    batch_size=64     
    weight_decay=1e-6  
    gradient_accumulation_steps=1
    max_grad_norm=1000
    qtransform_params={"sr": 2048, "fmin":24, "fmax":364, "hop_length": 6 , "bins_per_octave": 8}
    seed=42
    target_size=1
    target_col='target'
    n_fold=5
    trn_fold=[0,1,2,3,4]
    train=True

import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from nnAudio.Spectrogram import CQT1992v2
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.fft import fft, ifft

import scipy.signal as signal
import scipy.optimize as optimize
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import timm

from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = r'cwt_plus_cqt_aug_b1_plus_b4_b4/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

WINDOW=signal.tukey(4096, 0.25)[None,:]
LF=24 
HF=364 
def ts_window(ts):
    return ts * WINDOW

def ts_whiten(ts, lf=LF, hf=HF, order=4):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=2048)
    normalization = np.sqrt((hf - lf) / (2048 / 2))
    return signal.sosfiltfilt(sos, ts) / normalization


def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score

def init_logger(log_file=OUTPUT_DIR+'train.log'):
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


# train = train.sample(frac=0.1,random_state=42).reset_index(drop=True)
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_col])):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)
display(train.groupby(['fold', 'target']).size())

class TrainDataset(Dataset):
    def __init__(self, df,mode,transform=None):
        self.df = df
        self.mode = mode
        self.file_names = df['file_path'].values
        self.labels = df[CFG.target_col].values
        self.det = torch.tensor(np.load('det.npy'))
        
    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        ts = np.load(file_path)
        ts = ts_window(ts)
        ts = ts_whiten(ts)
        
        ts = torch.tensor(ts)
        fs = fft(ts) / self.det
        ts = torch.fft.ifft(fs).real 
        qs = ts
        ts = ts_window(ts) 
        ts = ts.to(torch.float32)
        qs = qs.to(torch.float32)
        label = torch.tensor(self.labels[idx]).float()
        return qs, ts , label


class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.batchnorm_cwt = nn.BatchNorm2d(3)
        self.batchnorm_cqt = nn.BatchNorm2d(3)
        self.model_cwt = timm.create_model(self.cfg.model_name, pretrained=pretrained, in_chans=3)
        self.model_cqt = timm.create_model(self.cfg.model_name1, pretrained=pretrained, in_chans=3)
        self.n_features_cwt = self.model_cwt.classifier.in_features
        self.n_features_cqt = self.model_cqt.classifier.in_features
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.n_features_cwt+self.n_features_cqt,self.cfg.target_size)
        self.cwt_transform = CWT(dt=1/2048, fmin=LF, fmax=HF, padding=0)
        self.cqt_transform = CQT1992v2(**cfg.qtransform_params)
        self.kornea_transform1 = K.RandomChannelShuffle(p=0.5)
        self.kornea_transform2 = RandomRollW()
        
    def forward(self , cqt , cwt , mode):
        transform_cwt = self.cwt_transform(cwt)
        transform_cwt = F.interpolate(transform_cwt,(64,1024),mode='bilinear')
        transform_cqt1 = self.cqt_transform(cqt[:,0,:])
        transform_cqt2 = self.cqt_transform(cqt[:,1,:])
        transform_cqt3 = self.cqt_transform(cqt[:,2,:])
        transform_cqt = torch.stack([transform_cqt1,transform_cqt2,transform_cqt3],axis=1)
        
        if mode == 'train':
            transform_cwt = self.kornea_transform1(transform_cwt)
            transform_cwt = self.kornea_transform2(transform_cwt)
            transform_cqt = self.kornea_transform1(transform_cqt)
            transform_cqt = self.kornea_transform2(transform_cqt)
        
        transform_cwt = self.batchnorm_cwt(transform_cwt)
        output_cwt = self.model_cwt.forward_features(transform_cwt)
        output_cwt = self.model_cwt.global_pool(output_cwt) 
        
        transform_cqt = self.batchnorm_cqt(transform_cqt)
        output_cqt = self.model_cqt.forward_features(transform_cqt)
        output_cqt = self.model_cqt.global_pool(output_cqt)
        
        output = torch.cat([output_cwt,output_cqt],axis=1)     
        output = self.dropout(output)
        output = self.classifier(output)
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


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    losses = AverageMeter()
    model.train()
    tk1 = tqdm(enumerate(train_loader),total=len(train_loader))
    for step, (images1 , images2 , labels) in tk1:
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds = model(images1,images2,'train')
        loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        tk1.set_postfix(loss=losses.avg)
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    tk2 = tqdm(enumerate(valid_loader),total=len(valid_loader))
    for step, (images1 , images2 , labels) in tk2:
        images1 = images1.to(device)
        images2 = images2.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(images1,images2,'valid')
        loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        tk2.set_postfix(loss=losses.avg)
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def train_loop(folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_col].values

    train_dataset = TrainDataset(train_folds,mode='train',transform=None)
    valid_dataset = TrainDataset(valid_folds,mode='val', transform=None)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size, 
                              shuffle=True, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=CFG.batch_size * 2, 
                              shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    model = CustomModel(CFG, pretrained=True)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-3,epochs=CFG.epochs,steps_per_epoch=len(train_loader),pct_start=0.05,div_factor=100,final_div_factor=100)

    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch,scheduler,device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)

        # scoring
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss.pth')
    
    valid_folds['preds'] = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth', 
                                      map_location=torch.device('cpu'))['preds']

    return valid_folds

def main():
    """
    Prepare: 1.train 
    """
    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')
    
    if CFG.train:
        # train 
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)




if __name__ == '__main__':
    main()




