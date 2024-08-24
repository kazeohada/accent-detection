##############################################
# Original code by Bjöorn Lindqvist
# https://github.com/bjourne/onset-replication
##############################################

from keras.layers import *
from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import Sequence
from keras.metrics import CategoricalAccuracy, F1Score  
from madmom.audio.filters import MelFilterbank
from madmom.audio.signal import FramedSignal, Signal
from madmom.audio.spectrogram import (FilteredSpectrogram,
                                      LogarithmicSpectrogram)
from madmom.audio.stft import ShortTimeFourierTransform
from madmom.features.onsets import peak_picking
from madmom.utils import combine_events
import numpy as np
from tensorflow import one_hot

# The model described in Schlüter and böck 2014 section 4.3 Initial
# Architecture with one caveat: the momentum is fixed at 0.8 instead
# of being linearly increased from 0.45 to 0.9 between epoch 10 and
# 20. The reason for this is because I needed to be able to resume
# training runs and hyper parameter scheduling works poorly in
# combination with resumtion in Keras.
def model():
    m = Sequential()
    m.add(Conv2D(10, (7, 3), input_shape = (15, 80, 3),
                 padding = 'valid', activation = 'relu'))
    m.add(MaxPooling2D(pool_size = (1, 3)))
    m.add(Conv2D(20, (3, 3), input_shape = (9, 26, 10),
                 padding = 'valid', activation = 'relu'))
    m.add(MaxPooling2D(pool_size = (1, 3)))
    m.add(Dropout(0.5))
    m.add(Flatten())
    m.add(Dense(256, activation = 'sigmoid'))
    m.add(Dense(2, activation='sigmoid'))

    optimizer = SGD(learning_rate= 0.05, momentum= 0.8, clipvalue= 5)
    m.compile(loss = 'binary_crossentropy', 
              optimizer = optimizer,
            #   metrics = [F1Score()]
              )
    return m

def preprocess_sig(sig, frame_size):
    frames = FramedSignal(sig, frame_size = frame_size, fps = 100)
    stft = ShortTimeFourierTransform(frames)
    filt = FilteredSpectrogram(stft,
                               filterbank = MelFilterbank,
                               num_bands = 80,
                               fmin = 27.5, fmax = 16000,
                               norm_filters = True,
                               unique_filters = False)
    log_filt = LogarithmicSpectrogram(filt,
                                      log = np.log,
                                      add = np.spacing(1))
    return log_filt
def preprocess_x(filename):
    sig = Signal(filename, sample_rate = 44100, num_channels = 1)
    D = [preprocess_sig(sig, fs) for fs in [2048, 1024, 4096]]
    D = np.dstack(D)
    # Pad left and right with 7 frames
    s = np.repeat(D[:1], 7, axis = 0)
    e = np.repeat(D[-1:], 7, axis = 0)
    D = np.concatenate((s, D, e))
    return D

def preprocess_y(anns, n_frames):
    # 14 frames of padding are added in preprocess_x
    n_frames -= 14
    for i in range(2): 
        for j in range(0, len(anns[i])): 
            anns[i][j] = (anns[i][j] * 100).astype(int) # convert to ms

    
    onsets, accents = anns[0].astype(int), anns[1].astype(int)

    categories = []
    for i in range(n_frames):
        if i in accents: categories.append(2)
        elif i in onsets: categories.append(1)
        else: categories.append(0)
    y = one_hot(indices=categories, depth=3, on_value=1, off_value=0)

    return y

def postprocess_y(y_true, y_guess):



    return

def samples_in_audio_sample(d):
    return np.array([d.x[i:i+15] for i in range(len(d.x) - 14)])

# The sequence logic is specific for the architecture.
class ArchSequence(Sequence):
    def __init__(self, D, batch_size = 256):
        self.D = D
        self.batch_size = batch_size
        self.n_samples = sum(len(d.y) for d in self.D)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.indices = np.arange(self.n_samples)

    def __len__(self):
        return self.n_batches

    def get_sample(self, i):
        at = 0
        for d in self.D:
            n_frames = len(d.y)
            assert len(d.x) == n_frames + 14
            if at + n_frames > i:
                ofs = i - at
                return d.x[ofs:ofs+15], d.y[ofs] 
            at += n_frames
        raise IndexError('Index `%d` is wrong...' % i)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        inds = self.indices[start:end]
        samples = [self.get_sample(i) for i in inds]
        xs = np.array([s[0] for s in samples])
        ys = np.array([s[1] for s in samples])
        return xs, ys

    # This trick comes from
    # https://github.com/keras-team/keras/issues/9707
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
