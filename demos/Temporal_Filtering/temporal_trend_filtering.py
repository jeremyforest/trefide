#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import trefide.preprocess as preprocess
from trefide.pmd import batch_decompose
import scipy.io as io


X = np.load("/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02/experiment_132/raw_data.npy").astype('double')
X = X.reshape(128,128,-1)
X = np.tile(X, (10, 1))
d1, d2, T = X.shape

K = 50
maxiter = 50
consec_failures = 3
tol = 5e-3
bheight = 16  #40
bwidth = 16  #40
spatial_cutoff = (bheight*bwidth / ((bheight*(bwidth-1) + bwidth*(bheight-1))))
w = .0025

U, V, K, indices = batch_decompose(d1, d2, T, X, bheight, bwidth, w, spatial_cutoff, K, consec_failures, maxiter, maxiter, tol)

stim = np.load("/mnt/home_nas/jeremy/Recherches/Postdoc/Projects/Memory/Computational_Principles_of_Memory/optopatch/data/2020_03_02/experiment_132/raw_data.npy")
signals, trends, stim, disc_idx = preprocess.detrend(X, stim, disc_idx=np.array([5]))

signals = signals.copy() # make sure signals contiguous in memory

T = len(stim)
idx = np.random.randint(0, N)
signal = signals[idx,:]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(np.arange(T), signal+trends[idx,:], 'r')
ax.plot(np.arange(T), trends[idx,:], 'b')

ax.set(xlabel='time', ylabel='Fluorescence',
       title='Slow Trend Fluorescence Neuron {}'.format(idx+1))
ax.grid()

plt.show()


# # Instantial and Fit A Trend Filtering Object for Each Signal
from trefide.temporal import TrendFilter

filters = [TrendFilter(len(signal)) for signal in signals]
denoised = np.asarray([filt.denoise(signal) for signal, filt in zip(signals, filters)])
fig, ax = plt.subplots(nrows=int(np.ceil(len(signals)/2)), ncols=2, figsize=(16, 3 * np.ceil(len(signals)/2)))

for idx, (noisey, smooth) in enumerate(zip(signals, denoised)):
    ax[int(np.floor(idx/2)), int(idx%2)].plot(np.arange(T), noisey + trends[idx], 'r')
    ax[int(np.floor(idx/2)), int(idx%2)].plot(np.arange(T), smooth + trends[idx,:], 'b')
    ax[int(np.floor(idx/2)), int(idx%2)].set(xlabel='time', ylabel='Fluorescence',
                                             title='Slow Trend Fluorescence Neuron {}'.format(idx+1))
    ax[int(np.floor(idx/2)), int(idx%2)].grid()

plt.tight_layout()
plt.show()
