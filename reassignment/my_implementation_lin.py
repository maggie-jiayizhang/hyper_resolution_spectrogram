# Linear frequency scan

import torch
import torch.fft as fft
import math
import numpy as np

import time
import os

torch.set_printoptions(precision=10)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def high_resolution_spectrogram(x, sigma, tdeci, over, minf, maxf, sr, ndiv, lint=0.2, device=torch.device('cuda'), dtype=torch.float32):
    eps = 1e-20 # bias to prevent NaN with division
    
    minf = minf*sr
    maxf = maxf*sr
    df = (maxf-minf)/(ndiv-1)

    N = len(x) # assumption: input mono sound
    x = torch.tensor(x, device=device, dtype=dtype)
    xf = fft.fft(x) # complex matrix
    
    # dimensions of the final histogram
    HT = math.ceil(N/tdeci) # time bins 
    HF = ndiv # freq bins

    histo = torch.zeros(HT, HF, device=device, dtype=dtype)
    histc = torch.zeros(HT, HF, device=device, dtype=dtype)

    f = torch.arange(N, device=device) / N 
    f = torch.where(f>0.5, f-1, f)
    f *= sr

    f_steps = np.linspace(minf, maxf, HF*over+1) # HERE
    acc_loop = 0

    tic_loop = time.perf_counter()
    for f0 in f_steps:
        gau = torch.exp(-torch.square(f-f0)/(2*sigma**2))
        gde = -1/sigma**1 * (f-f0) * gau
        
        xi = fft.ifft(gau.T * xf)
        eta = fft.ifft(gde.T * xf)

        mp = torch.div(eta, xi+eps)
        ener = (xi.abs())**2

        tins = torch.arange(1, N+1, device=device) + torch.imag(mp)/(2*math.pi*sigma)
        fins = f0 - torch.real(mp)*sigma

        mask = (torch.abs(mp)<lint) & (fins<maxf) & (fins>minf) & (tins>=1) & (tins<N)

        tins = tins[mask]
        fins = fins[mask]
        ener = ener[mask]

        #HERE: matlab code pipes gpu array into cpu here
        itins = torch.round(tins/tdeci + 0.5)-1    
        #QUESTION: why isn't this +0.5
        #ifins = torch.round(fins + 0.5)-1
        ifins = torch.round((maxf-fins)/df)
        #HERE: did bs here... fix it tmr
        idx = itins.long()*HF+ifins.long()
        # print(min(idx))    

	#HERE: might be even faster with cumulation matrix
	#but might not b/c there's no conversion between cpu/gpu
        histo.put_(idx, ener, accumulate=True)
        histc.put_(idx, (0*itins+1), accumulate=True)
        #histc[itins, ifins] += 1
    toc_loop = time.perf_counter()
    acc_loop += toc_loop-tic_loop

    mm = histc.max()
    histo[histc < torch.sqrt(mm)] = 0
    return histo



