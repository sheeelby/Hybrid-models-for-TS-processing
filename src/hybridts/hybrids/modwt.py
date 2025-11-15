"""Minimal MODWT implementation used inside HybridPlus."""
from __future__ import annotations

import numpy as np
import pywt
from scipy.ndimage import convolve1d


def upArrow_op(li, j):
    if j == 0:
        return [1]
    N = len(li)
    li_n = np.zeros(2 ** (j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2 ** (j - 1) * i] = li[i]
    return li_n


def period_list(li, N):
    n = len(li)
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    if len(li) < 2 * N:
        return np.array(li)
    li = np.array(li)
    li = np.reshape(li, [-1, N])
    li = np.sum(li, axis=0)
    return li


def circular_convolve_mra(h_j_o, w_j):
    return convolve1d(
        w_j,
        np.flip(h_j_o),
        mode="wrap",
        origin=(len(h_j_o) - 1) // 2,
    )


def circular_convolve_d(h_t, v_j_1, j):
    N = len(v_j_1)
    ker = np.zeros(len(h_t) * 2 ** (j - 1))
    for i, h in enumerate(h_t):
        ker[i * 2 ** (j - 1)] = h
    return convolve1d(v_j_1, ker, mode="wrap", origin=-len(ker) // 2)


def circular_convolve_s(h_t, g_t, w_j, v_j, j):
    N = len(v_j)
    h_ker = np.zeros(len(h_t) * 2 ** (j - 1))
    g_ker = np.zeros(len(g_t) * 2 ** (j - 1))
    for i, (h, g) in enumerate(zip(h_t, g_t)):
        h_ker[i * 2 ** (j - 1)] = h
        g_ker[i * 2 ** (j - 1)] = g
    v_j_1 = convolve1d(
        w_j,
        np.flip(h_ker),
        mode="wrap",
        origin=(len(h_ker) - 1) // 2,
    )
    v_j_1 += convolve1d(
        v_j,
        np.flip(g_ker),
        mode="wrap",
        origin=(len(g_ker) - 1) // 2,
    )
    return v_j_1


def modwt(x, filters, level):
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)


def imodwt(w, filters):
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    level = len(w) - 1
    v_j = w[-1]
    for jp in range(level):
        j = level - jp - 1
        v_j = circular_convolve_s(h_t, g_t, w[j], v_j, j + 1)
    return v_j


def modwtmra(w, filters):
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    level, N = w.shape
    level = level - 1
    D = []
    g_j_part = [1]
    for j in range(level):
        g_j_up = upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)
        h_j_up = upArrow_op(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)
        h_j_t = h_j / (2 ** ((j + 1) / 2.0))
        if j == 0:
            h_j_t = h / np.sqrt(2)
        h_j_t_o = period_list(h_j_t, N)
        D.append(circular_convolve_mra(h_j_t_o, w[j]))
    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2 ** ((j + 1) / 2.0))
    g_j_t_o = period_list(g_j_t, N)
    S = circular_convolve_mra(g_j_t_o, w[-1])
    D.append(S)
    return np.vstack(D)


__all__ = [
    "modwt",
    "imodwt",
    "modwtmra",
    "circular_convolve_d",
    "circular_convolve_mra",
    "circular_convolve_s",
]
