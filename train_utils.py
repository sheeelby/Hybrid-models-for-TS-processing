# train_utils.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class WindowDatasetStd(Dataset):
    def __init__(self, y, lookback, horizon, stride=1, scale=True):
        y = np.asarray(y, np.float32)
        self.mu, self.sd = (float(y.mean()), float(y.std()) + 1e-8) if scale else (0.0, 1.0)
        z = (y - self.mu) / self.sd if scale else y
        X, Y = [], []
        for t in range(0, len(z) - lookback - horizon + 1, stride):
            X.append(z[t:t+lookback]); Y.append(z[t+lookback:t+lookback+horizon])
        self.X = np.asarray(X, np.float32); self.Y = np.asarray(Y, np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        import torch
        return torch.from_numpy(self.X[i]).unsqueeze(0), torch.from_numpy(self.Y[i])
    @property
    def scaler(self): return (self.mu, self.sd)

class TrainConfig:
    def __init__(self, lookback, horizon, epochs=10, batch_size=64, lr=3e-3, weight_decay=1e-4,
                 clip=1.0, device=None):
        self.lookback = lookback
        self.horizon = horizon
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = clip
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, ds, cfg: TrainConfig):
    if len(ds)==0: return None
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    dev = cfg.device; model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            xb = xb.to(dev).float(); yb = yb.to(dev).float()
            opt.zero_grad(); yhat = model(xb); loss = lossf(yhat, yb); loss.backward()
            if cfg.clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
    model.eval(); return model
