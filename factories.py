# factories.py
from models.timesnet_v2 import TimesNetV2
from models.nbeats_v2 import NBEATSV2

def make_model(name, cfg):
    if name == "timesnet":
        return TimesNetV2(cfg.lookback, cfg.horizon, d_model=32, layers=2, topk=2)
    if name == "nbeats":
        return NBEATSV2(cfg.lookback, cfg.horizon, width=128, depth=2, nblocks=2)
    raise ValueError(name)