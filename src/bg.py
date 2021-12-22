#!/usr/bin/env python

import src
import src.bg_code
from src.bg_code.bg_model import bg_model
from src.bg_code.bg_train import bg_train
from src.bg_code.bg_transition import bg_transition


def bg(temp=300):
    bg = bg_model()
    if src.config.bg_train:
        bg_train(bg, temp)
    if src.config.bg_path:
        bg_transition(bg)
