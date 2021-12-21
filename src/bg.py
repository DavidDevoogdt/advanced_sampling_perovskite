#!/usr/bin/env python

import src
import src.bg_code
from src.bg_code.bg_model import bg_model
from src.bg_code.bg_train import bg_train
from src.bg_code.bg_transition import bg_transition

def bg(temp=300):
    bg = bg_model()
    bg_train(bg,temp)
    bg_transition(bg)

