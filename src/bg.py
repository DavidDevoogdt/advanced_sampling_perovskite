#!/usr/bin/env python

from genericpath import exists
import os
import src
import src.bg_code
from src.bg_code.bg_model import bg_model
from src.bg_code.bg_train import bg_train
from src.bg_code.bg_transition import bg_transition
import wandb


def bg(temp=300):

    if src.config.pf != "":
        os.system("cp ../{}/data.pt data.pt".format(src.config.pf))

    bg = bg_model()
    if src.config.bg_no_train:
        if exists("../{}/flow_state_dict.pt".format(src.config.pf)):
            os.system("cp ../{}/flow_state_dict.pt flow_state_dict.pt".format(
                src.config.pf))
        else:
            raise Exception("no flow state dict found in pf")
    else:
        bg_train(bg, temp)
    if not src.config.bg_no_path:
        bg_transition(bg)
