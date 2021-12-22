#!/usr/bin/env python

import ase
from ase.units import kB
from bgflow.bg import BoltzmannGenerator
import torch
from bgflow import GaussianMCMCSampler
import bgflow

from os.path import exists

import src


def bg_train(bg: BoltzmannGenerator, temp):

    target = bg._target

    # if src.config.debug:
    print("## Sampling Data", flush=True)
    target_sampler = GaussianMCMCSampler(target,
                                         init_state=target.init_state,
                                         temperature=temp * kB)
    # time intensive step, load from disk
    if exists("data.pt"):
        data = torch.load("data.pt")
        print("loaded data params")
    else:
        for i in range(src.config.bg_n_presample):
            data = target_sampler.sample(5)
            if i % 10:
                torch.save(data, "data.pt")

            torch.save(data, "data.pt")

    if exists("flow_state_dict.pt"):
        bg.flow.load_state_dict(torch.load("flow_state_dict.pt"))
        bg.flow.eval()
        print("loaded prior params")
    else:  # quite fast because sampling is already done
        # if src.config.debug:
        print("## starting NLL optimiser", flush=True)

        # first training
        nll_optimizer = torch.optim.Adam(bg.parameters(), lr=1e-3)
        nll_trainer = bgflow.KLTrainer(bg,
                                       optim=nll_optimizer,
                                       train_energy=False)

        for i in range(src.config.bg_kll_rounds):
            nll_trainer.train(
                n_iter=src.config.bg_kll_rounds,
                data=data,
                batchsize=src.config.bg_batch_size,
                n_print=5,
                w_energy=0.0,
                temperature=temp * kB,
            )

            torch.save(bg.flow.state_dict(), "flow_state_dict.pt")

    # # mixed training
    # if src.config.debug:
    print("## starting mixed training", flush=True)

    mixed_optimizer = torch.optim.Adam(bg.parameters(), lr=1e-4)
    mixed_trainer = bgflow.KLTrainer(bg,
                                     optim=mixed_optimizer,
                                     train_energy=True)

    for i in range(src.config.bg_kll_rounds):
        mixed_trainer.train(
            n_iter=1,
            data=data,
            batchsize=src.config.bg_batch_size,
            n_print=1,
            w_energy=0.2,
            w_likelihood=0.8,
            temperature=temp * kB,
        )

        torch.save(bg.flow.state_dict(), "flow_state_dict.pt")
