#!/usr/bin/env python

import ase
from ase.units import kB
from bgflow.bg import BoltzmannGenerator
import torch
from bgflow import GaussianMCMCSampler
import bgflow

from os.path import exists


def bg_train(bg:BoltzmannGenerator,temp):
    
    target = bg._target

    # if src.config.debug:
    print("## Sampling Data", flush=True)
    target_sampler = GaussianMCMCSampler(target, init_state=target.init_state,temperature= temp*kB )

     # time intensive step, load from disk
    if exists('data.pt'):
        data = torch.load('data.pt')
        print("loaded data params")
    else:
        data = target_sampler.sample(500)
        torch.save(data, 'data.pt')


    if exists("flow_state_dict.pt"):
        bg.flow.load_state_dict(torch.load("flow_state_dict.pt"))
        bg.flow.eval()
        print("loaded prior params")
    else: #quite fast because sampling is already done
        # if src.config.debug:
        print("## starting NLL optimiser", flush=True)

        # first training
        nll_optimizer = torch.optim.Adam(bg.parameters(), lr=1e-2)
        nll_trainer = bgflow.KLTrainer(bg,
                                       optim=nll_optimizer,
                                       train_energy=False)

        nll_trainer.train(n_iter=200,
                          data=data,
                          batchsize=10,
                          n_print=5,
                          w_energy=0.0,
                          temperature=temp*kB)

        torch.save(bg.flow.state_dict(), "flow_state_dict.pt")

    # # mixed training
    # if src.config.debug:
    print("## starting mixed training", flush=True)

    mixed_optimizer = torch.optim.Adam(bg.parameters(), lr=1e-3)
    mixed_trainer = bgflow.KLTrainer(bg,
                                     optim=mixed_optimizer,
                                     train_energy=True)

    for i in range(200):
        mixed_trainer.train(
            n_iter=2,
            data=data,
            batchsize=5,
            n_print=1,
            w_energy=0.5,
            w_likelihood=0.5,
            temperature=temp*kB,
        )

        torch.save(bg.flow.state_dict(), "flow_state_dict.pt")

