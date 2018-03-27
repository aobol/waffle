#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np

import dnest4

from waffle.management import LocalFitManager, FitConfiguration
import multiprocessing as mp

def main():

    directory = "8wf_600"
    wf_file = "training_data/chan600_8wfs.npz"
    conf_name = "bege.config"
    # conf_name = "P42574A.config"
    wf_idxs = np.arange(8)

    if os.path.isdir(directory):
        print("Directory {} already exists: exiting rather than over-writing")
        sys.exit()
    else:
        os.makedirs(directory)

    datadir= os.environ['DATADIR']
    conf_file = datadir +"/siggen/config_files/" + conf_name
    field_file = None

    print("Setting up conf options...")
    conf = FitConfiguration(
        wf_file, conf_file, wf_idxs,
        directory = directory,
        alignType="timepoint",
        align_idx = 125,
        num_samples = 250,
        #imp_grad_guess= 0.1,
        #avg_imp_guess= -0.408716,
        interpType = "linear",
        #smooth_type = "gen_gaus",
        time_step_calc=1
    )

    print("Setting up local fit manager...")
    fm = LocalFitManager(conf, num_threads=48) #mp.cpu_count())

    print("Saving config...")
    conf.save_config()

    print("Starting fit...")
    fm.fit(numLevels=1000, directory = directory,new_level_interval=10000, numParticles=3)


if __name__=="__main__":
    main()
