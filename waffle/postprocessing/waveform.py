#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import re
import datetime

import matplotlib.pyplot as plt
# plt.style.use('presentation')
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns
# import corner

import pandas as pd
import numpy as np
import scipy
from scipy import signal

from waffle.management import FitConfiguration, WaveformBatchConfiguration
from waffle.models import *
from waffle.models import Model, PulserTrainingModel

from .base import ResultBase

from siggen import PPC

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "brown", "deeppink", "goldenrod", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]


class WaveformFitResult(ResultBase):
    """
    WaveformFitResult looks in a directory for the relavent files and figures out
    how the fit went. Can produce plots of the waveforms with residual, as well as
    looking at the trace from the chain.

    Parameters
    ----------
    batch_directory: string
        The directory containing the whole batch of fits
    num_samples: int
        Number of samples from the fit/chain to process
    sample_dec:
        Decimation factor, to sample more broadly from the chain faster
    index:
        Which waveform fit you want to look at from the batch
    posterior:
        If true, samples from posterior rather than the chain directly

    Returns
    -------

    Raises
    ------

    """
    def __init__(self, batch_directory, num_samples, sample_dec, index=None, posterior=False):
        # Does not call the super init...

        # ***** Set up directories:
        self.batch_dir = batch_directory

        if index is not None:
            # handle the case for a single fit or specific set of fits
            if(type(index) is int):
                self.index = [index]
            elif(type(index) is list):
                self.index = index
            else:
                print("Not sure how to parse the input you gave as 'index'...")
                exit()
        else:
            # Look at all fits in batch together
            self.index = self.get_indices()

        if self.index is None:
            print("Index list is empty...")
            exit()
        print("Looking at indices: {}".format(self.index))
        
        self.fit_dir = {}
        for i in self.index:
            self.fit_dir[i] = os.path.join(self.batch_dir,"wf{}".format(i))
        # self.fit_dir = [os.path.join(self.batch_dir,"wf{}".format(i)) for i in self.index]


        # print(self.fit_dir)
        # print(self.index)
        # exit()

        self.num_samples = num_samples
        self.sample_dec = sample_dec
        self.posterior = posterior

        self.wf_batch_conf = WaveformBatchConfiguration(None)
        self.wf_batch_conf.load_config(self.batch_dir)

        self.setup_detector()
        self.setup_configurations()

        self.best_params = None

        # for d in self.fit_dir:
        self.total_samples = {}
        self.result_data = {}
        for i in self.index:
            d = self.fit_dir[i]
            if self.posterior:
                print("Getting DNest4 posterior...")
                self.dnest_postprocess(d)
                self.parse_samples("posterior_sample.txt", i)
            else:
            # for d in self.fit_dir:
                print("Using the raw chain, not the posterior...")
                self.parse_samples("sample.txt", i) #d, num_samples, sample_dec)


    def get_indices(self):
        """Pulls out all the directories within the batch dir where fits were performed
        Then, returns a list of the indices themselves
        """
        fits = []

        old_pwd = os.getcwd()
        os.chdir(self.batch_dir)

        ls = os.listdir()
        # filter out any responses that are not directory names
        ls = [value for value in ls if os.path.isdir(value)]

        indices = [int(re.findall(r'\d+', dir_name)[0]) for dir_name in ls]

        fits = [os.path.join(self.batch_dir,dir) for dir in ls]

        os.chdir(old_pwd)
        # return fits,indices
        return indices


    def parse_samples(self, sample_file_name, i):
        """Load the data from CSV
        Uses pandas so its fast
        """
        directory = self.fit_dir[i]
        num_to_read = self.num_samples
        sample_dec = self.sample_dec

        sample_file_name = os.path.join(directory, sample_file_name)

        if not os.path.isfile(sample_file_name):
            print("The file {} does not exist...".format(sample_file_name))
            print("Quitting...")
            raise FileExistsError

        data = pd.read_csv(sample_file_name, delim_whitespace=True, header=None)
        total_samples = len(data.index)
        self.total_samples[i] = total_samples

        print("Found {} samples... ".format(total_samples), end='')

        # Automatically guess how many samples I want to plot/work with
        if num_to_read is None:
            if(total_samples < 2000):
                num_to_read = total_samples
            elif(total_samples >= 2000 and total_samples < 5000):
                num_to_read = 2000
            elif(total_samples >= 5000):
                # take the last 3/4, rounded to the nearest 100
                num_to_read = int(np.floor(total_samples*0.75/100)*100)

        if num_to_read == -1:
            self.result_data[i] = data
        elif total_samples >= num_to_read:
            total_samples = num_to_read
            end_idx = len(data.index) - 1
            self.result_data[i] = data.iloc[(end_idx - total_samples):end_idx:sample_dec]
        elif total_samples < num_to_read:
            self.result_data[i] = data

        print( "Using the last {} samples".format( len(self.result_data[i])) )

    def setup_detector(self):
        # ***** Get the detector set up:

        self.detector = PPC(self.wf_batch_conf.detector_conf, wf_padding=100)
        for m in self.wf_batch_conf.models:
            
            model_name,params = m[:]

            print("Applying {} to detector using:".format(model_name))
            print("   Params: {}".format(params))

            num_params = len(params)

            if model_name == "VelocityModel":
                if (num_params == 4):
                    include_beta = False
                else:
                    include_beta = True
                model = VelocityModel(include_beta=include_beta)
            elif model_name == "ImpurityModelEnds":
                model = ImpurityModelEnds(self.detector)
            elif model_name == "HiPassFilterModel":
                model = HiPassFilterModel(
                    detector=self.detector,
                )
            elif model_name == "LowPassFilterModel":
                model = LowPassFilterModel(
                    self.detector, 
                    # order=the_model.order, 
                    # include_zeros=the_model.include_zeros, 
                    # pmag_lims=the_model.pmag_lims,
                    # pphi_lims=the_model.pphi_lims, 
                    # zmag_lims=the_model.zmag_lims, 
                    # zphi_lims=the_model.zphi_lims, 
                )
            elif model_name ==  "OvershootFilterModel":
                model = OvershootFilterModel(self.detector)
            elif model_name ==  "OscillationFilterModel":
                model = OscillationFilterModel(self.detector)
            elif model_name == "AntialiasingFilterModel":
                model = AntialiasingFilterModel(self.detector)
            elif model_name == "FirstStageFilterModel":
                model = FirstStageFilterModel(self.detector)
            elif model_name ==  "TrappingModel":
                model = TrappingModel()
            else:
                continue
                # raise ValueError("model_name {} is not a valid model".format(model_name))

            print("   Applying these values to the model...")
            model.apply_to_detector(params,self.detector)
            
            # For plots:
            self.width = 18

    def setup_configurations(self):
        # self.fit_configuration = [FitConfiguration.from_file(fd) for fd in self.fit_dir]

        print("Loading target wf from conf...")
        self.wf_model = {}
        for i in self.index:
            wf = self.get_wf_from_conf(i)
            wf.window_waveform(
                time_point=self.wf_batch_conf.align_percent, 
                early_samples=200, 
                num_samples=self.wf_batch_conf.num_samples #200
                )
            # wf = wf_batch_conf

            self.wf_model[i] = (WaveformModel(
                wf,
                align_percent=self.wf_batch_conf.align_percent,
                detector = self.detector,
                align_idx = 200,
            ))

    def get_wf_from_conf(self,index):
        """Just a handy way to pull out the waveform from the file we started from.
        This is designed to operate on the WaveformBatchConfiguration conf file

        Parameters
        ----------
        index: int
            The index number of the waveform you want out of the file

        Returns
        -------
        wf: WaveformModel
            The waveform you asked for

        Raises
        ------
        FileNotFoundError
            You probably moved the files around, or something is really wrong

        """
        
        print("Loading waveform from file {}...\n".format(self.wf_batch_conf.wf_file_name))
        data = np.load(self.wf_batch_conf.wf_file_name, encoding="latin1")
        wfs = data['wfs']
        # wf = wfs[conf.wf_idxs[0]]
        wf = wfs[index]

        return wf

    def get_best_result(self):
        """
        
        r, z, phi, scale, maxt, smooth

        best_params is a nested array
        best_params[model_index][param_index] = param_avg
        """
        # self.result_data
        # raise NotImplementedError
        param_names = ['r', 'z', 'phi', 'scale', 'align', 'cloud']
 
        self.best_params = {}#[[] for i in self.index]

        for i in self.index:
            self.best_params[i] = []
            #loops over all the fit waveforms
        # for model_num,single_wf_model in enumerate(self.wf_model):
            model_num = i
            single_wf_model = self.wf_model[i]
            print("Calculating best values for {}".format(model_num))

            for j in range(single_wf_model.num_params):
                # loops over all the parameters
                param_values = self.result_data[i][j]

                param_avg = np.mean(param_values)
                param_std = np.std(param_values)

                self.best_params[model_num].append(param_avg)

                print("{name}: {avg:4.4f} +/- {std:4.4f}".format(
                    name=param_names[j], 
                    avg=param_avg, 
                    std=param_std)
                    )


    def plot_best_waveform(self):

        if(self.best_params is None):
            self.get_best_result()

        # for wf_model, best_params in zip(self.wf_model,self.best_params):
        for i in self.index:

            wf_model = self.wf_model[i]
            best_params = self.best_params[i]

            target_wf = wf_model.target_wf.windowed_wf
            data_len = len(target_wf)

            plt.figure(figsize=(self.width,8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1], sharex=ax0)
            ax1.set_xlabel("Digitizer Time [ns]")
            ax0.set_ylabel("Voltage [Arb.]")
            ax1.set_ylabel("Residual")

            # plot the target wf with a time axis
            data_len = wf_model.target_wf.window_length
            t_data = np.arange(data_len) * 10
            ax0.plot(t_data, target_wf, color='red', ls = "-")

            # generate the fit-determined best wf
            best_wf = wf_model.make_waveform(data_len, best_params, )

            if best_wf is None:
                #cry
                return

            # plot the best waveform
            t_model = np.arange(data_len) * 10
            ax0.plot(t_data, best_wf, color='blue', alpha=0.5)

            # plot a residual
            resid = best_wf -  target_wf
            ax1.plot(t_data, resid, color='red', alpha=0.5,)# linestyle="steps")

            ax0.set_ylim(-20, wf_model.target_wf.amplitude*1.1)
            ax0.axhline(y=0,color="black", ls=":")

    def plot_best_fft(self):
        """ Plots the FFT of the target and best waveforms next to each other
        """

        if(self.best_params is None):
            self.get_best_result()

        # for wf_model, best_params in zip(self.wf_model,self.best_params):
        for i in self.index:

            wf_model = self.wf_model[i]
            best_params = self.best_params[i]

            target_wf = wf_model.target_wf.windowed_wf
            data_len = len(target_wf)

            plt.figure(figsize=(self.width,8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1], sharex=ax0)
            ax1.set_xlabel("Frequency [MHz]")
            ax0.set_ylabel("Power [Arb.]")
            ax1.set_ylabel("Residual")

            # Calculate the target_wf FFT
            target_fft = np.fft.fft(target_wf)
            target_fft_freq = np.fft.fftfreq(len(target_wf),10*1e-9) * 1e-6

            ## only use the positive real part of the FFT
            fft_len_pos = int(np.floor(len(target_wf)/2))
            target_fft = np.abs(target_fft[:fft_len_pos])
            target_fft_freq = target_fft_freq[:fft_len_pos]

            # plot the target wf FFT with a freq axis
            # data_len = wf_model.target_wf.window_length
            # t_data = np.arange(data_len) * 10
            ax0.semilogy(target_fft_freq, target_fft, color='red', ls = "-")

            # generate the fit-determined best wf
            best_wf = wf_model.make_waveform(data_len, best_params, )

            if best_wf is None:
                #cry
                return

            # Calculate the best wf FFT
            best_fft = np.fft.fft(best_wf)
            best_fft_freq = np.fft.fftfreq(len(best_wf),10*1e-9) * 1e-6

            ## only use the positive real part of the FFT
            fft_len_pos = int(np.floor(len(target_wf)/2))
            best_fft = np.abs(best_fft[:fft_len_pos])
            best_fft_freq = best_fft_freq[:fft_len_pos]

            # plot the best waveform FFT
            t_model = np.arange(data_len) * 10
            ax0.semilogy(best_fft_freq, best_fft, color='blue', alpha=0.5)


            # plot a residual
            resid = best_fft -  target_fft
            ax1.plot(best_fft_freq, resid, color='red', alpha=0.5,)# linestyle="steps")

            # ax0.set_ylim(-20, wf_model.target_wf.amplitude*1.1)
            ax0.axhline(y=0,color="black", ls=":")

    def plot_all_wfs(self):
        """ Plots all waveforms together
        Calculates each residual
        plots them all together
        """
        if(self.best_params is None):
            self.get_best_result()

        plt.figure(figsize=(self.width,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
        ax0 = plt.subplot(gs[0])                # holds the best wfs
        ax1 = plt.subplot(gs[1], sharex=ax0)    # holds the residuals
        ax1.set_xlabel("Digitizer Time [ns]")
        ax0.set_ylabel("Voltage [Arb.]")
        ax1.set_ylabel("Residual")

        residual = []

        # for i,(wf_model, best_params) in enumerate(zip(self.wf_model,self.best_params)):
        for i in self.index:
            wf_model = self.wf_model[i]
            best_params = self.best_params[i]

            i_color = i % len(colors)

            target_wf = wf_model.target_wf.windowed_wf
            data_len = len(target_wf)

            # plot the target wf with a time axis
            data_len = wf_model.target_wf.window_length
            t_data = np.arange(data_len) * 10
            # ax0.plot(t_data, target_wf, color='red', ls = "-")

            # generate the fit-determined best wf
            best_wf = wf_model.make_waveform(data_len, best_params, )

            if best_wf is None:
                #cry
                return

            # plot the residuals together
            # plot a residual
            resid = best_wf -  target_wf
            residual.append(resid)

            ax0.plot(t_data, best_wf, color=colors[i_color], alpha=0.2,)# linestyle="steps")
            ax1.plot(t_data, resid, color=colors[i_color], alpha=0.2,)# linestyle="steps")

        # ax0.set_ylim(-20, wf_model.target_wf.amplitude*1.1)
        avg_residual = np.mean(residual,0)

        ax0.axhline(y=0,color="black", ls=":")
        # t_model = np.arange(data_len) * 10
        # ax0.plot(t_data, avg_residual, color='blue', alpha=0.9)

        return avg_residual


    def plot_avg_residual(self):
        """ Calculates each residual
        plots them all together
        plots the average of all f them
        plots the average of them after scaling? to compare different sized wfs?
        """
        if(self.best_params is None):
            self.get_best_result()

        plt.figure(figsize=(self.width,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_xlabel("Digitizer Time [ns]")
        ax0.set_ylabel("Voltage [Arb.]")
        ax1.set_ylabel("Residual")

        residual = []

        # for i,(wf_model, best_params) in enumerate(zip(self.wf_model,self.best_params)):
        for i in self.index:
            wf_model = self.wf_model[i]
            best_params = self.best_params[i]

            i_color = i % len(colors)

            target_wf = wf_model.target_wf.windowed_wf
            data_len = len(target_wf)

            # plot the target wf with a time axis
            data_len = wf_model.target_wf.window_length
            t_data = np.arange(data_len) * 10
            # ax0.plot(t_data, target_wf, color='red', ls = "-")

            # generate the fit-determined best wf
            best_wf = wf_model.make_waveform(data_len, best_params, )

            if best_wf is None:
                #cry
                return

            # plot the residuals together
            # plot a residual
            resid = best_wf -  target_wf
            residual.append(resid)

            ax1.plot(t_data, resid, color=colors[i_color], alpha=0.2,)# linestyle="steps")

        # ax0.set_ylim(-20, wf_model.target_wf.amplitude*1.1)
        avg_residual = np.mean(residual,0)

        ax0.axhline(y=0,color="black", ls=":")
        t_model = np.arange(data_len) * 10
        ax0.plot(t_data, avg_residual, color='blue', alpha=0.9)

        return avg_residual

    def plot_best_waveform_residual_fft(self):

        if(self.best_params is None):
            self.get_best_result()

        # for wf_model, best_params in zip(self.wf_model,self.best_params):
        for i in self.index:

            wf_model = self.wf_model[i]
            best_params = self.best_params[i]

            target_wf = wf_model.target_wf.windowed_wf
            data_len = len(target_wf)

            plt.figure(figsize=(self.width,8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1], sharex=ax0)
            ax1.set_xlabel("Digitizer Time [ns]")
            ax0.set_ylabel("Voltage [Arb.]")
            ax1.set_ylabel("Residual")


            # Calculate the target_wf FFT
            target_fft = np.fft.fft(target_wf)
            target_fft_freq = np.fft.fftfreq(len(target_wf),10*1e-9) * 1e-6

            ## only use the positive real part of the FFT
            fft_len_pos = int(np.floor(len(target_wf)/2))
            target_fft = np.abs(target_fft[:fft_len_pos])
            target_fft_freq = target_fft_freq[:fft_len_pos]

            # plot the target wf FFT with a freq axis
            # data_len = wf_model.target_wf.window_length
            # t_data = np.arange(data_len) * 10
            # ax0.plot(target_fft_freq, target_fft, color='red', ls = "-")
            ax0.semilogy(target_fft_freq, target_fft, color='red', ls = "-")

            # generate the fit-determined best wf
            best_wf = wf_model.make_waveform(data_len, best_params, )

            if best_wf is None:
                #cry
                return

            # Calculate the best wf FFT
            best_fft = np.fft.fft(best_wf)
            best_fft_freq = np.fft.fftfreq(len(best_wf),10*1e-9) * 1e-6

            ## only use the positive real part of the FFT
            fft_len_pos = int(np.floor(len(target_wf)/2))
            best_fft = np.abs(best_fft[:fft_len_pos])
            best_fft_freq = best_fft_freq[:fft_len_pos]

            # plot the best waveform FFT
            t_model = np.arange(data_len) * 10
            # ax0.plot(best_fft_freq, best_fft, color='blue', alpha=0.5)
            ax0.semilogy(best_fft_freq, best_fft, color='blue', alpha=0.5)

            # Calculate the residual fft and the frequency basis (in MHz)
            resid = best_wf -  target_wf
            resid_fft = np.fft.fft(resid)
            resid_fft_freq = np.fft.fftfreq(len(resid),10*1e-9) * 1e-6

            # only use the positive part of the FFT
            fft_len_pos = int(np.floor(len(resid)/2))
            resid_fft = resid_fft[:fft_len_pos]
            resid_fft_freq = resid_fft_freq[:fft_len_pos]

            ax1.plot(resid_fft_freq, resid_fft, color='red', alpha=0.5,)# linestyle="steps")

            # ax0.set_ylim(-20, wf_model.target_wf.amplitude*1.1)
            # ax0.axhline(y=0,color="black", ls=":")


    def plot_avg_fft_residual(self):
        """ Calculates the FFT of each residual
        plots them all together
        plots the average of all of them
        """
        if(self.best_params is None):
            self.get_best_result()

        plt.figure(figsize=(self.width,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_xlabel("Frequency [MHz]")
        ax0.set_ylabel("Power [Arb.]")
        ax1.set_ylabel("Residual")

        residual = []
        residual_fft = []

        resid_fft_freq = None

        # for i,(wf_model, best_params) in enumerate(zip(self.wf_model,self.best_params)):
        for i in self.index:
            wf_model = self.wf_model[i]
            best_params = self.best_params[i]

            i_color = i % len(colors)

            target_wf = wf_model.target_wf.windowed_wf
            data_len = len(target_wf)

            # plot the target wf with a time axis
            # data_len = wf_model.target_wf.window_length
            # t_data = np.arange(data_len) * 10
            # ax0.plot(t_data, target_wf, color='red', ls = "-")

            # generate the fit-determined best wf
            best_wf = wf_model.make_waveform(data_len, best_params, )

            if best_wf is None:
                #cry
                return

            resid = best_wf -  target_wf

            # Calculate the residual fft and the frequency basis (in MHz)
            resid_fft = np.fft.fft(resid)
            resid_fft_freq = np.fft.fftfreq(len(resid),10*1e-9) * 1e-6

            # only use the positive real part of the FFT
            fft_len_pos = int(np.floor(len(resid)/2))
            resid_fft = np.abs(resid_fft[:fft_len_pos])
            resid_fft_freq = resid_fft_freq[:fft_len_pos]

            residual.append(resid)
            residual_fft.append(resid_fft)

            # ax1.plot(t_data, resid, color=colors[i_color], alpha=0.2,)# linestyle="steps")
            ax1.plot(resid_fft_freq, resid_fft, color=colors[i_color], alpha=0.3,)# linestyle="steps")


        # Plot the average of the FFTs of the residuals
        avg_residual_fft = np.mean(residual_fft,0)

        # only use the positive part of the FFT
        # fft_len_pos = len(avg_residual_fft)

        # resid_fft_freq = np.fft.fftfreq(fft_len_pos*2,10*1e-9) * 1e-6
        # resid_fft_freq = resid_fft[:fft_len_pos]

        ax0.axhline(y=0,color="black", ls=":")
        # t_model = np.arange(data_len) * 10

        ax0.plot(resid_fft_freq, avg_residual_fft, color='blue', alpha=0.9)



    def plot_trace(self):
        f, ax = plt.subplots(self.wf_model[self.index[0]].num_params, 2, figsize=(self.width,10), sharex=True)

        for i in range(self.wf_model[self.index[0]].num_params):
            tf_data = self.result_data[i]
            ax[i].plot(tf_data)


    def plot_waveform(self):
        """Look at how the fit waveforms... look

        TODO: This relies on self.wf_model being a non-array, but its an array now?

        """
        # data = self.result_data
        # wf_model = self.wf_model
        data = self.wf_model.target_wf.windowed_wf
        data_len = len(data)

        plt.figure(figsize=(self.width,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_xlabel("Digitizer Time [ns]")
        ax0.set_ylabel("Voltage [Arb.]")
        ax1.set_ylabel("Residual")


        data_len = self.wf_model.target_wf.window_length
        t_data = np.arange(data_len) * 10
        ax0.plot(t_data, data, color=colors[0], ls = "-")

        num_to_plot = len(self.result_data.index)
        if num_to_plot > self.num_to_use:
            num_to_plot = self.num_to_use
        
        for (idx) in range(num_to_plot):
            # print ("wf %d max %d" % (idx, np.amax(data)))
            wf_params = self.result_data.iloc[idx].as_matrix()
            model = self.wf_model.make_waveform(data_len, wf_params, )

            if model is None:
                continue

            t_model = np.arange(data_len) * 10
            color_idx = idx % len(colors)
            ax0.plot(t_data, model, color=colors[color_idx], alpha=0.1)

            resid = model -  data
            ax1.plot(t_data, resid, color=colors[color_idx],alpha=0.1,)# linestyle="steps")

        ax0.set_ylim(-20, self.wf_model.target_wf.amplitude*1.1)
        ax0.axhline(y=0,color="black", ls=":")





class WaveformFitPlotter(ResultBase):
    def __init__(self, result_directory, num_samples, sample_dec, wf_model):
        # super().__init__(result_directory=result_directory, num_samples=num_samples, model_type="Waveform")
        print("I'm pretty sure this class is broken")
        raise NotImplementedError

        self.wf_model = wf_model

        self.dnest_postprocess(result_directory)
        self.parse_samples("posterior_sample.txt", result_directory, num_samples, sample_dec)
        # self.parse_samples("sample.txt", result_directory, num_samples, sample_dec)

        self.configuration = FitConfiguration.from_file(result_directory)

        # self.num_wf_params = self.model.num_wf_params
        # self.wf_conf = self.model.conf.wf_conf
        # self.model_conf = self.model.conf.model_conf

        # For plots:
        self.width = 18

    # def __init__(self, result_directory, num_samples=None, wf_model, posterior=False):
    #     super().__init__(result_directory=result_directory, num_samples=num_samples, posterior=posterior, waveform=True)


    def plot_trace(self):
        f, ax = plt.subplots(self.wf_model.num_params, 1, figsize=(self.width,10), sharex=True)
        for i in range(self.wf_model.num_params):
            tf_data = self.result_data[i]
            ax[i].plot(tf_data)

        # plt.show()

    def plot_waveform(self):
        data = self.result_data
        wf_model = self.wf_model
        wf = wf_model.target_wf
        wf_idx = 0

        plt.figure(figsize=(self.width,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_xlabel("Digitizer Time [ns]")
        ax0.set_ylabel("Voltage [Arb.]")
        ax1.set_ylabel("Residual")


        dataLen = wf.window_length
        t_data = np.arange(dataLen) * 10
        ax0.plot(t_data, wf.windowed_wf, color=colors[wf_idx], ls = ":")
        print ("wf %d max %d" % (wf_idx, np.amax(wf.windowed_wf)))

        for (idx) in range(len(data.index)):
            wf_params = data.iloc[idx].as_matrix()

            # wf_model.calc_likelihood(wf_params)
            # continue

            fit_wf = wf_model.make_waveform(wf.window_length,wf_params)
            if fit_wf is None:
                continue

            t_data = np.arange(wf.window_length) * 10
            color_idx = wf_idx % len(colors)
            ax0.plot(t_data,fit_wf, color=colors[color_idx], alpha=0.1)

            resid = fit_wf -  wf.windowed_wf
            ax1.plot(t_data, resid, color=colors[color_idx],alpha=0.1,)# linestyle="steps")

        ax0.set_ylim(-20, wf.amplitude*1.1)
        ax0.axhline(y=0,color="black", ls=":")

    def plot_waveform_old(self):
        data = self.result_data
        wf_model = self.wf_model
        wf = wf_model.target_wf
        wf_idx = 0

        data_wf = wf_model.target_wf.windowed_wf


        plt.figure(figsize=(self.width,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_xlabel("Digitizer Time [ns]")
        ax0.set_ylabel("Voltage [Arb.]")
        ax1.set_ylabel("Residual")


        dataLen = wf.window_length
        t_data = np.arange(dataLen) * 10
        # t_data = np.arange(1000) * 10
        # t_data = np.arange(100) * 100
        # t_data = t_data[::10]

        print("target wf has sample period of {} ns".format(wf.sample_period))
        # fit_target_wf = wf.windowed_wf
        fit_target_wf = wf
        # fit_target_wf.window_waveform(time_point=0.95, early_samples=200, num_samples=dataLen)
        fit_target_wf = wf.windowed_wf

        # ax0.plot(t_data, fit_target_wf, color=colors[wf_idx], ls = ":")
        ax0.plot(fit_target_wf, color=colors[wf_idx], ls = ":")
        
        print ("wf %d max %d" % (wf_idx, np.amax(fit_target_wf)))

        for (idx) in range(len(data.index)):
            wf_params = data.iloc[idx].as_matrix()
            # model_wf = self.wf_model.make_waveform(len(data_wf),wf_params)
            # if model_wf is None:
            #     continue
            fit_wf = wf_model.make_waveform(wf.window_length,wf_params)
            if fit_wf is None:
                continue
            
            t_data = np.arange(wf.window_length) * 10
            color_idx = wf_idx % len(colors)
            # ax0.plot(t_data,fit_wf, color=colors[color_idx], alpha=0.1)
            ax0.plot(fit_wf, color=colors[color_idx], alpha=0.1)

            # resid = fit_wf -  fit_target_wf
            # ax1.plot(t_data, resid, color=colors[color_idx],alpha=0.1,)# linestyle="steps")

        ax0.set_ylim(-20, wf.amplitude*1.1)
        ax0.axhline(y=0,color="black", ls=":")
        # plt.show()
        # ax0.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.set_ylim(-bad_wf_thresh, bad_wf_thresh)