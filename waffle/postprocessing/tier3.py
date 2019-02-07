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
from waffle.management import WaveformT3BatchConfiguration
from waffle.models import *
from waffle.models import Model, PulserTrainingModel

from .base import ResultBase

from pygama.processing import *
import pygama.decoders as dl

from siggen import PPC

import dnest4

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "brown", "deeppink", "goldenrod", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]


class Tier3FitResult(ResultBase):
    """
    Tier3itResult looks in a directory for the relavent files and figures out
    how the fit went. Can produce plots of the waveforms with residual, as well as
    looking at the trace from the chain.

    The fit results are stored in directories like:
    /data4/majorana/sjmeijer/mjd/wf_fits/DS1-1/11510/600/2450
                                           | |   |    |   |
    ds_____________________________________| |   |    |   |
    subset___________________________________|   |    |   |
    run__________________________________________|    |   |
    channel___________________________________________|   |
    index_________________________________________________|

    These directories are parsed out automatically when given the DS and all that


    Parameters
    ----------
    ds: int
        The dataset
    subset: int
        The subset
    run:
        The run number to analyze
    chan_list:
        A python list of channel numbers
    posterior:
        If true, samples from posterior rather than the chain directly.

    Returns
    -------

    Raises
    ------

    """
    # def __init__(self, batch_directory, num_samples, sample_dec, index=None, posterior=False):
    def __init__(self, ds,subset,run, chan_list=None, posterior=True):
        # Does not call the super init...


        # Set up the class variables
        self.ds = ds
        self.subset = subset
        self.run = run
        self.chan_list = chan_list
        self.posterior = posterior

        if self.posterior:
            self.sample_file_name = "posterior_sample.txt"
        else:
            self.sample_file_name = "sample.txt"



        # ***** Set up some directories:
        # self.batch_dir = batch_directory

        self.mjd_data_dir = os.path.join(os.getenv("DATADIR", "."), "mjd")
        # raw_data_dir = os.path.join(mjd_data_dir,"raw")
        self.t1_data_dir = os.path.join(self.mjd_data_dir,"t1")
        self.t2_data_dir = os.path.join(self.mjd_data_dir,"t2")
        self.t3_data_dir = os.path.join(self.mjd_data_dir,"t3")
        self.wf_fit_dir = os.path.join(self.mjd_data_dir,"wf_fits")
        # training_dir = os.path.join(mjd_data_dir,"training_fits")

        self.ds_subdir = os.path.join(
            self.wf_fit_dir,
            "DS{d}-{s}".format(d=self.ds,s=self.subset))
        
        self.run_subdir = os.path.join(
            self.wf_fit_dir,
            "DS{d}-{s}/{r}".format(d=self.ds,s=self.subset,r=self.run))



        if self.chan_list is None:
            self.chan_list = self.get_channels()

        print("Using the following channels:\n{}\n".format(self.chan_list))

        # setup the places to store results
        self.total_samples = {}
        self.result_data = {}
        # self.best_params = None
        self.best_params = {}

        for chan in self.chan_list:
            self.total_samples[chan] = {}
            self.result_data[chan] = {}
            self.best_params[chan] = {}

        # open the tier3 data file for writing into
        # t1_file = os.path.join(t1_data_dir,"t1_run{}.h5".format(run))
        # t2_file = os.path.join(t2_data_dir,"t2_run{}.h5".format(run))
        # t3_file = os.path.join(self.t3_data_dir,"t3_run{}.h5".format(run))

        # df1 = pd.read_hdf(t1_file,key="ORGretina4MWaveformDecoder")
        # df2 = pd.read_hdf(t2_file)

        # Set up df1 and df2 later if I need them
        self.df1 = None
        self.df2 = None
        self.load_df3()
        # self.df3 = pd.read_hdf(t3_file)


        print("Setting up the batch configuration stuff")
        self.wf_batch_conf = {}
        self.detector = {}
        for chan in self.chan_list:
            chan_subdir = "DS{d}-{s}/{r}/{c}".format(
                d=self.ds,
                s=self.subset,
                r=self.run,
                c=chan)
            chan_dir = os.path.join(self.wf_fit_dir,chan_subdir)

            self.wf_batch_conf[chan] = WaveformT3BatchConfiguration(None,None,None,None)
            self.wf_batch_conf[chan].load_config(chan_dir)
            self.detector[chan] = None

    def postprocess(self):
        """Loops over every run and 

        if it was smart it would check the file touched date and only postprocess if
        the sample.txt file was newer than the posterior_sample.txt file.

        """
        print("Postprocessing the results...")
        # I probably don't want to do this every time, maybe should check if it
        # has been done, and optionally force it to postprocess. 
        for chan in self.chan_list:
            chan_indices = self.get_indices(chan)
            chan_indices.sort()

            chan_subdir = "DS{d}-{s}/{r}/{c}".format(
                d=self.ds,
                s=self.subset,
                r=self.run,
                c=chan)
            chan_dir = os.path.join(self.wf_fit_dir,chan_subdir)

            # self.wf_batch_conf = WaveformT3BatchConfiguration(None,None,None,None)
            # self.wf_batch_conf.load_config(chan_dir)

            # self.setup_detector()
            # self.setup_configurations()

            for index in chan_indices:
                idx_subdir = "DS{d}-{s}/{r}/{c}/{i}".format(
                    d=self.ds,
                    s=self.subset,
                    r=self.run,
                    c=chan,
                    i=index)
                idx_dir = os.path.join(self.wf_fit_dir,idx_subdir)

                # old_pwd = os.getcwd()
                # os.chdir(idx_dir)

                log_z,info,weights = self.dnest_postprocess(idx_dir)
                # log_z,info,weights = dnest4.postprocess(plot=False,)
                # Weirdly, this doesn't give back the number of effective samples
                # but the source reads like it should? I'll just count it straight
                # from the output file (now its an integer though...)
                
                if(log_z is not None):
                    with open(os.path.join(
                        idx_dir,
                        'posterior_sample.txt')
                        ) as f:
                        eff_samples = sum(1 for _ in f)
                else:
                    # The postprocessing failed for this run, so we'll just let it be None
                    eff_samples = 0

                print(F"Writing values for {index}")
                self.df3.loc[index,'logZ'] = log_z
                self.df3.loc[index,'information'] = info
                self.df3.loc[index,'eff_samples'] = eff_samples

                # self.df3.loc[index]['logZ'] = log_z
                # self.df3.loc[index]['information'] = info
                # self.df3.loc[index]['eff_samples'] = eff_samples


    def find_failed_fits(self,thresh=1,verbose=False):
        """Identifies fits that did not get good results

        This usually means that the dnest postprocessing failed, or that the 
        number of effective samples was too low

        """

        self.failed_fits = {}

        for chan in self.chan_list:
            self.failed_fits[chan] = []
            df = self.df3[(self.df3.fittable==True)&(self.df3.channel==chan)]
            if(verbose):
                print(F"{chan}: ")
            for index,row in df.iterrows():
                if (row.eff_samples <= thresh) or (row.logZ == np.nan): 
                    self.failed_fits[chan].append(index)
                    if(verbose):
                        print(F"   {index}: {row.eff_samples}")

    def list_rerun_jobs(self):
        """Gives a nice formatted output to re-run failed fits
        """

        for chan in self.failed_fits.keys():
            for idx in self.failed_fits[chan]:
                command = "qsub t3_fit.py -l mem=16GB:walltime=04:00:00 -F\"-c {c} -d {d} -s {s} -r {r} -i {i} -f\"".format(
                    d=self.ds,
                    s=self.subset,
                    r=self.run,
                    c=chan,
                    i=idx
                )
                print(command)

        print("\nOr instead, as a bash one liner that should be safer to paste in\n")

        for chan in self.failed_fits.keys():
            print(F"channel {chan}, run {self.run}, DS{self.ds}-{self.subset}:")
            idx_list = [str(i)+',' for i in self.failed_fits[chan]]
            idx_str = "".join(idx_list)[:-1]

            qsub_command = "qsub t3_refit.py -F\"-c {c} -d {d} -s {s} -r {r} -i $idx -f\"".format(
                        d=self.ds,
                        s=self.subset,
                        r=self.run,
                        c=chan
                    )
            command = F"for idx in {{{idx_str}}};do echo {qsub_command}; {qsub_command}; done"

            # command = "for idx in {{{index_str}}};do echo $idx:; qsub t3_fit.py -l mem=16GB:walltime=04:00:00 -F'-c {c} -d {d} -s {s} -r {r} -i $idx -f'; done".format(
            #             index_str=idx_str,
            #             d=self.ds,
            #             s=self.subset,
            #             r=self.run,
            #             c=chan
            #         )
            print(command)

        print("\nSummary:")
        for chan in self.failed_fits.keys():
            print(F"Channel {chan}: {len(self.failed_fits[chan])} fails")



    def get_best_result(self):
        """
        
        r, z, phi, scale, maxt, smooth

        best_params is a nested array
        best_params[model_index][param_index] = param_avg
        """
        # self.result_data
        # raise NotImplementedError
        param_names = ['r', 'z', 'phi', 'scale', 'align', 'cloud']
 

        for chan in self.chan_list:
            for i in self.index:
                self.best_params[chan][i] = []
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

    def load_df1(self):
        # self.t1_data_dir = os.path.join(self.mjd_data_dir,"t1")
        # self.t2_data_dir = os.path.join(self.mjd_data_dir,"t2")
        # self.t3_data_dir = os.path.join(self.mjd_data_dir,"t3")
        # self.wf_fit_dir = os.path.join(self.mjd_data_dir,"wf_fits")
        # # training_dir = os.path.join(mjd_data_dir,"training_fits")

        print("Reading in t1 dataframe...")
        t1_file = os.path.join(self.t1_data_dir,"t1_run{}.h5".format(self.run))
        # t2_file = os.path.join(self.t2_data_dir,"t2_run{}.h5".format(run))
        # t3_file = os.path.join(self.t3_data_dir,"t3_run{}.h5".format(run))

        self.df1 = pd.read_hdf(t1_file,key="ORGretina4MWaveformDecoder")
        # df2 = pd.read_hdf(t2_file)
        # self.df3 = pd.read_hdf(t3_file)


    def load_df2(self):
        # self.t1_data_dir = os.path.join(self.mjd_data_dir,"t1")
        # self.t2_data_dir = os.path.join(self.mjd_data_dir,"t2")
        # self.t3_data_dir = os.path.join(self.mjd_data_dir,"t3")
        # self.wf_fit_dir = os.path.join(self.mjd_data_dir,"wf_fits")
        # # training_dir = os.path.join(mjd_data_dir,"training_fits")

        print("Reading in t2 dataframe...")
        # t1_file = os.path.join(self.t1_data_dir,"t1_run{}.h5".format(run))
        t2_file = os.path.join(self.t2_data_dir,"t2_run{}.h5".format(self.run))
        # t3_file = os.path.join(self.t3_data_dir,"t3_run{}.h5".format(run))

        # df1 = pd.read_hdf(t1_file,key="ORGretina4MWaveformDecoder")
        self.df2 = pd.read_hdf(t2_file)
        # self.df3 = pd.read_hdf(t3_file)


    # def save_t3(run_df,runNumber,save_dir=None):
    #     if save_dir is None: save_dir = t3_data_dir

    #     t3_file = os.path.join(save_dir, "t3_run{}.h5".format(runNumber))
    #     run_df.to_hdf(t3_file, key="data", format='table', mode='w', data_columns=run_df.columns.tolist() )

    def load_df3(self):
        print("Reading in t3 dataframe...")

        t3_file = os.path.join(self.t3_data_dir,"t3_run{}.h5".format(self.run))
        self.df3 = pd.read_hdf(t3_file)


    def save_tier3_dataframe(self,save_dir=None):
        """Saves the tier3 dataframe back in place, overwriting with the new
        calculated parameters inside.

        Should be called after doing all the push_samples_to_df() is done.
        """
        if save_dir is None: 
            save_dir = self.t3_data_dir 

        # self.df3.reset_index(inplace=True)
        # self.df3.set_index("event_number", inplace=True)

        for key in self.df3.keys():
            self.df3[key] = pd.to_numeric(self.df3[key])

        # t3f.df3.r = pd.to_numeric(t3f.df3.r)
        # t3f.df3.z = pd.to_numeric(t3f.df3.z)
        # t3f.df3.phi = pd.to_numeric(t3f.df3.phi)
        # t3f.df3.scale = pd.to_numeric(t3f.df3.scale)
        # t3f.df3.t0 = pd.to_numeric(t3f.df3.t0)
        # t3f.df3.cloud = pd.to_numeric(t3f.df3.cloud)

        # t3f.df3.r_unc = pd.to_numeric(t3f.df3.r)
        # t3f.df3.z_unc = pd.to_numeric(t3f.df3.z)
        # t3f.df3.phi_unc = pd.to_numeric(t3f.df3.phi)
        # t3f.df3.scale_unc = pd.to_numeric(t3f.df3.scale)
        # t3f.df3.t0_unc = pd.to_numeric(t3f.df3.t0)
        # t3f.df3.cloud_unc = pd.to_numeric(t3f.df3.cloud)

        t3_file = os.path.join(save_dir, "t3_run{}.h5".format(self.run))
        self.df3.to_hdf(t3_file, key="data", format='table', mode='w', data_columns=self.df3.columns.tolist() )



    def get_indices(self,channel):
        """Pulls out all the directories within the channel batch dir where 
        the fits were performed
        Then, returns a list of the indices themselves for that channel
        """
        fits = []

        fit_subdir = "DS{d}-{s}/{r}/{c}".format(d=self.ds,s=self.subset,r=self.run,c=channel)
        chan_subdir = os.path.join(
            self.wf_fit_dir,
            fit_subdir)

        old_pwd = os.getcwd()
        os.chdir(chan_subdir)

        ls = os.listdir()
        # filter out any responses that are not directory names
        indices = [int(value) for value in ls if os.path.isdir(value)]

        os.chdir(old_pwd)
        return indices

    def get_channels(self):
        """Pulls out all the directories within the run batch dir where 
        the fits were performed
        Then, returns a list of the channels themselves for that run
        """
        fits = []

        fit_subdir = "DS{d}-{s}/{r}".format(d=self.ds,s=self.subset,r=self.run)
        chan_subdir = os.path.join(
            self.wf_fit_dir,
            fit_subdir)

        old_pwd = os.getcwd()
        os.chdir(chan_subdir)

        ls = os.listdir()
        # filter out any responses that are not directory names
        channels = [int(value) for value in ls if os.path.isdir(value)]

        os.chdir(old_pwd)
        return channels



    # def extract_params(self):
        


    def parse_samples(self, channel=None, index=None):
        """Load the sample data as CSV (from .txt) format into the result_data dict
        This doesn't process the data at all, just takes it in for extracting later

        If you have it set up to do posterior sampling, it will run the postprocess
        which will then add the logZ, information, and effective sample values
        into the dataframe (although it isn't saved yet)

        Probably will get called once for each fit waveform 
        Uses pandas so its fast!
        """

        if isinstance(channel,int):
            # if its a single value, make it look like a list
            channel = [channel]
        if channel is None:
            # if left blank we'll just do them all for that channel
            channel = self.chan_list


        for chan in channel:

            if isinstance(index,int):
                # if its a single value, make it look like a list
                index = [index]
            if index is None:
                # if left blank we'll just do them all for that channel
                index = self.get_indices(chan)

            for idx in index:

                idx_subdir = "DS{d}-{s}/{r}/{c}/{i}".format(
                    d=self.ds,
                    s=self.subset,
                    r=self.run,
                    c=chan,
                    i=idx)
                idx_dir = os.path.join(self.wf_fit_dir,idx_subdir)

                sample_file_path = os.path.join(idx_dir, self.sample_file_name)

                if not os.path.isfile(sample_file_path):
                    # print("The file {} does not exist...".format(sample_file_path))
                    print('\033[91m'+"The file {} does not exist...".format(sample_file_path)+'\033[0m')
                    continue
                    # print("Quitting...")
                    # raise FileExistsError

                data = pd.read_csv(sample_file_path, delim_whitespace=True, header=None)
                total_samples = len(data.index)
                self.total_samples[chan][idx] = total_samples

                print("Found {} samples... ".format(total_samples), end='')
                self.result_data[chan][idx] = data
                print( "Using the last {} samples".format( len(self.result_data[chan][idx])) )

        # directory = idx_dir
        # num_to_read = self.num_samples
        # sample_dec = self.sample_dec


        # if self.posterior:
        #     print("Evaluating DNest4 posterior...")
        #     log_z,info,weights = self.dnest_postprocess(directory)
        #     # log_z,info,weights = dnest4.postprocess(plot=False,)
        #     # Weirdly, this doesn't give back the number of effective samples
        #     # but the source reads like it should? I'll just count it straight
        #     # from the output file (now its an integer though...)
            
        #     with open('posterior_sample.txt') as f:
        #         eff_samples = sum(1 for _ in f)

        #     self.df3.loc[index]['logZ'] = log_z
        #     self.df3.loc[index]['information'] = info
        #     self.df3.loc[index]['eff_samples'] = eff_samples

        # else:
        #     print("Using the raw chain, not the posterior...")
        #     # self.parse_samples("sample.txt", i) #d, num_samples, sample_dec)




    def push_samples_to_df(self):
        param_names = ['r', 'z', 'phi', 'scale', 'align', 'cloud']
        param_names = ['r', 'z', 'phi', 'scale', 't0', 'cloud']
        for chan in self.result_data:
            for index in self.result_data[chan]:
                params = self.result_data[chan][index]

                for i,param in enumerate(params):
                    param_name = param_names[i]

                    param_mean = np.mean(params[i])
                    param_unc = np.std(params[i])

                    # print(F"Writing values for {index}")
                    self.df3.loc[index,param_name] = param_mean
                    self.df3.loc[index,param_name+'_unc'] = param_unc


    def push_likelihood_to_df(self):
        """Adds the fit likelihood measure into the df

        """
        name = 'ln_like'

        for chan in self.ln_like:
            for index in self.ln_like[chan]:
                value = self.ln_like[chan][index]

                # print(F"Writing values for {index}")
                self.df3.loc[index,name] = value

    def setup_detector(self):
        # ***** Get the detector set up:

        self.detector = {}
        for chan in self.chan_list:
            self.detector[chan] = PPC(self.wf_batch_conf[chan].detector_conf, wf_padding=100)
            for m in self.wf_batch_conf[chan].models:
                
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
                    model = ImpurityModelEnds(self.detector[chan])
                elif model_name == "HiPassFilterModel":
                    model = HiPassFilterModel(
                        detector=self.detector[chan],
                    )
                elif model_name == "LowPassFilterModel":
                    model = LowPassFilterModel(
                        self.detector[chan], 
                        # order=the_model.order, 
                        # include_zeros=the_model.include_zeros, 
                        # pmag_lims=the_model.pmag_lims,
                        # pphi_lims=the_model.pphi_lims, 
                        # zmag_lims=the_model.zmag_lims, 
                        # zphi_lims=the_model.zphi_lims, 
                    )
                elif model_name ==  "OvershootFilterModel":
                    model = OvershootFilterModel(self.detector[chan])
                elif model_name ==  "OscillationFilterModel":
                    model = OscillationFilterModel(self.detector[chan])
                elif model_name == "AntialiasingFilterModel":
                    model = AntialiasingFilterModel(self.detector[chan])
                elif model_name == "FirstStageFilterModel":
                    model = FirstStageFilterModel(self.detector[chan])
                elif model_name ==  "TrappingModel":
                    model = TrappingModel()
                else:
                    continue
                    # raise ValueError("model_name {} is not a valid model".format(model_name))

                print("   Applying these values to the model...")
                model.apply_to_detector(params,self.detector[chan])
            
            # For plots:
            self.width = 18


    def get_wf(self,channel,index,verbose=False):
        """Pulls a given waveform from the tier 1 data

        """
        if(verbose):
            print(F"Getting wf {index} directly for chan {channel}...")

        t1_file = os.path.join(self.t1_data_dir,"t1_run{}.h5".format(self.run))

        g4 = dl.Gretina4MDecoder(t1_file)

        if self.df1 is None:
            self.load_df1()
        if self.df2 is None:
            self.load_df2()

        row1 = self.df1.loc[index]
        row2 = self.df2.loc[index]

        wf = g4.parse_event_data(row1)

        wf.training_set_index = index
        wf.amplitude = row2.trap_max
        wf.bl_slope = row2.bl_slope
        wf.bl_int = row2.bl_int
        wf.t0_estimate = row2.t0est
        wf.tp_50 = row2.tp_50

        wf.window_waveform(
            time_point=self.wf_batch_conf[channel].align_percent, 
            early_samples=200, 
            num_samples=self.wf_batch_conf[channel].num_samples #200
            )

        return wf


    def sim_best_waveform(self, chan, index, data_len=None, charge_type=None):
        """Simulates a waveform with the parameters that were fit out of the wf
        These "best" guess parameters are 

        """
        r = self.df3.loc[index].r
        z = self.df3.loc[index].z
        phi = self.df3.loc[index].phi
        scale = self.df3.loc[index].scale
        maxt = self.df3.loc[index].t0

        smooth = self.df3.loc[index].cloud
        skew = None

        if r is None:
            print("Attempting to simulate for a wf with no info in dataframe")
            print("Check that the dataframe has had samples pushed into it")
            return None

        # if self.do_smooth:
        # smooth = wf_params[5]
        if smooth < 0:
            raise ValueError("Smooth should not be below 0 (value {})".format(smooth))
        # if self.smoothing_type == "skew":
        #     skew = wf_params[6]

        if not self.detector[chan]:
            self.setup_detector()
        if data_len is None:
            data_len = self.wf_batch_conf[chan].num_samples

        # r = rad * np.cos(theta)
        # z = rad * np.sin(theta)

        if scale < 0:
            raise ValueError("Scale should not be below 0 (value {})".format(scale))

        if not self.detector[chan].IsInDetector(r, phi, z):
            raise ValueError("Point {},{},{} is outside detector.".format(r,phi,z))
            # import time
            # print('\033[91m'+"Point {},{},{} is outside detector.+'\033[0m'".format(r,phi,z))
            # print("I'll attempt to continue, assuming this is a valid point..")
            # time.sleep(3)

        if charge_type is None:
                model = self.detector[chan].MakeSimWaveform(
                    r, phi, z, scale, 
                    maxt,self.wf_batch_conf[chan].align_percent, 
                    data_len, 
                    smoothing=smooth, 
                    skew=skew)
                # model = self.detector.GetWaveform(r, phi, z, scale)
        elif charge_type == 1:
            model = self.detector[chan].MakeWaveform(r, phi, z,1)[0,:]
        elif charge_type == -1:
            model = self.detector[chan].MakeWaveform(r, phi, z,-1)[0,:]
        else:
            raise ValueError("Not a valid charge type! {0}".format(charge_type))

        if model is None or np.any(np.isnan(model)):
            return None


        return model

    def calc_likelihood(self, channel=None, index=None):
        """ Calculate the total likelihood for the resulting fit

        Gives you a ln_like[chan][index], with the likelihood for each wf
        """

        if isinstance(channel,int):
            # if its a single value, make it look like a list
            channel = [channel]
        if channel is None:
            # if left blank we'll just do them all for that channel
            channel = self.chan_list

        ln_like = {}

        for chan in channel:
            if isinstance(index,int):
                # if its a single value, make it look like a list
                index = [index]
            if index is None:
                # if left blank we'll just do them all for that channel
                index = self.get_indices(chan)

            ln_like[chan] = {}

            for idx in index:
                data = self.get_wf(channel=chan,index=idx).windowed_wf
                # model_err = 0.57735027 * wf.baselineRMS
                model_err = 2.5 #TODO: get this from the waveform itself
                data_len = len(data)

                model = self.sim_best_waveform(chan,idx)

                if model is None:
                    ln_like[chan][idx] = -np.inf
                else:
                    inv_sigma2 = 1.0/(model_err**2)
                    ln_like[chan][idx] = -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))

        self.ln_like = ln_like

        return ln_like

    def setup_configurations(self):
        # self.fit_configuration = [FitConfiguration.from_file(fd) for fd in self.fit_dir]

        raise NotImplementedError

        print("Loading target wf from conf...")
        self.wf_model = {}
        # for chan in 
        for i in self.index:
            wf = self.get_wf_from_conf(i)
            wf.window_waveform(
                time_point=self.wf_batch_conf[chan].align_percent, 
                early_samples=200, 
                num_samples=self.wf_batch_conf[chan].num_samples #200
                )
            # wf = wf_batch_conf

            self.wf_model[i] = (WaveformModel(
                wf,
                align_percent=self.wf_batch_conf[chan].align_percent,
                detector = self.detector[chan],
                align_idx = 200,
            ))

    # def get_wf(self,index):
    #     """Just a handy way to pull out the waveform from the tier1 file

    #     Parameters
    #     ----------
    #     index: int
    #         The index number of the waveform you want out of the file

    #     Returns
    #     -------
    #     wf: WaveformModel
    #         The waveform you asked for

    #     Raises
    #     ------
    #     FileNotFoundError
    #         You probably moved the files around, or something is really wrong

    #     """
        
    #     print("Loading waveform from file {}...\n".format(self.wf_batch_conf.wf_file_name))
    #     data = np.load(self.wf_batch_conf.wf_file_name, encoding="latin1")
    #     wfs = data['wfs']
    #     # wf = wfs[conf.wf_idxs[0]]
    #     wf = wfs[index]

    #     return wf




    # def plot_best_waveform(self):

    #     if(self.best_params is None):
    #         self.get_best_result()

    #     # for wf_model, best_params in zip(self.wf_model,self.best_params):
    #     for i in self.index:

    #         wf_model = self.wf_model[i]
    #         best_params = self.best_params[i]

    #         target_wf = wf_model.target_wf.windowed_wf
    #         data_len = len(target_wf)

    #         plt.figure(figsize=(self.width,8))
    #         gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    #         ax0 = plt.subplot(gs[0])
    #         ax1 = plt.subplot(gs[1], sharex=ax0)
    #         ax1.set_xlabel("Digitizer Time [ns]")
    #         ax0.set_ylabel("Voltage [Arb.]")
    #         ax1.set_ylabel("Residual")

    #         # plot the target wf with a time axis
    #         data_len = wf_model.target_wf.window_length
    #         t_data = np.arange(data_len) * 10
    #         ax0.plot(t_data, target_wf, color='red', ls = "-")

    #         # generate the fit-determined best wf
    #         best_wf = wf_model.make_waveform(data_len, best_params, )

    #         if best_wf is None:
    #             #cry
    #             return

    #         # plot the best waveform
    #         t_model = np.arange(data_len) * 10
    #         ax0.plot(t_data, best_wf, color='blue', alpha=0.5)

    #         # plot a residual
    #         resid = best_wf -  target_wf
    #         ax1.plot(t_data, resid, color='red', alpha=0.5,)# linestyle="steps")

    #         ax0.set_ylim(-20, wf_model.target_wf.amplitude*1.1)
    #         ax0.axhline(y=0,color="black", ls=":")

    # def plot_best_fft(self):
    #     """ Plots the FFT of the target and best waveforms next to each other
    #     """

    #     if(self.best_params is None):
    #         self.get_best_result()

    #     # for wf_model, best_params in zip(self.wf_model,self.best_params):
    #     for i in self.index:

    #         wf_model = self.wf_model[i]
    #         best_params = self.best_params[i]

    #         target_wf = wf_model.target_wf.windowed_wf
    #         data_len = len(target_wf)

    #         plt.figure(figsize=(self.width,8))
    #         gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    #         ax0 = plt.subplot(gs[0])
    #         ax1 = plt.subplot(gs[1], sharex=ax0)
    #         ax1.set_xlabel("Frequency [MHz]")
    #         ax0.set_ylabel("Power [Arb.]")
    #         ax1.set_ylabel("Residual")

    #         # Calculate the target_wf FFT
    #         target_fft = np.fft.fft(target_wf)
    #         target_fft_freq = np.fft.fftfreq(len(target_wf),10*1e-9) * 1e-6

    #         ## only use the positive real part of the FFT
    #         fft_len_pos = int(np.floor(len(target_wf)/2))
    #         target_fft = np.abs(target_fft[:fft_len_pos])
    #         target_fft_freq = target_fft_freq[:fft_len_pos]

    #         # plot the target wf FFT with a freq axis
    #         # data_len = wf_model.target_wf.window_length
    #         # t_data = np.arange(data_len) * 10
    #         ax0.semilogy(target_fft_freq, target_fft, color='red', ls = "-")

    #         # generate the fit-determined best wf
    #         best_wf = wf_model.make_waveform(data_len, best_params, )

    #         if best_wf is None:
    #             #cry
    #             return

    #         # Calculate the best wf FFT
    #         best_fft = np.fft.fft(best_wf)
    #         best_fft_freq = np.fft.fftfreq(len(best_wf),10*1e-9) * 1e-6

    #         ## only use the positive real part of the FFT
    #         fft_len_pos = int(np.floor(len(target_wf)/2))
    #         best_fft = np.abs(best_fft[:fft_len_pos])
    #         best_fft_freq = best_fft_freq[:fft_len_pos]

    #         # plot the best waveform FFT
    #         t_model = np.arange(data_len) * 10
    #         ax0.semilogy(best_fft_freq, best_fft, color='blue', alpha=0.5)


    #         # plot a residual
    #         resid = best_fft -  target_fft
    #         ax1.plot(best_fft_freq, resid, color='red', alpha=0.5,)# linestyle="steps")

    #         # ax0.set_ylim(-20, wf_model.target_wf.amplitude*1.1)
    #         ax0.axhline(y=0,color="black", ls=":")


    # def plot_avg_residual(self):
    #     """ Calculates each residual
    #     plots them all together
    #     plots the average of all f them
    #     plots the average of them after scaling? to compare different sized wfs?
    #     """
    #     if(self.best_params is None):
    #         self.get_best_result()

    #     plt.figure(figsize=(self.width,8))
    #     gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    #     ax0 = plt.subplot(gs[0])
    #     ax1 = plt.subplot(gs[1], sharex=ax0)
    #     ax1.set_xlabel("Digitizer Time [ns]")
    #     ax0.set_ylabel("Voltage [Arb.]")
    #     ax1.set_ylabel("Residual")

    #     residual = []

    #     # for i,(wf_model, best_params) in enumerate(zip(self.wf_model,self.best_params)):
    #     for i in self.index:
    #         wf_model = self.wf_model[i]
    #         best_params = self.best_params[i]

    #         i_color = i % len(colors)

    #         target_wf = wf_model.target_wf.windowed_wf
    #         data_len = len(target_wf)

    #         # plot the target wf with a time axis
    #         data_len = wf_model.target_wf.window_length
    #         t_data = np.arange(data_len) * 10
    #         # ax0.plot(t_data, target_wf, color='red', ls = "-")

    #         # generate the fit-determined best wf
    #         best_wf = wf_model.make_waveform(data_len, best_params, )

    #         if best_wf is None:
    #             #cry
    #             return

    #         # plot the residuals together
    #         # plot a residual
    #         resid = best_wf -  target_wf
    #         residual.append(resid)

    #         ax1.plot(t_data, resid, color=colors[i_color], alpha=0.2,)# linestyle="steps")

    #     # ax0.set_ylim(-20, wf_model.target_wf.amplitude*1.1)
    #     avg_residual = np.mean(residual,0)

    #     ax0.axhline(y=0,color="black", ls=":")
    #     t_model = np.arange(data_len) * 10
    #     ax0.plot(t_data, avg_residual, color='blue', alpha=0.9)

    #     return avg_residual

    # def plot_best_waveform_residual_fft(self):

    #     if(self.best_params is None):
    #         self.get_best_result()

    #     # for wf_model, best_params in zip(self.wf_model,self.best_params):
    #     for i in self.index:

    #         wf_model = self.wf_model[i]
    #         best_params = self.best_params[i]

    #         target_wf = wf_model.target_wf.windowed_wf
    #         data_len = len(target_wf)

    #         plt.figure(figsize=(self.width,8))
    #         gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    #         ax0 = plt.subplot(gs[0])
    #         ax1 = plt.subplot(gs[1], sharex=ax0)
    #         ax1.set_xlabel("Digitizer Time [ns]")
    #         ax0.set_ylabel("Voltage [Arb.]")
    #         ax1.set_ylabel("Residual")


    #         # Calculate the target_wf FFT
    #         target_fft = np.fft.fft(target_wf)
    #         target_fft_freq = np.fft.fftfreq(len(target_wf),10*1e-9) * 1e-6

    #         ## only use the positive real part of the FFT
    #         fft_len_pos = int(np.floor(len(target_wf)/2))
    #         target_fft = np.abs(target_fft[:fft_len_pos])
    #         target_fft_freq = target_fft_freq[:fft_len_pos]

    #         # plot the target wf FFT with a freq axis
    #         # data_len = wf_model.target_wf.window_length
    #         # t_data = np.arange(data_len) * 10
    #         # ax0.plot(target_fft_freq, target_fft, color='red', ls = "-")
    #         ax0.semilogy(target_fft_freq, target_fft, color='red', ls = "-")

    #         # generate the fit-determined best wf
    #         best_wf = wf_model.make_waveform(data_len, best_params, )

    #         if best_wf is None:
    #             #cry
    #             return

    #         # Calculate the best wf FFT
    #         best_fft = np.fft.fft(best_wf)
    #         best_fft_freq = np.fft.fftfreq(len(best_wf),10*1e-9) * 1e-6

    #         ## only use the positive real part of the FFT
    #         fft_len_pos = int(np.floor(len(target_wf)/2))
    #         best_fft = np.abs(best_fft[:fft_len_pos])
    #         best_fft_freq = best_fft_freq[:fft_len_pos]

    #         # plot the best waveform FFT
    #         t_model = np.arange(data_len) * 10
    #         # ax0.plot(best_fft_freq, best_fft, color='blue', alpha=0.5)
    #         ax0.semilogy(best_fft_freq, best_fft, color='blue', alpha=0.5)

    #         # Calculate the residual fft and the frequency basis (in MHz)
    #         resid = best_wf -  target_wf
    #         resid_fft = np.fft.fft(resid)
    #         resid_fft_freq = np.fft.fftfreq(len(resid),10*1e-9) * 1e-6

    #         # only use the positive part of the FFT
    #         fft_len_pos = int(np.floor(len(resid)/2))
    #         resid_fft = resid_fft[:fft_len_pos]
    #         resid_fft_freq = resid_fft_freq[:fft_len_pos]

    #         ax1.plot(resid_fft_freq, resid_fft, color='red', alpha=0.5,)# linestyle="steps")

    #         # ax0.set_ylim(-20, wf_model.target_wf.amplitude*1.1)
    #         # ax0.axhline(y=0,color="black", ls=":")


    # def plot_avg_fft_residual(self):
    #     """ Calculates the FFT of each residual
    #     plots them all together
    #     plots the average of all of them
    #     """
    #     if(self.best_params is None):
    #         self.get_best_result()

    #     plt.figure(figsize=(self.width,8))
    #     gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    #     ax0 = plt.subplot(gs[0])
    #     ax1 = plt.subplot(gs[1], sharex=ax0)
    #     ax1.set_xlabel("Frequency [MHz]")
    #     ax0.set_ylabel("Power [Arb.]")
    #     ax1.set_ylabel("Residual")

    #     residual = []
    #     residual_fft = []

    #     resid_fft_freq = None

    #     # for i,(wf_model, best_params) in enumerate(zip(self.wf_model,self.best_params)):
    #     for i in self.index:
    #         wf_model = self.wf_model[i]
    #         best_params = self.best_params[i]

    #         i_color = i % len(colors)

    #         target_wf = wf_model.target_wf.windowed_wf
    #         data_len = len(target_wf)

    #         # plot the target wf with a time axis
    #         # data_len = wf_model.target_wf.window_length
    #         # t_data = np.arange(data_len) * 10
    #         # ax0.plot(t_data, target_wf, color='red', ls = "-")

    #         # generate the fit-determined best wf
    #         best_wf = wf_model.make_waveform(data_len, best_params, )

    #         if best_wf is None:
    #             #cry
    #             return

    #         resid = best_wf -  target_wf

    #         # Calculate the residual fft and the frequency basis (in MHz)
    #         resid_fft = np.fft.fft(resid)
    #         resid_fft_freq = np.fft.fftfreq(len(resid),10*1e-9) * 1e-6

    #         # only use the positive real part of the FFT
    #         fft_len_pos = int(np.floor(len(resid)/2))
    #         resid_fft = np.abs(resid_fft[:fft_len_pos])
    #         resid_fft_freq = resid_fft_freq[:fft_len_pos]

    #         residual.append(resid)
    #         residual_fft.append(resid_fft)

    #         # ax1.plot(t_data, resid, color=colors[i_color], alpha=0.2,)# linestyle="steps")
    #         ax1.plot(resid_fft_freq, resid_fft, color=colors[i_color], alpha=0.3,)# linestyle="steps")


    #     # Plot the average of the FFTs of the residuals
    #     avg_residual_fft = np.mean(residual_fft,0)

    #     # only use the positive part of the FFT
    #     # fft_len_pos = len(avg_residual_fft)

    #     # resid_fft_freq = np.fft.fftfreq(fft_len_pos*2,10*1e-9) * 1e-6
    #     # resid_fft_freq = resid_fft[:fft_len_pos]

    #     ax0.axhline(y=0,color="black", ls=":")
    #     # t_model = np.arange(data_len) * 10

    #     ax0.plot(resid_fft_freq, avg_residual_fft, color='blue', alpha=0.9)



    # def plot_trace(self):
    #     f, ax = plt.subplots(self.wf_model[self.index[0]].num_params, 2, figsize=(self.width,10), sharex=True)

    #     for i in range(self.wf_model[self.index[0]].num_params):
    #         tf_data = self.result_data[i]
    #         ax[i].plot(tf_data)


    # def plot_waveform(self):
    #     """Look at how the fit waveforms... look

    #     """
    #     # data = self.result_data
    #     # wf_model = self.wf_model
    #     data = self.wf_model.target_wf.windowed_wf
    #     data_len = len(data)

    #     plt.figure(figsize=(self.width,8))
    #     gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    #     ax0 = plt.subplot(gs[0])
    #     ax1 = plt.subplot(gs[1], sharex=ax0)
    #     ax1.set_xlabel("Digitizer Time [ns]")
    #     ax0.set_ylabel("Voltage [Arb.]")
    #     ax1.set_ylabel("Residual")


    #     data_len = self.wf_model.target_wf.window_length
    #     t_data = np.arange(data_len) * 10
    #     ax0.plot(t_data, data, color=colors[0], ls = "-")

    #     num_to_plot = len(self.result_data.index)
    #     if num_to_plot > self.num_to_use:
    #         num_to_plot = self.num_to_use
        
    #     for (idx) in range(num_to_plot):
    #         # print ("wf %d max %d" % (idx, np.amax(data)))
    #         wf_params = self.result_data.iloc[idx].as_matrix()
    #         model = self.wf_model.make_waveform(data_len, wf_params, )

    #         if model is None:
    #             continue

    #         t_model = np.arange(data_len) * 10
    #         color_idx = idx % len(colors)
    #         ax0.plot(t_data, model, color=colors[color_idx], alpha=0.1)

    #         resid = model -  data
    #         ax1.plot(t_data, resid, color=colors[color_idx],alpha=0.1,)# linestyle="steps")

    #     ax0.set_ylim(-20, self.wf_model.target_wf.amplitude*1.1)
    #     ax0.axhline(y=0,color="black", ls=":")

