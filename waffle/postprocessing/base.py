import sys, os, shutil
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

import dnest4

from waffle.management import FitConfiguration, WaveformConfiguration
from waffle.models import *
from waffle.models import Model, PulserTrainingModel

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "brown", "deeppink", "goldenrod", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]



class ResultBase():
    def __init__(self, result_directory, num_samples=None, sample_dec=1, model_type="Model"):#, posterior=False, waveform=False):
        """The base class used for all the postprocessing checks.
        result_directory: The directory containing the sample.txt etc
        num_samples: The number of samples to look at from the selected chain (will use the most recent N)
        sample_dec: The decimation factor, by which to only look at some fraction of the result
        model_type: Probably just want to use "Model"
        posterior: Sample the chain or the posterior.
        waveform: Was it a waveform fit (true) or a training fit (false)?
        """
        self.parse_samples("sample.txt", result_directory, num_samples, sample_dec)

        self.configuration = FitConfiguration(directory=result_directory, loadSavedConfig=True)

        if model_type == "Model":
            self.model = Model(self.configuration)
        elif model_type == "PulserTrainingModel":
            self.model = PulserTrainingModel(self.configuration)
        # elif model_type == "Waveform":
        #     self.model = 

        self.num_wf_params = self.model.num_wf_params
        self.wf_conf = self.model.conf.wf_conf
        self.model_conf = self.model.conf.model_conf

        self.id = result_directory.replace("/","")

        # For plots:
        self.width = 18



        # # if(not posterior):
        #     # This samples the chain
        # self.parse_samples("sample.txt", result_directory, num_samples, sample_dec)
        # # else:
        # #     # This samples the posterior
        # #     self.dnest_postprocess(directory=result_directory)
        # #     self.parse_samples("posterior_sample.txt", result_directory, num_samples, sample_dec)

        # if model_type == 'Waveform':
        #     self.configuration = FitConfiguration.from_file(result_directory)
        #     self.model = WaveformModel(self.configuration)
        # else:
        #     self.configuration = WaveformConfiguration(wf_file_name=None,directory=result_directory, loadSavedConfig=True)
        #     if model_type == "Model":
        #         self.model = Model(self.configuration)
        #     elif model_type == "PulserTrainingModel":
        #         self.model = PulserTrainingModel(self.configuration)

        # self.num_wf_params = self.model.num_wf_params
        # self.wf_conf = self.model.conf.wf_conf
        # self.model_conf = self.model.conf.model_conf

        # self.id = result_directory.replace("/","")
        # self.num_samples = num_samples

        # # For plots:
        # self.width = 18

    def parse_samples(self, sample_file_name, directory, num_to_read, sample_dec=1):
        """Load the data from CSV
        Uses pandas so its fast
        """
        sample_file_name = os.path.join(directory, sample_file_name)

        if not os.path.isfile(sample_file_name):
            print("The file {} does not exist...".format(sample_file_name))
            print("Quitting...")
            raise FileExistsError

        data = pd.read_csv(sample_file_name, delim_whitespace=True, header=None)
        total_samples = len(data.index)
        self.total_samples = total_samples

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
            self.result_data = data
        elif total_samples >= num_to_read:
            total_samples = num_to_read
            end_idx = len(data.index) - 1
            self.result_data = data.iloc[(end_idx - total_samples):end_idx:sample_dec]
        elif total_samples < num_to_read:
            self.result_data = data

        print( "Using the last {} samples".format( len(self.result_data)) )

    # def average_param_vals(self):

    #     params_avg_values = {}
    #     pv = self.params_values
    #     for model_key in pv:
    #         for param_key in pv[model_key]:
    #             params_avg_values  = np.mean(pv[model_key][param_key])


    def dnest_postprocess(self,directory):
        # Moves into the result directory to postprocess, then moves back to wherever we were
        # TODO: Should probably check that directories and files are valid...

        old_dir = os.getcwd()
        os.chdir(directory)
        dnest4.postprocess(plot=False)
        os.chdir(old_dir)

        return


    def multipage_plot(self,filename, figs=None, dpi=200):
        pp = PdfPages(filename)
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()