#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from waffle.management import FitConfiguration
from waffle.models import *
from waffle.models import Model, PulserTrainingModel

from .base import ResultBase

colors = ["red" ,"blue", "green", "purple", "orange", "cyan", "magenta", "brown", "deeppink", "goldenrod", "lightsteelblue", "maroon", "violet", "lawngreen", "grey", "chocolate" ]


class TrainingPlotter(ResultBase):
    def __init__(self, result_directory, num_samples=None, sample_dec=1, model_type="Model"):
        super().__init__(result_directory, num_samples, sample_dec)

        # configuration = FitConfiguration(directory=result_directory, loadSavedConfig=True)

        # if model_type == "Model":
        #     self.model = Model(configuration)
        # elif model_type == "PulserTrainingModel":
        #     self.model = PulserTrainingModel(configuration)

        # self.num_wf_params = self.model.num_wf_params

        # end_idx = 10000

    def plot_waveforms(self, print_det_params=False):
        data = self.result_data
        model = self.model

        bad_wf_thresh = 1000

        plt.figure(figsize=(self.width,8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_xlabel("Digitizer Time [ns]")
        ax0.set_ylabel("Voltage [Arb.]")
        ax1.set_ylabel("Residual")

        num_det_params = model.num_det_params
        resid_arr = np.zeros(model.wfs[0].window_length)


        for wf_idx, wf in enumerate(model.wfs):
          dataLen = wf.window_length
          t_data = np.arange(dataLen) * 10
          ax0.plot(t_data, wf.windowed_wf, color=colors[wf_idx % len(colors)], ls = ":")
          print ("wf %d max %d" % (wf_idx, np.amax(wf.windowed_wf)))

        for (idx) in range(len(data.index)):
            params = data.iloc[idx].as_matrix()
            if print_det_params: print(params[:self.model.num_det_params])

            self.model.joint_models.apply_params(params)

            for (wf_idx,wf) in enumerate(model.wfs):
                # if wf_idx < 4: continue
                # wfs_param_arr[-1,wf_idx] = 1
                wf_params =  params[model.num_det_params + wf_idx*model.num_wf_params: model.num_det_params + (wf_idx+1)*self.num_wf_params]

                fit_wf = model.wf_models[wf_idx].make_waveform(wf.window_length,wf_params)
                if fit_wf is None:
                    continue

                # if idx == 1 and wf_idx == 0:
                #     print("\n\n")
                #     print(model.detector.lp_num)
                #     print(model.detector.lp_den)
                #     print(model.detector.hp_num)
                #     print(model.detector.hp_den)
                #     print(wf_params[num_det_params:])

                t_data = np.arange(wf.window_length) * 10
                color_idx = wf_idx % len(colors)
                ax0.plot(t_data,fit_wf, color=colors[color_idx], alpha=0.1)

                resid = fit_wf -  wf.windowed_wf
                resid_arr += resid
                ax1.plot(t_data, resid, color=colors[color_idx],alpha=0.1,)# linestyle="steps")

                # em_lo_idx = self.model.joint_models.name_map["LowPassFilterModel"]
                # em_lo = self.model.joint_models.models[em_lo_idx]
                # lo_data = params[em_lo.start_idx: em_lo.start_idx + em_lo.num_params]
                # lo_pz = em_lo.get_pz(lo_data)
                #
                # if resid[130] > 0:
                #     print ("over: {}".format(lo_pz))
                # else:
                #     print ("under: {}".format(lo_pz))


        # ax0.set_ylim(-20, np.amax([ np.amax(wf.data) for wf in model.wfs])*1.1)
        ax1.axhline(y=0,color="black", ls=":")
        # ax0.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.axvline(x=model.conf.align_idx*10,color="black", ls=":")
        # ax1.set_ylim(-bad_wf_thresh, bad_wf_thresh)

        avg_resid = resid_arr/len(model.wfs)/len(data.index)
        plt.figure(figsize=(self.width,4))
        plt.plot(avg_resid, ls="steps", color = "blue")
        plt.axhline(y=0,color="black", ls=":")
        plt.xlabel("Sample number [10s of ns]")
        plt.ylabel("Average residual [adc]")
        plt.savefig("average_residual.pdf")

    def plot_waveform_components(self):
        data = self.result_data
        model = self.model

        plt.figure(figsize=(self.width,8))
        plt.xlabel("Digitizer Time [ns]")

        num_det_params = model.num_det_params

        # for wf_idx, wf in enumerate(model.wfs):
        #   dataLen = wf.window_length
        #   t_data = np.arange(dataLen) * 10
        #   plt.plot(t_data, wf.windowed_wf / wf.amplitude, color=colors[wf_idx], ls = ":")
        #   print ("wf %d max %d" % (wf_idx, np.amax(wf.windowed_wf)))

        for (idx) in range(len(data.index)):
            print("index {}".format(idx))
            params = data.iloc[idx].as_matrix()
            self.model.joint_models.apply_params(params)

            for (wf_idx,wf) in enumerate(model.wfs):
                # if wf_idx < 4: continue
                # wfs_param_arr[-1,wf_idx] = 1

                print("wf index {}".format(wf_idx), end="")
                wf_params =  params[model.num_det_params + wf_idx*model.num_wf_params: model.num_det_params + (wf_idx+1)*self.num_wf_params]

                h_wf = np.copy(model.wf_models[wf_idx].make_waveform(wf.window_length,wf_params, charge_type=1))
                e_wf = np.copy(model.wf_models[wf_idx].make_waveform(wf.window_length,wf_params, charge_type=-1))

                print(h_wf.shape)

                t_data = np.arange(len(h_wf))
                color_idx = wf_idx % len(colors)
                plt.plot(t_data,h_wf, color=colors[color_idx],ls="steps", alpha=0.1)
                plt.plot(t_data,e_wf, color=colors[color_idx], ls="steps", alpha=0.1)
                # plt.plot(t_data,h_wf+e_wf, color=colors[color_idx], ls="--", alpha=0.1)

        # plt.ylim(-0.05, 1.05)

    def plot_tf(self):
        plt.figure(figsize=(self.width,8))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
        ax0 = plt.subplot(gs[:,0])
        ax1 = plt.subplot(gs[0,1])
        ax2 = plt.subplot(gs[0,2])

        ax_decay_rc = plt.subplot(gs[1,1])
        ax_square = plt.subplot(gs[1,2])

        square_fig = plt.figure()
        square_ax = plt.gca()

        for idx, row in self.result_data.iterrows():
            mod_idx = -1

            #show the square response for the whole thing
            square = np.zeros(1000)
            square[100:] = 1
            sp = np.copy(square)

            for model in self.model.joint_models.models:
                if not (isinstance(model, DigitalFilterModel) or isinstance(model, FirstStageFilterModel)or isinstance(model, AntialiasingFilterModel)): continue
                mod_idx +=1

                data = row[model.start_idx: model.start_idx + model.num_params].values

                model.apply_to_detector(data, None)
                # square = model.digital_filter.apply_to_signal(square)

                try:
                    pz = model.get_pz(data)

                    for i, pole in enumerate(pz["poles"]):
                        ax0.scatter(np.real(pole), np.imag(pole), color=colors[mod_idx], alpha=0.3)

                    for i, zero in enumerate(pz["zeros"]):
                        ax0.scatter(np.real(zero), np.imag(zero), color=colors[mod_idx], alpha=0.3)
                except AttributeError: pass

                if isinstance(model, HiPassFilterModel):
                    w_hi = np.logspace(-15, -6, 500, base=np.pi)
                    (w_hi, h) = model.get_freqz(data, w_hi)
                    p = ax1.loglog( w_hi, np.abs(h), alpha = 0.2, color=colors[mod_idx])

                    #Pick out the effective RC constant of the decay
                    decay_const = -1/np.log(-1*model.digital_filter.den[-1])/1E9/1E-6
                    ax_decay_rc.axvline(decay_const, alpha=0.2, color=colors[mod_idx])

                elif isinstance(model, LowPassFilterModel) or isinstance(model, AntialiasingFilterModel) or isinstance(model, FirstStageFilterModel):

                    w_lo = np.logspace(-6, 1, 500, base=np.pi)
                    (w_lo, h3) = model.get_freqz(data, w_lo)

                    h3_db = 10*np.log10(np.abs(h3))

                    p3 = ax2.semilogx( w_lo, h3_db, alpha = 0.2, color=colors[mod_idx])

                    spi = model.apply_to_signal(square)
                    sp = model.apply_to_signal(sp)
                    square_ax.plot(spi, color=colors[mod_idx], alpha=0.1   )

                else:
                    w_lo = np.logspace(-15, 1, 500, base=np.pi)
                    (w_lo, h3) = model.get_freqz(data, w_lo)
                    ax_square.loglog( w_lo, np.abs(h3), alpha = 0.2, color=colors[mod_idx])

                    spi = model.apply_to_signal(square)
                    sp = model.apply_to_signal(sp)
                    square_ax.plot(spi, color=colors[mod_idx], alpha=0.1   )


            square_ax.plot(sp, color="k", alpha=0.2)


        an = np.linspace(0,np.pi,200)
        ax0.plot(np.cos(an), np.sin(an), c="k")
        ax0.axis("equal")

        ax_decay_rc.set_xscale("log")
        ax_decay_rc.set_xlabel("RC constant (us)")


    def plot_imp(self):
        im = self.model.imp_model
        imp_data = self.result_data.iloc[:, self.model.imp_first_idx: self.model.imp_first_idx + im.get_num_params()]

        imp_z0 = imp_data.iloc[:,0].as_matrix()
        imp_zmax = imp_data.iloc[:,1].as_matrix()

        g = sns.jointplot(x=imp_z0, y=imp_zmax, kind="kde", stat_func=None)

        avgs = self.model.detector.imp_avg_points
        grads = self.model.detector.imp_grad_points

        # plt.figure()

        for avg in avgs:
            y1 = 2*avg - im.imp_max
            y2 = 2*avg - 0
            g.ax_joint.plot((im.imp_max, 0), (y1, y2), c="k", ls="--")
        for grad in grads:
            y1 = (grad*self.model.detector.detector_length/10) + im.imp_max
            y2 = (grad*self.model.detector.detector_length/10) + 0

            g.ax_joint.plot((im.imp_max, 0), (y1, y2), c="k", ls="--")

        g.ax_joint.set_xlim(im.imp_max, 0)
        g.ax_joint.set_ylim(im.imp_min, 0)

        #overlay the grid of calculated points

    def plot_imp_ends(self):
        im = self.model.imp_model
        imp_data = self.result_data.iloc[:, self.model.imp_first_idx: self.model.imp_first_idx + im.get_num_params()]

        imp_z0 = imp_data.iloc[:,0].as_matrix()
        imp_zmax = imp_data.iloc[:,1].as_matrix()

        g = sns.jointplot(x=imp_z0, y=imp_zmax, kind="kde", stat_func=None)

        avgs = self.model.detector.imp_avg_points
        grads = self.model.detector.imp_grad_points

        # plt.figure()

        for avg in avgs:
            y1 = 2*avg - im.imp_max
            y2 = 2*avg - 0
            g.ax_joint.plot((im.imp_max, 0), (y1, y2), c="k", ls="--")
        for grad in grads:
            y1 = (grad*self.model.detector.detector_length/10) + im.imp_max
            y2 = (grad*self.model.detector.detector_length/10) + 0

            g.ax_joint.plot((im.imp_max, 0), (y1, y2), c="k", ls="--")

        g.ax_joint.set_xlim(im.imp_max, 0)
        g.ax_joint.set_ylim(im.imp_min, 0)

        #overlay the grid of calculated points

    def plot_trace(self):
        f, ax = plt.subplots(self.model.num_det_params, 1, figsize=(self.width,10), sharex=True)
        for i in range(self.model.num_det_params):
            tf_data = self.result_data[i]
            ax[i].plot(tf_data, ls="steps")

            #corresponding model
            model_idx = self.model.joint_models.index_map[i]
            param_model = self.model.joint_models.models[model_idx]

            model_param_idx = i - param_model.start_idx
            param = param_model.params[model_param_idx]
            ax[i].axhline(param.lim_lo,c="k", ls=":")
            ax[i].axhline(param.lim_hi,c="k", ls=":")
            ax[i].set_ylabel(param.name)

        # plt.xlim(0, len(tf_data))

    def plot_detector_pair(self):
        g = sns.pairplot(self.result_data.iloc[:, :self.model.num_det_params], diag_kind="kde")
        plt.savefig("pairplot.png")

    def plot_waveform_trace(self):
        f, ax = plt.subplots(self.num_wf_params, 1, figsize=(self.width,10), sharex=True)
        for i in range(self.num_wf_params):
            for j in range (self.model.num_waveforms):
                tf_data = self.result_data[self.model.num_det_params + self.num_wf_params*j+i]
                try:
                    ax[i].plot(tf_data, color=colors[j%len(colors)], ls="steps")
                except TypeError:
                    ax.plot(tf_data, color=colors[j%len(colors)], ls="steps")

class TrainingResultSummary(ResultBase):
    def __init__(self, result_directory, num_samples=None, sample_dec=1, model_type="Model"):
        super().__init__(result_directory, num_samples, sample_dec)

        self.params_values = {}
        self.params_values_avg = {}
        self.params_values_std = {}
    def extract_model_values(self):
        """Dump the contents of the fit results into a dictionary, where
        the outputs are labeled by parameter names.
        Results are stored in the self.params_values dict
        """
        # for idx, row in self.result_data.iterrows():

        for mod_idx, model in enumerate(self.model.joint_models.models):

            data = self.result_data.iloc[:, model.start_idx: model.start_idx + model.num_params]

            model_name = type(model).__name__
            param_names = [model.params[i].name for i in range(model.num_params)]

            if model_name in self.params_values:
                model_name = model_name + "_2"

            self.params_values[model_name] = {}
            self.params_values_avg[model_name] = {}
            self.params_values_std[model_name] = {}

            for i,name in enumerate(param_names):
                self.params_values[model_name][name] = data.iloc[:,i].values
                self.params_values_avg[model_name][name] = np.mean(data.iloc[:,i].values)
                self.params_values_std[model_name][name] = np.std(data.iloc[:,i].values)

    def extract_filter_values(self):
        """Specifically in the context of electronics filters. This looks for
        any electronics models, and then prints the best values for the numerator and
        denominator, so they can be easily applied to another detector somewhere down
        the line. 
        Dump the contents of the fit results into a dictionary, where
        the outputs are labeled by parameter names.
        Results are stored in the self.params_values dict
        """
        # for idx, row in self.result_data.iterrows():


        # should be able to assume that params_values has already been filled
        # Fill it myself if it is empty
        if self.params_values is None:
            self.extract_model_values()
        

        # Search through the models in there, and check if each is a DigitalFilterModel
        # If it is, then we should sort it into a different dict
        # for model in self.params_values:
        #     if not (isinstance(model, DigitalFilterModel)):
        #         continue
            


        # Then, I'll need to give the average/best values from elsewhere as an input, to
        # be able to read out the values of the numerator/denominator
        
        # Then, I should be able to read out model.den() and model.num() directly


        for model in self.model.joint_models.models:
            if not (isinstance(model, DigitalFilterModel) or isinstance(model, FirstStageFilterModel)or isinstance(model, AntialiasingFilterModel)): continue
            mod_idx +=1

            data = row[model.start_idx: model.start_idx + model.num_params].values

            model.apply_to_detector(data, None)
            # square = model.digital_filter.apply_to_signal(square)

            try:
                pz = model.get_pz(data)

                for i, pole in enumerate(pz["poles"]):
                    ax0.scatter(np.real(pole), np.imag(pole), color=colors[mod_idx], alpha=0.3)

                for i, zero in enumerate(pz["zeros"]):
                    ax0.scatter(np.real(zero), np.imag(zero), color=colors[mod_idx], alpha=0.3)
            except AttributeError: pass

            if isinstance(model, HiPassFilterModel):
                w_hi = np.logspace(-15, -6, 500, base=np.pi)
                (w_hi, h) = model.get_freqz(data, w_hi)
                p = ax1.loglog( w_hi, np.abs(h), alpha = 0.2, color=colors[mod_idx])

                #Pick out the effective RC constant of the decay
                decay_const = -1/np.log(-1*model.digital_filter.den[-1])/1E9/1E-6
                ax_decay_rc.axvline(decay_const, alpha=0.2, color=colors[mod_idx])

            elif isinstance(model, LowPassFilterModel) or isinstance(model, AntialiasingFilterModel) or isinstance(model, FirstStageFilterModel):

                w_lo = np.logspace(-6, 1, 500, base=np.pi)
                (w_lo, h3) = model.get_freqz(data, w_lo)

                h3_db = 10*np.log10(np.abs(h3))

                p3 = ax2.semilogx( w_lo, h3_db, alpha = 0.2, color=colors[mod_idx])

                spi = model.apply_to_signal(square)
                sp = model.apply_to_signal(sp)
                square_ax.plot(spi, color=colors[mod_idx], alpha=0.1   )

            else:
                w_lo = np.logspace(-15, 1, 500, base=np.pi)
                (w_lo, h3) = model.get_freqz(data, w_lo)
                ax_square.loglog( w_lo, np.abs(h3), alpha = 0.2, color=colors[mod_idx])

                spi = model.apply_to_signal(square)
                sp = model.apply_to_signal(sp)
                square_ax.plot(spi, color=colors[mod_idx], alpha=0.1   )







        for mod_idx, model in enumerate(self.model.joint_models.models):

            data = self.result_data.iloc[:, model.start_idx: model.start_idx + model.num_params]

            model_name = type(model).__name__
            param_names = [model.params[i].name for i in range(model.num_params)]

            if model_name in self.params_values:
                model_name = model_name + "_2"

            self.params_values[model_name] = {}

            for i,name in enumerate(param_names):
                self.params_values[model_name][name] = data.iloc[:,i].values
    

    def list_model_params(self):
        """Just print out the models and all of the parameters for
        each model without having to loop over the data or read anything in
        """

        if not self.params_values: # check if the extract_model_values has already filled the params_values
            for model in self.model.joint_models.models:
                model_name = type(model).__name__
                param_names = [model.params[i].name for i in range(model.num_params)]
                print("{}".format(model_name))
                for the_name in param_names:
                    print("    {}".format(the_name))
        else:
            for the_model,the_params in self.params_values.items():
                print("{}".format(the_model))
                for val_name,val_array in the_params.items():
                    print("    {}".format(val_name))

    def summarize_params(self,do_plots=False):
        """ Plot out histograms of the parameters 
        """
        for the_model,the_params in self.params_values.items():
            print("Model Results: {}".format(the_model))
            if(do_plots):
                plt.figure().suptitle(the_model)
                plot_num = 0
            for val_name,val_array in the_params.items():
                avg = np.average(val_array)
                stdev = np.std(val_array)
                print("    {}:".format(val_name))
                print("         avg:{}".format(avg))
                print("         std:{}".format(stdev))

                if(do_plots):
                    plot_num += 1
                    plt.subplot(len(the_params.keys()),1,plot_num)
                    plt.hist(val_array,bins='rice')
                    plt.axvline(avg,color='r',linestyle='dashed')
                    plt.axvline(avg+stdev,color='r',linestyle='dashed')
                    plt.axvline(avg-stdev,color='r',linestyle='dashed')
                    plt.title(val_name)
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.85)
        if(do_plots):
            plt.show()

    def summarize_chains(self,do_plots=False,save_dir=None):
        """ Plot out histograms of the parameters 
        save_dir : If included, the directory where output plots and results will be saved
        """
        if(save_dir is not None):
            out_name = self.id + "_chain_summary.txt"
            out_file = open(os.path.join(save_dir,out_name), 'w',newline='\n')
            out_file.write("Summary for {}\n".format(self.id))
            out_file.write("Using the last {} samples ({} total)\n".format(self.num_samples,self.total_samples))
            out_file.write("Generated at {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
            

        for the_model,the_params in self.params_values.items():
            print("Model Results: {}".format(the_model))
            if(save_dir is not None):
                out_file.write("Model Results: {}\n".format(the_model))
                
            if(do_plots):
                plt.figure().suptitle(the_model)
                gs = gridspec.GridSpec(len(the_params.keys()), 2,width_ratios=[1, 4])
                plot_num = 0
            
            for val_name,val_array in the_params.items():
                avg = np.average(val_array)
                stdev = np.std(val_array)
                print("    {}:".format(val_name))
                print("         avg:{}".format(avg))
                print("         std:{}".format(stdev))
                if(save_dir is not None):
                    out_file.write("    {}:\n".format(val_name))
                    out_file.write("         avg:{}\n".format(avg))
                    out_file.write("         std:{}\n".format(stdev))

                if(do_plots):
                    
                    plt.subplot(gs[plot_num, 0])
                    
                    # plt.subplot(len(the_params.keys()),1,plot_num)
                    plt.hist(val_array,bins='rice',orientation=u'horizontal',)
                    plt.axhline(avg,color='r',linestyle='dashed')
                    plt.axhline(avg+stdev,color='r',linestyle='dashed')
                    plt.axhline(avg-stdev,color='r',linestyle='dashed')
                    plt.title(val_name,loc='left')

                    plt.subplot(gs[plot_num, 1])
                    # plt.subplot(len(the_params.keys()),2,plot_num)
                    plt.plot(val_array)
                    plt.axhline(avg,color='r',linestyle='dashed')
                    plt.axhline(avg+stdev,color='r',linestyle='dashed')
                    plt.axhline(avg-stdev,color='r',linestyle='dashed')
                    plt.tight_layout()
                    # plt.subplots_adjust(top=0.85)
                    plot_num += 1

        if(do_plots):
            if(save_dir is not None):
                filename = os.path.join(save_dir,self.id + "_chain_summary.pdf")
                # filename = self.id + "_chain_summary.pdf"
                # filename = "chain_summary.pdf"
                # filename = 'multipage.pdf'
                print("Saving figures to {}".format(filename))
                self.multipage_plot(filename)
                plt.close('all')
            else:
                plt.show()



    def summarize_chains_kde(self,do_plots=False,save_dir=None):
        """ Plot out kde figs of the parameters 
        save_dir : If included, the directory where output plots and results will be saved
        """

        import seaborn as sns
        # sample = np.hstack((np.random.randn(30), np.random.randn(20)+5))
        fig, ax = plt.subplots(figsize=(8,4))
        # sns.distplot(sample, rug=True, hist=False, rug_kws={"color": "g"},
        #     kde_kws={"color": "k", "lw": 3})
        # plt.show()

        if(save_dir is not None):
            out_name = self.id + "_chain_summary.txt"
            out_file = open(os.path.join(save_dir,out_name), 'w',newline='\n')
            out_file.write("Summary for {}\n".format(self.id))
            out_file.write("Using the last {} samples ({} total)\n".format(self.num_samples,self.total_samples))
            out_file.write("Generated at {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
            

        for the_model,the_params in self.params_values.items():

            print("Model Results: {}".format(the_model))
            if(save_dir is not None):
                out_file.write("Model Results: {}\n".format(the_model))
                
            if(do_plots):
                plt.figure().suptitle(the_model)
                gs = gridspec.GridSpec(len(the_params.keys()), 2,width_ratios=[1, 3])
                plot_num = 0
            
            for val_name,val_array in the_params.items():
                avg = np.average(val_array)
                stdev = np.std(val_array)
                print("    {}:".format(val_name))
                print("         avg:{}".format(avg))
                print("         std:{}".format(stdev))
                if(save_dir is not None):
                    out_file.write("    {}:\n".format(val_name))
                    out_file.write("         avg:{}\n".format(avg))
                    out_file.write("         std:{}\n".format(stdev))

                if(do_plots):
                    
                    plt.subplot(gs[plot_num, 0])
                    
                    # plt.subplot(len(the_params.keys()),1,plot_num)
                    sns.distplot(val_array, rug=True, hist=False, rug_kws={"color": "g"},
                        kde_kws={"color": "k", "lw": 1},vertical=True)
                    # plt.hist(val_array,bins='rice',orientation=u'horizontal',)
                    plt.axhline(avg,color='r',linestyle='dashed')
                    plt.axhline(avg+stdev,color='r',linestyle='dashed')
                    plt.axhline(avg-stdev,color='r',linestyle='dashed')
                    plt.title(val_name,loc='left')

                    plt.subplot(gs[plot_num, 1])
                    # plt.subplot(len(the_params.keys()),2,plot_num)
                    plt.plot(val_array)
                    plt.axhline(avg,color='r',linestyle='dashed')
                    plt.axhline(avg+stdev,color='r',linestyle='dashed')
                    plt.axhline(avg-stdev,color='r',linestyle='dashed')
                    plt.tight_layout()
                    # plt.subplots_adjust(top=0.85)
                    plot_num += 1


        if(do_plots):
            if(save_dir is not None):
                filename = os.path.join(save_dir,self.id + "_chain_summary_kde.pdf")
                # filename = self.id + "_chain_summary.pdf"
                # filename = "chain_summary.pdf"
                # filename = 'multipage.pdf'
                print("Saving figures to {}".format(filename))
                self.multipage_plot(filename)
                plt.close('all')
            else:
                plt.show()

    def kde(data):
        """
        Perform a KDE on the given data chain
        """
        import seaborn as sns
        p=sns.kdeplot(data, shade=True)

        x,y = p.get_lines()[0].get_data()
        
        kde = [x,y]

        #care with the order, it is first y
        #initial fills a 0 so the result has same length than x
        cdf = scipy.integrate.cumtrapz(y, x, initial=0)

        nearest_05 = np.abs(cdf-0.5).argmin()

        x_median = x[nearest_05]
        y_median = y[nearest_05]

        nearest_095 = np.abs(cdf-0.95).argmin()
        nearest_005 = np.abs(cdf-0.05).argmin()



        return kde,median,l,u

class ResultComparison(ResultBase):
    # def __init__(self, result_directory, num_samples, sample_dec=1, model_type="Model"):
        # super().__init__(result_directory, num_samples, sample_dec)
    def __init__(self,results):
        self.results = results
        # I require that the result object is a list of result summaries
        assert isinstance(results,list)
        assert isinstance(results[0],TrainingResultSummary)

        self.num = len(results)

        if(self.num < 2):
            raise ValueError("You are trying to compare fewer than 2 results (only {}).".format(self.num))        

        self.ids = [res.id for res in results]

        # Figure out how many samples I want to plot/work with
        # self.min_samples = min([res.total_samples for res in results])
        # if(self.min_samples < 2000):
        #     self.num_samples = self.min_samples
        # elif(self.min_samples >= 2000 and self.min_samples < 5000):
        #     self.num_samples = 2000
        # elif(self.min_samples >= 5000):
        #     # take the last 3/4, rounded to the nearest 100
        #     self.num_samples = np.floor(self.min_samples*0.75/100)*100
        
        self.verify_compatible_models()

        if(self.compatible is not True):
            print("Not all requested models are compatible. Continue at your own risk...")

        self.model_list = [(the_model, list(the_params.keys())) for the_model,the_params in self.results[0].params_values.items()]

        print("Assuming that the following model set is correct:")
        print(self.model_list)

    def verify_compatible_models(self):
        """ Check that the models we are comparing have the same models.
        Has not been rigorously tested, but seems to work. 
        Just verifies that the lists of models and parameters are the same.
        There's probably a better way to do this but ¯\_(ツ)_/¯
        """
        # models = [type(model).__name__ for model in self.results.model.joint_models.models]
        first = {(the_model, frozenset(the_params.keys())) for the_model,the_params in self.results[0].params_values.items()}

        self.compatible = None
        for i,m in enumerate(self.results[1:]):
            m_set = {(the_model, frozenset(the_params.keys())) for the_model,the_params in m.params_values.items()}
            
            if(first == m_set):
                print("Models 0 and {} are compatible".format(i+1))
            else:
                print("Models 0 and {} are NOT compatible".format(i+1))
                print(first)
                print(m_set)
                self.compatible = False
        
        if self.compatible is not None:
            self.compatible = False
        else:
            self.compatible = True
            # print("All models appear to contain the following:")
            # print(first)
        return self.compatible

    def compare_chains(self,do_plots=True):
        """ Plot out histograms of the parameters 
        save_dir : If included, the directory where output plots and results will be saved
        """

        # initialize the complete result with just the first (we'll append the rest later)
        self.total_result = self.results[0]

        # for i_model,res in enumerate(self.results):
        #     if i_model==1:
        #         continue
        for the_model,the_params in self.model_list:
            # for the_model,the_params in res.params_values.items():
            if(do_plots):
                plt.figure().suptitle(the_model)
                gs = gridspec.GridSpec(len(the_params), 2,width_ratios=[1, 4])
                plot_num = 0
            
            for val_name in the_params:
                for res in self.results:
                    val_array = res.params_values[the_model][val_name]
                    avg = np.average(val_array)
                    stdev = np.std(val_array)

                    if(do_plots):
                        
                        plt.subplot(gs[plot_num, 0])
                        
                        # plt.subplot(len(the_params.keys()),1,plot_num)
                        plt.hist(val_array,bins='rice',orientation=u'horizontal',)
                        plt.axhline(avg,color='r',linestyle='dashed')
                        # plt.axhline(avg+stdev,color='r',linestyle='dashed')
                        # plt.axhline(avg-stdev,color='r',linestyle='dashed')
                        plt.title(val_name,loc='left')

                        plt.subplot(gs[plot_num, 1])
                        # plt.subplot(len(the_params.keys()),2,plot_num)
                        plt.plot(val_array)
                        plt.axhline(avg,color='r',linestyle='dashed')
                        # plt.axhline(avg+stdev,color='r',linestyle='dashed')
                        # plt.axhline(avg-stdev,color='r',linestyle='dashed')
                        plt.tight_layout()
                        # plt.subplots_adjust(top=0.85)
                plot_num += 1
        plt.show()

    def compare_params(self,do_plots=False,save_dir=None):
        """ Plot out histograms of the parameters 
        """
        for the_model,the_params in self.model_list:
            plt.figure().suptitle(the_model)
            gs = gridspec.GridSpec(len(the_params), 1)
            plot_num = 0
            
            for val_name in the_params:
                p_max = []
                p_min = []
                for i,res in enumerate(self.results):
                    # loop over once to get the max + min, so I can plot them 
                    # all with the same histogram bins
                    val_array = res.params_values[the_model][val_name]
                    p_max.append(max(val_array))
                    p_min.append(min(val_array))

                for i,res in enumerate(self.results):
                    # loop over the second time to do the actual plotting
                    val_array = res.params_values[the_model][val_name]
                    avg = np.average(val_array)
                    stdev = np.std(val_array)

                    plt.subplot(gs[plot_num, 0])
                    
                    # plt.subplot(len(the_params.keys()),1,plot_num)
                    plt.hist(val_array, color=colors[i], alpha=0.5, bins='rice', density=True,label=res.id,range=(min(p_min),max(p_max)))
                    plt.axvline(avg,color=colors[i],alpha=0.6,linestyle='dashed')
                    plt.axvline(avg-stdev,color=colors[i],alpha=0.3,linestyle='dashed')
                    plt.axvline(avg+stdev,color=colors[i],alpha=0.3,linestyle='dashed')
                plt.title(val_name,loc='left')
                plt.tight_layout()

                plot_num += 1
            plt.legend()

        if(save_dir is not None):
            id_string = '-and-'.join(self.ids)
            filename = os.path.join(save_dir,id_string + "_param_comparison.pdf")
            print("Saving figures to {}".format(filename))
            self.multipage_plot(filename)

        plt.show()
