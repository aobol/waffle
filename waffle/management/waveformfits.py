import numpy as np
import sys, os, shutil
import dnest4

from ..models import WaveformModel

class WaveformFitManager(object):
    '''
    This is really simple for now.
    '''

    def __init__(self, *args, **kwargs):
        self.model = WaveformModel( *args, **kwargs )
    # def __init__(self, target_wf, align_percent, detector, align_idx=125, do_smooth=True, smoothing_type="gauss"):
    #     self.model = WaveformModel(
    #                                 target_wf=target_wf,
    #                                 align_percent=align_percent, 
    #                                 detector=detector,
    #                                 do_smooth=do_smooth, 
    #                                 align_idx=align_idx,
    #                                 smoothing_type=smoothing_type
    #                                 )
    # def __init__(self, fit_configuration, num_threads=None, model_type="Model"):
    #     align_percent=wf_conf.align_percent
    #     detector=self.detector
    #     do_smooth=wf_conf.do_smooth
    #     smoothing_type=wf_conf.smoothing_type

    
    def fit(self, numLevels, directory="",numPerSave=1000,numParticles=5,new_level_interval=10000 ):

        sampler = dnest4.DNest4Sampler(self.model,
                                     backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                        sep=" "))

        # Set up the sampler. The first argument is max_num_levels
        gen = sampler.sample(max_num_levels=numLevels, num_steps=200000, new_level_interval=new_level_interval,
                            num_per_step=numPerSave, thread_steps=100,
                            num_particles=numParticles, lam=10, beta=100, seed=1234)

        # Do the sampling (one iteration here = one particle save)
        for i, sample in enumerate(gen):
          print("# Saved {k} particles.".format(k=(i+1)))
