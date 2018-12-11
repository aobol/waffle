import numpy as np
import sys, os
import pickle

class WaveformConfiguration(object):
    """This class is used to store information about the waveform(s) in a fit and how they are set up.
    The fit configuration takes this as a parameter

    Doesn't need to have the ability to be saved or loaded because the Fit Configuration should have the 
    appropriate information inside to re-make this at will. 

    """
    def __init__(self,
        #params for setting up & aligning waveforms
        wf_file_name,
        align_idx = 200,
        num_samples = 400,
        align_percent = 0.95,
        do_smooth=True,
        wf_idxs=None,
        smoothing_type="gauss"
    ):
        self.wf_file_name=wf_file_name
        self.align_idx = align_idx
        self.align_percent = align_percent
        self.num_samples = num_samples
        self.wf_idxs = wf_idxs
        self.do_smooth=do_smooth
        self.smoothing_type=smoothing_type

    # def __init__(self,
    #     #params for setting up & aligning waveforms
    #     wf_file_name,
    #     align_idx = 200,
    #     num_samples = 400,
    #     align_percent = 0.95,
    #     do_smooth=True,
    #     wf_idxs=None,
    #     smoothing_type="gauss",
    #     loadSavedConfig = False,
    #     directory = None
    # ):
    #     self.wf_file_name=wf_file_name
    #     self.align_idx = align_idx
    #     self.align_percent = align_percent
    #     self.num_samples = num_samples
    #     self.wf_idxs = wf_idxs
    #     self.do_smooth=do_smooth
    #     self.smoothing_type=smoothing_type
    #     self.directory = directory

    #     if loadSavedConfig:
    #         self.load_config(directory)
    #     else:
    #         pass
            # self.wf_config = WaveformConfiguration(**wf_conf)
    # @classmethod
    # def from_other(cls,other):


        #downsampling the decay portion
    # def save_config(self):
    #     saved_file=os.path.join(self.directory, "fit_params.npy")
    #     pickle.dump(self.__dict__.copy(),open(saved_file, 'wb'))

    # def load_config(self,directory):
    #     saved_file=os.path.join(directory, "fit_params.npy")
    #     if not os.path.isfile(saved_file):
    #         print ("Saved configuration file {0} does not exist".format(saved_file))
    #         exit()

    #     self.__dict__.update(pickle.load(open(saved_file, 'rb')))        
    #     # print("Loading saved WaveformConfiguation objects is not yet supported!!! Quitting...")
    #     # raise NotImplementedError

class WaveformSingleConfiguration(object):
    def __init__(self, wf_idx=None, wf_batch_config=None,saved_file_name="fit_params", **kwargs):

        self.wf_batch_config = wf_batch_config
        self.wf_idx = wf_idx
        self.saved_file_name = saved_file_name

        if(wf_batch_config is None):
            try:
                self.wf_file_name=wf_file_name
            except NameError:
                pass
            try:
                self.align_idx = align_idx
            except NameError:
                pass
            try:
                self.align_percent = align_percent
            except NameError:
                pass
            try:
                self.num_samples = num_samples
            except NameError:
                pass
            try:
                self.wf_idxs = wf_idxs
            except NameError:
                pass
            try:
                self.do_smooth=do_smooth
            except NameError:
                pass
            try:
                self.smoothing_type=smoothing_type
            except NameError:
                pass
        # self.directory = wf_batch_config.directory
        # self.wf_file_name = self.wf_batch_config.wf_file_name
        # self.align_idx = align_idx
        # self.align_percent = align_percent
        # self.num_samples = num_samples
        # self.wf_idxs = wf_idxs
        # self.do_smooth=do_smooth
        # self.smoothing_type=smoothing_type
        
        #downsampling the decay portion
    def save_config(self):
        saved_file=os.path.join(self.directory, self.saved_file_name)
        pickle.dump(self.__dict__.copy(),open(saved_file, 'wb'))

    def load_data(self):
        data = np.load(wf_file, encoding="latin1")
        wfs = data['wfs']

        wf = wfs[wf_idx]
        wf.window_waveform(time_point=align_point, early_samples=100, num_samples=125)
    
    # def set_directories(self):
    #     wf_directory = os.path.join(directory, "wf{}".format(wf_idx))
    #     if os.path.isdir(wf_directory):
    #         if len(os.listdir(wf_directory)) >0:
    #             raise OSError("Directory {} already exists: not gonna over-write it".format(wf_directory))
    #     else:
    #         os.makedirs(wf_directory)

class WaveformBatchConfiguration(object):
    def __init__(self,
        #params for setting up & aligning waveforms
        wf_file_name,
        directory = "",
        params = None,
        detector_conf = None,
        models = None,
        # wf_idx,
        align_idx = 200,
        num_samples = 1000,
        align_percent = 0.95,
        do_smooth=True,
        wf_idxs=None,
        smoothing_type="gauss",
        saved_file_name = "batch_fit_params.npy"
    ):
        """
        WaveformBatchConfiguration is used to store information for running single 
        waveform fits, and for understanding the data context later. 

        Parameters
        ----------
        wf_file_name: string
            full path to the file containing waveforms (probably an npz)
        directory: string
            the top level directory where all the fits will get saved (doesn't need to be a full path)
        align_idx:
            The sample index where the align percent will be aligned to
        align_percent: 
            The fractional value 0-1 indicating where on the wf will be aligned to
        do_smooth: bool 
            Turns on the charge cloud smoothing, which is probably what you want
        wf_idxs: 
            A list of all the indices to fit, such as range(100). Presently unused in any real situation

        """
        self.directory = directory
        self.params = params
        self.detector_conf = detector_conf
        self.models = models
        self.wf_file_name = wf_file_name
        self.align_idx = align_idx
        self.align_percent = align_percent
        self.num_samples = num_samples
        self.wf_idxs = wf_idxs
        self.do_smooth=do_smooth
        self.smoothing_type=smoothing_type
        self.saved_file_name = saved_file_name

    # def parse_result(self):
    #     res = TrainingResultSummary(result_directory=fit_name, num_samples=1, sample_dec=1, model_type="Model")
    #     res.parse_samples(
    #         sample_file_name="posterior_sample.txt",
    #         directory=fit_name, 
    #         num_to_read=1, 
    #         sample_dec=1)
    #     res.extract_model_values()
    #     params_values = res.params_values


    def save_config(self):
        print("Saving configuration file...")
        saved_file=os.path.join(self.directory, self.saved_file_name)
        try:
            os.makedirs(self.directory)
        except FileExistsError as e:
            pass
        if(os.path.isfile(saved_file)):
            print ("A configuration file already exists at: {0}".format(saved_file))
            print ("Remove the directory/file manually or select a different file.")
            exit()
        pickle.dump(self.__dict__.copy(),open(saved_file, 'wb'))
        print("Saved configuration at {}".format(saved_file))

    def load_config(self,directory):
        saved_file=os.path.join(directory, "batch_fit_params.npy")
        if not os.path.isfile(saved_file):
            print ("Saved configuration file {0} does not exist".format(saved_file))
            exit()

        self.__dict__.update(pickle.load(open(saved_file, 'rb')))
        # self.wf_config = WaveformConfiguration(**self.wf_conf)


class FitConfiguration(object):
    """
    This takes in the detector conf and the wf conf.
    Used for keeping all the data together, and is fed to the Fit Manager to do the actual fit.
    This produces the saved file fit_params.npy which should contain all that you need.

    """
    def __init__(self,
        #data files
        # wf_file_name,
        conf_file="",

        #save path
        directory = "",

        #fit parameters
        wf_conf={},
        model_conf={},

        loadSavedConfig=False,

        time_step_calc=1,
        **kwargs
    ):
        # self.wf_file_name = wf_file_name
        self.siggen_conf_file=conf_file
        self.directory = directory

        self.wf_conf = wf_conf

        self.model_conf=model_conf

        self.time_step_calc=time_step_calc

        for key, value in kwargs.items():
            setattr(self, key, value)

        if loadSavedConfig:
            self.load_config(directory)
        else:
            # self.wf_config = wf_conf
            self.wf_config = WaveformConfiguration(**wf_conf)

    @classmethod
    def from_file(cls,directory):
        # raise NotImplementedError
        saved_file=os.path.join(directory, "fit_params.npy")
        print("Loading FitConfiguration from file ({})...".format(saved_file))
        if not os.path.isfile(saved_file):
            print ("Saved configuration file {0} does not exist".format(saved_file))
            exit()

        loaded = pickle.load(open(saved_file, 'rb'))

        try:
            return cls(
                siggen_conf_file = loaded['siggen_conf_file'],
                directory = loaded['directory'],
                wf_conf = loaded['wf_conf'],
                model_conf = loaded['model_conf'],
                time_step_calc = loaded['time_step_calc'],
                loadSavedConfig=False
                )
        except AttributeError as e:
            print("One of the variables you tried to use from the loaded file was probably not matching the expected ones...")
            print(loaded)
            print(e)
            exit()

    @classmethod
    def wf_from_file(cls,file):
        raise NotImplementedError
        saved_file=os.path.join(directory, "fit_params.npy")
        if not os.path.isfile(saved_file):
            print ("Saved configuration file {0} does not exist".format(saved_file))
            exit()

        loaded = pickle.load(open(saved_file, 'rb'))

        return cls(
            conf_file = loaded.siggen_conf_file,
            directory = loaded.directory,
            wf_conf = loaded.wf_conf,
            model_conf = loaded.model_conf,
            time_step_calc = loaded.time_step_calc,
            loadSavedConfig=False
            )

    def save_config(self):
        saved_file=os.path.join(self.directory, "fit_params.npy")
        pickle.dump(self.__dict__.copy(),open(saved_file, 'wb'))

    def load_config(self,directory):
        saved_file=os.path.join(directory, "fit_params.npy")
        if not os.path.isfile(saved_file):
            print ("Saved configuration file {0} does not exist".format(saved_file))
            exit()

        self.__dict__.update(pickle.load(open(saved_file, 'rb')))
        self.wf_config = WaveformConfiguration(**self.wf_conf)

    def plot_training_set(self):
        import matplotlib.pyplot as plt

        if os.path.isfile(self.wf_file_name):
            print("Loading wf file {0}".format(self.wf_file_name))
            data = np.load(self.wf_file_name, encoding="latin1")
            wfs = data['wfs']
            wfs = wfs[self.wf_idxs]

            plt.figure()
            for wf in wfs:
                plt.plot(wf.data)
            plt.show()
