import numpy as np
import sys, os, shutil
import pickle
import dnest4
from multiprocessing import Pool, cpu_count
# from mpi4py import MPI

from ..models import Model, PulserTrainingModel

def init_parallelization(conf, model_type):
    global model
    if model_type=="Model":
        model = Model( conf)
    elif model_type=="PulserTrainingModel":
        model = PulserTrainingModel( conf)

def WaveformLogLikeStar(a_b):
  return model.calc_wf_likelihood(*a_b)

class LocalFitManager():
    '''Does the fit using one machine -- either multicore or single-threaded'''

    def __init__(self, fit_configuration, num_threads=None, model_type="Model"):
        if model_type=="Model":
            self.model = Model( fit_configuration, fit_manager=self)
        elif model_type=="PulserTrainingModel":
            self.model = PulserTrainingModel( fit_configuration, fit_manager=self)

        self.num_waveforms = self.model.num_waveforms
        self.num_det_params = self.model.num_det_params
        self.num_wf_params = self.model.num_wf_params#fit_configuration.num_wf_params

        if num_threads is None: num_threads = cpu_count()

        if num_threads > self.model.num_waveforms: num_threads = self.model.num_waveforms

        self.num_threads = num_threads

        if num_threads > 1:
            self.pool = Pool(num_threads, initializer=init_parallelization, initargs=(fit_configuration,model_type))
        else:
            init_parallelization(fit_configuration,model_type)

    def calc_likelihood(self, params):
        num_det_params = self.num_det_params
        lnlike = 0

        #parallelized calculation
        if self.num_threads > 1:
            args = []
            for wf_idx in range(self.num_waveforms):
                args.append( [self.model.get_wf_params(params, wf_idx), wf_idx] )
                # print ("shipping {0}: {1}".format(wf_idx, wf_params[num_det_params:, wf_idx]))

            results = self.pool.map(WaveformLogLikeStar, args)
            # exit()
            for result in (results):
                lnlike += result
        else:
            for wf_idx in range(self.num_waveforms):
                result = model.calc_wf_likelihood(self.model.get_wf_params(params, wf_idx), wf_idx)
                lnlike += result
            # print (result)
        return lnlike

    def fit(self, numLevels, directory=None, numPerSave=1000,numParticles=5,new_level_interval=10000,debug=False ):

      if directory is None:
        directory = "./"  

      if not os.access(directory, os.W_OK):
        raise OSError("Directory {} either does not exist or is not writeable to the fitter...".format(directory))

    
    #   mpi = dnest4.MPISampler()

      sampler = dnest4.DNest4Sampler(self.model,
                                     backend=dnest4.backends.CSVBackend(basedir = directory,
                                                                        sep=" "))
                                    # MPISampler=mpi)

      # Set up the sampler. The first argument is max_num_levels
      gen = sampler.sample(max_num_levels=numLevels, num_steps=200000, new_level_interval=new_level_interval,
                            num_per_step=numPerSave, thread_steps=100,
                            num_particles=numParticles, lam=10, beta=100, seed=1234)

      # Set up the sampler. The first argument is max_num_levels
    #   gen = sampler.sample(model=self.model,
    #                         max_num_levels=numLevels, 
    #                         num_steps=-1, 
    #                         num_per_step=numPerSave,
    #                         new_level_interval=new_level_interval,
    #                         thread_steps=1,
    #                         lam=10, beta=100, seed=1234)


      # Do the sampling (one iteration here = one particle save)
      for i, sample in enumerate(gen):
          print("# Saved {k} particles.".format(k=(i+1)))


class MPIFitManager():
    def __init__(self, fit_configuration, comm=None, doParallelParticles = False, debug=False):
        # from mpi4py import MPI

        self.model = Model( fit_configuration, fit_manager=self)
        self.num_waveforms = self.model.num_waveforms
        self.num_det_params = self.model.num_det_params

        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        self.rank = self.comm.Get_rank()
        self.num_workers = self.comm.Get_size() - 1

        self.doParallelParticles = doParallelParticles

        self.tags = self.enum('CALC_LIKE', 'CALC_WF', 'EXIT')

        self.numCalls = 0
        self.LastMem = memory_usage_psutil()

        self.debug = debug
        self.debug_mem_file = "memory_info.txt"

        self.num_wf_params = self.model.num_wf_params

    def is_master(self):
        """
        Is the current process the master?
        """
        return self.rank == 0


    def calc_likelihood(self, params_in):
        params = np.copy(params_in)
        num_det_params = self.num_det_params
        tags = self.tags

        if self.debug:
            self.numCalls +=1
            if self.numCalls % 1000 == 0:
                meminfo = "Particle {0} (call {1}) memory: {2}\n".format(MPI.COMM_WORLD.Get_rank(), self.numCalls , memory_usage_psutil())
                with open(self.debug_mem_file, "a") as f:
                    f.write(meminfo)

        wfs_param_arr = params[num_det_params:].reshape((self.num_wf_params, self.num_waveforms))

        wf_params = np.empty(num_det_params+self.num_wf_params)
        wf_params[:num_det_params] = params[:num_det_params]

        #nonparallelized: should only be called on init
        if self.doParallelParticles and MPI.COMM_WORLD.Get_rank() == 0:
            ln_like = 0
            for wf_idx in range(self.num_waveforms):
                wf_params[num_det_params:] = wfs_param_arr[:,wf_idx]
                ln_like += self.model.calc_wf_likelihood(wf_params, wf_idx)
            return ln_like

        #parallelized calculation
        for wf_idx in range(self.num_waveforms):
            worker = np.int(wf_idx + 1)
            wf_params[num_det_params:] = wfs_param_arr[:,wf_idx]
            self.comm.send(wf_params, dest=worker, tag=self.tags.CALC_LIKE)

        wf_likes = np.empty(self.num_waveforms)
        for i in range(self.num_waveforms):
                worker = i + 1
                wf_likes[i] = self.comm.recv(source=worker, tag=MPI.ANY_TAG)

        return np.sum(wf_likes)


    def wait_and_process(self):
        tags = self.tags
        status = MPI.Status()   # get MPI status object

        if self.is_master():
            raise RuntimeError("Master node told to await jobs.")

        while True:
            # Blocking receive to wait for instructions.
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

            if self.debug:
                self.numCalls +=1
                if self.numCalls % 1000 == 0:
                    meminfo = "Particle {0} (call {1}) memory: {2}\n".format(MPI.COMM_WORLD.Get_rank(), self.numCalls , memory_usage_psutil())
                    with open(self.debug_mem_file, "a") as f:
                        f.write(meminfo)

            if status.tag == self.tags.CALC_LIKE:
                # if self.debug:
                #     print( "rank %d (local rank %d) calcing like %d" % (MPI.COMM_WORLD.Get_rank(), self.rank, self.rank - 1) )

                wf_idx = self.rank - 1
                ln_like = self.model.calc_wf_likelihood(task, wf_idx)
                self.comm.send(ln_like, dest=0, tag=status.tag)

            if status.tag == self.tags.CALC_WF:
                data_len = self.model.output_wf_length
                model = self.model.make_waveform(data_len, task)
                self.comm.send(model, dest=0, tag=status.tag)

            elif status.tag == self.tags.EXIT:
                break

            del task


    def enum(self, *sequential, **named):
        """Handy way to fake an enumerated type in Python
        http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
        """
        enums = dict(zip(sequential, range(len(sequential))), **named)
        return type('Enum', (), enums)

    def fit(self, numLevels, directory="",numPerSave=1000,numParticles=5 ):

      sampler = dnest4.DNest4Sampler(self.model,
                                     backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                        sep=" "))

      # Set up the sampler. The first argument is max_num_levels
      gen = sampler.sample(max_num_levels=numLevels, num_steps=200000, new_level_interval=10000,
                            num_per_step=numPerSave, thread_steps=100,
                            num_particles=numParticles, lam=10, beta=100, seed=1234)

      # Do the sampling (one iteration here = one particle save)
      for i, sample in enumerate(gen):
          print("# Saved {k} particles.".format(k=(i+1)))

      # Run the postprocessing
      # dnest4.postprocess()

    def fit_particle(self, manager_comm,  numLevels, directory="", numPerSave=1000, numParticles=5, new_level_interval=10000):

      mpi_sampler = dnest4.MPISampler(comm=manager_comm, debug=False)

      if manager_comm.rank == 0:
          # Set up the sampler. The first argument is max_num_levels
          sampler = dnest4.DNest4Sampler(self.model, backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                          sep=" "), MPISampler=mpi_sampler)

          gen = sampler.sample(max_num_levels=numLevels, num_steps=200000, new_level_interval=new_level_interval,
                                num_per_step=numParticles, thread_steps=100,
                                lam=10, beta=100, seed=1234)

          # Do the sampling (one iteration here = one particle save)
          for i, sample in enumerate(gen):
            #   print("# Saved {k} particles.".format(k=(i+1)))

              if self.debug:
                  meminfo = "Particle {0} memory: {1}\n".format(MPI.COMM_WORLD.Get_rank(),  memory_usage_psutil())
                  with open(self.debug_mem_file, "a") as f:
                      f.write(meminfo)

      else:
          mpi_sampler.wait(self.model, max_num_levels=numLevels, num_steps=200000, new_level_interval=new_level_interval,
                                num_per_step=numPerSave, thread_steps=100,
                                lam=10, beta=100, seed=1234)
          return

    def close(self):
        if self.is_master():
            for i in range(self.num_workers):
                self.comm.send(None, dest=i + 1, tag=self.tags.EXIT)

    def __exit__(self, *args):
        self.close()

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem