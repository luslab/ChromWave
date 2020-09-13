import matplotlib


matplotlib.use('Agg')

import os
import numpy

from hyperopt.fmin import fmin
from hyperopt import space_eval
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice,uniform

import glob

from chromwave.functions import utils

from chromwave import chromwavenet, runtime_dataset, filesystem

import pickle
from hyperopt.base import Trials

max_rec = 0x100000
import resource
import sys
# May segfault without this line. 0x100 is a guess at the size of each stack frame.
resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
sys.setrecursionlimit(max_rec)


# taken from https://github.com/dimitry12/hyperopt/tree/filetrials_320
class FileTrials(Trials):
    def __init__(self, exp_key=None, refresh=True, persisted_location=None):
        super(FileTrials, self).__init__(exp_key=exp_key, refresh=refresh)
        if persisted_location is not None:
            self._persisted_file = open(persisted_location, 'a+b')
            try:
                self._persisted_file.seek(0)
                docs = pickle.load(self._persisted_file)
                #docs = dill.load(self._persisted_file)
                super(FileTrials, self)._insert_trial_docs(docs)
                self.refresh()
            except EOFError:
                None
    def _insert_trial_docs(self, docs):
        rval = super(FileTrials, self)._insert_trial_docs(docs)
        self._persisted_file.seek(0)
        self._persisted_file.truncate()
        pickle.dump(self._dynamic_trials, self._persisted_file)
        #dill.dump(self._dynamic_trials, self._persisted_file)
        self._persisted_file.flush()
        return rval
###########################################
# Setting up the directories
#########################################
working_dir = '../../../'

data_dir = os.path.join(working_dir,'data')
model_dir = os.path.join(working_dir,'models')

out_dir = os.path.join(working_dir,'Test/Hyperopt/Human_Nucleosomes')

genome_dir = os.path.join(data_dir,'genomes/hg38_promoter')

nuc_profile = os.path.join(data_dir,'gaffney2012','nucleosomes_all_TSS_upstream_downstream_1000.compMat.mat')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
run_dir = os.path.join(out_dir, '01_hyperopt')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

###########################################
# Setting up the hyperparameters
#########################################
ALGO= tpe.suggest
# MAX_EPOCHS=30
# MAX_EPOCHS_FULL = 100
# STEP_SIZE = 2000
# STEP_SIZE_TRAIN = 2000
# MAX_EVALS =100
# GPU=0
# N_TRAIN=6

MAX_EPOCHS=5
MAX_EPOCHS_FULL = 10
STEP_SIZE = 2000
STEP_SIZE_TRAIN = 2000
MAX_EVALS =3
GPU=0
N_TRAIN=2

files = filesystem.FileSystem(genome_dir ,run_dir,source_profile= nuc_profile,
                              overwrite=False, test_fraction = 0.1, val_fraction=0.2,resume = True)

kernel_length_choices = [16, 24,32]

conv_n_filters_choices = [64,82,128,256,360,480,520]
dilation_depth_choices = [9]
dilation_kernel_size_choices = [2]
weights_min_choices = [0.,0.1]
weights_max_choices= [0.7,0.8]

space = { 'optimizer': choice('optimizer', ['adam', 'rmsprop','nadam']),
          'amsgrad':choice('amsgrad',['True','False']),
        'dropout' :uniform('dropout',0.0, 0.7),
        'res_dropout' :uniform('res_dropout',0.0, 0.7),
        'res_l2' :uniform('res_l2',0.0, 0.0001),
        'final_l2' :uniform('final_l2',0.0, 0.0001),
        'kernel_lengths' : choice('kernel_lengths',kernel_length_choices),
        'conv_n_filters':choice('conv_n_filters',conv_n_filters_choices),
        'dilation_depth':choice('dilation_depth',dilation_depth_choices),
        'dilation_kernel_size': choice('dilation_kernel_size',dilation_kernel_size_choices),
        'learning_rate': uniform('learning_rate',0.00001, 0.01),
        'reduceLR_rate' :uniform('reduceLR_rate',0.0, 0.99),
         'weights_min' :choice('weights_min',weights_min_choices),
         'weights_max' :choice('weights_max',weights_min_choices)
        }


def train_genomic_wavenet_model(params):

    no_trial=len(glob.glob(files.get_output_directory()+'/hyperas*'))
    print('Running Trial '+str(no_trial)+' /'+str(MAX_EVALS))
    run_dir = os.path.join(files.get_output_directory(),'hyperas_trial_' + str(no_trial))

    print('Creating running dir: ' + run_dir)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    """Function to do perform a hyperparameter search for a nucleosome chromwavenet  model """
    print("Hyper opt parameter: ")
    print(params)


    cmd = '-o ' + run_dir + " "+\
    '--genome_dir ' + genome_dir +" "+ \
    '--profiles ' +nuc_profile+" "+ \
      '--kernel_lengths ' + str(params['kernel_lengths']) +" "+\
    "--conv_n_filters "+ str(params['conv_n_filters']) +" "+\
     "--pool_sizes " +  "global" +" "+\
     "--learning_rate "+ str(params['learning_rate']) +" "+ \
     "--reduceLR_rate " + str(params['reduceLR_rate']) + " " + \
     "--dropout "+ str(params['dropout']) +" "+\
     "--res_dropout "+ str(params['res_dropout']) +" "+\
     "--res_l2 "+ str(params['res_l2']) +" "+\
     "--final_l2 "+ str(params['final_l2']) +" "+ \
     "--weights_min " + str(params['weights_min']) + " " + \
    "--weights_max " + str(params['weights_max']) + " " + \
     "--max_epochs "+ str(MAX_EPOCHS) +" "+ \
     "--step_size " + str(STEP_SIZE_TRAIN) + " " + \
    "--optimizer " + params['optimizer'] + " " + \
     "--amsgrad " + params['amsgrad'] + " " + \
     "--early_stopping_patience " + str(int(MAX_EPOCHS/3)) + " " + \
     "--which_gpu " + str(GPU)#+ " " + \


    # train model loads data, trains model and stores the results of the hyperparameter search in the hyperopt_result.json file
    os.system('python -u train_model.py ' + cmd)
    results = utils.load_json(os.path.join(run_dir,'chromwave_output','hyperopt_result.json'))
    model = chromwavenet.ChromWaveNet()
    model.deserialize(os.path.join(run_dir,'chromwave_output','hyperopt_result.json'))
    results['model'] = model
    return results


trial_file = os.path.join(files.get_output_directory(),'trials.hyperopt')

trials = FileTrials(persisted_location=trial_file)
if len(trials.trials)<MAX_EVALS:
    try:
        best_run = fmin(train_genomic_wavenet_model, space, algo=ALGO, max_evals=MAX_EVALS, trials=trials)
        best_params=space_eval(space, best_run)
        print('best: ')
        print(best_params)
        utils.save_json(best_params, os.path.join(files.get_output_directory(),'best_hyperparameters.json'))
    except Exception as ex:
         print("Failure in the hyperoptimization run. ")
         print(ex)

print('Loading best hyper-parameters from file')
best_params = utils.load_json(os.path.join(files.get_output_directory(),'best_hyperparameters.json'))


run_dir = os.path.join(out_dir, '02_Full_Training')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

###########################################
# hyperparameter optimization
#########################################

 #
params=best_params

no_existing_trial=len(glob.glob(run_dir+'/Trial*'))
if no_existing_trial>0:
    losses = [utils.load_json(os.path.join(run_dir,'Trial_'+str(i), 'chromwave_output', 'hyperopt_result.json'))['loss'] for i in range(no_existing_trial)]
else:
    losses = []

for i in range(no_existing_trial,N_TRAIN):
    print('Running Trial ' + str(i) + ' /' + str(N_TRAIN))
    output_dir = os.path.join(run_dir,'Trial_'+str(i))
    cmd = '-o ' + output_dir + " " + \
          '--genome_dir ' + genome_dir + " " + \
          '--profiles '+nuc_profile+" "+\
          '--kernel_lengths ' + str(params['kernel_lengths']) + " " + \
          "--conv_n_filters " + str(params['conv_n_filters']) + " " + \
          "--pool_sizes " + "global" + " " + \
        "--learning_rate " + str(params['learning_rate']) + " " + \
        "--reduceLR_rate " + str(params['reduceLR_rate']) + " " + \
          "--dropout " + str(params['dropout']) + " " + \
          "--res_dropout " + str(params['res_dropout']) + " " + \
          "--res_l2 " + str(params['res_l2']) + " " + \
          "--final_l2 " + str(params['final_l2']) + " " + \
          "--weights_min " + str(params['weights_min']) + " " + \
          "--weights_max " + str(params['weights_max']) + " " + \
          "--dilation_depth " + str(params['dilation_depth']) + " " + \
          "--dilation_kernel_size " + str(params['dilation_kernel_size']) + " " + \
          "--max_epochs " + str(MAX_EPOCHS_FULL) + " " + \
          "--step_size " + str(STEP_SIZE) + " " + \
          "--early_stopping_patience " + str(int(MAX_EPOCHS/3)) + " " + \
          "--optimizer " + params['optimizer'] + " " + \
          "--amsgrad " + params['amsgrad'] + " " + \
          "--which_gpu " + str(GPU) #+ " " + \

    # train model loads data, trains model and stores the results of the hyperparameter search in the hyperopt_result.json file
    os.system('python -u train_model.py ' + cmd)
    results = utils.load_json(os.path.join(run_dir, 'Trial_' + str(i), 'chromwave_output', 'hyperopt_result.json'))
    losses.append(results['loss'])

if not os.path.exists(os.path.join(run_dir, 'Best_Trial')):
    os.makedirs(os.path.join(run_dir, 'Best_Trial'))
utils. copytree(os.path.join(run_dir, 'Trial_' + str(numpy.argmin(losses)), 'chromwave_output'),
             os.path.join(run_dir, 'Best_Trial'))


# to run this script run
# mkdir ../../../Test/Hyperopt/Human_Nucleosomes/01_hyperopt
#python -u hyperopt_main.py 2>../../../Test/Hyperopt/Human_Nucleosomes/01_hyperopt/out.log &>../../../Test/Hyperopt/Human_Nucleosomes/01_hyperopt/out.log


