import matplotlib
matplotlib.use('Agg')
import os

import numpy
from hyperopt import Trials, STATUS_OK, tpe
from hyperopt.fmin import fmin
from hyperopt import space_eval
from hyperas.distributions import choice,uniform

import glob
import pickle

from chromwave.functions import utils

from chromwave import chromwavenet, runtime_dataset, filesystem
from hyperopt.base import Trials
import shutil
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

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
###########################################
# Setting up the directories
#########################################

working_dir = '../../../'

data_dir = os.path.join(working_dir,'data')
model_dir = os.path.join(working_dir,'models')


out_dir = os.path.join(working_dir,'Test/Hyperopt/TFNucleosomes')

genome_dir = os.path.join(data_dir,'genomes/sacCer3')

TF_profile1=os.path.join(data_dir,'henikoff2011','nuclei_20min_rep1_nucFree_fpm_unrolled.bed')
TF_profile2=os.path.join(data_dir,'henikoff2011','nuclei_20min_rep2_nucFree_fpm_unrolled.bed')
nuc_profile1=os.path.join(data_dir,'henikoff2011','nuclei_20min_rep1_monoNuc_fpm_unrolled.bed')
nuc_profile2=os.path.join(data_dir,'henikoff2011','nuclei_20min_rep2_monoNuc_fpm_unrolled.bed')

pbm_dir = os.path.join(data_dir,'ScerTF','PFM')
Nucleosome_deplacing_factor_list = ['Abf1', 'Cbf1','McM1','Rap1','Reb1','Orc1','Asg1','Azf1','Bas1','Ecm22','Ino4','Leu3','Rfx1','Rgm1','Rgt1','Rsc3','Sfp1','Stb4','Stb5','Stp1','Sum1','Sut1','Tbf1','Tbs1','Tea1','Uga3','Ume6','Urc2']
all_dbf = os.listdir(pbm_dir)
select_dbf = [[s  for s in all_dbf if s.__contains__(selection) or s.__contains__(str.upper(selection))] for selection in Nucleosome_deplacing_factor_list]
select_dbf = [s[0] for s in select_dbf]

n_pwm = 2 * len(select_dbf)


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


nuc_model=os.path.join(model_dir,'nucleosomes/brogaard2012_nucModel_6.h5')
dbf_directory = os.path.join(model_dir,'tf','zhu2009')
assert os.path.exists(nuc_model),'Error: Basic nucleosome model does not exist!'

run_dir = os.path.join(out_dir, '01_hyperopt')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)


##########################################
ALGO= tpe.suggest
MAX_EPOCHS=50
MAX_EPOCHS_FULL = 100
STEP_SIZE = 500
MAX_EVALS =100
N_TRAIN=6
MINIMUM_BATCH_SIZE = 32


u=[20,10]
source_profiles=[TF_profile1,nuc_profile1]

f = filesystem.FileSystem(genome_dir ,run_dir,source_profile= source_profiles,overwrite=False, test_fraction = 0.1, val_fraction=0.2,resume = True)

kernel_length_choices = [6, 16, 24,32]
pool_size_choices = [1, 2, 6, 'global']
conv_n_filters_choices = [64, 82, 128, 256, 360, 480, 520]
dilation_depth_choices = [9, 10]
dilation_kernel_size_choices = [2,3]
weights_min_choices = [0.,2.0]
weights_max_choices= [0.0,2.0]
fragment_size_choices = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]


space = { 'optimizer': choice('optimizer', ['adam', 'rmsprop','nadam']),
          'amsgrad':choice('amsgrad',['True','False']),
        'dropout' :uniform('dropout',0.0, 0.7),
        'res_dropout' :uniform('res_dropout',0.0, 0.7),
        'res_l2' :uniform('res_l2',0.0, 0.0001),
        'final_l2' :uniform('final_l2',0.0, 0.0001),
        'kernel_lengths' : choice('kernel_lengths',kernel_length_choices),
        'pool_sizes' : choice('pool_sizes',pool_size_choices),
        'conv_n_filters':choice('conv_n_filters',conv_n_filters_choices),
        'dilation_depth':choice('dilation_depth',dilation_depth_choices),
        'dilation_kernel_size': choice('dilation_kernel_size',dilation_kernel_size_choices),
        'learning_rate': uniform('learning_rate',0.00001, 0.01),
        'reduceLR_rate' :uniform('reduceLR_rate',0.0, 0.99),
        'inject_dbf_pwms': choice('inject_dbf_pwms', ['True', 'False']),
        'use_pretrained_models': choice('use_pretrained_models', ['True', 'False']),
    	'fragment_size': choice('fragment_size', fragment_size_choices),
         'weights_min' :uniform('weights_min',0,2.0),
         'weights_max' :uniform('weights_max',0,2.0)
        }


def train_genomic_wavenet_model(params):

    no_trial=len(glob.glob(f.get_output_directory()+'/hyperas*'))
    print('Running Trial '+str(no_trial)+' /'+str(MAX_EVALS))
    run_dir = os.path.join(f.get_output_directory(),'hyperas_trial_' + str(no_trial))

    print('Creating running dir: ' + run_dir)

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    """Function to do perform a hyperparameter search for a nucleosome wavenet  model """
    print("Hyper opt parameter: ")
    print(params)

    cmd = '-o ' + run_dir + " " + \
          '--genome_dir ' + genome_dir + " " + \
          '--profiles ' + TF_profile1 + " " + nuc_profile1 + " " + \
          '--kernel_lengths ' + str(params['kernel_lengths']) + " " + \
          "--pool_sizes " + "global" + " " + \
          "--learning_rate " + str(params['learning_rate']) + " " + \
          "--dropout " + str(params['dropout']) + " " + \
          "--res_dropout " + str(params['res_dropout']) + " " + \
          "--res_l2 " + str(params['res_l2']) + " " + \
          "--final_l2 " + str(params['final_l2']) + " " + \
          "--minimum_batch_size " + str(MINIMUM_BATCH_SIZE) + " " + \
          "--u " + str(u[0]) + " " + str(u[1]) + " " \
                                                 "--max_epochs " + str(MAX_EPOCHS) + " " + \
          "--step_size " + str(params['fragment_size']) + " " + \
          "--fragment_length " + str(params['fragment_size']) + " " + \
          "--weights_max " + str(params['weights_max']) + " " + \
          "--weights_min " + str(params['weights_min']) + " " + \
          "--optimizer " + params['optimizer'] + " " + \
          "--amsgrad " + params['amsgrad'] + " " + \
          "--early_stopping_patience " + str(int(MAX_EPOCHS / 3)) + " "

    if params['use_pretrained_models'] == 'True':
        print('Using pretrained model ' + nuc_model)
        cmd = cmd + '--keras  ' + nuc_model + " "

    if params['inject_dbf_pwms'] == 'True':
        print('Injecting PWMs ')
        cmd = cmd + '--inject_pwm_select ' + " ".join([str(s) for s in select_dbf]) + " " + '--pwm_dir ' + pbm_dir + " "
        cmd = cmd + "--conv_n_filters " + str(params['conv_n_filters'] + (params['conv_n_filters'] - n_pwm)) + " "
    else:
        cmd = cmd + "--conv_n_filters " + str(params['conv_n_filters']) + " "
    # train model loads data, trains model and stores the results of the hyperparameter search in the hyperopt_result.json file
    os.system('python -u train_model.py ' + cmd)
    if os.path.exists(os.path.join(run_dir, 'chromwave_output', 'hyperopt_result.json')):
        results = utils.load_json(os.path.join(run_dir, 'chromwave_output', 'hyperopt_result.json'))
        model = chromwavenet.ChromWaveNet()
        model.deserialize(os.path.join(run_dir, 'chromwave_output', 'hyperopt_result.json'))
        results['model'] = model
    else:
        results = {'model': None, 'loss': None, 'status': None}
    return results


trial_file = os.path.join(f.get_output_directory(),'trials.hyperopt')

trials = FileTrials(persisted_location=trial_file)
if len(trials.trials)<MAX_EVALS:
    try:
        best_run = fmin(train_genomic_wavenet_model, space, algo=ALGO, max_evals=MAX_EVALS, trials=trials)
        best_params=space_eval(space, best_run)
        print('best: ')
        print(best_params)
        utils.save_json(best_params, os.path.join(f.get_output_directory(),'best_hyperparameters.json'))
    except Exception as ex:
          print("Failure in the hyperoptimization run. ")
          print(ex)

print('Loading best hyper-parameters from file')
best_params = utils.load_json(os.path.join(f.get_output_directory(),'best_hyperparameters.json'))


run_dir = os.path.join(out_dir, '02_Full_Training')

if not os.path.exists(run_dir):
    os.makedirs(run_dir)

###########################################
# hyperparameter optimization
#########################################

 #

params = best_params

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
          '--profiles ' + TF_profile1 + " " + nuc_profile1 + " " + \
          '--kernel_lengths ' + str(params['kernel_lengths']) + " " + \
          '--keras_models ' + nuc_model + " " + \
          "--pool_sizes " + "global" + " " + \
          "--learning_rate " + str(params['learning_rate']) + " " + \
          "--dropout " + str(params['dropout']) + " " + \
          "--res_dropout " + str(params['res_dropout']) + " " + \
          "--res_l2 " + str(params['res_l2']) + " " + \
          "--final_l2 " + str(params['final_l2']) + " " + \
          "--u " + str(u[0]) + " " + str(u[1]) + " " \
                                                 "--minimum_batch_size " + str(MINIMUM_BATCH_SIZE) + " " + \
          "--max_epochs " + str(MAX_EPOCHS_FULL) + " " + \
          "--step_size " + str(STEP_SIZE) + " " + \
          "--fragment_length " + str(params['fragment_size']) + " " + \
          "--weights_max " + str(params['weights_max']) + " " + \
          "--weights_min " + str(params['weights_min']) + " " + \
          "--early_stopping_patience " + str(int(MAX_EPOCHS / 3)) + " " + \
          "--optimizer " + params['optimizer'] + " " + \
          "--amsgrad " + params['amsgrad'] + " "

    if params['use_pretrained_models'] == 'True':
        print('Using pretrained model ' + nuc_model)
        cmd = cmd + '--keras  ' + nuc_model + " "

    if params['inject_dbf_pwms'] == 'True':
        print('Injecting PWMs ')
        cmd = cmd + '--inject_pwm_select ' + " ".join([str(s) for s in select_dbf]) + " " + '--pwm_dir ' + pbm_dir + " "
        cmd = cmd + "--conv_n_filters " + str(params['conv_n_filters'] + (params['conv_n_filters'] - n_pwm)) + " "
    else:
        cmd = cmd + "--conv_n_filters " + str(params['conv_n_filters']) + " "
    # train model loads data, trains model and stores the results of the hyperparameter search in the hyperopt_result.json file
    # train model loads data, trains model and stores the results of the hyperparameter search in the hyperopt_result.json file
    os.system('python -u train_model.py ' + cmd)
    results = utils.load_json(os.path.join(run_dir, 'Trial_' + str(i), 'chromwave_output', 'hyperopt_result.json'))
    losses.append(results['loss'])

    if not os.path.exists(os.path.join(run_dir, 'Best_Trial')):
        os.makedirs(os.path.join(run_dir, 'Best_Trial'))
    copytree(os.path.join(run_dir, 'Trial_' + str(numpy.argmin(losses)), 'chromwave_output'),
             os.path.join(run_dir, 'Best_Trial'))


# to run this script run
# mkdir ../../Test/Hyperopt/TFNucleosomes/
#python -u hyperopt_main.py 2> ../../../Test/Hyperopt/TFNucleosomes/out.log &> ../../../Test/Hyperopt/TFNucleosomes/out.log


