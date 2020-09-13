import os
import pandas
import numpy
from ..functions import utils
from keras.models import load_model


def load_existing_keras_conv_weights( model_path):
    base_model = load_model(model_path, custom_objects = {'mcc': utils.mcc,'r_squared':utils.r_squared,'cor_pcc':utils.cor_pcc})
    return base_model.layers[0].get_weights()



def load_scerTF_pwm_conv_weights(pbm_dir, dbf_list=None):
    if not dbf_list:
        print('loading all existing pwm files in directory')
        dbf_list=os.listdir(pbm_dir)
    pbm_files = [os.path.join(pbm_dir, f) for f in dbf_list]
    weights=[]
    for file in pbm_files:
        pwm=pandas.read_table(file,  header =None, sep=" ")
        assert pwm.shape[0]==4
        pwm=pwm.drop(pwm.columns[[0]],axis=1)
        pwm = pwm.drop(pwm.columns[[0]], axis=1)
        pwm = pwm.drop(pwm.columns[[-1]], axis=1)
        weights.append(numpy.array(pwm).transpose())

    return weights