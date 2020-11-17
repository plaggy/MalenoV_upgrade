from train import train_wrapper
from predict import predict
import os
from os.path import dirname, abspath
import datetime
import numpy as np

mode = 'predict'

### common parameters ###
homedir = dirname(dirname(abspath(__file__)))
data_dir = homedir + '/data/'
train_data_dir = data_dir + 'interpretation_points/F3/IL339/il_xl/'
test_data_dir = data_dir + 'interpretation_points/F3/XL500/il_xl/'
facies_names = ['else', 'grizzly', 'high_amp_cont', 'high_amp_dips', 'high_amplitude', 'low_amp_dips',
                     'low_amplitude', 'low_coherency', 'salt']
segy_filename = data_dir + 'F3_entire.segy'
inp_res = np.float32
now = datetime.datetime.now()

### predict mode parameters ###
homedir = dirname(dirname(abspath(__file__)))
data_dir = homedir + '/data/'
predict_dict = {
    'model_path': homedir + '/output/malenov/train_variogram_17_10_4_2020-05-13_14-57/trained.h5',
    'segy_filename': segy_filename,
    'facies_names': facies_names,
    'inp_res': inp_res,
    'test_files': [test_data_dir + x for x in os.listdir(test_data_dir)],
    'save_location': homedir + '/output/malenov/predict_17_10_4_' + now.strftime('%Y-%m-%d_%H-%M') + '/',
    'batch_size': 128,
    'cube_step_interval': 2
}

### train mode parameters ###
train_dict = {
    'train_files': [train_data_dir + x for x in os.listdir(train_data_dir)],
    'epochs': 2,  # number of epochs we run on each training ensemble/mini-batch
    'num_train_ex': 'all',
    'batch_size': 128,  # number of training examples fed to the optimizer as a batch
    'save_location': '/output/malenov/train_variogram_17_10_4_' + now.strftime('%Y-%m-%d_%H-%M') + '/',
    'test_files': [test_data_dir + x for x in os.listdir(test_data_dir)],
    'facies_names': facies_names,
    'validation_split': 0.1,
    'cube_incr_x': [17],
    'cube_incr_y': [10],
    'cube_incr_z': [4],
    'test_split': 0,
    'subcube_step_interval': 2,
    'n_model_samples': 2,
}

if mode == 'train':
    train_wrapper(
        segy_filename=segy_filename,  # Seismic filenames
        inp_res=inp_res,  # Format of input seismic
        train_dict=train_dict,  # Input training dictionary
    )

elif mode == 'predict':
    predict(predict_dict)

else:
    raise ValueError("mode should be train or predict")

