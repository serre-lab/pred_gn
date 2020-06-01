
""" random motion generator
Generates video sequences of random 2D object motion without physics simulation. 
Used for evaluating object motion detection circuit model on different types of motion: translation, rotation, and scaling

Workflow:

A dictiornary specifies random generation parameters for a subset and a random seed.
Sequence specificaions are saved and used for generating batches on the fly.

"""
import numpy as np

import torch
import torch.nn.functional as F

import kornia

import matplotlib.pyplot as plt
import time

from simple_motion_generator import generate_random_sequences

import copy

print('generating meta')

train_size = 30000
val_size = 1000

data_names = [
    'one_bars_t',
    'one_bars_r',
    'one_bars_s',
    'one_rnd_t',
    'one_rnd_r',
    'one_rnd_s',
]
data_configs = []

basic_cfg = {
    'n_sequences': train_size + val_size, #31000
    'n_frames': 20,
    'seed':50,
    'size': 96,
    'boundary': 'walls',
    'shape': {
        'type': ['bars'], #, 'geometric', 'alphabet', 'digits', 'icons'
        'number': 1, # 20
        'size': [0.3,0.5],
        'motion':{
            'translate':    {'v': [-0,0],'a': 0},
            'rotate':       {'v': [-0,0],'a': 0}, 
            'expand':       {'v': [-0,0],'a': 0}, 
        },
    },
    'background':{
        'type': ['empty'], #, 'objects'
        'motion': {
            'translate': [0, 0],
            'rotate': [0, 0],
            'expand': [0, 0],
            'shear': [0, 0],
        },
    }
}

####################################################################################
# 1 object
####################################################################################
# bars 
####################################################################################
# Velocity
####################################################################################

# translation
# seed : 0

# configs = basic_cfg.copy() 
# configs = dict(basic_cfg)
configs =copy.deepcopy(basic_cfg)

configs['seed'] = 0
configs['shape']['motion']['translate'] = {'v': [-0.2,0.2],'a': 0}
data_configs.append(configs)

# rotation
# seed : 1

# configs = basic_cfg.copy() 
# configs = dict(basic_cfg)
configs =copy.deepcopy(basic_cfg)

configs['seed'] = 1
configs['shape']['motion']['rotate'] = {'v': [-0.1,0.1],'a': 0}
data_configs.append(configs)

# size
# seed : 2

# configs = basic_cfg.copy() 
# configs = dict(basic_cfg)
configs =copy.deepcopy(basic_cfg)

configs['seed'] = 2
configs['shape']['motion']['expand'] = {'v': [-0.05,0.05],'a': 0}
data_configs.append(configs)

####################################################################################
# Velocity - Acceleration
####################################################################################
# TODO

####################################################################################
# random objects 
####################################################################################
# Velocity
####################################################################################

# translation
# seed : 3


# configs = basic_cfg.copy() 
# configs = dict(basic_cfg)
configs =copy.deepcopy(basic_cfg)
configs['seed'] = 3
configs['shape']['motion']['translate'] = {'v': [-0.2,0.2],'a': 0}
configs['shape']['type'] = ['geometric', 'digits', 'icons']
data_configs.append(configs)

print(basic_cfg)

# rotation
# seed : 4

# configs = basic_cfg.copy() 
# configs = dict(basic_cfg)
configs =copy.deepcopy(basic_cfg)
configs['seed'] = 4
configs['shape']['motion']['rotate'] = {'v': [-0.2,0.2],'a': 0}
configs['shape']['type'] = ['geometric', 'digits', 'icons']
data_configs.append(configs)

# size
# seed : 5

# configs = basic_cfg.copy() 
# configs = dict(basic_cfg)
configs =copy.deepcopy(basic_cfg)
configs['seed'] = 5
configs['shape']['motion']['expand'] = {'v': [-0.05,0.05],'a': 0}
configs['shape']['type'] = ['geometric', 'digits', 'icons']
data_configs.append(configs)


for i in range(6):
    cfg = data_configs[i]
    name = data_names[i]
    
    specs = generate_random_sequences(cfg)

    np.save(name+'_train.npy', {'cfg':cfg, 'specs':specs[:train_size]})
    np.save(name+'_val.npy', {'cfg':cfg, 'specs':specs[train_size:]} )

####################################################################################
# Velocity - Acceleration
####################################################################################
# TODO



####################################################################################
# many objects
####################################################################################
# bars 
####################################################################################
# TODO