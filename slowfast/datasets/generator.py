

# random motion generator

# confgurations
configs = {
    'colored': False,
    'shape':{
        'type': ['bars', 'geometric', 'alphabet', 'digits', 'icons'],
        'variability': 'random', # or specify object name
        'number': 1, # 20
        'size': [0.1,0.3]
        'motion':{
            'translate':    {'v': [0.1,0.2],'a': [0, 'fixed', 'friction', 'circular', 'elliptical', 'sinusoidal']},
            'rotate':       {'v': [0.1,0.2],'a': [0, 'fixed', 'friction']},
            'expand':       {'v': [0.1,0.2],'a': [0, 'fixed', 'friction']},
        },
    },
    'background':{
        'type': ['empty', 'checkerboard', 'noise', 'objects']
        'motion': {
            'translate': [0.1, 0.2],
            'rotate': [0.1, 0.2],
            'expand': [0.1, 0.2],
            'shear': [0.1, 0.2],
        }
    },
}


sequence = {
    'size':128,
    'background':{
        'type': 'empty',
        'motion': None,
    }
    'shapes':[{
        'type': 'bars',
        'dims': [0.5,0.5],
        'motion': {
            'position': [0.1, 0.2],
            'size': 0.15,
            'expand': {'v':0.02},
        }
        'positions':[[0.1, 0.2],
                    [0.1, 0.2],
                    [0.1, 0.2],
                    [0.1, 0.2],
                    [0.1, 0.2]],
        'angles':[0,0,0,0,0],
        'sizes':[0.15, 0.17, 0.19, 0.21, 0.23]
    }],
}


# process:

# 1- for each object:
    # if rotate is true
        # generate rotations
    # if expand is true
        # generate expantions
    # put all in the same uniform background zeros
# 2- if generate background:
    # generate background
    # add each frame to the background


# generate sequences and save them in npys.
# use sequences to generate frames on the fly.



configs = {
    'colored': False,
    'shape':{
        'type': ['bars', 'geometric', 'alphabet', 'digits', 'icons'],
        'variability': 'random', # or specify object name
        'number': 1, # 20
        'size': [0.1,0.3]
        'motion':{
            'translate':    {'v': [0.1,0.2],'a': [0, 'fixed', 'friction', 'circular', 'elliptical', 'sinusoidal']},
            'rotate':       {'v': [0.1,0.2],'a': [0, 'fixed', 'friction']},
            'expand':       {'v': [0.1,0.2],'a': [0, 'fixed', 'friction']},
        },
    },
    'background':{
        'type': ['empty', 'checkerboard', 'noise', 'objects']
        'motion': {
            'translate': [0.1, 0.2],
            'rotate': [0.1, 0.2],
            'expand': [0.1, 0.2],
            'shear': [0.1, 0.2],
        }
    },
}

def generate_random_sequences(config):
    """ generates random sequences sepcified as a dictionary following a configuration
    """
    pass

def generate_background(spec):
    """ creates a background from a sequence specification, outputs either one frame or many frames for a moving background
    """
    pass

def generate_background(spec):
    """ creates a background from a sequence specification, outputs either one frame or many frames for a moving background
    """
    pass

def generate_sequence(spec):
    """ creates a sequence from a sequence specification, outputs video frames
    """
    pass