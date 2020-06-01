
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



# import os

# folders = ['money', 'disk', 'envelope', 'moon', 'water_polo', 'boat', 'numbers', 'cartwheeling', 'surfing', 'kiss', 'hand', 'vehicle', 'emotion_face', 'airplane', 'phone', 'mountain', 'holding_hands', 'prohibit_sign', 'worker', 'books', 'flag', 'bunny_ears', 'marine_animals', 'monkey', 'clock', 'bird', 'star', 'feline', 'tree', 'blade', 'writing_utensil', 'fast_train', 'arrow_directions', 'flower', 'biking', 'drinks', 'hat', 'heart', 'medal', 'footwear', 'family', 'golfing', 'lock', 'umbrella', 'japanese_ideograph', 'ball', 'building', 'mailbox', 'cloud', 'wrestling']

# data_path = "/users/azerroug/scratch/icons-50/Icons-50/"

# from PIL import Image
# import numpy as np

# im_data = {}
# for folder in folders:
#     images = os.listdir(os.path.join(data_path, folder))
#     im_data[folder] = []
#     for im_name in images:
#         image_path = os.path.join(data_path, folder, im_name)
#         im = np.array(Image.open(image_path).convert('L').resize((28,28), resample=0))
#         im_data[folder].append(im)
#     im_data[folder] = np.stack(im_data[folder])

# np.save('icons.npy', im_data)

icons = np.load('icons.npy', allow_pickle=True).item()
mnist = np.load('mnist.npy', allow_pickle=True).item()
# confgurations
configs = {
    'n_sequences': 10,
    'n_frames': 5,
    'seed':1,
    'size': 96,
    'boundary': 'walls',
    'shape':{
        'type': ['bars', 'geometric', 'alphabet', 'digits', 'icons'],
        # 'variability': 'random', # or specify object name
        'number': 2, # 20
        'size': [0.3,0.5],
        'motion':{
            'translate':    {'v': [-0.05,0.05],'a': [0, [-0.1, 0.1], 'friction', 'elliptical', 'sinusoidal']},
            'rotate':       {'v': [-0.05,0.05],'a': [0, [-0.1, 0.1], 'friction']},
            'expand':       {'v': [-0.05,0.05],'a': [0, [-0.1, 0.1], 'friction']},
        },
    },
    'background':{
        'type': ['empty', 'checkerboard', 'noise', 'objects'],
        'motion': {
            'translate': [-0.05, 0.05],
            'rotate': [-0.05, 0.05],
            'expand': [-0.05, 0.05],
            'shear': [-0.05, 0.05],
        },
    },
}

# configs = {
#     'n_sequences': 10,
#     'n_frames': 5,
#     'seed':1,
#     'size': 96,
#     'shape':{
#         'type': [ 'digits'], #'bars', 'geometric', 'alphabet', 'digits',
#         # 'variability': 'random', # or specify object name
#         'number': 5, # 20
#         'size': [0.1,0.5],
#         'motion':{
#             'translate':    {'v': [-0.15,0.15],'a': 0},
#             # 'rotate':       {'v': [-0.05,0.05],'a': [0, [-0.1, 0.1], 'friction']},
#             # 'expand':       {'v': [-0.05,0.05],'a': [0, [-0.1, 0.1], 'friction']},
#         },
#     },
#     'background':{
#         'type': ['empty', 'checkerboard', 'noise', 'objects'],
#         'motion': {
#             'translate': [-0.05, 0.05],
#             'rotate': [-0.05, 0.05],
#             'expand': [-0.05, 0.05],
#             'shear': [-0.05, 0.05],
#         },
#     },
# }


sequence = {
    'size':128,
    'background':{
        'type': 'empty',
        'motion': None,
    },
    'shapes':[{
        'type': 'bars',
        'dims': [0.5,0.5],
        'motion': {
            'position': [0.1, 0.2],
            'size': 0.15,
            'expand': {'v':0.02},
        },
        'positions':[[0.1, 0.2],
                    [0.1, 0.2],
                    [0.1, 0.2],
                    [0.1, 0.2],
                    [0.1, 0.2]],
        'angles':[0,0,0,0,0],
        'sizes':[0.15, 0.17, 0.19, 0.21, 0.23]
    }],
}

shape_types ={
    'bars': None,
    'geometric': [
        'ellipse',
        'rectangle',
        'star',
        'triangle',
    ],
    'alphabet': ['a','b','c','d','e','f'], # get images froma dataset and sample randomly 
    'digits': ['0','1','2','3','4','5','6','7','8','9'], # sample from mnist
    'icons': list(icons.keys()), # get list from the kaggle dataset
}

def get_random_path(shape_type, shape_class):
    """ returns a path to a random object
    """
    if shape_type =='icons':
        list(icons.keys()) 
        path = ['icons', shape_class, np.random.choice(len(icons[shape_class]))]
    elif shape_type =='digits':
        list(icons.keys()) 
        path = ['digits', shape_class, np.random.choice(len(mnist[int(shape_class)]))]
    elif shape_type =='geometric':
        
        path = 'fiddle/circle.png'
    # elif shape_type =='digits':
    #     path = ['digits', shape_class, np.random.choice(len(mnist[shape_class]))]
    else:
        path = 'fiddle/circle.png'
    return path

def get_translation_sequence(position, v_init, a, n_frames, boundary='walls'):
    """ simulates an object's trajectory from an initial velocity and an acceleration type
    """
    x, y = position[0], position[1]
    vx, vy = v_init[0], v_init[1]
    if isinstance(a,list):
        ax, ay = a[0], a[1]
    sequence = [[x,y]]
    for i in range(n_frames-1):
        x = x+vx    
        y = y+vy    
        # if (x>1 or x<0) and a != 'elliptical':
        #     x = x%1
        # if (y>1 or y<0) and a != 'elliptical':
        #     y = y%1

        # TODO
        # 'friction' 'sinusoidal'
        if a == 0:
            sequence.append([x,y])
        elif a == 'fixed':
            vx = vx+ax
            vy = vy+ay
            sequence.append([x,y])
        elif isinstance(a,list):
            vx = vx+ax
            vy = vy+ay
            sequence.append([x,y])
        elif a == 'elliptical':
            ax = (x - 0.5)/10
            ay = (y - 0.5)/10
            vx = vx+ax
            vy = vy+ay
            sequence.append([x,y])
        elif a == 'friction':
            ax = -vx/10
            ay = -vy/10
            vx = vx+ax
            vy = vy+ay
            sequence.append([x,y])
        elif a == 'sinusoidal':
            ax = np.sin(i/10)/10
            ay = -vy/10
            vx = vx+ax
            vy = vy+ay
            sequence.append([x,y])
        if boundary=='walls':
            if (x <= 0 and vx<0) or (x >= 1 and vx > 1):
                vx = -vx
            if (y <= 0 and vy<0) or (y >= 1 and vy > 1):
                vy = -vy
                
    return sequence

def get_rotation_sequence(angle_0, v_init, a, n_frames):
    """ simulates an object's rotation from an initial velocity and an acceleration type
    """
    angle = angle_0
    sequence = [angle]
    # TODO
    # friction, sinusoidal
    for i in range(n_frames-1):
        angle = angle + v_init
        if not isinstance(a, str):
            v_init = v_init + a
        
        sequence.append(angle)

    return sequence

def get_scale_sequence(scale_0, v_init, a, n_frames):
    """ simulates an object's size change from an initial velocity and an acceleration type
    """
    scale = scale_0
    sequence = [scale]
    # TODO
    # friction, sinusoidal
    for i in range(n_frames-1):
        scale = max(0.05, scale + v_init)
        if not isinstance(a, str):
            v_init = v_init + a
        sequence.append(scale)

    return sequence

# TODO
# add object paths
# generating trajectory from initial variables
def generate_random_sequences(config, save_file=None):
    """ generates random sequences sepcified as a dictionary following a configuration
    """
    n_sequences = config['n_sequences']
    n_frames = config['n_frames']
    size = config['size']
    np.random.seed(config['seed'])
    bg_cfg = config['background']
    s_cfg = config['shape']
    n_objects = s_cfg['number']
    data = []
    for i in range(n_sequences):
        sequence = {'n_frames': n_frames, 'size': size}
        
        #####################################################################
        # background configs
        #####################################################################
        if isinstance(bg_cfg['type'], list):
            bg_type = bg_cfg['type'][np.random.choice(len(bg_cfg['type']))]
        else:
            bg_type = bg_cfg['type']

        if bg_type == 'checkerboard':
            # add checkerboard params
            sequence['background'] = {
                'type': 'checkerboard',
                'scale': np.random.uniform(0.1,0.3),
            }
        elif bg_type == 'noise':
            sequence['background'] = {
                'type': 'noise',
                'noise_type': 'gaussian', #'salt pepper'
                'scale': np.random.uniform(0.1,0.4), # scale by which to multiply the gaussian noise highest point 
            }
        elif bg_type == 'objects':
            sequence['background'] = {
                'type': 'objects',
                'n_objects': np.random.choice(100),
                'objects': [], # generate n_object object specs
                'scale': np.random.uniform(0.1,0.4), # scale by which to multiply the gaussian noise highest point 
                'motion': None,
            }
        else:
            sequence['background'] = {
                'type': 'empty',
                'motion': None,
            }
            
        if 'motion' in bg_cfg:
            sequence['background']['motion'] = {}
            if 'translate' in bg_cfg['motion']:
                sequence['background']['motion']['translate'] = np.random.uniform(*bg_cfg['motion']['translate'], 2)
            if 'rotate' in bg_cfg['motion']:
                sequence['background']['motion']['translate'] = np.random.uniform(*bg_cfg['motion']['translate'], 2)
            if 'expand' in bg_cfg['motion']:
                sequence['background']['motion']['translate'] = np.random.uniform(*bg_cfg['motion']['translate'], 2)
            if 'shear' in bg_cfg['motion']:
                sequence['background']['motion']['translate'] = np.random.uniform(*bg_cfg['motion']['translate'], 2)

        #####################################################################
        # image configs
        #####################################################################
        
        sequence['shapes'] = []
        
        n_objects = s_cfg['number']

        for i in range(n_objects):
            shape_seq={}
            # specify object shape and type
            shape_seq['boundary'] = config['boundary']
            if isinstance(s_cfg['type'], list):
                k = np.random.choice(len(s_cfg['type']))
                shape_seq['type'] = s_cfg['type'][k]
            else:
                shape_seq['type'] = s_cfg['type']
            
            if shape_seq['type'] == 'bars':
                shape_seq['dims'] = np.random.uniform(0,1,2)
            else:
                if 'shape' in s_cfg:
                    shape_seq['shape'] = s_cfg['shape']
                else:
                    st = shape_types[shape_seq['type']]
                    shape_seq['shape'] = st[np.random.choice(len(st))]
                    # shape_seq['shape'] = shape_types[shape_seq['type']][np.random.choice(len(shape_types[s_cfg['type']]))]
                shape_seq['path'] = get_random_path(shape_seq['type'], shape_seq['shape'])
            
            # initial parameters
            shape_seq['motion'] = {
                'position': [np.random.uniform(), np.random.uniform()],
                'size': np.random.uniform()*(s_cfg['size'][1] - s_cfg['size'][0])+s_cfg['size'][0],
            }

            # motion parameters

            if 'translate' in s_cfg['motion']:
                shape_seq['motion']['translate'] = {'v': s_cfg['motion']['translate']['v'], 'a':s_cfg['motion']['translate']['a']}

                a = shape_seq['motion']['translate']['a']
                
                if isinstance(a, list):
                    a = a[np.random.choice(len(a))]
                v_init = shape_seq['motion']['translate']['v']
                v_init = [np.random.uniform() * (v_init[1] - v_init[0]) + v_init[0], 
                        np.random.uniform() * (v_init[1] - v_init[0]) + v_init[0]]
                shape_seq['motion']['translate'] = {'v':v_init, 'a': a}
                positions = get_translation_sequence(shape_seq['motion']['position'], v_init, a, n_frames, boundary=config['boundary'])
                shape_seq['positions'] = positions

            if 'rotate' in s_cfg['motion']:
                shape_seq['motion']['rotate'] = {'v': s_cfg['motion']['rotate']['v'], 'a':s_cfg['motion']['rotate']['a']}
                a = shape_seq['motion']['rotate']['a']
                # type of acceleration
                if isinstance(a, list):
                    a = a[np.random.choice(len(a))]
                # random sample value of acceleration if is list
                if isinstance(a, list):
                    a = np.random.uniform(a[0], a[1])
                
                v_init = shape_seq['motion']['rotate']['v']
                v_init = np.random.uniform() * (v_init[1] - v_init[0]) + v_init[0]
                shape_seq['motion']['rotate'] = {'v':v_init, 'a': a}
                angles = get_rotation_sequence(0, v_init, a, n_frames)
                shape_seq['angles'] = angles

            if 'expand' in s_cfg['motion']:
                shape_seq['motion']['expand'] = {'v': s_cfg['motion']['expand']['v'], 'a':s_cfg['motion']['expand']['a']}
                a = shape_seq['motion']['expand']['a']
                # type of acceleration
                if isinstance(a, list):
                    a = a[np.random.choice(len(a))]
                # random sample value of acceleration if is list
                if isinstance(a, list):
                    a = np.random.uniform(a[0], a[1])
                v_init = shape_seq['motion']['expand']['v']
                v_init = np.random.uniform() * (v_init[1] - v_init[0]) + v_init[0]
                shape_seq['motion']['expand'] = {'v':v_init, 'a': a}
                sizes = get_scale_sequence(shape_seq['motion']['size'], v_init, a, n_frames)
                
                shape_seq['sizes'] = sizes
                
            sequence['shapes'].append(shape_seq)
        
        data.append(sequence)

    if save_file is not None:
        np.save(save_file, data)
    return data


def generate_background(spec, size, n_frames):
    """ creates a background from a sequence specification, outputs either one frame or many frames for a moving background
    """
    # TODO
    # objects
    # add motion

    if spec['type'] == 'checkerboard':
        scale = int(spec['scale']*size)
        # if 'motion' in spec:
        #     size_r = size
        #     size = size+size//4

        #     bg = torch.zeros([1,size,size])
            
        #     bg[:, np.arange(size)%(scale*2)<scale] = 1
        #     bg[:, :, np.arange(size)%(scale*2)>scale] = 1 - bg[:, :, np.arange(size)%(scale*2)>scale]
        #     bg = torch.stack([bg]*n_frames)
        #     if 'tram'
        # else:
        bg = torch.zeros([1,size,size])
        
        bg[:, np.arange(size)%(scale*2)<scale] = 1
        bg[:, :, np.arange(size)%(scale*2)>scale] = 1 - bg[:, :, np.arange(size)%(scale*2)>scale]

    elif spec['type'] == 'noise':
        
        if spec['noise_type'] == 'gaussian':
            bg = torch.randn([1,size,size])
            bg = bg - bg.min()
            bg = bg / bg.max()
        elif spec['noise_type'] == 'salt_pepper':
            bg = torch.rand([1,size,size])
            bg[bg>0.5] = 1
            bg[bg<0.5] = 0

    # elif spec['type'] == 'objects':
    else:
        bg = torch.zeros([1,size,size])
    
    return bg

def load_image(spec):
    """ load object image
    """ 
    size = 28
    center = size//2
    r = center-4

    icons

    # print(spec['type'])
    if spec['type'] == 'bars':
        image = torch.zeros([1,size,size])
        dims = spec['dims'] 
        x, y = dims[0]/max(dims), dims[1]/max(dims)
        image[:,int(center-x*r): int(center+x*r) ,int(center-y*r): int(center+y*r)] = 1
    elif spec['type'] == 'cross':
        image = torch.zeros([1,size,size])
        x, y = spec['dims'][0], spec['dims'][1]
        image[:,:,int(center-y*r): int(center+y*r)] = 1
        image[:,int(center-x*r): int(center+x*r)] = 1
    elif spec['type'] == 'icons':
        path = spec['path']
        image = torch.Tensor(icons[path[1]][path[2]])[None,:]
    
    elif spec['type'] == 'digits':
        path = spec['path']
        image = torch.Tensor(mnist[int(path[1])][int(path[2])])[None,:]
        
    else:
        image = torch.zeros([1,size,size])
        image[:,center-r:center+r,center-r:center+r] = 1
        
    return image

def object_within_boundary(position, obj_shape, frame_shape):
    """ checks whether the object is within the boundaries of the image
    """
    out = True
    out = out and (position[0] + obj_shape[0]//2 > 0)
    out = out and (position[0] - obj_shape[0]//2 < frame_shape[0])
    out = out and (position[1] + obj_shape[1]//2 > 0)
    out = out and (position[1] - obj_shape[1]//2 < frame_shape[1])
    return out

def object_borders_within_frame(position, obj_frame_size, frame_shape):
    x, y = position[0], position[1]
    h, w = obj_frame_size[0], obj_frame_size[1]
    h_f, w_f = frame_shape[0], frame_shape[1]

    x2 = x+h//2 if h%2==0 else x+h//2+1 
    y2 = y+w//2 if h%2==0 else y+w//2+1
    x1 = int(max(0, x-h//2))
    x2 = int(min(h_f, x2))
    y1 = int(max(0, y-w//2))
    y2 = int(min(w_f, y2))

    x1_s = -int(min(0, x-h//2))
    y1_s = -int(min(0, y-w//2))

    x2_s = x1_s + x2 - x1
    y2_s = y1_s + y2 - y1  
    
    # x2_s = h + int(min(0, h_f - (x+h//2)))#+1
    # y2_s = w + int(min(0, w_f - (y+w//2)))#+1  

    # if (x1-x2)%2 != (x1_s-x2_s)%2:
    #     x1_s-=1
    # if (y1-y2)%2 != (y1_s-y2_s)%2:
    #     y1_s-=1
    return (x1, x2, y1, y2, x1_s, x2_s, y1_s, y2_s)

def generate_object_sequence(frames, spec, n_frames):
    """ creates a background from a sequence specification, outputs either one frame or many frames for a moving background
    1- for each object:
        if rotate is true
            generate rotations
        if expand is true
            generate expantions
        put all in the same uniform background zeros

    """
    # read image
    im = load_image(spec)
    frame_shape = frames.shape[-2:]
    frame_size = min(frames.shape[-2:])

    # if rotations create rotations
    if 'angles' in spec:
        # group all angles and images in batch  
        angles = spec['angles']
        im = kornia.geometry.transform.rotate(torch.stack([im]*len(angles)),torch.Tensor(angles)*360)
    
    for i in range(n_frames):
        # print(spec['angles'])
        obj_frame = im[i] if 'angles' in spec else im
        size = spec['sizes'][i] if 'sizes' in spec else spec['motion']['size']
        #print(spec['positions'], i)
        position = spec['positions'][i] if 'positions' in spec else spec['motion']['position']
        position = [int(position[0]*frame_shape[0]), int(position[1]*frame_shape[1])]
        obj_frame_size = (int(frame_size*size), int(frame_size*size))
        
        if object_within_boundary(position, obj_frame_size, frame_shape):
            obj_frame = F.interpolate(obj_frame[None,:], obj_frame_size, mode='bilinear', align_corners=True)[0]
            # cut shape
            (x1, x2, y1, y2, x1_s, x2_s, y1_s, y2_s) = \
                object_borders_within_frame(position, obj_frame_size, frame_shape)
            # print((x1, x2, y1, y2, x1_s, x2_s, y1_s, y2_s), i)
            # print(x1_s,x2_s, y1_s,y2_s)
            obj_frame = obj_frame[:,x1_s:x2_s, y1_s:y2_s]
            frames[i, :, x1:x2, y1:y2] = frames[i, :, x1:x2, y1:y2] + obj_frame#[:,x1_s:x2_s, y1_s:y2_s]

            # TODO 
            # add case where background isn't zeros
    
    frames = torch.clamp(frames, 0,1)

    return frames

def generate_sequence(spec):
    """ creates a sequence from a sequence specification, outputs video frames
    1- if generate background:
        generate background
        add each frame to the background
    2- for each object generate the corresponding frames and add them to the sequence
    """
    bg = generate_background(spec['background'], spec['size'], spec['n_frames']) #, spec['n_frames']
    if len(bg.shape)==3:
        bg = torch.stack([bg]*spec['n_frames'], 0)
    for object_shape_spec in spec['shapes']:
        bg = generate_object_sequence(bg, object_shape_spec, spec['n_frames']) 
    return bg



# f_sh = [50,50]
# o_sh = [20,20]
# pos = [0,0]
# (x1, x2, y1, y2, x1_s, x2_s, y1_s, y2_s) = \
#     object_borders_within_frame(pos, o_sh, f_sh)

# if (x1-x2)%2 != (x1_s-x2_s)%2 or (y1-y2)%2 != (y1_s-y2_s)%2:
#     print((x1, x2, y1, y2, x1_s, x2_s, y1_s, y2_s))

import matplotlib.pyplot as plt
import time

print('generating meta')


configs = {
    'n_sequences': 300,
    'n_frames': 20,
    'seed':50,
    'size': 96,
    'shape':{
        'type': ['bars', 'geometric', 'alphabet', 'digits', 'icons'],
        # 'variability': 'random', # or specify object name
        'number': 1, # 20
        'size': [0.3,0.5],
        'motion':{
            'translate':    {'v': [-0.1,0.1],'a': 0}, #[0, [-0.1, 0.1], 'friction', 'elliptical', 'sinusoidal']
            'rotate':       {'v': [-0.1,0.1],'a': 0}, #[0, [-0.1, 0.1], 'friction']
            'expand':       {'v': [-0.1,0.1],'a': 0}, #[0, [-0.1, 0.1], 'friction']
        },
    },
    'background':{
        'type': ['empty', 'checkerboard', 'noise'], #, 'objects'
        'motion': {
            'translate': [-0.1, 0.1],
            'rotate': [-0.1, 0.1],
            'expand': [-0.1, 0.1],
            'shear': [-0.1, 0.1],
        },
    },
}


# specs = generate_random_sequences(configs)

# print('generating sequences')


# # f, ax = plt.subplots(10,5,figsize=(10,20))
# # for j in range(10):
# #     # print(specs[j])
# #     frames = generate_sequence(specs[j])
# #     for i in range(5):
# #         ax[j,i].imshow(frames.data.numpy()[i,0])

# # plt.savefig('examples.png')
# times = []

# for j in range(configs['n_sequences']):
    
#     if j % 10==0:
#         print(j)
#     try:
#         start = time.perf_counter()
#         frames = generate_sequence(specs[j])
#         gen = time.perf_counter() - start
#         times.append(gen)
#     except:
#         print(specs[j])
        
# print(np.mean(times))
    