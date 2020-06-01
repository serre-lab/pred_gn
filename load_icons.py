




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
        
#         # print(im.max())
#         im = 1-im/im.max()

#         im_data[folder].append(im)
        
#     im_data[folder] = np.stack(im_data[folder])

# np.save('icons.npy', im_data)
# # print(im.shape)

import torchvision
import numpy as np

data = torchvision.datasets.MNIST('./mnist', train=True, transform=None, target_transform=None, download=True)
# im = data[0][0]
# im = np.array(im)
# print(im.shape)
im_data = {}

for i in range(10):
    im_data[i] = []
for i in range(len(data)):
    im, l = data[i]
    im = np.array(im)
    im_data[l].append(im)

np.save('mnist.npy', im_data)