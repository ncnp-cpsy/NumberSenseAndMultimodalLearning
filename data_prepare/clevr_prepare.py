from PIL import Image
import numpy as np
import json
import torch
from torchvision.utils import save_image, make_grid

######  TRAIN #######
#image
img_all = []
for i in range(60000):
    original_path = '/home/ubuntu/CLEVR_v1.0/images/train_ncnp/' + str(i).zfill(6) +'.png'
    image = Image.open(original_path)
    im = np.array(image).T / 255.0
    img_all.append([im])
    
clevr_train_images = torch.FloatTensor(img_all)
print('image prepare daone!')

#label
json_open = open('/home/ubuntu/CLEVR_v1.0/scenes/CLEVR_train_scenes.json', 'r')
json_load = json.load(json_open)

len_all = []
for i in range(60000):
    len_all.append(len(json_load['scenes'][i]['objects']))
    
clevr_train_labels = torch.FloatTensor(len_all)
print('label prepare daone!')

#save
torch.save(clevr_train_images, '../data/clevr_train_images.pt')
torch.save(clevr_train_labels, '../data/clevr_train_labels.pt')



######  TEST #######


#image
img_all = []
for i in range(1000):
    original_path = '/home/ubuntu/CLEVR_v1.0/images/test_ncnp/' + str(i).zfill(6) +'.png'
    image = Image.open(original_path)
    im = np.array(image).T / 255.0
    img_all.append([im])
clevr_test_images = torch.FloatTensor(img_all)

print('image prepare daone!')

#label
json_open = open('/home/ubuntu/CLEVR_v1.0/scenes/CLEVR_val_scenes.json', 'r')
json_load = json.load(json_open)

len_all = []
for i in range(1000):
    len_all.append(len(json_load['scenes'][i]['objects']))
clevr_test_labels = torch.FloatTensor(len_all)

print('label prepare daone!')

#save
torch.save(clevr_test_images, '../data/clevr_test_images.pt')
torch.save(clevr_test_labels, '../data/clevr_test_labels.pt')



##### CHECK #####
inds = [np.random.randint(0, len(clevr_train_images)) for i in range(10)]
save_image(clevr_train_images[inds], 'generated_images/test_clevr.jpg')
