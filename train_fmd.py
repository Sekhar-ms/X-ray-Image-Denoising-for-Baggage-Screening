

import numpy as np
from nets import *
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import join
from imageio import imread
import glob
import cv2
from tqdm import trange

import argparse

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.7
#session = tf.compat.v1.Session(config=config)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.random.seed(1234)
tf.random.set_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--path',required=True,help='path to dataset root')
parser.add_argument('--dataset',required=True,help='dataset name e.g. DV2K_1')
parser.add_argument('--mode',default='uncalib',help='noise model: uncalib, gaussian, poisson, or poissongaussian')
parser.add_argument('--reg',type=float,default=10,help='regularization weight on prior std. dev.')
parser.add_argument('--crop',type=int,default=128,help='crop size')
parser.add_argument('--batch',type=int,default=4,help='batch size')
parser.add_argument('--epoch',type=int,default=300,help='num epochs')
parser.add_argument('--steps',type=int,default=50,help='steps per epoch')
parser.add_argument('--lr',type=float,default=0.0003,help='learning rate')

args = parser.parse_args()

""" Load dataset """

def load_images(noise):
    basepath = args.path + '/' + args.dataset + '/' + noise
    images = []
    for i in range(1,21):
        if i==19: continue
        for path in sorted(glob.glob(basepath + '/%d/*.png'%i)):
    #         images.append(cv2.imread(path,-1))
    #         # print((cv2.imread(path,-1)).shape)
    # return np.stack(images,axis=0)[:,:,:,None]/65535.0
            image = cv2.imread(path,-1)
            image = cv2.resize(image, (512,512))
            # images.append((cv2.imread(path,-1)).resize((512, 512)))
            images.append(image)
            # print("input1", cv2.imread(path,-1).dtype)
            # print('original shape = ',cv2.imread(path).shape)
    return np.stack(images,axis=0)[:,:,:,None]/65535.


train_images = load_images('raw')
np.random.shuffle(train_images)

X = train_images[:-5]
X_val = train_images[-5:]
print('%d training images'%len(X))
print('%d validation images'%len(X_val))


# def augment_images(images):
#   augmented = np.concatenate((images,
#                               np.rot90(images, k=1, axes=(1, 2)),
#                               np.rot90(images, k=2, axes=(1, 2))))
#   augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
# #   print("aug=",augmented.shape)
# #   print("images = ", images.shape)
#   return augmented

def augment_images(images):
  augmented = np.concatenate((images,
                              np.rot90(images, k=1, axes=(1, 2)),
                              np.rot90(images, k=2, axes=(1, 2)),
                              np.rot90(images, k=3, axes=(1, 2))))
  augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
#   print("aug=",augmented.shape)
#   print("images = ", images.shape)
  return augmented

X = augment_images(X)
print("X shape=",X.shape)
X_val = augment_images(X_val)
print('%d training images after augmenting'%len(X))

""" Training """
""" Train on random crops of the training image."""

def random_crop_generator(data, crop_size, batch_size):
    while True:
        # print("data shape=",data.shape)
        inds = np.random.randint(data.shape[0],size=batch_size)
        y = np.random.randint(data.shape[1]-crop_size,size=batch_size)
        x = np.random.randint(data.shape[2]-crop_size,size=batch_size)
        batch = np.zeros((batch_size,crop_size,crop_size,1),dtype=data.dtype)
        for i,ind in enumerate(inds):
            batch[i] = data[ind,y[i]:y[i]+crop_size,x[i]:x[i]+crop_size]
        yield batch, None
        # print("batch:",batch.shape)

model = gaussian_blindspot_network((args.crop, args.crop, 1),args.mode,args.reg)

model.compile(optimizer=Adam(args.lr))

os.makedirs('weights',exist_ok=True)

if args.mode == 'uncalib' or args.mode == 'mse':
    weights_path = 'weights/weights.%s.%s.latest.hdf5'%(args.dataset,args.mode)
else:
    weights_path = 'weights/weights.%s.%s.%0.3f.latest.hdf5'%(args.dataset,args.mode,args.reg)

callbacks = []
callbacks.append(ModelCheckpoint(filepath=weights_path, monitor='val_loss',save_best_only=1,verbose=1))
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))

gen = random_crop_generator(X,args.crop,args.batch)
val_crops = []
for y in range(0,X_val.shape[1],args.crop):
    if y+args.crop > X_val.shape[1]: continue
    for x in range(0,X_val.shape[2],args.crop):
        if x+args.crop > X_val.shape[2]: continue
        val_crops.append(X_val[:,y:y+args.crop,x:x+args.crop])
val_data = np.concatenate(val_crops,axis=0)
# print("val data = ",val_data.shape)

history = model.fit(x=gen, y=None,
                    steps_per_epoch=args.steps,
                    validation_data=(val_data,None),
                    epochs=args.epoch,
                    verbose=1,
                    callbacks=callbacks)

# Get training and test loss histories
training_loss = history.history['loss']
valid_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r-')
plt.plot(epoch_count, valid_loss, 'b-')
plt.grid()
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

