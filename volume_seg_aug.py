import numpy as np
import scipy
import random

# upside-down flip
def aug_flip_ud(image, mask):
  image = image[:,::-1,:,:]
  mask = mask[:,::-1,:]

  return image, mask

# left-right flip
def aug_flip_lr(image, mask):
  image = image[::-1,:,:,:]
  mask = mask[::-1,:,:,]

  return image, mask

# bottom-up flip
def aug_flip_bu(image, mask):
  image = image[:,:,::-1,:]
  mask = mask[:,:,::-1]

  return image, mask

# add value to the image.
# default channel is 0
# default add value is 0.1 (please set it between 0 and 1)
def aug_add(image, mask, channel=0, addValue=0.1):
  image[:,:,:,channel] = image[:,:,:,channel] + addValue
  mask = mask

  return image, mask

# shuffle the channel of the image
def aug_shuffleChannel(image, mask):
  arraySize = image.shape
  shuffleIndex = np.random.permutation(arraySize[-1])
  image = image[:,:,:,shuffleIndex]
  mask = mask

  return image, mask

# randomly crop the original image and mask to the final Size [depth, row, column]
# the image will be rescaled to the original size in the end
def aug_crop(image, mask, finalSize):  

  start_x = np.random.randint(0, mask.shape[0] - finalSize[0] + 1)
  start_y = np.random.randint(0, mask.shape[1] - finalSize[1] + 1)
  start_z = np.random.randint(0, mask.shape[2] - finalSize[2] + 1)

  # print("after crop innerFunction")
  # plt.imshow(np.sqrt(np.square(image[:,:,0,1])+np.square(image[:,:,0,0])))
  # plt.show()
  
  cropped_image = image[start_x:start_x+finalSize[0], start_y:start_y+finalSize[1], start_z:start_z+finalSize[2], :]
  cropped_mask = mask[start_x:start_x+finalSize[0], start_y:start_y+finalSize[1], start_z:start_z+finalSize[2]]

  # print("after crop innerFunction")
  # plt.imshow(np.sqrt(np.square(cropped_image[:,:,0,1])+np.square(cropped_image[:,:,0,0])))
  # plt.show()

  img_scale_factor = [mask.shape[0]/finalSize[0],mask.shape[1]/finalSize[1],mask.shape[2]/finalSize[2],1]
  msk_scale_factor = [mask.shape[0]/finalSize[0],mask.shape[1]/finalSize[1],mask.shape[2]/finalSize[2]]

  image = scipy.ndimage.zoom(cropped_image, img_scale_factor, order=2)
  mask = scipy.ndimage.zoom(cropped_mask, msk_scale_factor, order=2)
  mask = np.where(mask>0.1, 1, 0)

  return image, mask

# Pad value to the image.
def add_pad(image, mask, padWidth=((0, 1), (0, 1), (0, 1)), padValue=0):
  
  padWidth4image = padWidth + ((0,0),)

  image = np.pad(image, padWidth4image, mode='constant', constant_values = padValue)
  mask = np.pad(mask, padWidth, mode='constant', constant_values = padValue)

  return image, mask


# Affine transfomation with rotation angle theta and translation
def aug_affine(image, mask, theta_x=0, theta_y=0, theta_z=np.pi/4, tx=0, ty=0, tz=0):

  # Define the individual rotation matrices

  if(theta_x!=0):
    theta_x = random.uniform(-theta_x, theta_x)

  if(theta_y!=0):
    theta_y = random.uniform(-theta_y, theta_y)

  if(theta_z!=0):
    theta_z = random.uniform(-theta_z, theta_z)

  if(tx!=0):
    tx = random.uniform(-tx, tx)

  if(ty!=0):
    ty = random.uniform(-ty, ty)

  if(tz!=0):
    tz = random.uniform(-tz, tz)


  Rx = np.array([[1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]])

  Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]])

  Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]])

  # Define the translation vector
  t = np.array([tx, ty, tz])

  # Define the 4-by-4 affine transformation matrix
  T = np.eye(4)
  T[:3, :3] = Rz.dot(Ry).dot(Rx)
  T[:3, 3] = t

  mask = scipy.ndimage.affine_transform(mask, T, order=3)
  image = scipy.ndimage.affine_transform(image, T)

  return image, mask




# ----------------------------------------%%%%%%%%%%-----------------------------------------------
# The functions below are used for stacked dataset (image Size of [number of sample, depth, row, column, channel])

def flipImage(image, mask, rowFlip=1, columnFlip=1, depthFlip=0, p=0.5):
  if(rowFlip):
    if np.random.rand()<p:
      image, mask = aug_flip_ud(image, mask)

  if(columnFlip):
    if np.random.rand()<p:
      image, mask = aug_flip_lr(image, mask)

  if(depthFlip):
    if np.random.rand()<p:
      image, mask = aug_flip_bu(image, mask)

  return image, mask


def cropImage(image, mask, cropRatio=0.8, p=0.5):
  remainSize = [np.random.randint(cropRatio*image.shape[0], image.shape[0]), np.random.randint(cropRatio*image.shape[1], image.shape[1]),image.shape[2]]

  if np.random.rand()<p:
    image, mask = aug_crop(image, mask, remainSize)

  return image, mask

def affineTrans(image, mask, theta_x=0, theta_y=0, theta_z=np.pi/4, tx=0, ty=0, tz=0, p=0.5):
  if np.random.rand()<p:
    image, mask = aug_affine(image, mask, theta_x, theta_y, theta_z, tx, ty, tz)
    mask = np.where(mask>0.5, 1, 0)
  return image, mask

def preprocessData(batch_input):
  batch_input, batch_mask = batch_input
  batch_input, batch_mask = flipImage(batch_input, batch_mask,p=0.3)          
  batch_input, batch_mask = cropImage(batch_input, batch_mask,p=0.3)
  batch_input, batch_mask = affineTrans(batch_input, batch_mask, theta_z=np.pi/16,p=0.2)
  # print("in preprocessing:",batch_input.shape, batch_mask.shape)
  return batch_input, batch_mask


  image = np.transpose(image, [0,3,2,1,4])
  mask = np.transpose(mask, [0,3,2,1])
  return image, mask