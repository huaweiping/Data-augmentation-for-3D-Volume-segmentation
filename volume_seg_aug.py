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
  image = image[:,:,::-1,:]
  mask = mask[:,:,::-1]

  return image, mask

# bottom-up flip
def aug_flip_bu(image, mask):
  image = image[::-1,:,:,:]
  mask = mask[::-1,:,:]

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
  if(np.any(np.greater(np.array(finalSize),mask.shape))):
    raise ValueError("Crop size is larger then the original image")
  
  start_x = np.random.randint(0, mask.shape[1] - finalSize[1] + 1)
  start_y = np.random.randint(0, mask.shape[2] - finalSize[2] + 1)
  start_z = np.random.randint(0, mask.shape[0] - finalSize[0] + 1)

  cropped_image = image[start_z:start_z+finalSize[0], start_x:start_x+finalSize[1], start_y:start_y+finalSize[2], :]
  cropped_mask = mask[start_z:start_z+finalSize[0], start_x:start_x+finalSize[1], start_y:start_y+finalSize[2]]

  img_scale_factor = [mask.shape[0]/finalSize[0],mask.shape[1]/finalSize[1],mask.shape[2]/finalSize[2],1]
  msk_scale_factor = [mask.shape[0]/finalSize[0],mask.shape[1]/finalSize[1],mask.shape[2]/finalSize[2]]

  image = scipy.ndimage.zoom(cropped_image, img_scale_factor, order=2)
  mask = scipy.ndimage.zoom(cropped_mask, msk_scale_factor, order=2)

  print("croppedmask:\n",cropped_mask)
  print("rescaledmask:\n",mask)

  return image, mask

# Pad value to the image.
def add_pad(image, mask, padWidth=((0, 1), (0, 1), (0, 1)), padValue=0):
  
  padWidth4image = padWidth + ((0,0),)
  print(padWidth4image)

  image = np.pad(image, padWidth4image, mode='constant', constant_values = padValue)
  mask = np.pad(mask, padWidth, mode='constant', constant_values = padValue)

  return image, mask


# Affine transfomation with rotation angle theta and translation
def aug_affine(image, mask, theta_x=0, theta_y=0, theta_z=np.pi/4, tx=0, ty=0, tz=0):
  # Define the rotation angles and translations
  # theta_x = np.pi/4  # rotation around x axis
  # theta_y = np.pi/6  # rotation around y axis
  # theta_z = np.pi/3  # rotation around z axis
  # tx = 1.0           # translation in x direction
  # ty = 2.0           # translation in y direction
  # tz = 3.0           # translation in z direction

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



  mask = scipy.ndimage.affine_transform(mask, T, order=2)
  image[:,:,:,0] = scipy.ndimage.affine_transform(image[:,:,:,0], T)
  image[:,:,:,1] = scipy.ndimage.affine_transform(image[:,:,:,1], T)
  image[:,:,:,2] = scipy.ndimage.affine_transform(image[:,:,:,2], T)

  return image, mask



# The functions below are used for stacked dataset (image Size of [number of sample, depth, row, column, channel])

def flipImage(image, mask, rowFlip=1, columnFlip=1, depthFlip=0, p=0.5):
  if(rowFlip):
    for i in range(image.shape[0]):
      if np.random.rand()<p:
        image[i], mask[i] = aug_flip_ud(image[i], mask[i])

  if(columnFlip):
    for i in range(image.shape[0]):
      if np.random.rand()<p:
        image[i], mask[i] = aug_flip_lr(image[i], mask[i])

  if(depthFlip):
    for i in range(image.shape[0]):
      if np.random.rand()<p:
        image[i], mask[i] = aug_flip_bu(image[i], mask[i])

  return image, mask_img_ext


def cropImage(image, mask, remainSize, p=0.5):
  for i in range(image.shape[0]):
    if np.random.rand()<p:
      image[i], mask[i] = aug_crop(image[i], mask[i], remainSize)

  return image, mask

def affineTrans(image, mask, theta_x=0, theta_y=0, theta_z=np.pi/4, tx=0, ty=0, tz=0, p=0.5):
  image = np.transpose(image, [0,3,2,1,4])
  mask = np.transpose(mask, [0,3,2,1])
  for i in range(image.shape[0]):
    if np.random.rand()<p:
      image[i], mask[i] = aug_affine(image[i], mask[i], theta_x, theta_y, theta_z, tx, ty, tz)
      mask[i] = (mask[i]>0.5)


  image = np.transpose(image, [0,3,2,1,4])
  mask = np.transpose(mask, [0,3,2,1])
  return image, mask