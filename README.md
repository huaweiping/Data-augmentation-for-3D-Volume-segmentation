# Data-augmentation-for-Volume-segmentation
This is a data augmentation tool for the 3D volume segmentation including Image and mask

Thank you so much, ChatGPT!



# Current Version provides:


## For single image-mask pair
<!-- upside-down/left-right/bottom-up flip-->
aug_flip(image, mask) 

<!-- add value to the image.
 default channel is 0
 default add value is 0.1 (please set it between 0 and 1)-->
aug_add(image, mask, channel=0, addValue=0.1)

<!-- shuffle the channel of the image-->
aug_shuffleChannel(image, mask)

<!-- randomly crop the original image and mask to the Size [depth, row, column]
 the image and the mask will be rescaled to the original size-->
aug_crop(image, mask, finalSize)

<!-- Pad value to the image. padWidth = (depth, row, column)-->
aug_pad(image, mask, padWidth=((0, 1), (0, 1), (0, 1)), padValue=0)

<!-- affine transfomation to the image.-->
aug_affine(image, mask, theta_x=0, theta_y=0, theta_z=np.pi/4, tx=0, ty=0, tz=0)

## The functions below are used for stacked dataset (Size of [number of sample, depth, row, column, channel])

flipImage(image, mask, rowFlip=1, columnFlip=1, depthFlip=0, p=0.5):

cropImage(image, mask, remainSize, p=0.5):

affineTrans(image, mask, theta_x=0, theta_y=0, theta_z=np.pi/4, tx=0, ty=0, tz=0, p=0.5):
