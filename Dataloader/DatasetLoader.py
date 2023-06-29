# import library
import os
import json
from glob import glob
from PIL import Image
import numpy as np
import tensorflow as tf

#########################################################################################################
# Datasets link
# UECFOODPIXCOMPLETE
# https://mm.cs.uec.ac.jp/uecfoodpix/#downloads
#
# MyFood
# https://zenodo.org/record/4041488
# https://arxiv.org/pdf/2012.03087.pdf
#
# CamerFood10
# https://drive.google.com/drive/u/1/folders/1MugfmVehtIjjyqtphs-4u0GksuHy3Vjz
#########################################################################################################

class CreateDataset() :
    
    def __init__(self, dataset_name, dataset_path, img_size=512, batch_size=2, num_classes=11):
       
        if dataset_name not in ['camerfood10', 'brazillian', 'uecfoodpix']:
            print("ERROR! Dataset name should be : 'camerfood10', 'brazillian', 'uecfoodpix'")
            raise NotImplementedError
        
        # The name of dataset in ['camerfood10', 'brazillian', 'uecfoodpix']
        self.dataset_name = dataset_name
        # The dataset absolute path
        self.DATASET_PATH = dataset_path
        self.images_path = os.path.join(self.DATASET_PATH, "images")
        self.masks_path = os.path.join(self.DATASET_PATH, "masks")
        # Only Camerfood10 have and annotation file
        self.annotation_path = None
        if self.dataset_name == "camerfood10":
            self.annotation_path = os.path.join(self.DATASET_PATH, "via_annotations.json")

        # Dataset parameters
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = img_size
        self.NB_CLASS = num_classes
        
    def load_data(self):
        """
        Returns 2 lists for original and masked files respectively
        """
        # Make a list for images and masks absolute path
        images_list = []
        masks_list = []

        if self.dataset_name == "camefood10":
            # load annotations using json.load()
            annotations = json.load(open(self.annotation_path))
            # convert annotations into a list
            annotations = list(annotations.values())
            # we only require the regions in the annotations
            annotations = [a for a in annotations if a['regions']]
            # Add images
            for a in annotations:
                # Get image name
                filename = a['filename']
                image_name = os.path.splitext(filename)[0]
                if os.path.exists(os.path.join(self.masks_path, image_name+".png")) and os.path.exists(os.path.join(self.images_path, image_name+".jpg")):
                    images_list.append(os.path.join(self.images_path, image_name+".jpg"))
                    masks_list.append(os.path.join(self.masks_path, image_name+".png"))

        elif self.dataset_name == "brazillian":
            images_list = glob(os.path.join(self.images_path, "*.png"))
            masks_list = glob(os.path.join(self.masks_path, "*.png"))

        elif self.dataset_name == "uecfoodpix":
            images_list = glob(os.path.join(self.images_path, "*.jpg"))
            masks_list = glob(os.path.join(self.masks_path, "*.png"))
        
        images_list = sorted(images_list)
        masks_list = sorted(masks_list)
        print("Number of images:", len(images_list))
        print("Number of masks:", len(masks_list))
        return images_list, masks_list

    # initialization data
    def get(self):

        def read_image(path):
            # Load image
            image = Image.open(path).convert('RGB')
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), resample=Image.BILINEAR)
            image = np.asarray(image)
            image = image[..., :3]
            image = image / 255.0
            image = image.astype(np.float32)
            return image

        def read_mask(path):

            if self.dataset_name == "camefood10":
                # Load mask
                y = Image.open(path).convert('L')
                # NEAREST to avoid the problem with pixel value changing in mask
                y = y.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), resample=Image.NEAREST)
                y = np.asarray(y)
                n = 255 // (self.NB_CLASS-1)
                y = y / n
                # print(np.unique(y))
                y = np.expand_dims(y, axis=-1)
                y = y.astype(np.uint8)
                
            elif self.dataset_name == "brazillian":
                # Load mask
                y = Image.open(path).convert('L')
                # NEAREST to avoid the problem with pixel value changing in mask
                y = y.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), resample=Image.NEAREST)
                y = np.asarray(y)
                # y = y / 255.0
                # print(np.unique(y))
                y = np.expand_dims(y, axis=-1)
                y = y.astype(np.uint8)

            elif self.dataset_name == "uecfoodpix":
                # Load mask
                y = Image.open(path).convert('RGB')
                y = y.getchannel('R')  #Retrieve only the Red Channel
                # NEAREST to avoid the problem with pixel value changing in mask
                y = y.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), resample=Image.NEAREST)
                y = np.asarray(y)
                # y = y / 255.0
                # print(np.unique(y))
                y = np.expand_dims(y, axis=-1)
                y = y.astype(np.uint8)

            return y

        def preprocess(x, y):
            def f(x,y):
                x = x.decode()
                y = y.decode()
                x = read_image(x)
                y = read_mask(y)
                return x, y

            images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
            images.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
            masks.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 1])
            return images, masks
        
        images, masks = self.load_data()
       
        dataset = tf.data.Dataset.from_tensor_slices((images,masks))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(preprocess)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(2)        
        return dataset

    