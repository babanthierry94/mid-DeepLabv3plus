# for data load
import os
import json
from glob import glob

# import library
from PIL import Image
import numpy as np
import tensorflow as tf

# https://mm.cs.uec.ac.jp/uecfoodpix/#downloads
class CreateDataset() :
    
    def __init__(self, dataset_name, dataset_path, img_size=512, batch_size=8, num_classes=11, class_names=None):
        if dataset_name not in ['camerfood10', 'brazillian', 'uecfoodpix']:
          print('encoder_name ERROR!')
          print("Please input: 'camerfood10', 'brazillian', 'uecfoodpix'")
          raise NotImplementedError
        

       
        # Dictionary {name_of_class: class_id} remember background has id 0
        self.class_names = class_names

        self.dataset_name = dataset_name

        self.DATASET_PATH = dataset_path
        self.images_path = os.path.join(self.DATASET_PATH, "images")
        self.masks_path = os.path.join(self.DATASET_PATH, "masks")
        self.annotation_path = None
        if self.dataset_name == "camerfood10":
            self.annotation_path = os.path.join(self.DATASET_PATH, "via_annotations.json")

        # Dataset parameters
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = img_size
        self.NB_CLASS = num_classes
        
        # Make a list for images and masks filenames
        # self.images_list = []
        # self.masks_list = []
        
    def load_data(self):
        """
        Returns 2 lists for original and masked files respectively
        """
        if self.dataset_name == "camefood10":
            # load annotations using json.load()
            annotations = json.load(open(self.annotation_path))
            # convert annotations into a list
            annotations = list(annotations.values())
            # we only require the regions in the annotations
            annotations = [a for a in annotations if a['regions']]

            # Add images
            for a in annotations:
                # extracting shape attributes and region attributes
                polygons = [r['shape_attributes'] for r in a['regions']] 
                objects = [s['region_attributes']['name'] for s in a['regions']]
                # all the ids/classes in a image
                num_ids = [self.class_names[a] for a in objects]
                # read image and get height and width
                filename = a['filename']
                image_name = os.path.splitext(filename)[0]
                if os.path.exists(os.path.join(self.masks_path, image_name+".png")) and os.path.exists(os.path.join(self.images_path, image_name+".jpg")):
                    self.images_list.append(os.path.join(self.images_path, image_name+".jpg"))
                    self.masks_list.append(os.path.join(self.masks_path, image_name+".png"))

            images_list = sorted(self.images_list)
            masks_list = sorted(self.masks_list)
        
        elif self.dataset_name == "brazillian":
            images_list = sorted(glob(os.path.join(self.images_path, "*.png")))
            masks_list = sorted(glob(os.path.join(self.masks_path, "*.png")))

        elif self.dataset_name == "uecfoodpix":
            images_list = sorted(glob(os.path.join(self.images_path, "*.jpg")))
            masks_list = sorted(glob(os.path.join(self.masks_path, "*.png")))

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
        # print("Dataset number of images  : ", len(images))
        
        return dataset

    