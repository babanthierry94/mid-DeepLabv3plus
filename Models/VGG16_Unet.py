#########################################################################################################
# This code implement the Unet model with VGG16 encoder
# We reproduced the architecture published in [1].
#
# [1] @inproceedings{ronneberger2015u,
#  title={U-net: Convolutional networks for biomedical image segmentation},
#  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
#  booktitle={Medical Image Computing and Computer-Assisted Intervention--MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18},
#  pages={234--241},
#  year={2015},
#  organization={Springer}
# }
# https://idiotdeveloper.com/vgg16-unet-implementation-in-tensorflow/
# https://www.kaggle.com/code/aithammadiabdellatif/vgg16-u-net
# https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/vgg16_unet.py
# https://github.com/zhoudaxia233/PyTorch-Unet/blob/master/README.md
# https://arxiv.org/pdf/1505.04597v1.pdf
#########################################################################################################

# VGG16 Unet
import tensorflow as tf 

def double_conv_block(x, n_filters):
    # Conv2D, BatchNormalization and ReLU activation
    x = tf.keras.layers.Conv2D(n_filters, kernel_size = (3,3), padding = "same", kernel_initializer = "he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    # Conv2D, BatchNormalization and ReLU activation
    x = tf.keras.layers.Conv2D(n_filters, kernel_size = (3,3), padding = "same", kernel_initializer = "he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size = (2,2), strides=(2,2), padding="same")(x)
    # concatenate 
    x = tf.keras.layers.concatenate([x, conv_features])
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x

def VGG16_Unet(input_shape=(512, 512, 3), num_classes=21):
    # inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor = inputs)
    # vgg16.summary()

    # encoder: contracting path - downsample
    # 1 - downsample 64
    f1 = vgg16.get_layer("block1_conv2").output
    print(f1.shape)
    # 2 - downsample 128
    f2 = vgg16.get_layer("block2_conv2").output
    # 3 - downsample 256
    f3 = vgg16.get_layer("block3_conv3").output
    # 4 - downsample 512
    f4 = vgg16.get_layer("block4_conv3").output

    # 5 - the center 1024
    center = vgg16.get_layer("block5_conv3").output

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(center, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    # outputs
    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding="same", activation = "softmax")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="VGG16_UNet")

    return unet_model

# model = VGG16_Unet(num_classes=10, input_shape=(512, 512, 3))
# model.summary()