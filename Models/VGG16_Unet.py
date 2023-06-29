#########################################################################################################
# This code implement the GourmetNet model with ResNet101 and ResNet50 backbone
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

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = tf.keras.layers.MaxPool2D(strides=(2,2))(f)
    return f, p

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size = (2,2), strides=(2,2), padding="same")(x)
    # concatenate 
    x = tf.keras.layers.concatenate([x, conv_features])
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x

def Unet(input_shape=(512, 512, 3), classes=21):
    # inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    # outputs
    outputs = tf.keras.layers.Conv2D(classes, 1, padding="same", activation = "softmax")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="VGG16_U_Net")

    return unet_model