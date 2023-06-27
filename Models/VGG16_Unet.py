#VGG16 Unet
import tensorflow as tf 

def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = tf.keras.layers.Conv2D(n_filters, kernel_size = (3,3), padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # Conv2D then ReLU activation
    x = tf.keras.layers.Conv2D(n_filters, kernel_size = (3,3), padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x

def downsample_block(x, n_filters, dropout_prob):
    f = double_conv_block(x, n_filters)
    p = tf.keras.layers.MaxPool2D(strides=(2,2))(f)
    # p = layers.Dropout(dropout_prob)(p)

    return f, p

def upsample_block(x, conv_features, n_filters, dropout_prob):
    # upsample
    x = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size = (2,2), strides=(2,2), padding="same")(x)
    # concatenate 
    x = tf.keras.layers.concatenate([x, conv_features])
    # dropout
    # x = layers.Dropout(dropout_prob)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x

def Unet(input_shape=(512, 512, 3), classes=21):
    # inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64, 0.1)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128, 0.1)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256, 0.2)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512, 0.2)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512, 0.2)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256, 0.2)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128, 0.1)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64, 0.1)

    # outputs
    outputs = tf.keras.layers.Conv2D(classes, 1, padding="same", activation = "softmax")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model