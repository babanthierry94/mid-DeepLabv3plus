import tensorflow as tf
# https://towardsdatascience.com/creating-deeper-bottleneck-resnet-from-scratch-using-tensorflow-93e11ff7eb02
# https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/resnet.py#L75

class SimAM(tf.keras.layers.Layer):
    def __init__(self, e_lambda=1e-7, trainable=True, **kwargs):
        super(SimAM, self).__init__(trainable=trainable, **kwargs)
        self.e_lambda = e_lambda
        self.data_format = tf.keras.backend.image_data_format()

    def call(self, inputs, **kwargs):
        input_shape = inputs.shape
        if self.data_format == "channels_first":
            self.height = input_shape[2]
            self.width = input_shape[3]
            self.channels = input_shape[1]
        else:
            self.height = input_shape[1]
            self.width = input_shape[2]
            self.channels = input_shape[3]

        # spatial size
        n = self.width * self.height - 1
        # square of (t - u)
        d = tf.math.square(inputs - tf.math.reduce_mean(inputs, axis=(1, 2), keepdims=True))
        # d.sum() / n is channel variance
        v = tf.math.reduce_sum(d, axis=(1, 2), keepdims=True) / n
        # E_inv groups all importance of X
        E_inv = d / (4 *  tf.maximum(v, self.e_lambda) + 0.5)
        return inputs * tf.keras.activations.sigmoid(E_inv)

    def get_config(self):
        config = {"e_lambda": self.e_lambda}
        base_config = super().get_config()
        return {**base_config, **config}


class DeepLabv3p_mid(object):
    """
    ResNet-101
    ResNet-50
    output_stride fixed 16
    """
    def __init__(self, num_classes=21, encoder_name="res101", input_shape=(512,512,3), finetune=False, output_stride=16):
        if encoder_name not in ['res101', 'res50']:
          print('encoder_name ERROR!')
          print("Please input: res101, res50")
          raise NotImplementedError

        self.encoder_name = encoder_name
        self.inputs = tf.keras.layers.Input(shape=input_shape)
        self.num_classes = num_classes
        self.channel_axis = 3
        if finetune :
            self.pretrained = "imagenet"
        else :
            self.pretrained = None
        self.output_stride = output_stride
        if self.output_stride not in [16]:
            raise NotImplementedError

        self.build_network()

    def __call__(self):
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name='Deeplabv3p_mid')
        return model

    def build_network(self):
        low_level_feat, middle_level_feat, high_level_feat = self.build_encoder()
        self.outputs = self.build_decoder(low_level_feat, middle_level_feat, high_level_feat)

    def build_encoder(self):
        print("-----------build encoder: %s-----------" % self.encoder_name)
        if self.encoder_name == 'res50':
            backbone_model = tf.keras.applications.ResNet50(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            first_layer_name = "conv2_block3_out" #After block 1
            middle_layer_name = "conv3_block4_out" #After block 2
            last_layer_name = "conv4_block6_out" #After block 4
        elif self.encoder_name == 'res101':
            backbone_model = tf.keras.applications.ResNet101(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
            first_layer_name = "conv2_block3_out" #After block 1
            middle_layer_name = "conv3_block4_out" #After block 2
            last_layer_name = "conv4_block23_out" #After block 4
            
        low_level_feat = backbone_model.get_layer(first_layer_name).output
        middle_level_feat = backbone_model.get_layer(middle_layer_name).output
        layer_block3 = backbone_model.get_layer(last_layer_name).output

        #block4
        outputs = self._bottleneck_resblock(layer_block3, 2048, 1, 2, 'block4_unit_1', identity_connection=False)
        for i in range(2, 4):
            outputs = self._bottleneck_resblock(outputs, 2048, 1, 1, 'block4_unit_%d'%i)
        high_level_feat = outputs

        print("low_level_feat:", low_level_feat.shape)
        print("middle_level_feat:", middle_level_feat.shape)
        print("high_level_feat:", high_level_feat.shape)

        return low_level_feat, middle_level_feat, high_level_feat

    def build_decoder(self, low_level_feat, middle_level_feat, high_level_feat):
        print("-----------build decoder-----------")
        dilations = [1, 6, 12, 18]

        low_level_feat = self._SimAM(low_level_feat)
        high_level_feat = self._SimAM(high_level_feat) 

        middle_features = self._SimAM(middle_level_feat)
        middle_features = tf.keras.layers.Conv2D(48, (1, 1), padding='same', kernel_initializer='he_normal')(middle_features)
        middle_features = tf.keras.layers.BatchNormalization()(middle_features)
        middle_features = tf.keras.layers.ReLU()(middle_features)
        middle_features =  tf.keras.layers.UpSampling2D(name="Decoder_Upsampling1b", size=(2,2), interpolation="bilinear")(middle_features) #Upsampling x2

        x1 = self._ASPPv2(high_level_feat, 256, dilations)
        print("after asppv2 block:", x1.shape)

        x1 = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', kernel_initializer='he_normal')(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Activation('relu')(x1)
        print("Before the first UpSampling:", x1.shape)
        x1 = tf.keras.layers.UpSampling2D(size=(self.inputs.shape[1]//4//x1.shape[1], self.inputs.shape[2]//4//x1.shape[2]), interpolation="bilinear")(x1)

        low_level = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', kernel_initializer='he_normal')(low_level_feat)
        low_level = tf.keras.layers.BatchNormalization()(low_level)
        low_level = tf.keras.layers.Activation('relu')(low_level)
        x = tf.keras.layers.Concatenate()([x1, middle_features, low_level])
        #-------------------------------------------------------------------------------#
        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.UpSampling2D(size=self.inputs.shape[1]//x.shape[1], interpolation="bilinear")(x)
        print("Before the last UpSampling:", x.shape)
        outputs = tf.keras.layers.Conv2D(self.num_classes, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

        return outputs

    def _bottleneck_resblock(self, x, filters, stride, dilation_factor, name, identity_connection=True):
          assert filters % 4 == 0, 'Bottleneck number of output ERROR!'
          # branch1
          if not identity_connection:
              o_b1 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', name='%s/shortcut'%name)(x)
              o_b1 = tf.keras.layers.BatchNormalization(name='%s/shortcut_bn'%name)(o_b1)
          else:
              o_b1 = x
          # branch2
          o_b2a = tf.keras.layers.Conv2D(filters / 4, kernel_size=1, strides=1, padding='same', name='%s/conv1'%name)(x)
          o_b2a = tf.keras.layers.BatchNormalization(name='%s/conv1_bn'%name)(o_b2a)
          o_b2a = tf.keras.layers.Activation("relu", name='%s/conv1_relu'%name)(o_b2a)

          o_b2b = tf.keras.layers.Conv2D(filters / 4, kernel_size=3, strides=stride, dilation_rate=dilation_factor, padding='same', name='%s/conv2'%name)(o_b2a)
          o_b2b = tf.keras.layers.BatchNormalization(name='%s/conv2_bn'%name)(o_b2b)
          o_b2b = tf.keras.layers.Activation("relu", name='%s/conv2_relu'%name)(o_b2b)

          o_b2c = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', name='%s/conv3'%name)(o_b2b)
          o_b2c = tf.keras.layers.BatchNormalization(name='%s/conv3_bn'%name)(o_b2c)

          # add
          outputs = tf.keras.layers.Add(name='%s/add'%name)([o_b1, o_b2c])
          # relu
          outputs = tf.keras.layers.Activation("relu", name='%s/relu'%name)(outputs)
          return outputs

    def _Atrous_SepConv(self, x, conv_type="sepconv2d", prefix="None", filters=256, kernel_size=3,  stride=1, dilation_rate=1, use_bias=False):
        conv_dict = {
            'conv2d': tf.keras.layers.Conv2D,
            'sepconv2d': tf.keras.layers.SeparableConv2D
        }
        conv = conv_dict[conv_type]
        x = conv(filters, kernel_size, name=prefix, strides=stride, dilation_rate=dilation_rate,
                                padding="same", kernel_initializer='he_normal', use_bias=use_bias)(x)
        x = tf.keras.layers.BatchNormalization(name='%s_bn'%prefix)(x)
        x = tf.keras.layers.Activation('relu', name='%s_relu'%prefix)(x)
        return x

    def _ASPPv2(self, x, nb_filters, d):
        x1 = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='aspp/sepconv1', filters=nb_filters, kernel_size=1, dilation_rate=d[0], use_bias=True)
        x2 = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='aspp/sepconv2', filters=nb_filters, kernel_size=3, dilation_rate=d[1], use_bias=True)
        x3 = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='aspp/sepconv3', filters=nb_filters, kernel_size=3, dilation_rate=d[2], use_bias=True)
        x4 = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='aspp/sepconv4', filters=nb_filters, kernel_size=3, dilation_rate=d[3], use_bias=True)

        x5 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='aspp/avg')(x)
        x5 = tf.keras.layers.Conv2D(256, kernel_size=1)(x5)
        x5 = tf.keras.layers.UpSampling2D(size=x.shape[1] // x5.shape[1], interpolation="bilinear", name='asspv2/avg_upsambling')(x5)
        out = tf.keras.layers.Concatenate(name='aspp/add')([x1, x2, x3, x4, x5])
        return out

    def _SimAM(self, inputs):
        height = inputs.shape[1]
        width = inputs.shape[2]
        e_lambda=1e-7
        # spatial size
        n = width * height - 1
        # square of (t - u)
        d = tf.math.square(inputs - tf.math.reduce_mean(inputs, axis=(1, 2), keepdims=True))
        # d.sum() / n is channel variance
        v = tf.math.reduce_sum(d, axis=(1, 2), keepdims=True) / n
        # E_inv groups all importance of X
        E_inv = d / (4 *  tf.maximum(v, e_lambda) + 0.5)
        return inputs * tf.keras.activations.sigmoid(E_inv)

