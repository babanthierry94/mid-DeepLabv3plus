# https://keras.io/examples/vision/deeplabv3_plus/
# https://github.com/tonandr/deeplabv3plus_keras/blob/master/bodhi/deeplabv3plus_keras/semantic_segmentation.py
# https://github.com/tensorflow/models/blob/master/research/deeplab/core/xception.py
# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py
# https://github.com/lattice-ai/DeepLabV3-Plus/blob/master/deeplabv3plus/model/deeplabv3_plus.py
import tensorflow as tf

class DeepLabv3p(object):
    """
    ResNet-101
    ResNet-50
    Xception
    output_stride fixed 16
    """
    def __init__(self, num_classes=21, encoder_name="res101", input_shape=(512,512,3), finetune=False, output_stride=16):
        if encoder_name not in ['res101', 'res50', 'xception']:
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
        if self.output_stride != 16:
            raise NotImplementedError

        self.build_network()

    def __call__(self):
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name='Deeplabv3p')
        return model

    def build_network(self):
        low_level_feat, high_level_feat = self.build_encoder()
        self.outputs = self.build_decoder(low_level_feat, high_level_feat)

    def build_encoder(self):
        print("-----------build encoder: %s-----------" % self.encoder_name)

        if self.encoder_name == 'res50' or self.encoder_name == 'res101':
            #starter_block
            outputs = self._start_block('conv1')
            print("after start block:", outputs.shape)

            #block1
            outputs = self._bottleneck_resblock(outputs, 256, self.strides[0], self.dilations[0], 'block1_unit_1', identity_connection=False)
            for i in range(2, self.blocks[0]+1):
                outputs = self._bottleneck_resblock(outputs, 256, 1, 1, 'block1_unit_%d'%i)
            print("after block1:", outputs.shape)
            low_level_feat = outputs

            #block2
            outputs = self._bottleneck_resblock(outputs, 512, self.strides[1], self.dilations[1], 'block2_unit_1', identity_connection=False)
            for i in range(2, self.blocks[1]+1):
                outputs = self._bottleneck_resblock(outputs, 512, 1, 1, 'block2_unit_%d'%i)
            print("after block2:", outputs.shape)

            #block3
            outputs = self._bottleneck_resblock(outputs, 1024, self.strides[2], self.dilations[2], 'block3_unit_1', identity_connection=False)
            for i in range(2, self.blocks[2]+1):
                outputs = self._bottleneck_resblock(outputs, 1024, 1, 1, 'block3_unit_%d'%i)
            print("after block3:", outputs.shape)
            #block4
            outputs = self._bottleneck_resblock(outputs, 2048, 1, 2, 'block4_unit_1', identity_connection=False)
            for i in range(2, 4):
                outputs = self._bottleneck_resblock(outputs, 2048, 1, 1, 'block4_unit_%d'%i)
            print("after block4:", outputs.shape)
            high_level_feat = outputs

        elif self.encoder_name == 'xception':
            ## Entry flow
            x = tf.keras.layers.Conv2D(32, kernel_size=3, name='entry_flow_Conv1', strides=2, padding='same', kernel_initializer='he_normal')(self.inputs)
            x = tf.keras.layers.BatchNormalization(name='entry_flow_Conv1_BN')(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(64, kernel_size=3, name='entry_flow_Conv2', strides=1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization(name='entry_flow_Conv2_BN')(x)
            x = tf.keras.layers.ReLU()(x)
            residual = tf.keras.layers.Conv2D(128, name='entry_flow_block2_Conv', kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal')(x)

            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='entry_flow_block2_SepConv1', filters=128, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='entry_flow_block2_SepConv2', filters=128, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='entry_flow_block2_SepConv3', filters=128, kernel_size=3,  stride=2)
            x = tf.keras.layers.Add(name="entry_flow_block2_Add")([residual, x])
            residual = tf.keras.layers.Conv2D(256, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal')(x)

            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='entry_flow_block3_SepConv1', filters=256, kernel_size=3)
            low_level_feat = x

            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='entry_flow_block3_SepConv2', filters=256, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='entry_flow_block3_SepConv3', filters=256, kernel_size=3,  stride=2)
            x = tf.keras.layers.Add(name="entry_flow_block3_Add")([residual, x])
            residual = tf.keras.layers.Conv2D(728, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal')(x)

            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='entry_flow_block4_SepConv1', filters=728, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='entry_flow_block4_SepConv2', filters=728, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='entry_flow_block4_SepConv3', filters=728, kernel_size=3,  stride=2)
            x = tf.keras.layers.Add(name="entry_flow_block4_Add")([residual, x])

            ## Middle flow
            for i in range(16):
                residual = x
                x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='middle_flow_unit_{}_SepConv1'.format(i + 1), filters=728, kernel_size=3)
                x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='middle_flow_unit_{}_SepConv2'.format(i + 1), filters=728, kernel_size=3)
                x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='middle_flow_unit_{}_SepConv3'.format(i + 1), filters=728, kernel_size=3)
                x = tf.keras.layers.Add(name="entry_flow_block_unit_{}_Add".format(i + 1))([residual, x])

            ##Exit flow
            residual = tf.keras.layers.Conv2D(1024, name='exit_flow_block1_Conv', kernel_size=1, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='exit_flow_block1_SepConv1', filters=728, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='exit_flow_block1_SepConv2', filters=1024, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='exit_flow_block1_SepConv3', filters=1024, kernel_size=3, dilation_rate=2)
            x = tf.keras.layers.Add(name='exit_flow_block1_Add')([residual, x])
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='exit_flow_block2_SepConv1', filters=1536, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='exit_flow_block2_SepConv2', filters=1536, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='exit_flow_block2_SepConv3', filters=2048, kernel_size=3)
            high_level_feat = x

        print("low_level_feat:", low_level_feat.shape)
        print("high_level_feat:", high_level_feat.shape)

        return low_level_feat, high_level_feat

    def build_decoder(self, low_level_feat, high_level_feat):
        print("-----------build decoder-----------")
        dilations = [1, 6, 12, 18]
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
        x = tf.keras.layers.Concatenate()([x1, low_level])
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
    
    # blocks
    def _start_block(self, name):
        outputs = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', name=name)(self.inputs)
        outputs = tf.keras.layers.BatchNormalization(name='%s_bn'%name)(outputs)
        outputs = tf.keras.layers.Activation("relu", name='%s_relu'%name)(outputs)
        outputs = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name="pool1")(outputs)
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


# model = DeepLabv3p(num_classes=10, encoder_name="xception", input_shape=(512, 512, 3))()
# model.summary()