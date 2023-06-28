############################################################################################################################
# This code implement the DeepLabv3+ model with Aligned Xception, ResNet101 and ResNet50 backbone
# We reproduced the architecture published in [1] and [2]
# 
# [1] @inproceedings{chen2018encoder,
#  title={Encoder-decoder with atrous separable convolution for semantic image segmentation},
#  author={Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
#  booktitle={Proceedings of the European conference on computer vision (ECCV)},
#  pages={801--818},
#  year={2018}
# }
# https://arxiv.org/pdf/1802.02611.pdf
#
# [2] @article{chen2017rethinking,
#  title={Rethinking atrous convolution for semantic image segmentation},
#  author={Chen, Liang-Chieh and Papandreou, George and Schroff, Florian and Adam, Hartwig},
#  journal={arXiv preprint arXiv:1706.05587},
#  year={2017}
# }
# https://arxiv.org/pdf/1706.05587.pdf
############################################################################################################################

import tensorflow as tf

class DeepLabv3p(object):
    """
    Arguments :
        num_classes   : number of classes for the model
        backbone_name : the backbone name [res101, res50, xception]
        input_shape   : input image shape (High, Width, Channel)
        output_stride : the ratio between input image size and the size of backbone last output features
                        Only output_stride=8 was implemented
    """

    def __init__(self, num_classes=21, backbone_name="res101", input_shape=(512,512,3), output_stride=16):

        if backbone_name not in ['res101', 'res50', 'xception']:
          print("backbone_name ERROR! Please input: xception, res101, res50")
          raise NotImplementedError
        
        self.inputs = tf.keras.layers.Input(shape=input_shape)
        # Number of blocks for ResNet50 and ResNet101
        self.backbone_name = backbone_name
        if self.backbone_name == 'res101':
            self.blocks = [3, 4, 23, 3]
        elif self.backbone_name == 'res50':
            self.blocks = [3, 4, 6, 3]

        self.num_classes = num_classes
        self.output_stride = output_stride
        if self.output_stride != 16:
            raise NotImplementedError
        
        # Stride and dilation rate for ResNet50 and ResNet101 depending of the outpout_stride
        if self.output_stride == 16:
            self.resnet_strides = [1, 2, 2, 1]
            self.resnet_dilations = [1, 1, 1, 2]
        elif self.output_stride == 8:
            self.resnet_strides = [1, 2, 1, 1]
            self.resnet_dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
        
        # Dilations rates for ASPP module
        self.aspp_dilations = [1, 6, 12, 18]
        self.build_network()

    def __call__(self):
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name='Deeplabv3p')
        return model

    def build_network(self):
        # Build the encoder path and get the high and low features layer
        low_level_feat, refined_high_level_feat = self.build_encoder()
        # Build the decoder path
        self.outputs = self.build_decoder(low_level_feat, refined_high_level_feat)

    """
    This fonction build the encoder path of the model.
    Including backbone features extraction (ResNet101, ResNet50, Xception) and Atrous Spatial Pyramid Pooling (ASPP)
    """
    def build_encoder(self):
        print("-----------build encoder: %s-----------" % self.backbone_name)
        if self.backbone_name == 'res50' or self.backbone_name == 'res101':
            #starter_block
            outputs = self._start_block('conv1')
            print("after start block:", outputs.shape)
            #block1
            outputs = self._bottleneck_resblock(outputs, 256, self.resnet_strides[0], self.resnet_dilations[0], 'conv2_block1', identity_connection=False)
            for i in range(2, self.blocks[0]+1):
                outputs = self._bottleneck_resblock(outputs, 256, 1, 1, 'conv2_block%d'%i)
            print("after block1:", outputs.shape)
            low_level_feat = outputs
            #block2
            outputs = self._bottleneck_resblock(outputs, 512, self.resnet_strides[1], self.resnet_dilations[1], 'conv3_block1', identity_connection=False)
            for i in range(2, self.blocks[1]+1):
                outputs = self._bottleneck_resblock(outputs, 512, 1, 1, 'conv3_block%d'%i)
            print("after block2:", outputs.shape)
            #block3
            outputs = self._bottleneck_resblock(outputs, 1024, self.resnet_strides[2], self.resnet_dilations[2], 'conv4_block1', identity_connection=False)
            for i in range(2, self.blocks[2]+1):
                outputs = self._bottleneck_resblock(outputs, 1024, 1, 1, 'conv4_block%d'%i)
            print("after block3:", outputs.shape)
            #block4
            outputs = self._bottleneck_resblock(outputs, 2048, self.resnet_strides[3], self.resnet_dilations[3], 'conv5_block1', identity_connection=False)
            for i in range(2, 4):
                outputs = self._bottleneck_resblock(outputs, 2048, 1, 1, 'conv5_block%d'%i)
            print("after block4:", outputs.shape)
            high_level_feat = outputs

        elif self.backbone_name == 'xception':
            ## Entry flow
            x = tf.keras.layers.Conv2D(32, kernel_size=3, name='entry_Conv1', strides=2, padding='same', kernel_initializer='he_normal')(self.inputs)
            x = tf.keras.layers.BatchNormalization(name='entry_Conv1_BN')(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(64, kernel_size=3, name='entry_Conv2', strides=1, padding='same')(x)
            x = tf.keras.layers.BatchNormalization(name='entry_Conv2_BN')(x)
            x = tf.keras.layers.ReLU()(x)
            residual = tf.keras.layers.Conv2D(128, name='entry_block2_Conv', kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal')(x)

            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='entry_block2_SepConv1', filters=128, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='entry_block2_SepConv2', filters=128, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='entry_block2_SepConv3', filters=128, kernel_size=3,  stride=2)
            x = tf.keras.layers.Add(name="entry_block2_Add")([residual, x])
            residual = tf.keras.layers.Conv2D(256, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal')(x)

            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='entry_block3_SepConv1', filters=256, kernel_size=3)
            low_level_feat = x

            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='entry_block3_SepConv2', filters=256, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='entry_block3_SepConv3', filters=256, kernel_size=3,  stride=2)
            x = tf.keras.layers.Add(name="entry_block3_Add")([residual, x])
            residual = tf.keras.layers.Conv2D(728, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal')(x)

            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='entry_block4_SepConv1', filters=728, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='entry_block4_SepConv2', filters=728, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='entry_block4_SepConv3', filters=728, kernel_size=3,  stride=2)
            x = tf.keras.layers.Add(name="entry_block4_Add")([residual, x])
            print("After Entry flow block:", x.shape)

            ## Middle flow
            for i in range(16):
                residual = x
                x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='middle_unit_{}_SepConv1'.format(i + 1), filters=728, kernel_size=3)
                x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='middle_unit_{}_SepConv2'.format(i + 1), filters=728, kernel_size=3)
                x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='middle_unit_{}_SepConv3'.format(i + 1), filters=728, kernel_size=3)
                x = tf.keras.layers.Add(name="entry_block_unit_{}_Add".format(i + 1))([residual, x])
            print("After Middle flow block:", x.shape)

            ##Exit flow
            residual = tf.keras.layers.Conv2D(1024, name='exit_block1_Conv', kernel_size=1, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='exit_block1_SepConv1', filters=728, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='exit_block1_SepConv2', filters=1024, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='exit_block1_SepConv3', filters=1024, kernel_size=3, dilation_rate=2)
            x = tf.keras.layers.Add(name='exit_block1_Add')([residual, x])
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='exit_block2_SepConv1', filters=1536, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='exit_block2_SepConv2', filters=1536, kernel_size=3)
            x = self._Atrous_SepConv(x, conv_type="sepconv2d", name='exit_block2_SepConv3', filters=2048, kernel_size=3)
            high_level_feat = x
            print("After Exit flow block:", x.shape)

        # Build ASPP module
        x1 = self._ASPPv2(high_level_feat, 256, self.aspp_dilations)
        x1 = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', kernel_initializer='he_normal')(x1)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Activation('relu')(x1)
        refined_high_level_feat = tf.keras.layers.UpSampling2D(size=(self.inputs.shape[1]//4//x1.shape[1], self.inputs.shape[2]//4//x1.shape[2]), interpolation="bilinear")(x1)

        return low_level_feat, refined_high_level_feat
    
    """ 
    Decoder path
    """
    def build_decoder(self, low_level_feat, refined_high_level_feat):
        print("-----------build decoder-----------")
        low_level = tf.keras.layers.Conv2D(48, kernel_size=1, padding='same', kernel_initializer='he_normal')(low_level_feat)
        low_level = tf.keras.layers.BatchNormalization()(low_level)
        low_level = tf.keras.layers.Activation('relu')(low_level)
        x = tf.keras.layers.Concatenate()([refined_high_level_feat, low_level])

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.UpSampling2D(size=self.inputs.shape[1]//x.shape[1], interpolation="bilinear")(x)
        outputs = tf.keras.layers.Conv2D(self.num_classes, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

        return outputs
    
    # ResNet Starter Block
    def _start_block(self, name):
        outputs = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', name='%s_conv'%name)(self.inputs)
        outputs = tf.keras.layers.BatchNormalization(name='%s_bn'%name)(outputs)
        outputs = tf.keras.layers.Activation("relu", name='%s_relu'%name)(outputs)
        outputs = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name="pool1_pool")(outputs)
        return outputs
    
    # ResNet Bottleneck Block
    def _bottleneck_resblock(self, x, filters, stride, dilation_factor, name, identity_connection=True):
        assert filters % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', name='%s_1_conv'%name)(x)
            o_b1 = tf.keras.layers.BatchNormalization(name='%s_bn'%name)(o_b1)
        else:
            o_b1 = x
        # branch2
        o_b2a = tf.keras.layers.Conv2D(filters//4, kernel_size=1, strides=1, padding='same', name='%s_2_conv'%name)(x)
        o_b2a = tf.keras.layers.BatchNormalization(name='%s_2_bn'%name)(o_b2a)
        o_b2a = tf.keras.layers.Activation("relu", name='%s_2_relu'%name)(o_b2a)

        o_b2b = tf.keras.layers.Conv2D(filters//4, kernel_size=3, strides=stride, dilation_rate=dilation_factor, padding='same', name='%s/conv2'%name)(o_b2a)
        o_b2b = tf.keras.layers.BatchNormalization(name='%s_3_bn'%name)(o_b2b)
        o_b2b = tf.keras.layers.Activation("relu", name='%s_3_relu'%name)(o_b2b)

        o_b2c = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', name='%s_0_conv'%name)(o_b2b)
        o_b2c = tf.keras.layers.BatchNormalization(name='%s_0_bn'%name)(o_b2c)

        # add
        outputs = tf.keras.layers.Add(name='%s_add'%name)([o_b1, o_b2c])
        # relu
        outputs = tf.keras.layers.Activation("relu", name='%s_out'%name)(outputs)
        return outputs

    def _Atrous_SepConv(self, x, conv_type="sepconv2d", name="None", filters=256, kernel_size=3,  stride=1, dilation_rate=1, use_bias=False):
        conv_dict = {
            'conv2d': tf.keras.layers.Conv2D,
            'sepconv2d': tf.keras.layers.SeparableConv2D
        }
        conv = conv_dict[conv_type]
        x = conv(filters, kernel_size, name=name, strides=stride, dilation_rate=dilation_rate,
                                padding="same", kernel_initializer='he_normal', use_bias=use_bias)(x)
        x = tf.keras.layers.BatchNormalization(name='%s_bn'%name)(x)
        x = tf.keras.layers.Activation('relu', name='%s_relu'%name)(x)
        return x


    def _ASPPv2(self, x, nb_filters, d):
        x1 = self._Atrous_SepConv(x, conv_type="sepconv2d", name='aspp_conv_r%d'%d[0], filters=nb_filters, kernel_size=1, dilation_rate=d[0], use_bias=True)
        x2 = self._Atrous_SepConv(x, conv_type="sepconv2d", name='aspp_conv_r%d'%d[1], filters=nb_filters, kernel_size=3, dilation_rate=d[1], use_bias=True)
        x3 = self._Atrous_SepConv(x, conv_type="sepconv2d", name='aspp_conv_r%d'%d[2], filters=nb_filters, kernel_size=3, dilation_rate=d[2], use_bias=True)
        x4 = self._Atrous_SepConv(x, conv_type="sepconv2d", name='aspp_conv_r%d'%d[3], filters=nb_filters, kernel_size=3, dilation_rate=d[3], use_bias=True)

        x5 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='aspp_avg_pool')(x)
        x5 = tf.keras.layers.Conv2D(256, kernel_size=1, name='aspp_avg_conv')(x5)
        x5 = tf.keras.layers.UpSampling2D(size=x.shape[1] // x5.shape[1], interpolation="bilinear", name='aspp_avg_up2D')(x5)
        out = tf.keras.layers.Concatenate(name='aspp_add')([x1, x2, x3, x4, x5])
        return out


model = DeepLabv3p(num_classes=10, backbone_name="res50", input_shape=(512, 512, 3))()
model.summary()

# https://keras.io/examples/vision/deeplabv3_plus/
# https://github.com/tonandr/deeplabv3plus_keras/blob/master/bodhi/deeplabv3plus_keras/semantic_segmentation.py
# https://github.com/tensorflow/models/blob/master/research/deeplab/core/xception.py
# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py
# https://github.com/lattice-ai/DeepLabV3-Plus/blob/master/deeplabv3plus/model/deeplabv3_plus.py
