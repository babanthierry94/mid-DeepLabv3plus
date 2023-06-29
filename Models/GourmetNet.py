############################################################################################################################
# This code implement the GourmetNet model with ResNet101 and ResNet50 backbone
# We reproduced the architecture published in [1].
# 
# [1] @article{sharma2021gourmetnet,
#  title={Gourmetnet: Food segmentation using multi-scale waterfall features with spatial and channel attention},
#  author={Sharma, Udit and Artacho, Bruno and Savakis, Andreas},
#  journal={Sensors},
#  volume={21},
#  number={22},
#  pages={7504},
#  year={2021},
#  publisher={Multidisciplinary Digital Publishing Institute}
# }
# https://www.mdpi.com/1424-8220/21/22/7504
#
# [2] @inproceedings{he2016deep,
#  title={Deep residual learning for image recognition},
#  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
#  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
#  pages={770--778},
#  year={2016}
# }
# https://arxiv.org/pdf/1512.03385.pdf
############################################################################################################################

import tensorflow as tf

class GourmetNet(object):

    def __init__(self, num_classes=21, backbone_name="res101", finetune=False, input_shape=(512,512,3), output_stride=16):
        if backbone_name not in ['res101', 'res50']:
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
        self.waspp_dilations = [1, 6, 12, 18]

        self.build_network()

    def __call__(self):
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name='GourmetNet')
        return model

    def build_network(self):
        low_level_feat, high_level_feat = self.build_encoder()
        refined_low_level_feat, refined_high_level_feat = self._DualAttention(high_level_feat, low_level_feat)
        self.outputs = self.build_decoder(refined_low_level_feat, refined_high_level_feat)

    def build_encoder(self):
        print("-----------build encoder: %s-----------" % self.backbone_name)
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
        
        print("low_level_feat:", low_level_feat.shape)
        print("high_level_feat:", high_level_feat.shape)

        return low_level_feat, high_level_feat

    def build_decoder(self, refined_low_level, refined_high_level):
        print("-----------build decoder-----------")
        reduction = 256 // 8
        x = self._WASPv2(refined_high_level, 256, self.waspp_dilations)
        print("after waspv2 block:", x.shape)

        x = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.UpSampling2D(size=refined_low_level.shape[1]//x.shape[1], interpolation="bilinear")(x)

        refined_low_level = tf.keras.layers.Conv2D(reduction, kernel_size=1, padding='same', kernel_initializer='he_normal')(refined_low_level)
        refined_low_level = tf.keras.layers.BatchNormalization()(refined_low_level)
        refined_low_level = tf.keras.layers.Activation('relu')(refined_low_level)
        x = tf.keras.layers.Concatenate()([x, refined_low_level])

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

        o_b2b = tf.keras.layers.Conv2D(filters//4, kernel_size=3, strides=stride, dilation_rate=dilation_factor, padding='same', name='%s_3_conv'%name)(o_b2a)
        o_b2b = tf.keras.layers.BatchNormalization(name='%s_3_bn'%name)(o_b2b)
        o_b2b = tf.keras.layers.Activation("relu", name='%s_3_relu'%name)(o_b2b)

        o_b2c = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same', name='%s_0_conv'%name)(o_b2b)
        o_b2c = tf.keras.layers.BatchNormalization(name='%s_0_bn'%name)(o_b2c)

        # add
        outputs = tf.keras.layers.Add(name='%s_add'%name)([o_b1, o_b2c])
        # relu
        outputs = tf.keras.layers.Activation("relu", name='%s_out'%name)(outputs)
        return outputs


    def _DualAttention(self, high_level_feat, low_level_feat):
        # Channel Attention
        x = tf.keras.layers.Conv2D(512, kernel_size=1, padding='same', kernel_initializer='he_normal')(high_level_feat)

        ch_mask = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
        ch_mask = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(ch_mask)
        ch_mask = tf.keras.layers.BatchNormalization()(ch_mask)
        ch_mask = tf.keras.layers.Activation('sigmoid')(ch_mask)

        ch_mask = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(ch_mask)
        ch_mask = tf.keras.layers.BatchNormalization()(ch_mask)
        ch_mask = tf.keras.layers.Activation('relu')(ch_mask)

        low_level_feat = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(low_level_feat)
        low_level_feat = tf.keras.layers.BatchNormalization()(low_level_feat)
        low_level_feat = tf.keras.layers.Activation('relu')(low_level_feat)
        out_channel = tf.multiply(low_level_feat, ch_mask)

        # Spatial attention
        mask_sp_init = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(low_level_feat)
        mask_sp_init = tf.keras.layers.BatchNormalization()(mask_sp_init)
        mask_sp_init = tf.keras.layers.Activation('relu')(mask_sp_init)

        max_branch= tf.reduce_max(mask_sp_init, axis=-1, keepdims=True)
        avg_branch = tf.reduce_mean(mask_sp_init, axis=-1, keepdims=True)

        merge_branches = tf.keras.layers.Concatenate()([max_branch, avg_branch])
        mask_spatial = tf.keras.layers.Conv2D(1, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal')(merge_branches)
        mask_spatial = tf.keras.layers.BatchNormalization()(mask_spatial)
        mask_spatial = tf.keras.layers.Activation('sigmoid')(mask_spatial)

        upsample_x = tf.keras.layers.UpSampling2D(size=mask_spatial.shape[1]//high_level_feat.shape[1], interpolation="bilinear")(high_level_feat)
        out_spatial = tf.multiply(upsample_x, mask_spatial)

        return out_channel, out_spatial


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

    def _WASPv2(self, x, nb_filters, d):

        x1 = self._Atrous_SepConv(x, conv_type="sepconv2d", name='waspv2_conv_r%d'%d[0], filters=nb_filters, kernel_size=1, dilation_rate=d[0], use_bias=True)
        x2 = self._Atrous_SepConv(x1, conv_type="sepconv2d", name='waspv2_conv_r%d'%d[1], filters=nb_filters, kernel_size=3, dilation_rate=d[1], use_bias=True)
        x3 = self._Atrous_SepConv(x2, conv_type="sepconv2d", name='waspv2_conv_r%d'%d[2], filters=nb_filters, kernel_size=3, dilation_rate=d[2], use_bias=True)
        x4 = self._Atrous_SepConv(x3, conv_type="sepconv2d", name='waspv2_conv_r%d'%d[3], filters=nb_filters, kernel_size=3, dilation_rate=d[3], use_bias=True)

        x5 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='waspv2_avg')(x)
        x5 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='waspv2_avg_conv', kernel_initializer='he_normal')(x5)
        x5 = tf.keras.layers.BatchNormalization(name='waspv2_avg_bn')(x5)
        x5 = tf.keras.layers.Activation('relu', name='waspv2_avg_relu')(x5)
        x5 =  tf.keras.layers.UpSampling2D(size=x4.shape[1]//x5.shape[1], interpolation="bilinear", name='waspv2_avg_up2D')(x5)

        return tf.keras.layers.Concatenate(name='waspv2_add')([x1, x2, x3, x4, x5])


model = GourmetNet(num_classes=10, backbone_name="res50", output_stride=16, input_shape=(512,512,3))()
model.summary()

# https://github.com/uditsharma29/GourmetNet