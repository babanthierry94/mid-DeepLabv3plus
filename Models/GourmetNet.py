# https://github.com/uditsharma29/GourmetNet/blob/afc7b49096c5da90d9cd9a4c3cadecc6c4b8909b/modeling/backbone/resnet.py

import tensorflow as tf

class GourmetNet(object):
    """
    Original ResNet-101 ('resnet_v1_101.ckpt')
    Original ResNet-50 ('resnet_v1_50.ckpt')
    """
    def __init__(self, num_classes=21, encoder_name="res101", finetune=False, input_shape=(512,512,3), output_stride=16):
        if encoder_name not in ['res101', 'res50']:
            print('encoder_name ERROR!')
            print("Please input: res101, res50")
            sys.exit(-1)

        self.encoder_name = encoder_name
        self.inputs = tf.keras.layers.Input(shape=input_shape)
        self.num_classes = num_classes

        self.output_stride = output_stride
        if self.output_stride == 16:
            self.strides = [1, 2, 2, 1]
            self.dilations = [1, 1, 1, 2]
        elif self.output_stride == 8:
            self.strides = [1, 2, 1, 1]
            self.dilations = [1, 1, 2, 4]

        self.encoder_name = encoder_name
        if self.encoder_name == 'res101':
            self.blocks = [3, 4, 23, 3]
        elif self.encoder_name == 'res50':
            self.blocks = [3, 4, 6, 3]
        
        if finetune :
            self.pretrained = "imagenet"
        else :
            self.pretrained = None

        self.build_network()

    def __call__(self):
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name='GourmetNet')
        return model

    def build_network(self):
        low_level_feat, high_level_feat = self.build_encoder()
        refined_low_level_feat, refined_high_level_feat = self._DualAttention(high_level_feat, low_level_feat)
        self.outputs = self.build_decoder(refined_low_level_feat, refined_high_level_feat)

    def build_encoder(self):
        print("-----------build encoder: %s-----------" % self.encoder_name)
        # if self.encoder_name == 'res50':
        #     backbone_model = tf.keras.applications.ResNet50(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
        #     low_layer_name = "conv2_block3_out" #After block 1
        #     high_layer_name = "conv4_block6_out" #After block 3
        # elif self.encoder_name == 'res101':
        #     backbone_model = tf.keras.applications.ResNet101(weights=self.pretrained, include_top=False, input_tensor=self.inputs)
        #     low_layer_name = "conv2_block3_out" #After block 1
        #     high_layer_name = "conv4_block23_out" #After block 3
        # low_level_feat = backbone_model.get_layer(low_layer_name).output #After block 1
        # outputs = backbone_model.get_layer(high_layer_name).output #After block 3

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
        outputs = self._bottleneck_resblock(outputs, 2048, self.strides[3], self.dilations[3], 'block4_unit_1', identity_connection=False)
        for i in range(2, self.blocks[3]+1):
            outputs = self._bottleneck_resblock(outputs, 2048, 1, 1, 'block4_unit_%d'%i)
        print("after block4:", outputs.shape)
        high_level_feat = outputs
        
        print("low_level_feat:", low_level_feat.shape)
        print("high_level_feat:", high_level_feat.shape)

        return low_level_feat, high_level_feat

    def build_decoder(self, refined_low_level, refined_high_level):
        print("-----------build decoder-----------")
        dilations = [1, 6, 12, 18]
        reduction = 256 // 8
        x = self._WASPv2(refined_high_level, 256, dilations)
        print("after waspv2 block:", x.shape)

        x = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        tf.random.set_seed(0)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.UpSampling2D(size=refined_low_level.shape[1]//x.shape[1], interpolation="bilinear")(x)

        refined_low_level = tf.keras.layers.Conv2D(reduction, kernel_size=1, padding='same', kernel_initializer='he_normal')(refined_low_level)
        refined_low_level = tf.keras.layers.BatchNormalization()(refined_low_level)
        refined_low_level = tf.keras.layers.Activation('relu')(refined_low_level)
        x = tf.keras.layers.Concatenate()([x, refined_low_level])
        #-------------------------------------------------------------------------------#
        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        print("Before the last UpSampling:", x.shape)
        x = tf.keras.layers.UpSampling2D(size=self.inputs.shape[1]//x.shape[1], interpolation="bilinear")(x)
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

    def _DualAttention(self, high_level_feat, low_level_feat):
        #============================================================
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

        #============================================================
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

    def _WASPv2(self, x, nb_filters, d):

        x1 = self._Atrous_SepConv(x, conv_type="sepconv2d", prefix='aspp/sepconv1', filters=nb_filters, kernel_size=1, dilation_rate=d[0], use_bias=True)
        x2 = self._Atrous_SepConv(x1, conv_type="sepconv2d", prefix='aspp/sepconv2', filters=nb_filters, kernel_size=3, dilation_rate=d[1], use_bias=True)
        x3 = self._Atrous_SepConv(x2, conv_type="sepconv2d", prefix='aspp/sepconv3', filters=nb_filters, kernel_size=3, dilation_rate=d[2], use_bias=True)
        x4 = self._Atrous_SepConv(x3, conv_type="sepconv2d", prefix='aspp/sepconv4', filters=nb_filters, kernel_size=3, dilation_rate=d[3], use_bias=True)

        x5 = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='waspv2/avg')(x)
        x5 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='waspv2/avg_conv', kernel_initializer='he_normal')(x5)
        x5 = tf.keras.layers.BatchNormalization(name='waspv2/avg_bn')(x5)
        x5 = tf.keras.layers.Activation('relu', name='waspv2/avg_relu')(x5)
        x5 =  tf.keras.layers.UpSampling2D(size=x4.shape[1]//x5.shape[1], interpolation="bilinear", name='waspv2/avg_upsambling')(x5)

        return tf.keras.layers.Concatenate(name='waspv2/add')([x1, x2, x3, x4, x5])


model = GourmetNet(num_classes=10, encoder_name="res50", output_stride=16, finetune=True, input_shape=(512,512,3))()
# model.summary()