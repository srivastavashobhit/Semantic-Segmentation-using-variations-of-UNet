import tensorflow as tf


class UNetTCED(tf.keras.Model):
    """
    Skip Connection to add encoder output to corresponding decoder input at same depth level.
    Similar to Copy and Crop from the original U-Net architecture.
    """

    @staticmethod
    def skip_connection(input1, input2):
        primary = tf.keras.layers.concatenate(
            inputs=[input1, input2],
            axis=3
        )

        return primary

    @staticmethod
    def skip_connections(primary, encoder_blocks_output):
        primary_shape = primary.shape
        for layer in encoder_blocks_output:
            layer_shape = layer.shape
            if primary_shape[1] == layer_shape[1] and primary_shape[2] == layer_shape[2]:
                primary = tf.keras.layers.concatenate(
                    inputs=[primary, layer],
                    axis=3
                )
            else:
                len_factor = layer_shape[1] / primary_shape[1]
                wid_factor = layer_shape[2] / primary_shape[2]

                if len_factor > 1 and wid_factor > 1:
                    len_factor = int(len_factor)
                    wid_factor = int(wid_factor)
                    transformed_layer = tf.keras.layers.MaxPool2D(
                        pool_size=(len_factor, wid_factor)
                    )(layer)
                elif 1 > len_factor > 0 and 1 > wid_factor > 0:

                    len_factor = int(1 / len_factor)
                    wid_factor = int(1 / wid_factor)
                    transformed_layer = tf.keras.layers.UpSampling2D(
                        size=(len_factor, wid_factor)
                    )(layer)
                else:
                    assert False

                assert (primary_shape[1:3] == transformed_layer.shape[1:3])

                primary = tf.keras.layers.concatenate(
                    inputs=[primary, transformed_layer],
                    axis=3
                )

        return primary

    """
    U-Net inspired architecture, with 4 encoder blocks, 1 bottle neck layer and 4 decoder block
    Each encoder block learn features and squeeze the input on height and width dimension, 
    while increasing the depth. 

    Bottom neck layer simple apply two convolution operation and gives input to first decoder block.

    Decoder blocks uses transpose convolution layer to upsample the input, taking input from previous 
    transpose convolution layer and output of corresponding encoder block which gives spatial information 
    about the image. Output layer has #classes each corresponding to a output class.

    """

    def __init__(self, filters, classes, input_size):
        super(UNetTCED, self).__init__()
        self.filters = filters
        self.classes = classes
        self.encoder_block_1_conv1, \
            self.encoder_block_1_conv2, \
            self.encoder_block_1_maxpool, \
            self.encoder_block_1_dropout1 = self.encoder_block(block_depth=0, dropout_rate=0.3)

        self.encoder_block_2_conv1, \
            self.encoder_block_2_conv2, \
            self.encoder_block_2_maxpool, \
            self.encoder_block_2_dropout2 = self.encoder_block(block_depth=1, dropout_rate=0.3)

        self.encoder_block_3_conv1, \
            self.encoder_block_3_conv2, \
            self.encoder_block_3_maxpool, \
            self.encoder_block_3_dropout3 = self.encoder_block(block_depth=2, dropout_rate=0.3)

        self.encoder_block_4_conv1, \
            self.encoder_block_4_conv2, \
            self.encoder_block_4_maxpool, \
            self.encoder_block_4_dropout4 = self.encoder_block(block_depth=3, dropout_rate=0.3)

        self.bottle_neck_block_conv1, \
            self.bottle_neck_block_conv2 = self.bottle_neck_block(block_depth=4)

        self.decoder_block_4_upconv, \
            self.decoder_block_4_conv1, \
            self.decoder_block_4_conv2 = self.decoder_block(block_depth=3)

        self.decoder_block_3_upconv, \
            self.decoder_block_3_conv1, \
            self.decoder_block_3_conv2 = self.decoder_block(block_depth=2)

        self.decoder_block_2_upconv, \
            self.decoder_block_2_conv1, \
            self.decoder_block_2_conv2 = self.decoder_block(block_depth=1)

        self.decoder_block_1_upconv, \
            self.decoder_block_1_conv1, \
            self.decoder_block_1_conv2 = self.decoder_block(block_depth=0)

        self.output_block_0_conv, \
            self.output_block_0_output, = self.output_block(classes=classes)

        self.build(input_size)
        self.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    """
    Encoder block function
    """

    def encoder_block(self, block_depth, dropout_rate, kernel_size=3):
        filters = self.filters * (2 ** block_depth)

        conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )

        conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )
        maxpool1 = tf.keras.layers.MaxPool2D(
            pool_size=2
        )

        dropout1 = tf.keras.layers.Dropout(
            rate=dropout_rate
        )

        return conv1, conv2, maxpool1, dropout1

    """
    Bottle Neck block
    """

    def bottle_neck_block(self, block_depth, kernel_size=3):
        filters = self.filters * (2 ** block_depth)

        conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )

        conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            kernel_initializer='he_normal'
        )

        return conv1, conv2

    """
    Decoder block function
    """

    def decoder_block(self, block_depth, kernel_size=3):
        filters = self.filters * (2 ** block_depth)
        upconv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            padding='same',
        )

        conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same"
        )

        conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same"
        )

        return upconv, conv1, conv2

    """
    Output block function
    """

    def output_block(self, classes=1, kernel_size=3):
        filters = self.filters

        conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same"
        )

        output = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=1,
            padding='same'
        )

        return conv, output

    def call(self, inputs, training=False):
        tpt = False
        tpt2 = False
        tpt3 = False
        if tpt:
            print("************************", inputs.shape)
        x = self.encoder_block_1_conv1(inputs)
        if tpt:
            print("encoder_block_1_conv1 shape", x.shape)
        x = self.encoder_block_1_conv2(x)
        if tpt:
            print("encoder_block_1_conv2 shape", x.shape)
        skip_connection_input_1 = tf.identity(x)
        if tpt2:
            print("E skip_connection_input _1 shape", x.shape)
        x = self.encoder_block_1_maxpool(x)
        if tpt:
            print("encoder_block_1_maxpool shape", x.shape)
        if training:
            x = self.encoder_block_1_dropout1(x)

        x = self.encoder_block_2_conv1(x)
        if tpt:
            print("encoder_block_2_conv1 shape", x.shape)
        x = self.encoder_block_2_conv2(x)
        if tpt:
            print("encoder_block_2_conv2 shape", x.shape)
        skip_connection_input_2 = tf.identity(x)
        if tpt2:
            print("E skip_connection_input _2 shape", x.shape)
        x = self.encoder_block_2_maxpool(x)
        if tpt:
            print("encoder_block_2_maxpool shape", x.shape)
        if training:
            x = self.encoder_block_2_dropout2(x)

        x = self.encoder_block_3_conv1(x)
        if tpt:
            print("encoder_block_3_conv1 shape", x.shape)
        x = self.encoder_block_3_conv2(x)
        if tpt:
            print("encoder_block_3_conv2 shape", x.shape)
        skip_connection_input_3 = tf.identity(x)
        if tpt2:
            print("E skip_connection_input _3 shape", x.shape)
        x = self.encoder_block_3_maxpool(x)
        if tpt:
            print("encoder_block_3_maxpool shape", x.shape)
        if training:
            x = self.encoder_block_3_dropout3(x)
        x = self.encoder_block_4_conv1(x)
        if tpt:
            print("encoder_block_4_conv1 shape", x.shape)
        x = self.encoder_block_4_conv2(x)
        if tpt:
            print("encoder_block_4_conv2 shape", x.shape)
        skip_connection_input_4 = tf.identity(x)
        if tpt2:
            print("E skip_connection_input _4 shape", x.shape)
        x = self.encoder_block_4_maxpool(x)
        if tpt:
            print("encoder_block_4_maxpool shape", x.shape)
        if training:
            x = self.encoder_block_4_dropout4(x)

        x = self.bottle_neck_block_conv1(x)
        if tpt:
            print("bottle_neck_block_conv1 shape", x.shape)
        x = self.bottle_neck_block_conv2(x)
        if tpt:
            print("bottle_neck_block_conv2 shape", x.shape)

        encoder_blocks_output = [skip_connection_input_1, skip_connection_input_2, skip_connection_input_3,
                                 skip_connection_input_4]

        x = self.decoder_block_4_upconv(x)
        if tpt:
            print("decoder_block_4_upconv shape", x.shape)

        # x = UNet.skip_connection(x, skip_connection_input_4)
        x = UNetTCED.skip_connections(x, encoder_blocks_output)

        if tpt3:
            print("D x", x.shape, 'and skip_connection_input_4 shape', skip_connection_input_4.shape)
        x = self.decoder_block_4_conv1(x)
        if tpt:
            print("decoder_block_4_conv1 shape", x.shape)
        x = self.decoder_block_4_conv2(x)
        if tpt:
            print("decoder_block_4_conv2 shape", x.shape)

        x = self.decoder_block_3_upconv(x)
        if tpt:
            print("decoder_block_3_upconv shape", x.shape)

        # x = UNet.skip_connection(x, skip_connection_input_3)
        x = UNetTCED.skip_connections(x, encoder_blocks_output)

        if tpt3:
            print("D x", x.shape, 'and skip_connection_input_3 shape', skip_connection_input_3.shape)
        x = self.decoder_block_3_conv1(x)
        if tpt:
            print("decoder_block_3_conv1 shape", x.shape)
        x = self.decoder_block_3_conv2(x)
        if tpt:
            print("decoder_block_3_conv2 shape", x.shape)

        x = self.decoder_block_2_upconv(x)
        if tpt:
            print("decoder_block_2_upconv shape", x.shape)

        # x = UNet.skip_connection(x, skip_connection_input_2)
        x = UNetTCED.skip_connections(x, encoder_blocks_output)

        if tpt3:
            print("D x", x.shape, 'and skip_connection_input_2 shape', skip_connection_input_2.shape)
        x = self.decoder_block_2_conv1(x)
        if tpt:
            print("decoder_block_2_conv1 shape", x.shape)
        x = self.decoder_block_2_conv2(x)
        if tpt:
            print("decoder_block_2_conv2 shape", x.shape)

        x = self.decoder_block_1_upconv(x)
        if tpt:
            print("decoder_block_1_upconv shape", x.shape)

        # x = UNet.skip_connection(x, skip_connection_input_1)
        x = UNetTCED.skip_connections(x, encoder_blocks_output)

        if tpt3:
            print("D x", x.shape, 'and skip_connection_input_1 shape', skip_connection_input_1.shape)
        x = self.decoder_block_1_conv1(x)
        if tpt:
            print("decoder_block_1_conv1 shape", x.shape)
        x = self.decoder_block_1_conv2(x)
        if tpt:
            print("decoder_block_1_conv2 shape", x.shape)

        x = self.output_block_0_conv(x)
        if tpt:
            print("output_block_0_conv shape", x.shape)
        x = self.output_block_0_output(x)
        if tpt:
            print("output_block_0_output shape", x.shape)

        return x
