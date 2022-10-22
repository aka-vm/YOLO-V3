import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Add, Concatenate


# Conv2d layer with optional batch normalization and leaky relu


class ConvBlock(Conv2D):

    def __init__(self, output_filters, kernel_size, normalize=True, downsample=False, **kwargs):
        strides, padding = (2, 'valid') if downsample else (1, 'same')
        super(ConvBlock, self).__init__(output_filters, kernel_size, strides, padding, **kwargs)
        if normalize:
            self.use_bias = False
            self.normalize = normalize
            self.bn = BatchNormalization()
            self.lrelu = tf.nn.leaky_relu

    def call(self, inputs):
        conv_layer = super(ConvBlock, self).call(inputs)
        if self.normalize:
            bn_layer = self.bn(conv_layer)
            ac_layer = self.lrelu(bn_layer, alpha=0.1)
            return ac_layer
        else:
            return conv_layer


class ResidualBlock(Layer):

    def __init__(self, output_filters:int, repetes:int = 1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.output_filters = output_filters
        self.repetes = repetes
        self.layes = [
            Sequential([
                ConvBlock(output_filters//2, 1),
                ConvBlock(output_filters, 3)
            ])
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.layes:
            x = Add()([x, layer(x)])
        return x


class ConvPassBlock(Layer):

    def __init__(self, output_filters, **kwargs):
        super(ConvPassBlock, self).__init__(**kwargs)
        layers = Sequential([
            ConvBlock(output_filters//2, 1),
            ConvBlock(output_filters, 3),
            ConvBlock(output_filters//2, 1),
            ConvBlock(output_filters, 3),
            ConvBlock(output_filters//2, 1),
        ])

    def call(self, inputs):
        return self.layers(inputs)


class OutputBlock(Layer):

    def __init__(self, filters:int, num_classes:int, num_anchors:int=3, **kwargs):
        super(OutputBlock, self).__init__(**kwargs)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.layer = Sequential([
            ConvBlock(filters, 3),
            ConvBlock(num_anchors*(num_classes+5), 1, normalize=False)
        ])

    def call(self, inputs):
        op = self.layer(inputs)
        op_shape = tf.shape(op)
        op = tf.reshape(op, (op_shape[0], op_shape[1], op_shape[2], self.num_anchors, 5+self.num_classes))      #(batch, h, w, n_anchors, 5+classes)
        return tf.transpose(op, (0, 3, 1, 2, 4))


class YOLOv3(Model):

    def __init__(
        self,
        input_shape=(416, 416, 3),
        num_classes=None,
        initial_filters=32,
        **kwargs
    ):
        self.num_classes = num_classes
        self.initial_filters = initial_filters

        input_layer = tf.keras.Input(shape=input_shape)
        outputs = self._create_model_layers(input_layer)
        super(YOLOv3, self).__init__(**kwargs, inputs=input_layer, outputs=outputs)

    def _create_model_layers(self, input_layer):
        FILTELS = self.initial_filters
        routes = []
        outputs = []

        # 1st block (416, 416, 3) -> (416, 416, 32)
        x = ConvBlock(FILTELS, 3)(input_layer)
        # 2st block (416, 416, 32) -> (208, 208, 64)
        x = ConvBlock(FILTELS*2, 3, downsample=True)(x)
        x = ResidualBlock(FILTELS*2)(x)
        # 3st block (208, 208, 64) -> (104, 104, 128)
        x = ConvBlock(FILTELS*4, 3, downsample=True)(x)
        x = ResidualBlock(FILTELS*4, repetes=2)(x)
        # 4st block (104, 104, 128) -> (52, 52, 256)
        x = ConvBlock(FILTELS*8, 3, downsample=True)(x)
        x = ResidualBlock(FILTELS*8, repetes=8)(x)
        routes.append(x)
        # 5st block (52, 52, 256) -> (26, 26, 512)
        x = ConvBlock(FILTELS*16, 3, downsample=True)(x)
        x = ResidualBlock(FILTELS*16, repetes=8)(x)
        routes.append(x)
        # 6st block (26, 26, 512) -> (13, 13, 1024)
        x = ConvBlock(FILTELS*32, 3, downsample=True)(x)
        x = ResidualBlock(FILTELS*32, repetes=4)(x)
        # Darknet53 Ends

        x = ConvPassBlock(FILTELS*32)(x)
        outputs.append(OutputBlock(FILTELS*32, self.num_classes)(x))    # Output 1 (13, 13, 3, 5+n_classes)

        x = ConvBlock(FILTELS*16, 1)(x)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, routes.pop()])
        x = ConvPassBlock(FILTELS*16)(x)
        outputs.append(OutputBlock(FILTELS*16, self.num_classes)(x))    # Output 2 (26, 26, 3, 5+n_classes)

        x = ConvBlock(FILTELS*8, 1)(x)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, routes.pop()])
        x = ConvPassBlock(FILTELS*8)(x)
        outputs.append(OutputBlock(FILTELS*8, self.num_classes)(x))     # Output 3 (52, 52, 3, 5+n_classes)

        return outputs





