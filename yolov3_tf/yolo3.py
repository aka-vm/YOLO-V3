import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
# from keras.layers import Layer, Conv2D, UpSampling2D, Add, Concatenate
from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D, Add, Concatenate

from utils import (
    anchor_box_convert,
    )
from config import (
    ANCHORS,
    NUM_CLASSES,
    CLASS_LABELS
)

# Conv2d layer with optional batch normalization and leaky relu


class ConvBlock(Conv2D):

    def __init__(self, output_filters, kernel_size, normalize=True, downsample=False, **kwargs):
        strides, padding = (2, 'same') if downsample else (1, 'same')
        super().__init__(output_filters, kernel_size, strides=strides, padding=padding, **kwargs)
        self.normalize = normalize
        if normalize:
            self.use_bias = False
            self.bn = tf.nn.batch_normalization
            self.lrelu = tf.nn.leaky_relu

    def call(self, inputs):
        x = self.convolution_op(inputs, self.kernel)
        if self.normalize:
            mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
            x = self.bn(x, mean, variance, offset=None, scale=None, variance_epsilon=1e-5)
            x = self.lrelu(x, alpha=0.1)
        return x


class ResidualBlock(Layer):

    def __init__(self, filters:int, repetes:int = 1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.repetes = repetes
        self.layes = [
            Sequential([
                ConvBlock(filters//2, 1),
                ConvBlock(filters, 3)
            ]) for _ in range(repetes)
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.layes:
            # x = Add()([x, layer(x)])
            x = tf.math.add(x, layer(x))
        return x


class ConvPassBlock(Layer):

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.layers = Sequential([
            ConvBlock(filters, 1),
            ConvBlock(filters*2, 3),
            ConvBlock(filters, 1),
            ConvBlock(filters*2, 3),
            ConvBlock(filters, 1),
        ])

    def call(self, inputs):
        return self.layers(inputs)


class OutputBlock(Layer):

    def __init__(self, filters:int, num_classes:int, num_anchors:int=3, **kwargs):
        super().__init__(**kwargs)
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
        num_classes:int = None,
        initial_filters=32,
        base_output_scale=(13, 13),
        anchors=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._input_shape = input_shape

        self.num_classes = num_classes
        self._initial_filters = initial_filters
        self.output_scales = [(base_output_scale[0]*2**i, base_output_scale[1]*2**i) for i in range(3)]
        anchors = anchors or ANCHORS
        self.anchors = anchor_box_convert(anchors, self.output_scales)

        self._model_layers = self._create_model()

    def call(self, inputs):
        outputs = []
        route_layers = []
        x = inputs

        for layer_name, layer in self._model_layers.items():
            if isinstance(layer, OutputBlock):
                outputs.append(layer(x))
                continue

            x = layer(x)
            if isinstance(layer, UpSampling2D):
                # x = Concatenate()([x, route_layers.pop()])
                x = tf.concat([x, route_layers.pop()], axis=-1)
            elif isinstance(layer, ResidualBlock) and layer.repetes == 8:
                route_layers.append(x)

        return outputs

    def summary(self):
        x = tf.keras.Input(shape=(self._input_shape))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def _create_model(self):
        FILTELS = self._initial_filters
        layers = dict(
            conv_1              = ConvBlock(FILTELS, 3, name='conv_1'),
            downsample_conv_1   = ConvBlock(FILTELS*2, 3, downsample=True, name='downsample_conv_1'),
            residual_block_1    = ResidualBlock(FILTELS*2, repetes=1, name='residual_block_1'),
            downsample_conv_2   = ConvBlock(FILTELS*4, 3, downsample=True, name='downsample_conv_2'),
            residual_block_2    = ResidualBlock(FILTELS*4, repetes=2, name='residual_block_2'),
            downsample_conv_3   = ConvBlock(FILTELS*8, 3, downsample=True, name='downsample_conv_3'),
            residual_block_3    = ResidualBlock(FILTELS*8, repetes=8, name='residual_block_3'),                   # Route 2
            downsample_conv_4   = ConvBlock(FILTELS*16, 3, downsample=True, name='downsample_conv_4'),
            residual_block_4    = ResidualBlock(FILTELS*16, repetes=8, name='residual_block_4'),                   # Route 1
            downsample_conv_5   = ConvBlock(FILTELS*32, 3, downsample=True, name='downsample_conv_5'),
            residual_block_5    = ResidualBlock(FILTELS*32, repetes=4, name='residual_block_5'),
            # Darknet53 ends here

            conv_pass_1     = ConvPassBlock(FILTELS*16, name='conv_pass_1'),
            output_1        = OutputBlock(FILTELS*32, self.num_classes, name='output_1'),

            conv_2          = ConvBlock(FILTELS*8, 1, name='conv_2'),
            upsample_1      = UpSampling2D(2, name='upsample_1'),
            # concat_1
            conv_pass_2     = ConvPassBlock(FILTELS*8, name='conv_pass_2'),
            output_2        = OutputBlock(FILTELS*16, self.num_classes, name='output_2'),

            conv_3          = ConvBlock(FILTELS*4, 1, name='conv_3'),
            upsample_2      = UpSampling2D(2, name='upsample_2'),
            # concat_2
            conv_pass_3     = ConvPassBlock(FILTELS*4, name='conv_pass_3'),
            output_3        = OutputBlock(FILTELS*8, self.num_classes, name='output_3'),
        )

        return layers


if __name__ == "__main__":
    num_classes = NUM_CLASSES
    class_labels = CLASS_LABELS
    IMAGE_SIZE = 416
    initial_filters = 32

    model = YOLOv3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=num_classes, initial_filters=initial_filters, name='YOLOv3')

    inputs = tf.random.normal((2, IMAGE_SIZE, IMAGE_SIZE, 3))
    op = model(inputs)

    assert op[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert op[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert op[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)

    model.summary()


    # plot_name = './YOLOv3-tf.png'
    # tf.keras.utils.plot_model(
    #     model,
    #     plot_name,
    #     show_shapes=True,
    #     show_layer_names=True,
    #     expand_nested=True,
    #     dpi=256
    #     )