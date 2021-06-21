from keras.layers import MaxPooling2D, BatchNormalization
from keras.layers import merge, Input, concatenate
from keras.utils import plot_model
from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation


class FCN:
    def __init__(self, img_height, img_width, classes) -> None:
        self.img_height = img_height
        self.img_width = img_width
        self.CLASSES = classes

    def create_model(self):
        # encoder
        # input_img
        input = Input(shape=(self.img_height, self.img_width, 3))
        # input_img_1 = Input(shape=(self.img_height, self.img_width, 3))
        # input_img_2 = Input(shape=(self.img_height, self.img_width, 1))
        # input = concatenate([input_img_1, input_img_2])

        # block1
        layer = Conv2D(filters=64, kernel_size=(3, 3),
                       padding='same')(input)
        layer = BatchNormalization()(layer)

        layer = Activation('relu')(layer)
        layer = Conv2D(filters=64, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling2D()(layer)

        # block2
        layer = Conv2D(filters=128, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=128, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling2D()(layer)

        # block3
        layer = Conv2D(filters=256, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=256, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=256, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling2D()(layer)

        output_block3 = layer

        # block4
        layer = Conv2D(filters=512, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=512, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=512, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling2D()(layer)

        output_block4 = layer

        # block5
        layer = Conv2D(filters=512, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=512, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=512, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = MaxPooling2D()(layer)

        # decoder
        # block5
        # Transpose
        layer = Conv2DTranspose(filters=512, kernel_size=(3, 3),
                                strides=(2, 2), padding='same')(layer)
        layer = Conv2D(filters=512, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=512, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=512, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        # sum
        layer = merge.add([layer, output_block4])
        # Transpose
        layer = Conv2DTranspose(filters=256, kernel_size=(3, 3),
                                strides=(2, 2), padding='same')(layer)
        # block4
        layer = Conv2D(filters=256, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=256, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=256, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        # sum
        layer = merge.add([layer, output_block3])
        # Transpose
        layer = Conv2DTranspose(filters=128, kernel_size=(3, 3),
                                strides=(8, 8), padding='same')(layer)
        # block3
        layer = Conv2D(filters=64, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=32, kernel_size=(3, 3),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(filters=self.CLASSES, kernel_size=(1, 1),
                       padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        # output
        layer = Activation('softmax')(layer)

        # model = Model([input_img_1, input_img_2], layer)
        model = Model(input, layer)

        # plot_model(model, to_file='model.png', show_shapes=True, dpi=300)

        return model
