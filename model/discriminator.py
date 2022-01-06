from data import Data
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class Discriminator:
    def __init__(self, data:Data, params) -> None:
        self.data = data
        self.initHyperparameters(params)

    def initHyperparameters(self, params) -> None:
        self.batchNorm = params.batchNorm
        self.kernelInitStdDev = params.kernelInitStdDev
        self.adamLearningRate = params.adamLearningRate
        self.adamBeta1 = params.adamBeta1
        self.convFilters = [int(x) for x in params.convFilters.split(',')]
        self.convLayerKernelSize = (3,3)
        self.leakyReluAlpha = params.leakyReluAlpha
        self.dropoutRate = params.dropoutRate

    def build(self) -> Model:
        # TODO: add parameter to distinguish discriminator type (i.e. for DCGAN or AuxGAN)

        init = RandomNormal(stddev=self.kernelInitStdDev)
        imageInput = Input(shape=self.data.imageShape)
        convLayer = self.buildConvLayers(self.batchNorm, init, imageInput)

        flattenLayer = Flatten()(convLayer)
        binaryOutputLayer = Dense(1, activation='sigmoid')(flattenLayer)
        labelsOutputLayer = Dense(self.data.classes, activation='softmax')(flattenLayer)
        model = Model(imageInput, [binaryOutputLayer, labelsOutputLayer])

        opt = Adam(learning_rate=self.adamLearningRate, beta_1=self.adamBeta1)
        # TODO: check how to get accuracy metrics for multi-output models
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])
        return model

    def buildConvLayers(self, batchNorm, kernelInit, inLayer):
        layer = inLayer
        for f in self.convFilters:
            layer = self.buildConvLayer(f, batchNorm, kernelInit, layer)
        return layer

    def buildConvLayer(self, filters, batchNorm, kernelInit, inLayer):
        # downsample layers
        layer = Conv2D(filters, self.convLayerKernelSize, (2,2), padding='same', kernel_initializer=kernelInit)(inLayer)
        if batchNorm:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU(self.leakyReluAlpha)(layer)
        layer = Dropout(self.dropoutRate)(layer)

        # normal sample layers
        layer = Conv2D(filters, self.convLayerKernelSize, padding='same', kernel_initializer=kernelInit)(layer)
        if batchNorm:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU(self.leakyReluAlpha)(layer)
        layer = Dropout(self.dropoutRate)(layer)

        return layer

    def train(self, params):
        raise Exception('Not supported')

    def summary(self):
        print('\nDiscriminator\n')
        model = self.build()
        model.summary()
