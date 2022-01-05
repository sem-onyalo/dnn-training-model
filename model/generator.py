from data import Data
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, Embedding
from tensorflow.keras.layers import ReLU, LeakyReLU, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

class Generator:
    def __init__(self, data:Data, params) -> None:
        self.data = data
        self.initHyperparameters(params)

    def initHyperparameters(self, params):
        self.latentDim = params.latentDim
        self.labelDim = params.labelDim
        self.batchNorm = params.batchNorm
        self.kernelInitStdDev = params.kernelInitStdDev
        self.generatorInputFilters = params.generatorInputFilters
        self.convTransposeFilters = [int(x) for x in params.convTransposeFilters.split(',')]
        self.convTransposeLayerKernelSize = (4,4)
        self.generatorOutputLayerKernelSize = (7,7)
        self.leakyReluAlpha = params.leakyReluAlpha
        self.dropoutRate = params.dropoutRate

    def buildModel(self) -> Model:
        labelInputNodes = self.data.imageDim * self.data.imageDim
        init = RandomNormal(stddev=self.kernelInitStdDev)

        labelInput = Input(shape=(1,))
        labelEmbedding = Embedding(self.data.classes, self.labelDim)(labelInput)
        labelDense = Dense(labelInputNodes, kernel_initializer=init)(labelEmbedding)
        labelShaped = Reshape((self.data.imageDim, self.data.imageDim, 1))(labelDense)

        imageInputNodes = self.generatorInputFilters * self.data.imageDim * self.data.imageDim
        imageInput = Input(shape=(self.latentDim,))
        imageDense = Dense(imageInputNodes, kernel_initializer=init)(imageInput)
        imageActv = LeakyReLU(self.leakyReluAlpha)(imageDense)
        imageShaped = Reshape((self.data.imageDim, self.data.imageDim, self.generatorInputFilters))(imageActv)

        imageLabelConcat = Concatenate()([imageShaped, labelShaped])

        convLayer = self.buildConvTransposeLayers(init, imageLabelConcat)

        outputLayer = Conv2D(1, self.generatorOutputLayerKernelSize, padding='same', activation='tanh', kernel_initializer=init)(convLayer)
        model = Model([imageInput, labelInput], outputLayer)
        return model

    def buildConvTransposeLayers(self, kernelInit, inLayer):
        layer = inLayer
        for f in self.convTransposeFilters:
            layer = self.buildConvTransposeLayer(f, kernelInit, layer)
        return layer

    def buildConvTransposeLayer(self, filters, kernelInit, inLayer):
        layer = Conv2DTranspose(filters, self.convTransposeLayerKernelSize, (2,2), padding='same', kernel_initializer=kernelInit)(inLayer)
        if self.batchNorm:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU(self.leakyReluAlpha)(layer)
        outLayer = Dropout(self.dropoutRate)(layer)
        return outLayer

    def train(self, params):
        raise Exception('Not supported')

    def summary(self):
        print('\nGenerator\n')
        model = self.buildModel()
        model.summary()
