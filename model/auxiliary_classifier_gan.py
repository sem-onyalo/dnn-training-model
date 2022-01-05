import os
import time
import datetime

from .discriminator import Discriminator
from .generator import Generator
from data import Data
from matplotlib import pyplot
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class AuxiliaryClassifierGAN:
    def __init__(self, data:Data, params=None) -> None:
        self.evalDirectoryName = params.evalDirLocal
        self.initHyperparameters(params)
        self.initMetricsVars()

        self.data = data

        discriminator = Discriminator(data, params)
        self.discriminator = discriminator.buildModel()
        
        generator = Generator(data, params)
        self.generator = generator.buildModel()

        self.gan = self.buildModel()

    def initHyperparameters(self, params):
        self.adamLearningRate = params.adamLearningRate
        self.adamBeta1 = params.adamBeta1

    def initMetricsVars(self):
        self.realBinaryLossHistory = list()
        self.realLabelsLossHistory = list()
        self.fakeBinaryLossHistory = list()
        self.fakeLabelsLossHistory = list()
        self.lossHistory = list()
        self.metricHistory = list()

    def buildModel(self) -> Model:
        for layer in self.discriminator.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        ganOutput = self.discriminator(self.generator.output)
        model = Model(self.generator.input, ganOutput)

        opt = Adam(learning_rate=self.adamLearningRate, beta_1=self.adamBeta1)
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
        return model

    def train(self, params):
        if not os.path.exists(self.evalDirectoryName):
            os.makedirs(self.evalDirectoryName)

        self.epochs = params.epochs
        self.batchSize = params.batchsize
        self.evalFreq = params.evalfreq
        self.batchesPerEpoch = int(self.data.getDatasetShape() / self.batchSize)
        self.halfBatch = int(self.batchSize / 2)
        self.writeTrainingParameters()

        self.plotStartingImageSamples()

        self.startTime = time.time()

        for i in range(self.epochs):
            for j in range(self.batchesPerEpoch):
                [xReal, xRealLabel], yReal = self.data.generateRealTrainingSamples(self.halfBatch)
                _, dRealLossBinary, dRealLossLabels, _, _ = self.discriminator.train_on_batch(xReal, [yReal, xRealLabel])

                [xFake, xFakeLabel], yFake = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, self.halfBatch)
                _, dFakeLossBinary, dFakeLossLabels, _, _ = self.discriminator.train_on_batch(xFake, [yFake, xFakeLabel])

                [xGan, xGanLabel], yGan = self.data.generateFakeTrainingGanSamples(self.latentDim, self.batchSize)
                _, gLossBinary, gLossLabels = self.gan.train_on_batch([xGan, xGanLabel], [yGan, xGanLabel])

                self.realBinaryLossHistory.append(dRealLossBinary)
                self.realLabelsLossHistory.append(dRealLossLabels)
                self.fakeBinaryLossHistory.append(dFakeLossBinary)
                self.fakeLabelsLossHistory.append(dFakeLossLabels)
                self.lossHistory.append(gLossBinary)

                metrics = ('> %d, %d/%d, dRealLossBinary=%.3f, dFakeLossBinary=%.3f, gLossBinary=%.3f' %
                    (i + 1, j, self.batchesPerEpoch, dRealLossBinary, dFakeLossBinary, gLossBinary))
                self.metricHistory.append(metrics)
                print(metrics)

            if (i + 1) % self.evalFreq == 0:
                self.evaluate(i + 1)

    def evaluate(self, epoch, samples=150):
        [xReal, xRealLabel], yReal = self.data.generateRealTrainingSamples(samples)
        # TODO: unpack results properly (i.e. figure out which output values are which)
        _, dRealAcc, _, _, _ = self.discriminator.evaluate(xReal, [yReal, xRealLabel])

        [xFake, xFakeLabel], yFake = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples)
        _, dFakeAcc, _, _, _ = self.discriminator.evaluate(xFake, [yFake, xFakeLabel])

        accuracyMetrics = '> %d, accuracy real: %.0f%%, fake: %.0f%%' % (epoch, dRealAcc * 100, dFakeAcc * 100)
        self.metricHistory.append(accuracyMetrics)
        print(accuracyMetrics)

        elaspedTime = f'> {epoch}, elapsed time: {self.getElapsedTime()}'
        self.metricHistory.append(elaspedTime)
        print(elaspedTime)

        modelFilename = '%s/generated_model_e%03d.h5' % (self.evalDirectoryName, epoch)
        self.generator.save(modelFilename)

        metricsFilename = '%s/metrics_e%03d.txt' % (self.evalDirectoryName, epoch)
        with open(metricsFilename, 'w') as fd:
            for i in self.metricHistory:
                fd.write(i + '\n')
            self.metricHistory.clear()

        outputPath = f'{self.evalDirectoryName}/generated_plot_e{epoch}_random.png'
        self.plotImageSamples([xFake, xFakeLabel], outputPath)

        xFakeOrdered, _ = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples, False)
        outputPath = f'{self.evalDirectoryName}/generated_plot_e{epoch}_ordered.png'
        self.plotImageSamples(xFakeOrdered, outputPath)

        self.plotHistory(epoch)

    def plotImageSamples(self, samples, outputPath, n=10):
        images, _ = samples
        scaledImages = (images + 1) / 2.0 # scale from -1,1 to 0,1

        for i in range(n * n):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(scaledImages[i, :, :, 0], cmap='gray_r')

        pyplot.savefig(outputPath)
        pyplot.close()

    def plotStartingImageSamples(self, samples=150):
        xReal, _ = self.data.generateRealTrainingSamples(samples)
        self.plotImageSamples(xReal, f'{self.evalDirectoryName}/target_plot.png')

        xFake, _ = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples)
        self.plotImageSamples(xFake, f'{self.evalDirectoryName}/generated_plot_e0.png')

    def plotHistory(self, epoch):
        pyplot.subplot(2, 1, 1)
        pyplot.plot(self.realBinaryLossHistory, label='dRealLoss')
        pyplot.plot(self.fakeBinaryLossHistory, label='dFakeLoss')
        pyplot.plot(self.lossHistory, label='gLoss')
        pyplot.legend()

        pyplot.subplot(2, 1, 2)
        pyplot.plot(self.realLabelsLossHistory, label='accReal')
        pyplot.plot(self.fakeLabelsLossHistory, label='accFake')
        pyplot.legend()

        pyplot.savefig('%s/loss_acc_history_e%03d.png' % (self.evalDirectoryName, epoch))
        pyplot.close()

    def getElapsedTime(self):
        elapsedTime = time.time() - self.startTime
        return str(datetime.timedelta(seconds=elapsedTime))

    def writeTrainingParameters(self):
        # TODO: write as JSON file
        params = 'Training Parameters\n'
        params += '--------------------------------------------------\n'
        params += f'latentDim: {self.latentDim}\n'
        params += f'convFilters: {self.convFilters}\n'
        params += f'convTransposeFilters: {self.convTransposeFilters}\n'
        params += f'generatorInputFilters: {self.generatorInputFilters}\n'
        params += f'convLayerKernelSize: {self.convLayerKernelSize}\n'
        params += f'convTransposeLayerKernelSize: {self.convTransposeLayerKernelSize}\n'
        params += f'generatorOutputLayerKernelSize: {self.generatorOutputLayerKernelSize}\n'
        params += f'adamLearningRate: {self.adamLearningRate}\n'
        params += f'adamBeta1: {self.adamBeta1}\n'
        params += f'kernelInitStdDev: {self.kernelInitStdDev}\n'
        params += f'leakyReluAlpha: {self.leakyReluAlpha}\n'
        params += f'dropoutRate: {self.dropoutRate}\n'
        params += f'epochs: {self.epochs}\n'
        params += f'batchSize: {self.batchSize}\n'
        params += f'evalFreq: {self.evalFreq}\n'
        params += f'batchesPerEpoch: {self.batchesPerEpoch}\n'
        params += f'halfBatch: {self.halfBatch}\n'
        with open(os.path.join(self.evalDirectoryName, 'training_parameters.txt'), mode='w') as fd:
            fd.write(params)

    def summary(self):
        print('\nDiscriminator\n')
        self.discriminator.summary()
        
        print('\nGenerator\n')
        self.generator.summary()
        
        print('\nGAN\n')
        self.gan.summary()
