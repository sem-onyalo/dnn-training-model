import datetime
import time

from .discriminator import Discriminator
from .generator import Generator
from data import Data
from matplotlib import pyplot
from storage import StorageLocal
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class AuxiliaryClassifierGAN:
    def __init__(self, data:Data, params=None) -> None:
        self.trainDateTimeUtc = datetime.MINYEAR
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
        self.latentDim = params.latentDim
        self.labelDim = params.labelDim
        self.batchNorm = params.batchNorm
        self.convFilters = [int(x) for x in params.convFilters.split(',')]
        self.convTransposeFilters = [int(x) for x in params.convTransposeFilters.split(',')]
        self.kernelInitStdDev = params.kernelInitStdDev
        self.generatorInputFilters = params.generatorInputFilters
        self.leakyReluAlpha = params.leakyReluAlpha
        self.dropoutRate = params.dropoutRate
        self.convLayerKernelSize = (3,3)
        self.convTransposeLayerKernelSize = (4,4)
        self.generatorOutputLayerKernelSize = (7,7)

    def initMetricsVars(self):
        self.metricHeader = ["Epoch", "Epochs", "Mini-Batch", "Mini-Batches", "Discriminator Loss: Real", "Discriminator Loss: Fake", "GAN Loss"]
        self.metricHistory = list()
        self.lossHistory = list()
        self.realBinaryLossHistory = list()
        self.realLabelsLossHistory = list()
        self.fakeBinaryLossHistory = list()
        self.fakeLabelsLossHistory = list()
        self.storage = list()

    def initStorage(self):
        self.trainDateTimeUtc = datetime.datetime.utcnow()
        self.storage.append(StorageLocal(root=self.evalDirectoryName, datetime=self.trainDateTimeUtc))

    def writeHyperparameters(self):
        hyperparameters = {
            "latentDim": self.latentDim,
            "labelDim": self.labelDim,
            "batchNorm": self.batchNorm,
            "adamLearningRate": self.adamLearningRate,
            "adamBeta1": self.adamBeta1,
            "kernelInitStdDev": self.kernelInitStdDev,
            "leakyReluAlpha": self.leakyReluAlpha,
            "dropoutRate": self.dropoutRate,
            "epochs": self.epochs,
            "batchSize": self.batchSize,
            "evalFreq": self.evalFreq,
            "batchesPerEpoch": self.batchesPerEpoch,
            "halfBatch": self.halfBatch,
            "convFilters": self.convFilters,
            "convTransposeFilters": self.convTransposeFilters,
            "generatorInputFilters": self.generatorInputFilters,
            "convLayerKernelSize": self.convLayerKernelSize,
            "convTransposeLayerKernelSize": self.convTransposeLayerKernelSize,
            "generatorOutputLayerKernelSize": self.generatorOutputLayerKernelSize
        }

        for item in self.storage:
            item.writeHyperparameters(hyperparameters)

    def writeSamplesInit(self, samples=150):
        xReal, _ = self.data.generateRealTrainingSamples(samples)
        xFake, _ = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples)

        for item in self.storage:
            item.writeTargetSamples(xReal)
            item.writeGeneratedSamples(0, xFake)

    def writeMetrics(self, epoch):
        for item in self.storage:
            item.writeMetrics(epoch, self.metricHistory)

        self.metricHistory.clear()

    def writeSamples(self, epoch, samples=150):
        xFake, _ = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples, False)

        for item in self.storage:
            item.writeGeneratedSamples(epoch, xFake)

    def writeSummary(self, epoch, **kwargs):
        summary = {
            "Epoch": epoch,
            "Real Accuracy": kwargs["realAcc"],
            "Fake Accuracy": kwargs["fakeAcc"],
            "Elapsed Time": kwargs["elapsedTime"]
        }

        for item in self.storage:
            item.writeSummary(epoch, summary)

        print(f'Summary: {summary}')

    def writeLossAndAccuracy(self, epoch):
        history = {
            "dLossReal": self.realBinaryLossHistory,
            "dLossFake": self.fakeBinaryLossHistory,
            "gLoss": self.lossHistory,
            "accReal": self.realLabelsLossHistory,
            "accFake": self.fakeLabelsLossHistory
        }

        for item in self.storage:
            item.writeLossAndAccuracy(epoch, history)

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
        self.initStorage()

        self.epochs = params.epochs
        self.batchSize = params.batchsize
        self.evalFreq = params.evalfreq
        self.batchesPerEpoch = int(self.data.getDatasetShape() / self.batchSize)
        self.halfBatch = int(self.batchSize / 2)

        self.writeHyperparameters()

        self.writeSamplesInit()

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

                metrics = [i + 1, self.epochs, j + 1, self.batchesPerEpoch, dRealLossBinary, dFakeLossBinary, gLossBinary]
                print(list(map(lambda x,y: {x: y}, self.metricHeader, metrics)))
                self.metricHistory.append(metrics)

                # - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG
                if j > 4:
                    break

            self.evaluate(i + 1)
            return
            # - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG - DEBUG

            if (i + 1) % self.evalFreq == 0:
                self.evaluate(i + 1)

    def evaluate(self, epoch, samples=150):
        [xReal, xRealLabel], yReal = self.data.generateRealTrainingSamples(samples)

        # TODO: unpack results properly (i.e. figure out which output values are which)
        _, dRealAcc, _, _, _ = self.discriminator.evaluate(xReal, [yReal, xRealLabel])

        [xFake, xFakeLabel], yFake = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples)
        _, dFakeAcc, _, _, _ = self.discriminator.evaluate(xFake, [yFake, xFakeLabel])

        modelFilename = '%s/generated_model_e%03d.h5' % (self.evalDirectoryName, epoch)
        # self.generator.save(modelFilename)

        self.writeMetrics(epoch)
        self.writeSamples(epoch)
        self.writeLossAndAccuracy(epoch)
        self.writeSummary(epoch, realAcc=dRealAcc, fakeAcc=dFakeAcc, elapsedTime=self.getElapsedTimeStr())

    def getElapsedTimeStr(self):
        elapsedTime = time.time() - self.startTime
        return str(datetime.timedelta(seconds=elapsedTime))

    def summary(self):
        print('\nDiscriminator\n')
        self.discriminator.summary()
        
        print('\nGenerator\n')
        self.generator.summary()
        
        print('\nGAN\n')
        self.gan.summary()
