import config
import csv
import datetime
import h5py
import json
import io
import math

from storage.azure import StorageAzure

from .discriminator import Discriminator
from .generator import Generator
from constants import CLOUD_STORAGE_AWS_S3, CLOUD_STORAGE_AZURE
from data import Data
from matplotlib import pyplot
from storage import StorageLocal, StorageAwsS3
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.saving import hdf5_format

class AuxiliaryClassifierGAN:
    def __init__(self, data:Data, params=None) -> None:
        self.trainingStartDateTimeUtc = datetime.MINYEAR
        self.evalDirectoryName = params.evalDirLocal
        self.cloudStorageType = params.cloudStorageType

        self.initHyperparameters(params)
        self.initMetricsVars()

        self.data = data

        discriminator = Discriminator(data, params)
        self.discriminator = discriminator.build()
        
        generator = Generator(data, params)
        self.generator = generator.build()

        self.gan = self.build()

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
        self.metricsHeader = ["Epoch", "Epochs", "Mini-Batch", "Mini-Batches", "Discriminator Loss: Real", "Discriminator Loss: Fake", "GAN Loss"]
        self.metricsHistory = list()
        self.lossHistory = list()
        self.realBinaryLossHistory = list()
        self.realLabelsLossHistory = list()
        self.fakeBinaryLossHistory = list()
        self.fakeLabelsLossHistory = list()
        self.storage = list()

    def initStorage(self):
        self.storage.clear()
        self.storage.append(StorageLocal(root=self.evalDirectoryName, datetime=self.trainingStartDateTimeUtc))

        if self.cloudStorageType == CLOUD_STORAGE_AWS_S3:
            self.storage.append(StorageAwsS3(datetime=self.trainingStartDateTimeUtc))
        elif self.cloudStorageType == CLOUD_STORAGE_AZURE:
            self.storage.append(StorageAzure(datetime=self.trainingStartDateTimeUtc))

    def getImagePlots(self, images, samples) -> io.BytesIO:
        scaledImages = (images + 1) / 2.0 # scale from -1,1 to 0,1

        n = int(math.sqrt(samples))
        for i in range(n * n):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(scaledImages[i, :, :, 0], cmap='gray_r')

        buffer = io.BytesIO()
        pyplot.savefig(buffer)
        pyplot.close()
        buffer.seek(0)

        return buffer

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

        serialized = json.dumps(hyperparameters, indent=4)
        bytesObj = serialized.encode()

        for item in self.storage:
            item.writeBytes(config.hyperparameters_file, bytesObj)

    def writeInitImageSamples(self, samples=100):
        [images, _], _ = self.data.generateRealTrainingSamples(samples)
        buffer = self.getImagePlots(images, samples)

        for item in self.storage:
            item.writeBytes(config.target_samples_file, buffer.getvalue())

        self.writeImageSamples(0)

    def writeImageSamples(self, epoch, samples=100):
        [images, _], _ = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples, False)
        buffer = self.getImagePlots(images, samples)

        for item in self.storage:
            item.writeBytes(config.generated_samples_file, buffer.getvalue(), [config.epoch_directory, str(epoch)])

    def writeSummary(self, epoch, **kwargs):
        summary = {
            "Epoch": epoch,
            "Real Accuracy": kwargs["realAcc"],
            "Fake Accuracy": kwargs["fakeAcc"],
            "Elapsed Time": str(datetime.datetime.utcnow() - self.trainingStartDateTimeUtc)
        }

        print(f'Summary: {summary}')

        serialized = json.dumps(summary, indent=4)
        bytesObj = serialized.encode()

        for item in self.storage:
            item.writeBytes(config.summary_file, bytesObj, [config.epoch_directory, str(epoch)])

    def writeMetrics(self, epoch):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(self.metricsHeader)
        writer.writerows(self.metricsHistory)

        for item in self.storage:
            item.writeBytes(config.metrics_file, buffer.getvalue().encode(), [config.epoch_directory, str(epoch)])

        self.metricsHistory.clear()

    def writeLossAndAccuracy(self, epoch):
        pyplot.subplot(2, 1, 1)
        pyplot.plot(self.realBinaryLossHistory, label='D-Loss-Real')
        pyplot.plot(self.fakeBinaryLossHistory, label='D-Loss-Fake')
        pyplot.plot(self.lossHistory, label='G-Loss')
        pyplot.legend()

        pyplot.subplot(2, 1, 2)
        pyplot.plot(self.realLabelsLossHistory, label='Acc-Real')
        pyplot.plot(self.fakeLabelsLossHistory, label='Acc-Fake')
        pyplot.legend()

        buffer = io.BytesIO()
        pyplot.savefig(buffer)
        pyplot.close()
        buffer.seek(0)

        for item in self.storage:
            item.writeBytes(config.loss_accuracy_file, buffer.getvalue(), [config.epoch_directory, str(epoch)])

    def writeModel(self, epoch):
        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as fd:
            hdf5_format.save_weights_to_hdf5_group(fd, self.generator.layers)

        for item in self.storage:
            item.writeBytes(config.model_file, buffer.getvalue(), [config.epoch_directory, str(epoch)])

    def build(self) -> Model:
        for layer in self.discriminator.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        ganOutput = self.discriminator(self.generator.output)
        model = Model(self.generator.input, ganOutput)

        opt = Adam(learning_rate=self.adamLearningRate, beta_1=self.adamBeta1)
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
        return model

    def train(self, params):
        self.epochs = params.epochs
        self.evalFreq = params.evalfreq
        self.batchSize = params.batchsize
        self.halfBatch = int(self.batchSize / 2)
        self.batchesPerEpoch = int(self.data.getDatasetShape() / self.batchSize)
        self.trainingStartDateTimeUtc = datetime.datetime.utcnow()

        self.initStorage()
        self.writeHyperparameters()
        self.writeInitImageSamples()

        for i in range(self.epochs):
            print(", ".join([str(i) for i in self.metricsHeader]))

            for j in range(self.batchesPerEpoch):
                [xReal, xRealLabel], yReal = self.data.generateRealTrainingSamples(self.halfBatch)
                _, dRealLossBinary, dRealLossLabels, _, _ = self.discriminator.train_on_batch(xReal, [yReal, xRealLabel])

                [xFake, xFakeLabel], yFake = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, self.halfBatch)
                _, dFakeLossBinary, dFakeLossLabels, _, _ = self.discriminator.train_on_batch(xFake, [yFake, xFakeLabel])

                [xGan, xGanLabel], yGan = self.data.generateFakeTrainingGanSamples(self.latentDim, self.batchSize)
                _, gLossBinary, _ = self.gan.train_on_batch([xGan, xGanLabel], [yGan, xGanLabel])

                self.realBinaryLossHistory.append(dRealLossBinary)
                self.realLabelsLossHistory.append(dRealLossLabels)
                self.fakeBinaryLossHistory.append(dFakeLossBinary)
                self.fakeLabelsLossHistory.append(dFakeLossLabels)
                self.lossHistory.append(gLossBinary)

                metrics = [i + 1, self.epochs, j + 1, self.batchesPerEpoch, dRealLossBinary, dFakeLossBinary, gLossBinary]
                self.metricsHistory.append(metrics)
                print(", ".join([str(i) for i in metrics]))

            if (i + 1) % self.evalFreq == 0:
                self.evaluate(i + 1)

    def evaluate(self, epoch, samples=150):
        [xReal, xRealLabel], yReal = self.data.generateRealTrainingSamples(samples)
        _, dRealAcc, _, _, _ = self.discriminator.evaluate(xReal, [yReal, xRealLabel])

        [xFake, xFakeLabel], yFake = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples)
        _, dFakeAcc, _, _, _ = self.discriminator.evaluate(xFake, [yFake, xFakeLabel])

        self.writeModel(epoch)
        self.writeMetrics(epoch)
        self.writeImageSamples(epoch)
        self.writeLossAndAccuracy(epoch)
        self.writeSummary(epoch, realAcc=dRealAcc, fakeAcc=dFakeAcc)

    def summary(self):
        print('\nDiscriminator\n')
        self.discriminator.summary()
        
        print('\nGenerator\n')
        self.generator.summary()
        
        print('\nGAN\n')
        self.gan.summary()
