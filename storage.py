import csv
import datetime
import json
import os

from matplotlib import pyplot

FILE_ENCODING = "utf-8"
METRICS_FILE = "metrics.csv"
SUMMARY_FILE = "summary.json"
HYPERPARAMETERS_FILE = "hyperparameters.json"
TARGET_SAMPLES_FILE = "target.png"
GENERATED_SAMPLES_FILE = "generated.png"
LOSS_ACCURACY_FILE = "loss_accuracy.png"
EPOCH_ROOT = "epoch"

class StorageLocal:
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("root", "")
        kwargs.setdefault("datetime", datetime.datetime.utcnow())

        self.rootDirectory = kwargs["root"]
        self.datetimeDirectory = kwargs["datetime"].strftime("%Y%m%dT%H%M%SZ")

    def getRootDirectory(self):
        if not os.path.exists(self.rootDirectory):
            os.makedirs(self.rootDirectory)

        root = os.path.join(self.rootDirectory, self.datetimeDirectory)
        if not os.path.exists(root):
            os.makedirs(root)

        return root

    def getEpochDirectory(self, epoch):
        root = self.getRootDirectory()
        
        epochRoot = os.path.join(root, EPOCH_ROOT)
        if not os.path.exists(epochRoot):
            os.makedirs(epochRoot)

        epochDir = os.path.join(epochRoot, str(epoch))
        if not os.path.exists(epochDir):
            os.makedirs(epochDir)

        return epochDir

    def writeHyperparameters(self, obj):
        jsonObj = json.dumps(obj, indent=4)
        path = os.path.join(self.getRootDirectory(), HYPERPARAMETERS_FILE)
        with open(path, mode="w", encoding=FILE_ENCODING) as fd:
            fd.write(jsonObj)

    def writeImageSamples(self, samples, path, n):
        images, _ = samples
        scaledImages = (images + 1) / 2.0 # scale from -1,1 to 0,1

        for i in range(n * n):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(scaledImages[i, :, :, 0], cmap='gray_r')

        pyplot.savefig(path)
        pyplot.close()

    def writeTargetSamples(self, samples, n=10):
        path = os.path.join(self.getRootDirectory(), TARGET_SAMPLES_FILE)
        self.writeImageSamples(samples, path, n)

    def writeGeneratedSamples(self, epoch, samples, n=10):
        path = os.path.join(self.getEpochDirectory(epoch), GENERATED_SAMPLES_FILE)
        self.writeImageSamples(samples, path, n)

    def writeSummary(self, epoch, obj):
        jsonObj = json.dumps(obj, indent=4)
        path = os.path.join(self.getEpochDirectory(epoch), SUMMARY_FILE)
        with open(path, mode="w", encoding=FILE_ENCODING) as fd:
            fd.write(jsonObj)

    def writeMetrics(self, epoch:int, metrics:list):
        path = os.path.join(self.getEpochDirectory(epoch), METRICS_FILE)
        with open(path, mode="w", newline="\n") as fd:
            writer = csv.writer(fd)
            # writer.writerow(metrics)
            writer.writerows(metrics)

    def writeLossAndAccuracy(self, epoch, obj):
        pyplot.subplot(2, 1, 1)
        pyplot.plot(obj["dLossReal"], label='D-Loss-Real')
        pyplot.plot(obj["dLossFake"], label='D-Loss-Fake')
        pyplot.plot(obj["gLoss"], label='G-Loss')
        pyplot.legend()

        pyplot.subplot(2, 1, 2)
        pyplot.plot(obj["accReal"], label='Acc-Real')
        pyplot.plot(obj["accFake"], label='Acc-Fake')
        pyplot.legend()

        path = os.path.join(self.getEpochDirectory(epoch), LOSS_ACCURACY_FILE)
        pyplot.savefig(path)
        pyplot.close()
