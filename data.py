from numpy import ones, zeros, asarray
from numpy.random import randint
from numpy.random import randn
from tensorflow import compat, executing_eagerly
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.models import Model

class Data():
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("subset", 0)

        self.classes = 10
        self.imageDim = 7
        self.imageShape = (28,28,1)
        self.dataSubset = kwargs["subset"]
        self.dataset = self.loadDataset()

    def getDatasetShape(self):
        images, _ = self.dataset
        return images.shape[0]

    def loadDataset(self):
        (x_train, y_train), (_, _) = load_data()
        
        if self.dataSubset > 0:
            ix = randint(0, x_train.shape[0], self.dataSubset)
            x_train, y_train = x_train[ix], y_train[ix]

        X = expand_dims(x_train, axis=-1)
        X = X.numpy() if executing_eagerly() else X.eval(session=compat.v1.Session())
        X = X.astype('float32')
        X = (X - 127.5) / 127.5 # scale from 0,255 to -1,1
        return [X, y_train]

    def generateRealTrainingSamples(self, samples):
        images, labels = self.dataset
        ix = randint(0, images.shape[0], samples)
        X, labels = images[ix], labels[ix]
        y = ones((samples, 1))
        return [X, labels], y

    def generateFakeTrainingSamples(self, generator:Model, latentDim, samples, random=True):
        if random == True:
            x, labels = self.generateLatentPointsAndRandomLabels(latentDim, samples)
        else:
            x, labels = self.generateLatentPointsAndOrderedLabels(latentDim, samples)

        X = generator.predict([x, labels])
        y = zeros((samples, 1))
        return [X, labels], y

    def generateFakeTrainingGanSamples(self, latentDim, samples):
        X, labels = self.generateLatentPointsAndRandomLabels(latentDim, samples)
        y = ones((samples, 1))
        return [X, labels], y

    def generateLatentPointsAndRandomLabels(self, latentDim, samples):
        x = self.generateLatentPoints(latentDim, samples)
        labels = randint(0, self.classes, samples)
        return [x, labels]

    def generateLatentPointsAndOrderedLabels(self, latentDim, samples):
        x = self.generateLatentPoints(latentDim, samples)
        labels = asarray([x for _ in range(int(samples/self.classes)) for x in range(self.classes)])
        return [x, labels]

    def generateLatentPoints(self, latentDim, samples):
        x = randn(latentDim * samples)
        x = x.reshape((samples, latentDim))
        return x