from data import Data
from matplotlib import pyplot
from tensorflow.keras.models import load_model

class Inference:
    def __init__(self, data:Data, modelPath:str, latentDim:int) -> None:
        self.data = data
        self.modelPath = modelPath
        self.latentDim = latentDim
        self.samples = 100
        self.evalDirectoryName = 'eval'

    def run(self):
        model = load_model(self.modelPath)
        input = self.data.generateLatentPointsAndOrderedLabels(self.latentDim, self.samples)
        output = model.predict(input)
        self.plotImageSamples(output)

    def plotImageSamples(self, images, n=10):
        scaledImages = (images + 1) / 2.0 # scale from -1,1 to 0,1

        for i in range(n * n):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(scaledImages[i, :, :, 0], cmap='gray_r')

        filename = f'{self.evalDirectoryName}/generated_samples.png'
        pyplot.savefig(filename)
        pyplot.close()