import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal
import scipy.misc
import scipy.io as scio
from scipy.sparse import csgraph
from PIL import Image

ALPHA = 5  # may choose alpha


def downsample(highSampled, alpha=ALPHA):  # may choose alpha
    return [[highSampled[y * alpha, x * alpha] for x in range(int(highSampled.shape[1] / alpha))] for y in
            range(int(highSampled.shape[0] / alpha))]


class imageVersions:
    def __init__(self, imgArr, evenlySpacedUnion=np.linspace(-5, 5, 6)):
        self.evenlySpacedUnion = evenlySpacedUnion
        self.gaussianImg = signal.convolve2d(imgArr, self.__gaussianMatrix())
        self.sincImg = signal.convolve2d(imgArr, self.__sincMatrix())

        # first practical assignment:
        self.lowResGaussian = downsample(self.gaussianImg)
        self.lowResSinc = downsample(self.sincImg)

    def __gaussianMatrix(self):
        mu = 0
        sigma = 1
        x, y = np.meshgrid(self.evenlySpacedUnion, self.evenlySpacedUnion)
        d = np.sqrt(x ** 2 + y ** 2)
        return np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    def __sincMatrix(self):
        xxMatrix = np.outer(self.evenlySpacedUnion, self.evenlySpacedUnion)
        return np.sinc(xxMatrix)


class patches:
    def __init__(self, imgArr, patchSize):
        self.r = self.__createPatches(imgArr, patchSize, 1)  # high ==> no use of alpha
        self.q = self.__createPatches(imgArr, patchSize)

    def __createPatches(self, imgArr, size, alpha=ALPHA):  # may choose alpha
        size = int(size / alpha)
        patches = []
        for i in range(0, imgArr.shape[0] - size, size):
            for j in range(0, imgArr.shape[1] - size, size):
                patches.append(imgArr[i:i + size, j:j + size])
        return patches


class RjCalculator:
    def __init__(self, k):
        self.k = k
        # additing epsilon diagonal to make sure it's invertible
        self.kInv = np.linalg.inv(k + np.eye(k.shape[0]) * 1e-10)

    def getRj(self, rPatches):
        return [self.__RjElement(patch) for patch in rPatches]

    ## return downsample(downsampleZeros(conv(k,rj))*kInv)
    def __RjElement(self, patch):
        k_rj = signal.convolve2d(self.k, patch, mode='same')
        downsampleZeros_k_rj = self.__sameSizeDownsampleWithZeros(k_rj)
        return downsample(downsampleZeros_k_rj.__matmul__(self.kInv))

    def __sameSizeDownsampleWithZeros(self, mat, alpha=ALPHA):  # may choose alpha
        for i in range(mat.shape[0]):
            if (i % alpha):
                mat[i, :] = 0  # line i is zeroes
                mat[:, i] = 0  # row i is zeroes
        return mat


class kCalculator:
    def __init__(self, allPatches, patchSize):
        self.allPatches = allPatches
        self.k = self.__iterativeAlgorithm(patchSize)

    def __iterativeAlgorithm(self, patchSize):  # כרגע ההתחלה של איטרציה אחת
        # init k with delta
        k = fftpack.fftshift(scipy.signal.unit_impulse((patchSize, patchSize)))
        Rj = RjCalculator(k).getRj(self.allPatches.r)
        neighborsWeights = self.__calculateWeights(k)


    def __calculateWeights(self, k):
        sigmaNN = 1
        distWeights = np.zeros((len(self.allPatches.q), len(self.allPatches.r)))
        for j in range(distWeights.shape[1]):
            rAlpha = downsample(signal.convolve2d(self.allPatches.r[j], k, mode='same'))
            for i in range(distWeights.shape[0]):
                distWeights[i, j] = np.exp(-0.5 * (np.linalg.norm(self.allPatches.q[i] - rAlpha) ** 2) / (sigmaNN ** 2))
        totalColumnsWeights = np.expand_dims(np.sum(distWeights, axis=0), axis=0)
        return np.divide(distWeights, totalColumnsWeights)


def main():
    t0 = time.time()

    imgArr = np.array(plt.imread("DIPSourceHW2.png"))[:, :, 0]
    filteredImage = imageVersions(imgArr)
    patchSize = 60  # how to decide?
    allPatches = patches(imgArr, patchSize)
    optimalK = kCalculator(allPatches, patchSize)






    plt.title('original')
    plt.imshow(imgArr, cmap='gray')
    plt.show()

    plt.title('gaussian')
    plt.imshow(filteredImage.gaussianImg, cmap='gray')
    plt.show()

    plt.title('gaussian + downsampled')
    plt.imshow(filteredImage.lowResGaussian, cmap='gray')
    plt.show()

    #
    plt.title('sinc')
    plt.imshow(filteredImage.sincImg, cmap='gray')
    plt.show()
    #
    plt.title('sinc + downsampled')
    plt.imshow(filteredImage.lowResSinc, cmap='gray')
    plt.show()







if __name__ == "__main__":
    main()
