import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal
from scipy.linalg import circulant
import scipy.misc
import sklearn
import sklearn.neighbors
import cv2

ALPHA = 3  # may choose alpha


def downsample(highSampled, alpha=ALPHA):
    (xSize, ySize) = highSampled.shape
    downsampled = np.zeros((int(xSize / alpha), int(ySize / alpha)))
    for i in range(downsampled.shape[0]):
        for j in range(downsampled.shape[1]):
            downsampled[i, j] = highSampled[alpha * i, alpha * j]
    return downsampled


def normalizeByRows(distWeights):
    neighborsWeightsSum = np.sum(distWeights, axis=1)
    for row in range(distWeights.shape[0]):
        rowSum = neighborsWeightsSum[row]
        if rowSum:
            distWeights[row] = distWeights[row] / rowSum
    return distWeights



def plotResults(restoredImg, blurredWith, restoredWith, origImg):
    plt.imshow(restoredImg, cmap='gray')
    PSNR = psnr(origImg[5:-5, 5:-5], restoredImg[5:-5, 5:-5])
    plt.title(f'image blurred with {blurredWith} and restored with {restoredWith}. PSNR={PSNR:.2f}')
    plt.show()


def psnr(orig, recycled):
    return 20 * np.log10(orig.max() / (np.sqrt(np.mean((orig - recycled) ** 2))))


class imageVersions:
    def __init__(self, imgArr, windowSize=15):

        self.gaussianKernel = self.__gaussianMatrix(np.linspace(-(windowSize / 16), windowSize / 16, windowSize))
        plt.imshow(self.gaussianKernel, cmap='gray')
        plt.title("Real gaussian PSF")
        plt.show()
        self.sincKernel = self.__sincMatrix(np.linspace(-(windowSize / 4), windowSize / 4, windowSize))

        self.gaussianImg = signal.convolve2d(imgArr, self.gaussianKernel, mode='same', boundary='wrap')
        self.sincImg = signal.convolve2d(imgArr, self.sincKernel, mode='same', boundary='wrap')

        # first practical assignment:
        self.lowResGaussian = downsample(self.gaussianImg)
        self.lowResSinc = downsample(self.sincImg)

        plt.title('low-res gaus-image')
        plt.imshow(self.lowResGaussian, cmap='gray')
        plt.show()

        plt.title('low-res sinc-image')
        plt.imshow(self.lowResSinc, cmap='gray')
        plt.show()


    def restoreSinc(self, k):
         return self.__wienerFilterToUpsample(self.lowResSinc, k)

    def restoreGaussian(self, k):
        return self.__wienerFilterToUpsample(self.lowResGaussian, k)


    def __gaussianMatrix(self, evenlySpacedUnion):
        mu = 0
        sigma = 1
        x, y = np.meshgrid(evenlySpacedUnion, evenlySpacedUnion)
        d = np.sqrt(x ** 2 + y ** 2)
        gaussian = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        return gaussian / gaussian.sum()

    def __sincMatrix(self, evenlySpacedUnion):
        xxMatrix = np.outer(evenlySpacedUnion, evenlySpacedUnion)
        sinc = np.sinc(xxMatrix)
        return sinc / sinc.sum()

    def __wienerFilterToUpsample(self, lowSampled, psf, k=0.1, alpha=ALPHA):
        if np.sum(psf):
            psf /= np.sum(psf)
        (xSize, ySize) = lowSampled.shape
        newSize = (int(ySize * alpha), int(xSize * alpha))
        img = cv2.resize(lowSampled, dsize=newSize, interpolation=cv2.INTER_CUBIC)
        psf = fftpack.fft2(psf, shape=img.shape)
        psf = np.conj(psf) / (np.abs(psf) ** 2 + k)
        return np.abs(fftpack.ifft2(fftpack.fft2(np.copy(img)) * psf))


class patches:
    def __init__(self, imgArr, patchSize):
        self.patchSize = patchSize
        self.r = self.__createPatches(imgArr, patchSize, 1)  # no use of alpha

        self.q = self.__createPatches(imgArr, patchSize)
        self.qVec = np.array([patch.reshape(patch.size) for patch in self.q])

        # self.qPCA = self.__pca()
        self.Rj = RjCalculator(self.r).getRj()

    def __createPatches(self, imgArr, size, alpha=ALPHA):
        size = int(size / alpha)
        step = int(size * alpha / self.patchSize)
        patches = []
        for i in range(0, imgArr.shape[0] - size, step):
            for j in range(0, imgArr.shape[1] - size, step):
                patches.append(imgArr[i:i + size, j:j + size])
        return patches

    def calcSumElement(self, i, j):
        return self.Rj[j].T @ self.Rj[j], self.Rj[j].T @ self.qVec[i]

    def calculateWeights(self, k, sigmaNN):
        numNeighbors = 5
        rAlpha = np.array([element @ k for element in self.Rj])
        tree = sklearn.neighbors.BallTree(rAlpha, leaf_size=2)
        distWeights = np.zeros((len(self.q), len(self.r)))

        for i, qi in enumerate(self.qVec):
            _, neighbor_indices = tree.query(np.expand_dims(qi, 0), k=numNeighbors)
            for j in neighbor_indices:
                distWeights[i, j] = np.exp(-0.5 * (np.linalg.norm(qi - rAlpha[j]) ** 2) / (sigmaNN ** 2))
        return normalizeByRows(distWeights)


class RjCalculator:
    def __init__(self, rPatches):
        self.rPatches = rPatches

    def getRj(self):
        return [self.__RjElement(patch) for patch in self.rPatches]

    def __RjElement(self, patch, alpha=ALPHA):
        return self.__downsampleShrinkMatrix1d(circulant(patch.reshape(patch.size)), alpha ** 2)

    def __downsampleShrinkMatrix1d(self, highSampled, alpha):
        (xSize, ySize) = highSampled.shape
        newSize = (int(xSize / alpha), int(ySize))
        downsampled = np.zeros(newSize)
        for i in range(newSize[0]):
            downsampled[i, :] = highSampled[alpha * i, :]
        return downsampled


class kCalculator:
    def __init__(self, allPatches):
        self.allPatchesFunctor = allPatches
        self.k = self.__iterativeAlgorithm(allPatches.r[0].__len__())

    def __iterativeAlgorithm(self, patchSize):
        CSquared = self.__squaredLaplacian(patchSize)
        # init k with delta
        delta = fftpack.fftshift(scipy.signal.unit_impulse((patchSize, patchSize)))
        k = delta.reshape(delta.size)
        for t in range(5):
            k = self.__oneIteration(k, CSquared)
        return k.reshape((patchSize, patchSize))

    def __oneIteration(self, k, CSquared):
        sigmaNN = 0.06
        neighborsWeights = self.allPatchesFunctor.calculateWeights(k, sigmaNN)
        size = k.shape[0]
        matEpsilon = np.ones((size, size)) * 1e-10
        sumLeft = np.zeros((size, size))
        sumRight = np.zeros_like(k)

        for i in range(neighborsWeights.shape[0]):
            for j in range(neighborsWeights.shape[1]):
                if neighborsWeights[i, j]:
                    left, right = self.allPatchesFunctor.calcSumElement(i, j)
                    sumLeft += neighborsWeights[i, j] * left + CSquared
                    sumRight += neighborsWeights[i, j] * right

        return np.linalg.inv(sumLeft * (sigmaNN ** -2) + matEpsilon) @ sumRight

    def __squaredLaplacian(self, length):

        val1 = -1
        val2 = 4

        # digonals init
        diag1 = np.zeros((length, length))
        diag2 = np.zeros((length, length))

        for i in range(length):
            if (i + 1) in range(length):
                diag1[i, i + 1] = val1
            if (i - 1) in range(length):
                diag1[i, i - 1] = val1

        for i in range(length):
            diag1[i, i] = val2
            diag2[i, i] = val1

        # first diagonal
        start = 0
        end = length
        C = np.zeros((length ** 2, length ** 2))

        for num_matrices in range(length):
            C[start: end, start: end] = diag1
            start += length
            end += length

        # second diagonal
        startX = 0
        endXstartY = length
        endY = 2 * length

        for num_matrices in range(length - 1):
            C[startX: endXstartY, endXstartY: endY] = diag2
            C[endXstartY: endY, startX: endXstartY] = diag2
            startX += length
            endXstartY += length
            endY += length

        return C.T @ C


def main():
    t0 = time.time()

    imgArr = np.array(plt.imread("DIPSourceHW2.png"))[:, :, 0]
    imgArr /= imgArr.max()
    expandImg = np.zeros((imgArr.shape[0]+2, imgArr.shape[1]+2))
    expandImg[1:-1, 1:-1] = imgArr
    filteredImage = imageVersions(expandImg)
    patchSize = 15  # how to decide?

    gaussianPatches = patches(filteredImage.lowResGaussian, patchSize)
    gaussianOptimalK = kCalculator(gaussianPatches).k
    gaussianRestoredOptimal = filteredImage.restoreGaussian(gaussianOptimalK)
    sincRestoredNotOptimal = filteredImage.restoreSinc(gaussianOptimalK)

    sincPatches = patches(filteredImage.lowResSinc, patchSize)
    sincOptimalK = kCalculator(sincPatches).k
    sincRestoredOptimal = filteredImage.restoreSinc(sincOptimalK)
    gaussianRestoredNotOptimal = filteredImage.restoreGaussian(sincOptimalK)

    ## plot results and PSNR with original high-res image
    plotResults(gaussianRestoredOptimal, "gauss-ker", "gauss-ker", expandImg)
    plotResults(gaussianRestoredNotOptimal, "gauss-ker", "sinc-ker", expandImg)
    plotResults(sincRestoredOptimal, "sinc-ker", "sinc-ker", expandImg)
    plotResults(sincRestoredNotOptimal, "sinc-ker", "gauss-ker", expandImg)

    plt.imshow(gaussianOptimalK, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
