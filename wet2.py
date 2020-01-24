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


def upsampleMatrix(lowSampled, alpha=ALPHA):
    (xSize, ySize) = lowSampled.shape
    newSize = (int(ySize * alpha), int(xSize * alpha))
    return cv2.resize(lowSampled, dsize=newSize, interpolation=cv2.INTER_CUBIC)


def wienerFilter(image, psf, k):
    imageDFT = fftpack.fft2(image)
    psf_dft = fftpack.fft2(fftpack.ifftshift(psf), shape=imageDFT.shape)
    filter_dft = np.conj(psf_dft) / (np.abs(psf_dft) ** 2 + k)
    return np.real(fftpack.ifft2(imageDFT * filter_dft))


def normalizeByRows(distWeights):
    neighborsWeightsSum = np.sum(distWeights, axis=1)
    for row in range(distWeights.shape[0]):
        rowSum = neighborsWeightsSum[row]
        if rowSum:
            distWeights[row] = distWeights[row] / rowSum
    return distWeights


class imageVersions:
    def __init__(self, imgArr, windowSize=15):
        self.evenlySpacedUnion = np.linspace(-(windowSize / 2), windowSize / 2, windowSize)

        self.gaussianKernel = self.__gaussianMatrix()
        plt.imshow(self.gaussianKernel, cmap='gray')
        plt.title("Real gaussian PSF")
        plt.show()

        self.gaussianImg = signal.convolve2d(imgArr, self.gaussianKernel, mode='same', boundary='wrap')
        self.sincImg = signal.convolve2d(imgArr, self.__sincMatrix())

        # first practical assignment:
        self.lowResGaussian = downsample(self.gaussianImg)
        self.lowResSinc = downsample(self.sincImg)

        #############
        gaussian_restored_true = wienerFilter(self.gaussianImg, self.gaussianKernel, 0.1)
        downsampled_low_res_gaussian = downsample(self.lowResGaussian)
        gaussian_img_high_res = upsampleMatrix(self.lowResGaussian)
        #############

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
        self.r = self.__createPatches(imgArr, patchSize, 1)  # no use of alpha

        self.q = self.__createPatches(imgArr, patchSize)
        self.qVec = np.array([patch.reshape(patch.size) for patch in self.q])

        # self.qPCA = self.__pca()
        self.Rj = RjCalculator(self.r).getRj()

    def __createPatches(self, imgArr, size, alpha=ALPHA):
        size = int(size / alpha)
        step = int(size / 2)
        patches = []
        for i in range(0, imgArr.shape[0] - size, step):
            for j in range(0, imgArr.shape[1] - size, step):
                patches.append(imgArr[i:i + size, j:j + size])
        return patches

    def calcSumElement(self, i, j):
        return self.Rj[j].T @ self.Rj[j], self.Rj[j].T @ self.qVec[i]

    def calculateWeights(self, k, sigmaNN):
        numNeighbors = 11
        rAlpha = np.array([element @ k for element in self.Rj])
        tree = sklearn.neighbors.BallTree(rAlpha, leaf_size=2)
        distWeights = np.zeros((len(self.q), len(self.r)))

        for i, qi in enumerate(self.qVec):
            _, neighbor_indices = tree.query(np.expand_dims(qi, 0), k=numNeighbors)
            for j in neighbor_indices:
                distWeights[i, j] = np.exp(-0.5 * (np.linalg.norm(qi - rAlpha[j]) ** 2) / (sigmaNN ** 2))
        return normalizeByRows(distWeights)
    # def __pca(self):
    #     pca = PCA(n_components=25, svd_solver='full')
    #     pca.fit(self.qVec)
    #     return pca.transform(self.qVec)


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
    #
    # def __sameSizeDownsampleWithZeros(self, mat, alpha=ALPHA):  # may choose alpha
    #     for i in range(mat.shape[0]):
    #         if (i % alpha):
    #             mat[i, :] = 0  # line i is zeroes
    #             mat[:, i] = 0  # row i is zeroes
    #     return mat


class kCalculator:
    def __init__(self, allPatches):
        self.allPatches = allPatches
        self.k = self.__iterativeAlgorithm(allPatches.r[0].__len__())

    def __iterativeAlgorithm(self, patchSize):
        CSquared = self.__squaredLaplacian(patchSize)
        # init k with delta
        delta = fftpack.fftshift(scipy.signal.unit_impulse((patchSize, patchSize)))
        k = delta.reshape(delta.size)
        for t in range(16):
            k = self.__oneIteration(k, CSquared)

            # curr_k_image = k.reshape((patchSize, patchSize))
            # plt.imshow(curr_k_image, cmap='gray')
            # plt.title(f'curr_k as an image ')
            # plt.show()

            # copied code, can be fitted, preferable in another func:
            # Wiener_Filter_Constant = 0.01
            # # print(f'curr_k shape: {curr_k.shape}')
            # factor = 1
            # for power in range(0, 4):
            #     gaussian_restored = wiener_filter(gaussian_img_high_res, curr_k.reshape((patchSize, patchSize)),
            #                                       Wiener_Filter_Constant * factor)
            #
            #     plt.imshow(gaussian_restored, cmap='gray')
            #     plt.title(
            #         f'restoration after iteration number: {t}, with wiener factor: {Wiener_Filter_Constant * factor}')
            #     plt.show()
            #     factor *= 10
            #     if (t == 7) and factor == 100:
            #         save_as_img(gaussian_restored, title_name, my_format)
            #         curr_k_image = curr_k.reshape((patch_size, patch_size))
            #
            #         save_as_img(curr_k_image, title_name + "_kernel", my_format)
        return k

    def __oneIteration(self, k, CSquared):
        sigmaNN = 0.1
        neighborsWeights = self.allPatches.calculateWeights(k, sigmaNN)
        size = k.shape[0]
        matEpsilon = np.ones((size, size)) * 1e-10
        sumLeft = np.zeros((size, size))
        sumRight = np.zeros_like(k)

        for i in range(neighborsWeights.shape[0]):
            for j in range(neighborsWeights.shape[1]):
                if neighborsWeights[i, j]:
                    left, right = self.allPatches.calcSumElement(i, j)
                    sumLeft += neighborsWeights[i, j] * left + CSquared
                    sumRight += neighborsWeights[i, j] * right

        return np.linalg.inv(sumLeft * (sigmaNN ** -2) + matEpsilon) @ sumRight

    def __squaredLaplacian(self, length):
        C = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        Cvec = np.zeros(length * length)
        rows_length_diff = abs(length - C.shape[1])
        index = 0
        for row in range(C.shape[0]):
            for col in range(C.shape[1]):
                Cvec[index] = C[row, col]
                index += 1
            for i in range(rows_length_diff):
                Cvec[index] = 0
                index += 1
        C = circulant(Cvec)
        return C.T @ C


def main():
    t0 = time.time()

    imgArr = np.array(plt.imread("DIPSourceHW2.png"))[:, :, 0]
    imgArr /= imgArr.max()
    filteredImage = imageVersions(imgArr)
    patchSize = 15  # how to decide?
    allPatches = patches(imgArr, patchSize)
    optimalK = kCalculator(allPatches).k

    curr_k_image = optimalK.reshape((patchSize, patchSize))
    plt.imshow(curr_k_image, cmap='gray')
    plt.show()

    plt.title('original')
    plt.imshow(imgArr, cmap='gray')
    plt.show()

    plt.title('gaussian')
    plt.imshow(filteredImage.gaussianImg, cmap='gray')
    plt.show()

    plt.title('gaussian + downsampled')
    plt.imshow(filteredImage.lowResGaussian, cmap='gray')
    plt.show()

    plt.title('sinc')
    plt.imshow(filteredImage.sincImg, cmap='gray')
    plt.show()

    plt.title('sinc + downsampled')
    plt.imshow(filteredImage.lowResSinc, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
