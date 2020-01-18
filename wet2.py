import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal
import scipy.misc
import scipy.io as scio
from scipy.sparse import csgraph
from scipy.linalg import circulant
import PIL
from PIL import Image
import sklearn
from sklearn.decomposition import PCA
import sklearn.neighbors
import cv2

ALPHA = 5  # may choose alpha


def downsample(highSampled, alpha=ALPHA):
    (xSize, ySize) = highSampled.shape
    downsampled = np.zeros((int(xSize / alpha), int(ySize / alpha)))
    for i in range(downsampled.shape[0]):
        for j in range(downsampled.shape[1]):
            downsampled[i, j] = highSampled[alpha * i, alpha * j]
    return downsampled


class imageVersions:
    def __init__(self, imgArr, windowSize=15):
        self.evenlySpacedUnion = np.linspace(-(windowSize / 2), windowSize / 2, windowSize)

        self.gaussianKernel = self.__gaussianMatrix()
        # plt.imshow(self.gaussianKernel, cmap='gray')
        # plt.title("Real gaussian PSF")
        # plt.show()

        self.gaussianImg = signal.convolve2d(imgArr, self.gaussianKernel, boundary='wrap')
        self.sincImg = signal.convolve2d(imgArr, self.__sincMatrix())

        # first practical assignment:
        self.lowResGaussian = downsample(self.gaussianImg)  # gaussian_img
        self.lowResSinc = downsample(self.sincImg)



        #############
        gaussian_restored_true = wiener_filter(self.gaussianImg, self.gaussianKernel, 0)
        downsampled_low_res_gaussian = downsample(self.lowResGaussian)
        gaussian_img_high_res = upsample_matrix(self.lowResGaussian)
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
        self.r = self.__createPatches(imgArr, patchSize, 1)  # high ==> no use of alpha

        self.q = self.__createPatches(imgArr, patchSize)
        self.qVec = [patch.reshape(patch.size) for patch in self.q]

        self.qPCA = self.__pca()


    def __pca(self):
        pca = PCA(n_components=15, svd_solver='full')
        pca.fit(self.qVec)
        return pca.transform(self.qVec)

    def __createPatches(self, imgArr, size, alpha=ALPHA):
        size = int(size / alpha)
        step = int(size / 2)
        patches = []
        for i in range(0, imgArr.shape[0] - size, step):  # -size -> -step ?
            for j in range(0, imgArr.shape[1] - size, step):  # -size -> -step ?
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
    def __init__(self, allPatches):
        self.allPatches = allPatches
        self.k = self.__iterativeAlgorithm(allPatches.r[0].__len__())

    def __iterativeAlgorithm(self, patchSize):  # כרגע ההתחלה של איטרציה אחת
        # init k with delta
        k = fftpack.fftshift(scipy.signal.unit_impulse((patchSize, patchSize)))
        Rj = RjCalculator(k).getRj(self.allPatches.r)
        neighborsWeights = self.__calculateWeights(k)

    def __calculateWeights(self, k):
        sigmaNN = 0.01
        distWeights = np.zeros((len(self.allPatches.q), len(self.allPatches.r)))
        for j in range(distWeights.shape[1]):
            rAlpha = downsample(signal.convolve2d(self.allPatches.r[j], k, mode='same'))
            for i in range(distWeights.shape[0]):
                distWeights[i, j] = np.exp(-0.5 * (np.linalg.norm(self.allPatches.q[i] - rAlpha) ** 2) / (sigmaNN ** 2))
        totalColumnsWeights = np.expand_dims(np.sum(distWeights, axis=0), axis=0)
        return np.divide(distWeights, totalColumnsWeights)


########
def downsample_shrink_matrix_1d(mat, alpha):
    # print(mat.shape)
    (mat_shape_x, mat_shape_y) = mat.shape
    new_size = (int(mat_shape_x / alpha), int(mat_shape_y))
    downsampled = np.zeros(new_size)
    for i in range(new_size[0]):
        downsampled[i, :] = mat[alpha * i, :]
    return downsampled


def upsample_matrix(mat, alpha=ALPHA):
    (mat_shape_x, mat_shape_y) = mat.shape
    new_size = (int(mat_shape_y * alpha), int(mat_shape_x * alpha))
    # FILTER = PIL.Image.BILINEAR
    upsampled_filtered_image = cv2.resize(mat, dsize=new_size, interpolation=cv2.INTER_CUBIC)
    # upsampled_filtered_image = mat.resize(new_size, resample=FILTER)
    return upsampled_filtered_image


## creates a laplacian matrix
def laplacian(window_size, range):
    x = np.linspace(-range, range, window_size)
    xx = np.outer(x, x)
    return csgraph.laplacian(xx, normed=False)


def wiener_filter(image, psf, k):
    image_dft = fftpack.fft2(image)
    psf_dft = fftpack.fft2(fftpack.ifftshift(psf), shape=image_dft.shape)
    filter_dft = np.conj(psf_dft) / (np.abs(psf_dft) ** 2 + k)
    recovered_dft = image_dft * filter_dft
    return np.real(fftpack.ifft2(recovered_dft))


###########


def main():
    t0 = time.time()

    imgArr = np.array(plt.imread("DIPSourceHW2.png"))[:, :, 0]
    imgArr /= imgArr.sum()
    filteredImage = imageVersions(imgArr)
    patchSize = 12  # how to decide?
    allPatches = patches(imgArr, patchSize)
    optimalK = kCalculator(allPatches).k

    ###########
    Wiener_Filter_Constant = 0.01
    num_neighbors = 11
    T = 16
    file_name = "patch20_alpha4_half_step_size_nn11_pca15"
    ###########

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
