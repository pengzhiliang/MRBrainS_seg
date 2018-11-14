import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
def dense_crf(img, prob):
    '''
    input:
      img: numpy array of shape (num of channels, height, width)
      prob: numpy array of shape (9, height, width), neural network last layer sigmoid output for img

    output:
      res: (height, width)

    Modified from:
      http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/
      https://github.com/yt605155624/tensorflow-deeplab-resnet/blob/e81482d7bb1ae674f07eae32b0953fe09ff1c9d1/inference_crf.py
    '''

    img = np.swapaxes(img, 0, 2)
    # img.shape: (width, height, num of channels)(224,224,3)

    num_iter = 50

    prob = np.swapaxes(prob, 1, 2)  # shape: (1, width, height) (9,224,224)
    num_classes = 9 #2

    d = dcrf.DenseCRF2D(img.shape[0] , img.shape[1], num_classes)

    unary = unary_from_softmax(prob)  # shape: (num_classes, width * height)
    unary = np.ascontiguousarray(unary)
    img = np.ascontiguousarray(img,dtype=np.uint8)

    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=img, compat=3)

    Q = d.inference(num_iter)  # set the number of iterations
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
    # res.shape: (width, height)

    res = np.swapaxes(res, 0, 1)  # res.shape:    (height, width)
    # res = res[np.newaxis, :, :]   # res.shape: (1, height, width)

    # func_end = time.time()
    # print('{:.2f} sec spent on CRF with {} iterations'.format(func_end - func_start, num_iter))
    # about 2 sec for a 1280 * 960 image with 5 iterations
    return res