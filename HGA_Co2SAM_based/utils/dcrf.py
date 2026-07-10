# ==============================================================================
# Standard DenseCRF wrapper for WSSS.
# Adapted from common open-source WSSS implementations.
# Uses the 'pydensecrf' library under the hood.
# ==============================================================================
# utils/dcrf.py
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels
import pydensecrf.utils as utils
import numpy as np

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    """
    Perform DenseCRF inference using soft probability maps as the unary potential.

    This method refines the continuous multi-class logit probabilities predicted by the 
    network. It constructs a pairwise Gaussian potential for spatial smoothing and a 
    pairwise bilateral potential guided by RGB color gradients to enforce sharp boundary 
    alignment along true image edges.

    Args:
        img (np.ndarray): Source RGB image array of shape [H, W, 3] in uint8 format.
        probs (np.ndarray): Softmax probability map of shape [C, H, W] from the model output.
        t (int): Number of mean-field approximation iterations (default: 10).
        scale_factor (float or int): Coordinate normalization factor for spatial kernels, 
            used to scale coordinate distances relative to image resolution.
        labels (int): Total number of semantic classes including background (default: 21).

    Returns:
        refined_probs (np.ndarray): Posterior probability map of shape [C, H, W] 
            after CRF optimization.
    """
    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img_c), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):
    """
    Perform DenseCRF inference using discrete hard labels to synthesize the unary potential.

    This method is typically utilized during offline pseudo-label generation. Since discrete 
    label indices lack raw continuous probability margins, this function constructs a 
    probabilistic distribution by assigning a dominant prior confidence (gt_prob) to the 
    target class and uniformly distributing the remaining probability mass among other classes.

    Args:
        img (np.ndarray): Source RGB image array of shape [H, W, 3] in uint8 format.
        labels (np.ndarray): 2D discrete label map of shape [H, W] containing class indices.
        t (int): Number of mean-field approximation iterations (default: 10).
        n_labels (int): Total number of semantic classes including background (default: 21).
        gt_prob (float): Prior confidence probability assigned to the provided label indices, 
            typically in the range [0.0, 1.0] (default: 0.7).

    Returns:
        refined_label (np.ndarray): 2D refined discrete class label map of shape [H, W] 
            obtained via argmax optimization.
    """
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)

class DenseCRF(object):
    """
    Object-oriented processor wrapping the DenseCRF2D optimization pipeline.

    This class encapsulates standard DenseCRF hyperparameters, allowing the user to 
    instantiate a pre-configured post-processing engine. It is optimized for batched 
    evaluation loops, applying consistent bilateral and spatial potentials across 
    multiple image-probability pairs.

    Attributes:
        iter_max (int): Number of mean-field approximation iterations.
        pos_w (float or int): Weight parameter for the pairwise spatial Gaussian kernel.
        pos_xy_std (float or int): Standard deviation of the spatial coordinates, 
            controlling the smoothing range.
        bi_w (float or int): Weight parameter for the pairwise bilateral kernel.
        bi_xy_std (float or int): Spatial standard deviation of the bilateral kernel, 
            controlling spatial range of color-sensitive smoothing.
        bi_rgb_std (float or int): Color standard deviation of the bilateral kernel, 
            controlling sensitivity to pixel value variations.
    """
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        """
        Initialize the DenseCRF wrapper with preset kernels and weights.

        Args:
            iter_max (int): Number of mean-field approximation iterations for optimization.
            pos_w (float or int): Weight of the pairwise spatial Gaussian kernel (controls spatial smoothing).
            pos_xy_std (float or int): Standard deviation of coordinates in the spatial Gaussian kernel.
            bi_w (float or int): Weight of the pairwise bilateral kernel (controls appearance-based grouping).
            bi_xy_std (float or int): Spatial standard deviation of coordinates in the bilateral kernel.
            bi_rgb_std (float or int): Color/RGB intensity standard deviation in the bilateral kernel.
        """
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        """
        Execute DenseCRF optimization on the input image and its soft probability map.

        Args:
            image (np.ndarray): Source RGB image array of shape [H, W, 3] in uint8 format.
            probmap (np.ndarray): Softmax probability map of shape [C, H, W] from the model.

        Returns:
            refined_probs (np.ndarray): Optimized posterior probability map of shape [C, H, W].
        """
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q