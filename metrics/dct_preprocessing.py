import tensorflow as tf
import numpy as np
import os

def dct2tf(array, batched=True):
    """DCT-2D transform of an array, by first applying dct-2 along rows then columns

    Arguments:
        array - The array to transform.
        batched - Is the input batched?

    Returns:
        DCT2D transformed array.
    """
    dtype = array.dtype
    array = tf.cast(array, tf.float32)

    if batched:
        # tensorflow computes over last axis (-1)
        # layout (B)atch, (R)ows, (C)olumns, (V)alue
        # BRCV
        array = tf.transpose(array, perm=[0, 3, 2, 1])
        # BVCR
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 1, 3, 2])
        # BVRC
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 2, 3, 1])
        # BRCV
    else:
        # RCV
        array = tf.transpose(array, perm=[2, 1, 0])
        # VCR
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[0, 2, 1])
        # VRC
        array = tf.signal.dct(array, type=2, norm="ortho")
        array = tf.transpose(array, perm=[1, 2, 0])
        # RCV

    array = tf.cast(array, dtype)

    return array

#----------------------------------------------------------------------------

def scale_by_absolute(image, current_max):
    '''Scale each frequency by its max value
    '''
    return image / current_max

#----------------------------------------------------------------------------

def normalize(image, mean, std):
    image = (image - mean) / std
    return image

#----------------------------------------------------------------------------

class PreprocessImages(tf.keras.layers.Layer):
    '''Convert, scale and normalize image
    '''
    def __init__(self, maxima=None, mean=None, var=None, **kwargs):
        kwargs["dtype"] = "float64"
        super(PreprocessImages, self).__init__(**kwargs)
        assert all(isinstance(arg, (np.ndarray, None)) for arg in (maxima, mean, var))
        self.scale = maxima is not None
        self.norm = mean is not None or var is not None

        weights = []

        if self.norm:
            assert not (mean is None) ^ (var is None), "both mean and std must exist"
            assert mean.shape == var.shape
            std = np.sqrt(var)
            self.mean = self.add_weight("mean", mean.shape, trainable=False)
            self.std = self.add_weight("std", std.shape, trainable=False)
            weights += [mean, std]
        if self.scale:
            if self.norm: assert mean.shape == maxima.shape
            self.maxima = self.add_weight("maxima", maxima.shape, trainable=False)
            weights += [maxima]
        
        self.set_weights(weights)

    def call(self, inputs):
        assert inputs.dtype == tf.float64
        outputs = dct2tf(inputs)
        if self.scale:
            outputs = scale_by_absolute(outputs, self.maxima)
        if self.norm:
            outputs = normalize(outputs, self.mean, self.std)
        return outputs

#----------------------------------------------------------------------------

def load_preprocessing_params(load_path):
    '''Load preprocessing parameters from mean.npy, var.npy, max.npy
    '''
    mean = var = maxima = None
    paths = [os.path.join(load_path, x) for x in ("mean.npy", "var.npy", "max.npy")]
    mean, var, maxima = [np.load(path) for path in paths if os.path.isfile(path)]
    return mean, var, maxima

#----------------------------------------------------------------------------

def save_preprocessing_params(params, save_path):
    '''Save preprocessing parameters to mean.npy, var.npy, max.npy
    '''
    assert isinstance(params, (tuple, list)) and len(params) == 3
    assert all(isinstance(x, (np.ndarray, None)) for x in params)

    paths = [os.path.join(save_path, x) for x in ("mean.npy", "var.npy", "max.npy")]
    for path, param in zip(paths, params):
        if param is not None: np.save(path, param)

#----------------------------------------------------------------------------

def add_preprocessing(model, load_preproc_from):
    '''Returns a model with initialized preprocessing layer.

    load_preproc_from: preprocessing params str or ndarrays (mean, var, maxima)
    '''
    if isinstance(load_preproc_from, str):
        assert os.path.isdir(load_preproc_from)
        mean, var, maxima = load_preprocessing_params(load_preproc_from)
    else:
        assert isinstance(load_preproc_from, (list, tuple)) and len(load_preproc_from) == 3

    # Add Preprocessing
    model = tf.keras.Sequential([
        tf.keras.Input(shape=model.input_shape[1:], dtype=tf.float64),
        PreprocessImages(maxima=maxima, mean=mean, var=var), 
        model])
    return model

#----------------------------------------------------------------------------