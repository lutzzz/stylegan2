"""DCT Fake Score (DCTFS)."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import dnnlib.tflib as tflib

from metrics import metric_base

#----------------------------------------------------------------------------

class DCTFS(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        
        # Construct TensorFlow graph.
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                classifier = tf.keras.models.load_model('nets/lutz_new_classifier_tf1.14.h5', compile=False)
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:], seed=42)
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, **Gs_kwargs)
                images = tflib.convert_images_to_uint8(images, nchw_to_nhwc=True)

        # Calculate activations for fakes.
        activations = np.empty([self.num_images, classifier.output_shape[1]], dtype=np.float32)
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            end = min(begin + minibatch_size, self.num_images)
            activations[begin:end] = classifier.predict_on_batch(images)[:end-begin]

        # Calculate DCT Fake Score.
        mean = np.mean(activations)
        std = np.std(activations)
        self._report_result(mean, suffix='_mean')
        self._report_result(std, suffix='_std')

#----------------------------------------------------------------------------
