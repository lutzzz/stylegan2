"""DCT Fake Score (DCTFS)."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import dnnlib.tflib as tflib
import os

from metrics import metric_base
from metrics.dct_preprocessing import add_preprocessing
from training import misc

#----------------------------------------------------------------------------

class DCTFS(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        classifier = tf.keras.models.load_model('nets/lutz_new_classifier_tf1.14.h5', compile=False)
        classifier = add_preprocessing(classifier, "nets")
        # if num_gpus > 1: classifier = tf.keras.utils.multi_gpu_model(classifier, num_gpus, cpu_relocation=True) # Runs with undeterministic output
        activations = np.zeros([self.num_images, classifier.output_shape[1]], dtype=np.float32)

        # Calculate statistics for reals (adversarial examples).
        cache_file = self._get_cache_file_for_reals(num_images=self.num_images)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.isfile(cache_file):
            mean_real, std_real = misc.load_pkl(cache_file)
        else:
            for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
                begin = idx * minibatch_size
                end = min(begin + minibatch_size, self.num_images)
                images = np.transpose(images, [0, 2, 3, 1]) # nchw to nhwc
                activations[begin:end] = classifier.predict_on_batch(images)[:end-begin]
                if end == self.num_images:
                    break
            mean_real = np.mean(activations)
            std_real = np.std(activations)
            misc.save_pkl((mean_real, std_real), cache_file)
        
        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:], seed=42)
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, labels, **Gs_kwargs)
                images = tflib.convert_images_to_uint8(images, nchw_to_nhwc=True)
                result_expr.append(images)
        result_expr = tf.concat(result_expr, axis=0)

        # Calculate statistics for fakes (generated examples).
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            end = min(begin + minibatch_size, self.num_images)
            activations[begin:end] = classifier.predict_on_batch(result_expr)[:end-begin]
        mean_fake = np.mean(activations)
        std_fake = np.std(activations)

        # Save DCT Fake Score.
        self._report_result(mean_fake, suffix='_mean_gen')
        self._report_result(std_fake, suffix='_std_gen')
        self._report_result(mean_real, suffix='_mean_adv')
        self._report_result(std_real, suffix='_std_adv')

#----------------------------------------------------------------------------