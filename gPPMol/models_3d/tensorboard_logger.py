import tensorflow as tf
import numpy as np

class Logger(object):
    """
    Logger class for writing scalar, image, and histogram summaries for TensorBoard.
    """

    def __init__(self, log_dir):
        """
        Initialize the Logger object and create a TensorFlow summary writer.
        
        Args:
            log_dir (str): The directory where log files will be saved.
        """
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """
        Log a scalar value for TensorBoard.
        
        Args:
            tag (str): The name of the scalar variable.
            value (float): The value of the scalar variable.
            step (int): The global step value to record with the scalar.
        """
        with self.writer.as_default():
            tf.summary.scalar(tag, data=value, step=step)

    def image_summary(self, tag, images, step):
        """
        Log a list of images to TensorBoard.
        
        Args:
            tag (str): The name of the image summary.
            images (numpy.ndarray): A numpy array of shape (Batch x C x H x W) where C is the number 
                                    of channels (e.g., 1 for grayscale, 3 for RGB).
            step (int): The global step value to record with the image summary.
        """
        with self.writer.as_default():
            imgs = None
            for i, j in enumerate(images):
                # Scale images from [-1.0, 1.0] to [0, 255] and convert to uint8
                img = ((j * 0.5 + 0.5) * 255).round().astype('uint8')
                
                # Adjust image shape if necessary (C x H x W to H x W x C)
                if len(img.shape) == 3:
                    img = img.transpose(1, 2, 0)
                else:
                    img = img[:, :, np.newaxis]  # Add channel axis if it's grayscale
                
                img = img[np.newaxis, :]  # Add batch axis

                # Concatenate images into a single batch for logging
                if imgs is not None:
                    imgs = np.append(imgs, img, axis=0)
                else:
                    imgs = img
            
            # Log the image batch to TensorBoard
            tf.summary.image(f'{tag}', imgs, max_outputs=len(imgs), step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """
        Log a histogram of values for TensorBoard.
        
        Args:
            tag (str): The name of the histogram.
            values (numpy.ndarray or Tensor): The values to create a histogram for.
            step (int): The global step value to record with the histogram.
            bins (int): The number of histogram bins.
        """
        with self.writer.as_default():
            tf.summary.histogram(f'{tag}', values, buckets=bins, step=step)
