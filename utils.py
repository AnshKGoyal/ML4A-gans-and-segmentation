import io
from PIL import Image
import numpy as np
from skimage.io import imread  # Import the imread function from skimage for image reading
import tensorflow as tf
import tensorflow_io as tfio
from keras import activations


def preprocess_image(image_file: io.BytesIO, streamlit_use=True) -> np.ndarray:
    """
    Preprocess the image for the model.

    Args:
        image_file (io.BytesIO): In-memory image file as bytes.
        streamlit_use (bool): Flag to determine if the image should be returned as a PIL Image 
                              (for Streamlit) or as a numpy array (for model prediction).

    Returns:
        np.ndarray or PIL.Image: Preprocessed image as a numpy array or PIL Image, 
                                 depending on the `streamlit_use` flag.
    """
    target_size = (480, 480)  # Define target height and width for resizing

    # Convert the BytesIO object to a byte string
    image_bytes = image_file.read()

    # Use TensorFlow to process the image
    img = tf.io.decode_image(image_bytes, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0

    if streamlit_use:
        # Convert the TensorFlow tensor back to a PIL Image for display in Streamlit
        img_numpy = img.numpy()  # Convert the tensor to a numpy array
        preprocessed_image = Image.fromarray((img_numpy * 255).astype(np.uint8))
        return preprocessed_image
    else:
        # Return the preprocessed image as a numpy array for model prediction
        return img.numpy()

def get_color_map():
    """
    Define a color map for visualizing segmentation masks.

    Returns:
        np.ndarray: Array mapping class indices to RGB colors.
    """
    return np.array([
        [0, 0, 0],       # Class 0: Black (Lunar Soil / Background)
        [255, 0, 0],     # Class 1: Red (Large Rocks)
        [0, 255, 0],     # Class 2: Green (Sky)
        [0, 0, 255]      # Class 3: Blue (Small Rocks)
        # Add more colors as needed for additional classes
    ], dtype=np.uint8)

def load_model(model_path: str):
    """
    Load a pre-trained TensorFlow model from a file.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        tf.keras.Model: Loaded TensorFlow model.
    """
    return tf.keras.models.load_model(model_path, compile=False)


class FixedDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config


custom_objects = {
    'swish': activations.swish,
    'FixedDropout': FixedDropout
}



'''def lab_to_rgb(l_chan, ab_chan):
    """
    Convert LAB color space to RGB.

    Args:
        l_chan (tf.Tensor): L channel
        ab_chan (tf.Tensor): ab channels

    Returns:
        tf.Tensor: RGB image
    """
    print(f"L channel shape: {l_chan.shape}, dtype: {l_chan.dtype}")
    print(f"AB channels shape: {ab_chan.shape}, dtype: {ab_chan.dtype}")

    # Ensure L channel has the same number of dimensions as ab channels
    if len(l_chan.shape) == 3 and len(ab_chan.shape) == 4:
        l_chan = tf.expand_dims(l_chan, axis=-1)
    elif len(l_chan.shape) == 4 and l_chan.shape[-1] == 1 and ab_chan.shape[-1] == 2:
        # If L channel already has 4 dimensions but with only 1 channel
        pass
    else:
        raise ValueError(f"Incompatible shapes: L {l_chan.shape}, ab {ab_chan.shape}")

    # Denormalize channels
    l_denorm = (l_chan + 1) * 50
    ab_denorm = ab_chan * 128
    
    # Combine channels
    lab_img = tf.concat([l_denorm, ab_denorm], axis=-1)
    print(f"Combined LAB image shape: {lab_img.shape}, dtype: {lab_img.dtype}")
    
    def convert_single_image(img):
        return tfio.experimental.color.lab_to_rgb(img)
    
    # Handle both batched and unbatched inputs
    if len(lab_img.shape) == 4:
        rgb_img = tf.map_fn(convert_single_image, lab_img, dtype=tf.float32)
    else:
        rgb_img = convert_single_image(lab_img)
    
    print(f"RGB image shape: {rgb_img.shape}, dtype: {rgb_img.dtype}")
    return tf.clip_by_value(rgb_img, 0.0, 1.0)'''