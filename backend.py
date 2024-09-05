# Imports
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import numpy as np
import io
from utils import preprocess_image, get_color_map, load_model, custom_objects
import tensorflow as tf
import tensorflow_io as tfio




# Initialize FastAPI app
app = FastAPI()
 
# Load the pre-trained segmentation model
model_path = 'models/LunarModel.h5'
model = load_model(model_path)

colorization_model_path = 'models/generator_60_efficientb4.h5'
colorization_model = tf.keras.models.load_model(colorization_model_path, custom_objects=custom_objects, compile=False)


@app.get("/")
async def read_root():
    """
    Root endpoint to test if the server is running.
    Returns a simple JSON response with a working message.
    """
    return {"App": "Working"}

@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    """
    Endpoint for segmenting an uploaded image.

    Args:
        file (UploadFile): The image file to be segmented.

    Returns:
        StreamingResponse: The segmentation result image in PNG format.
        JSONResponse: Error details if something goes wrong.
    """
    try:
        # Read the image file into a BytesIO object
        image_bytes = await file.read()
        image_file = io.BytesIO(image_bytes)

        # Preprocess the image for the model
        image_array = preprocess_image(image_file, streamlit_use=False)

        # Perform segmentation using the loaded model
        pred_mask = model.predict(np.expand_dims(image_array, axis=0))
        pred_mask = np.argmax(pred_mask, axis=-1)  # Determine the most probable class for each pixel
        pred_mask = pred_mask[0]  # Remove batch dimension

        # Map the predicted mask to colors
        color_map = get_color_map()
        segmentation_img = color_map[pred_mask]  # Convert class indices to color values

        # Convert the colored segmentation image to a PIL Image
        segmentation_img_pil = Image.fromarray(segmentation_img)

        # Save the PIL image to a BytesIO object
        img_byte_arr = io.BytesIO()
        segmentation_img_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the segmentation image as a streaming response
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except HTTPException as e:
        # Return an HTTP exception response with details
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        # Return a generic error response if an unexpected error occurs
        return JSONResponse(content={"error": str(e)}, status_code=500)

def enhance_ab_channels(ab_channels, factor=1.5):
    # Enhance the ab channels by scaling them
    return tf.clip_by_value(ab_channels * factor, -1, 1)


@app.post("/colorize/")
async def colorize_image(file: UploadFile = File(...)):
    try:
        # Read the image file into a BytesIO object
        image_bytes = await file.read()
        image_file = io.BytesIO(image_bytes)

        # Open the image as grayscale and convert to RGB
        # (This step ensures we have a 3-channel image, even if it's grayscale)
        img = Image.open(image_file).convert('L').convert('RGB')
        
        # Convert PIL Image to numpy array
        img_array = np.array(img)

        # Convert numpy array to TensorFlow tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)

        # Resize image
        target_size = (224, 224)  # Adjust this to match your model's input size
        img_resized = tf.image.resize_with_pad(img_tensor, target_size[0], target_size[1], method=tf.image.ResizeMethod.BILINEAR)

        # Normalize RGB to [0, 1] for LAB conversion
        img_float = tf.cast(img_resized, tf.float32) / 255.0

        # Convert RGB to LAB using TensorFlow I/O
        lab_img = tfio.experimental.color.rgb_to_lab(img_float)

        # Normalize LAB channels to [-1, 1]
        l_chan = lab_img[:,:,0] / 50.0 - 1.0  # L is in range [0, 100]
        
        # Prepare input for the model (add batch dimension)
        model_input = tf.expand_dims(tf.expand_dims(l_chan, axis=-1), axis=0)
        print(f"Model input shape: {model_input.shape}, dtype: {model_input.dtype}")
        print(f"Model input min: {tf.reduce_min(model_input)}, max: {tf.reduce_max(model_input)}")

        
        # Use the model to predict ab channels
        ab_channels = colorization_model.predict(model_input)
        print(f"Predicted ab channels shape: {ab_channels.shape}, dtype: {ab_channels.dtype}")
        print(f"Predicted ab channels min: {np.min(ab_channels)}, max: {np.max(ab_channels)}")

        # Enhance color vibrancy
        ab_channels_enhanced = enhance_ab_channels(ab_channels[0])
        
        # Denormalize the enhanced ab channels
        ab_channels_denorm = ab_channels_enhanced * 128.0

        # Combine L and predicted ab channels
        l_chan_denorm = (l_chan + 1.0) * 50.0
        lab_result = tf.stack([l_chan_denorm, ab_channels_denorm[:,:,0], ab_channels_denorm[:,:,1]], axis=-1)

        # Convert LAB back to RGB
        rgb_image = tfio.experimental.color.lab_to_rgb(lab_result)
        
        print(f"Final RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
        print(f"Final RGB image min: {tf.reduce_min(rgb_image)}, max: {tf.reduce_max(rgb_image)}")

        # Convert to uint8 and create PIL Image
        rgb_image_uint8 = tf.cast(tf.clip_by_value(rgb_image * 255, 0, 255), tf.uint8).numpy()
        colorized_img = Image.fromarray(rgb_image_uint8)

        # Save the colorized image to a BytesIO object
        img_byte_arr = io.BytesIO()
        colorized_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the colorized image as a streaming response
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)