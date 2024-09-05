from PIL import Image, ImageFilter
import io
import streamlit as st
import requests
from utils import preprocess_image

# Set title for the tab and icon
st.set_page_config(page_title="Space Image Processing", page_icon="🚀")

# URL of your FastAPI backend endpoints
SEGMENTATION_URL = "http://localhost:8000/segment/"
COLORIZATION_URL = "http://localhost:8000/colorize/"

def apply_unsharp_mask(image, amount, radius, threshold):
    """Apply unsharp mask filter to an image."""
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=amount, threshold=threshold))

# Sidebar menu
st.sidebar.title("Image Processing Options")
option = st.sidebar.radio("Select Operation", ["Lunar Surface Segmentation", "Space Image Colorization"])

# Apply background only for Lunar Surface Segmentation
if option == "Lunar Surface Segmentation":
    st.markdown(
        """
        <style>
        .full-page-background {
            background-image: url('https://static.vecteezy.com/system/resources/thumbnails/031/411/730/small_2x/ai-generated-ai-generative-evening-night-nature-outdoor-landscape-background-full-moon-and-mountains-graphic-art-photo.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        </style>
        <div class="full-page-background"></div>
        """,
        unsafe_allow_html=True
    )

# Main content
st.title("Space Image Processing Tool")

if option == "Lunar Surface Segmentation":
    st.header("Lunar Surface Segmentation")
    st.write("Upload a grayscale image of a lunar surface for segmentation. This tool identifies different features on the Moon's surface.")
    file_type = ["jpg", "jpeg", "png"]
    upload_prompt = "Choose a lunar surface image..."
elif option == "Space Image Colorization":
    st.header("Space Image Colorization")
    st.write("Upload any grayscale space image (e.g., nebulae, galaxies, planets) to add color using our GAN model.")
    file_type = ["jpg", "jpeg", "png"]
    upload_prompt = "Choose a space image..."

    # Unsharp mask parameters
    st.sidebar.header("Enhance Colors")
    apply_enhancement = st.sidebar.checkbox("Apply Color Enhancement", value=True)
    if apply_enhancement:
        amount = st.sidebar.slider("Strength", 0, 500, 50, 10)
        radius = st.sidebar.slider("Radius", 0.0, 5.0, 2.0, 0.1)
        threshold = st.sidebar.slider("Threshold", 0, 10, 2)
    change_res = st.sidebar.checkbox("Change Resolution",value=True)
    res=314
    if change_res:
        res = st.sidebar.slider("pixel:",224,512,314,10)

# File uploader widget to allow users to upload an image
uploaded_file = st.file_uploader(upload_prompt, type=file_type)

if uploaded_file:
    # Convert the uploaded file into an in-memory bytes object
    image_bytes = io.BytesIO(uploaded_file.read())
    
    try:
        # Open the image using PIL
        img = Image.open(image_bytes).convert('L')  # Ensure grayscale
        
        # Resize the image to 480x480 if it's for GAN colorization
        if option == "Space Image Colorization":
            img = img.resize((res, res), Image.LANCZOS)
        
        # Convert the resized image back to bytes
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        
        # Display the image
        st.image(img, caption="Uploaded Image", use_column_width=option == "Lunar Surface Segmentation")

        # Button to trigger the selected operation
        if st.button(f"Process Image"):
            # Prepare the file to send in the POST request
            files = {"file": ("image.png", image_bytes, "image/png")}
            
            if option == "Lunar Surface Segmentation":
                # Segmentation process
                response = requests.post(SEGMENTATION_URL, files=files)
                result_caption = "Segmented Lunar Surface"
                use_column_width = True
            else:
                # Colorization process
                response = requests.post(COLORIZATION_URL, files=files)
                result_caption = "Colorized Space Image"
                use_column_width = False
            
            if response.status_code == 200:
                # If the request is successful, display the processed image
                result_image = Image.open(io.BytesIO(response.content))
                
                if option == "Space Image Colorization" and apply_enhancement:
                    # Apply unsharp mask to colorized image
                    result_image = apply_unsharp_mask(result_image, amount, radius, threshold)
                
                st.image(result_image, caption=result_caption, use_column_width=use_column_width, width=res if option == "Space Image Colorization" else None)
                
                # Option to download the processed image
                buffered = io.BytesIO()
                result_image.save(buffered, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buffered.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )
            else:
                # If there's an error, display an error message
                st.error(f"Error: Unable to process the image. Please try again. (Status code: {response.status_code})")
    
    except Exception as e:
        # Display error message if processing fails
        st.error(f"Error: {e}")

# Additional information
st.sidebar.markdown("---")
st.sidebar.info(
    "This tool offers two main functions:\n\n"
    "1. **Lunar Surface Segmentation**: Specifically designed for lunar images, "
    "this option segments different features on the Moon's surface.\n\n"
    "2. **Space Image Colorization**: Uses a GAN model to add color to any "
    "grayscale space image, including nebulae, galaxies, and planets."
)