# Space Image Processing

This project provides a web application for processing space-related images, offering two main functionalities:
1. Lunar Surface Segmentation
2. Space Image Colorization

The application consists of a FastAPI backend for image processing and a Streamlit frontend for user interaction.

**Note:** For Lunar Surface Segmentation i am using my previous [repo](https://github.com/AnshKGoyal/lunar-segmentation-app) as a reference

## Features

- **Lunar Surface Segmentation**: Segments different features on lunar surface images.
- **Space Image Colorization**: Adds color to grayscale space images using a GAN model.
- Interactive web interface for uploading and processing images.
- Image enhancement options for colorized images.

## Project Structure

- `backend.py`: FastAPI server handling image processing requests.
- `frontend.py`: Streamlit web interface for user interaction.
- `utils.py`: Utility functions for image preprocessing and model loading.
- `requirements.txt`: List of Python dependencies.
- `report.md`: Report of this project.
- `models/`: Directory containing pre-trained models.
  - `LunarModel.h5`: Model for lunar surface segmentation
  - `generator_60_efficientb4.h5`: Model for space image colorization
- `notebooks/`: Directory containing Jupyter notebooks for model training.
  - `lunar_segmentation_training.ipynb`: Notebook for training the lunar surface segmentation model.
  - `space_colorization_gan_training.ipynb`: Notebook for training the space image colorization GAN model.

## Demo Video

To see the Space Image Processing tool in action, check out our demo video:


https://github.com/user-attachments/assets/e78be9fe-8bc9-4882-bea9-fa86222f3bd7





## Setup and Installation

1. Clone or Download the Repository and open the project directory in your editor (VS Code)
   
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   python -m pip install -r requirements.txt
   ```

4. The pre-trained models are included in the `models/` directory, so no additional download is necessary.

## Running the Application

1. Start the FastAPI backend:
   ```
   uvicorn backend:app --reload
   ```

2. In a separate terminal, run the Streamlit frontend:
   ```
   streamlit run frontend.py
   ```

3. Open a web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1. Select the desired operation: "Lunar Surface Segmentation" or "Space Image Colorization".
2. Upload an image using the file uploader.
3. For colorization, you can adjust enhancement settings in the sidebar.
4. Click the "Process Image" button to perform the selected operation.
5. View the processed image and download it if desired.

## Training Models

For instructions on training the models used in this project, please refer to the Jupyter notebooks provided in the `notebooks/` directory:

- `lunar_segmentation_training.ipynb`: Notebook for training the lunar surface segmentation model.
- `space_colorization_gan_training.ipynb`: Notebook for training the space image colorization GAN model.

You can also use Kaggle to train these models. Simply import the provided notebooks into Kaggle:

1. Create a Kaggle account if you don't have one.
2. Go to the "Notebooks" section on Kaggle.
3. Click on "New Notebook" or "Create".
4. Choose "File" > "Upload Notebook" and select the desired training notebook from this repository.
5. Run the notebook on Kaggle, which provides free GPU resources.
6. After training, download the model files and replace the existing ones in the `models/` directory.

Using Kaggle can be beneficial as it provides free GPU resources and easy access to datasets, which can significantly speed up the training process.

### Kaggle Integration

The GAN model for space image colorization is currently undergoing further refinement. Once the training process is complete and optimal results are achieved, the final version of the training notebook will be made public on Kaggle.

Future Kaggle Resources:

Finalized GAN training notebook: [Link will be provided upon completion]

Kaggle profile: [https://www.kaggle.com/anshkgoyal]

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.
