# Astronomical Image Colorization using WGAN-GP

## 1. Introduction

This project tackles the challenge of automatically colorizing grayscale astronomical images using advanced deep learning techniques. Specifically, we employ a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) architecture to transform grayscale images into vibrant, full-color representations.

Our approach leverages the LAB color space, using the L channel (luminance) as input and predicting the a and b channels (color information). This method allows the model to focus on color prediction while preserving the original image structure.

The best-performing model, featuring an EfficientNetB4 backbone in the generator and trained for 60 epochs, achieved a Peak Signal-to-Noise Ratio (PSNR) of 27.589277 on the validation set. This result was obtained using the WGAN-GP architecture with L1 loss incorporated in the generator.

## 2. Dataset

Our project utilizes a combination of three astronomical image datasets:

1. Top 100 Hubble Telescope Images
2. ESA Hubble Images
3. SpaceNet: A Comprehensive Astronomical Dataset

These datasets offer a diverse range of cosmic imagery, including galaxies, nebulae, star clusters, and various celestial phenomena.

Preprocessing Steps:
1. Images are converted from RGB to the LAB color space.
2. The L channel is extracted as the grayscale input.
3. The a and b channels serve as color targets for the model to predict.
4. All channels are normalized to the range [-1, 1] for optimal model performance.

Example images from the dataset:

![image](https://github.com/user-attachments/assets/1f374373-b9fb-4566-9ae0-e4eb93c7a58e)

![image](https://github.com/user-attachments/assets/0b8ddf53-a4ef-467c-97a1-9d5e10d686a7)


## 3. Model Architecture

### Generator
- Based on a U-Net architecture with an EfficientNetB4 backbone
- Incorporates skip connections to preserve spatial information
- Uses upsampling layers in the decoder to produce the final colorized output

![image](https://github.com/user-attachments/assets/50964adc-2967-4bb2-8e72-c6c794116de5)



### Discriminator
- PatchGAN architecture for local consistency in generated colors
- Convolutional layers for feature extraction
- No batch normalization (as per WGAN-GP guidelines)
- Outputs a scalar value for each image patch

 ![image](https://github.com/user-attachments/assets/7a20eabb-4e2d-4151-8f17-66f470280885)


## 4. Training Approach

The model was trained using the WGAN-GP framework, known for improved stability compared to traditional GANs. Key aspects include:

- Alternating training of generator and discriminator
- Gradient penalty to enforce the Lipschitz constraint
- L1 loss in the generator to encourage color fidelity
- Adam optimizer with β1 = 0 and β2 = 0.9
- Generator trained once every 3 steps

## 5. Results

The model's performance improved through several iterations:

| Changes Made | Steps/Epochs | Train PSNR | Validation PSNR |
|--------------|--------------|------------|--------------------|
| ResNet34 backbone for UNet generator, complex discriminator with dropout | 15k steps | 24.2 | 21.4 |
| ResNet34 backbone for UNet generator, basic discriminator | 15k steps | 26.65 | 23.06 |
| WGANs-gp with L1 loss in generator (lambda=10) | 5 epochs | 28 | 25 (fluctuating) |
| Generator trained once every 3 steps, EfficientNetB2 backbone, resolution (224,224) | 62 epochs | 30 | 27.453756 |
| EfficientNetB4 backbone | 60 epochs | 30 | 27.589277 |

The final model with EfficientNetB4 backbone demonstrated the best performance, achieving a validation PSNR of 27.589277.

![image](https://github.com/user-attachments/assets/f79c1ca9-d2c4-49b6-9fac-0f6086340f8a)


## 6. Future Scope

While the current results are promising, several avenues for potential improvement exist:

1. Extended Training: Increase training to 150 epochs or more to potentially enhance colorization quality and PSNR scores.

2. Larger Backbones: Experiment with ResNet50 or larger EfficientNet variants to capture more complex features.

3. Hyperparameter Tuning: Fine-tune learning rates, batch sizes, and loss balancing for better results.

4. Data Augmentation: Implement more aggressive augmentation techniques for better generalization.

5. Ensemble Methods: Combine predictions from multiple models with different architectures or initializations.

6. Attention Mechanisms: Incorporate attention in the generator to focus on relevant colorization features.

7. Perceptual Loss: Add a perceptual loss term to improve visual quality beyond PSNR measurements.

8. Filtering some images and classes from SpaceNet dataset.

By implementing these improvements, we anticipate pushing the PSNR beyond 28 and achieving more vivid and accurate colorizations, particularly for complex astronomical phenomena.
