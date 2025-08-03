# GAN

## Description of Dataset and Preprocessing

The project uses the MNIST dataset, which consists of 60,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels. 

To prepare the data:

Images are normalized to the range [-1, 1] to match the output activation of the generator (tanh).

Dimensions are expanded using np.expand_dims() to add a channel dimension, making the input shape compatible with convolutional layers if needed.

The dataset is shuffled and batched using TensorFlow’s tf.data.Dataset API for efficient training.

## Model Architecture Summary

This GAN consists of two models:

### 1. Generator

Input: 100-dimensional noise vector

Layers:

Dense(128) with ReLU

Dense(784) with tanh

Reshape to (28, 28) to form a grayscale image

### 2.Discriminator

Input: 28x28 image

Layers:

Flatten

Dense(128) with ReLU

Dense(1) with sigmoid to output a probability (real or fake)

The generator tries to produce images that resemble MNIST digits, while the discriminator tries to distinguish real from generated images.

### Training Stability Techniques
Binary Crossentropy loss is used for both generator and discriminator, ensuring stable and interpretable gradients.

Adam optimizer with a low learning rate (1e-4) is applied to both models to prevent rapid, unstable updates.

Shuffling and batching of the dataset increases training variance and avoids overfitting.

While this implementation doesn't explicitly include label smoothing or batch normalization, they are common techniques in more advanced GANs to improve training stability.

### Observations and Learning
The generator initially produces noise but gradually learns to generate digit-like images.

The discriminator converges quickly, often outperforming the generator in early epochs.

Visual feedback (plotting generated images) is crucial to monitor GAN learning progress.

The balance between generator and discriminator training is key — if one learns too fast, the other fails.

This project reinforced practical understanding of adversarial training, loss dynamics, and how architectural choices impact learning.

### Sample Output:

<img width="1761" height="548" alt="image" src="https://github.com/user-attachments/assets/8073c45b-1cd0-4f0b-83ba-2884595264e2" />
