# CNN for image classification from scratch using NumPy

This repository contains an implementation of a Convolutional Neural Network (CNN) using **numpy**. The network is designed for **image classification** tasks, and includes features such as **Batch Normalization** and **backpropagation**.

The model is trained on the **CIFAR-10** dataset, achieving a **test accuracy of 43% after just one epoch of training**. The training time was drastically reduced by parallelizing the loop over filters, which led to a decrease in training time from several hours to under 5 minutes for one epoch.

## Architecture

The network consists of three layers:
1. **Convolutional Layer**: Conv - Batch Normalization - ReLU - 2x2 Max Pooling
2. **Fully Connected Layer**: Affine - ReLU
3. **Output Layer**: Affine - Softmax

### Input Data Format

The network operates on minibatches of data with the following shape:
- **N**: Number of images in a minibatch
- **H**: Height of the images
- **W**: Width of the images
- **C**: Number of input channels

In this implementation, the default input size is (32, 32, 3), which corresponds to the CIFAR-10 dataset images.

## Key Features

- **Parallelized filter loops**: The loop over the filters in the convolutional layer was parallelized to significantly reduce training time.
- **Batch Normalization**: This helps in accelerating the training process and improves performance by normalizing the activations.
- **Backpropagation**: The forward and backward passes were implemented using custom functions to calculate gradients for weight updates.
- **Efficient Training**: Training time was reduced from several hours per epoch to under 5 minutes for one epoch on CIFAR-10.

## Results

- **Test Accuracy**: 43% after 1 epoch of training on CIFAR-10.
- **Training Time**: Reduced to under 5 minutes for one epoch, compared to several hours for naive implementation.

## Installation

1. Clone the repository:

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    The only external dependency required is **numpy** and **PIL** for image loading and manipulation.

## Training

To train the model on CIFAR-10:

1. Download the CIFAR-10 dataset:
   - You can use the helper function `download_cifar10()` to download the dataset or manually download and extract the CIFAR-10 data from [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

2. Run the training script:

3. Training progress will be displayed in the terminal, showing loss and accuracy at each epoch.
   
### Example Training Output
```bash
(Iteration 181 / 200) loss: 1.679291
(Epoch 1 / 1) train acc: 0.426000; val_acc: 0.408000
Training time: 299.986317s
```
The trained filters look as follows,

![image](https://github.com/user-attachments/assets/612f4068-80c7-438f-8b7b-9e7c72d1b3b2)

Further improvements can be done by parallelizing the remaining for loops. The repository also contains a closely similar CNN model built using PyTorch for reference.

