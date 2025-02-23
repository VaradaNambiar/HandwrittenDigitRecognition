# Handwritten Digit Recognition with MNIST Dataset

This project demonstrates handwritten digit recognition using a simple neural network built with TensorFlow and Keras in a Jupyter Notebook. It utilizes the MNIST dataset, a classic dataset in computer vision, containing 70,000 grayscale images of handwritten digits (0-9).

## Project Structure

-   `HandwrittenDigitRecognitionModel.ipynb`: Jupyter Notebook containing the code for training and evaluating the model.
-   `README.md`: This file, providing an overview of the project.

## Dependencies

-   Python 3.x
-   TensorFlow (>= 2.x)
-   NumPy
-   Matplotlib

You can install the required libraries using pip:

```bash
pip install tensorflow numpy matplotlib
```

## Dataset
The MNIST dataset is automatically downloaded by TensorFlow when you run the notebook. It consists of:

60,000 training images
10,000 test images
Each image is a 28x28 pixel grayscale image.

## Model Architecture
The neural network architecture is as follows:

Flatten Layer: Converts the 28x28 input image into a 784-element vector.
Dense Layer (Hidden Layer): 128 neurons with ReLU activation.
Dense Layer (Output Layer): 10 neurons (one for each digit) with Softmax activation.

## Training
The model is trained using the following parameters:

Optimizer: Adam
Loss function: Sparse Categorical Crossentropy
Metrics: Accuracy
Epochs: 10

## Usage
1. Clone the Repository (Optional): If you are viewing this on a platform like GitHub, clone the repository to your local machine:

```bash
git clone [repository_url]
cd [repository_directory]


2. Open the Jupyter Notebook: Launch Jupyter Notebook and open Handwritten_Digit_Recognition.ipynb.

3. Run the Notebook: Execute the cells in the notebook sequentially. The notebook will:

	-Load and preprocess the MNIST dataset.
	-Build the neural network model.
	-Train the model.
	-Evaluate the model's performance on the test set.
	-Display sample predictions and the loss vs epoch graph.

4. View Results: The notebook will output the test accuracy and display sample predictions, along with a graph showing the training loss over epochs.

