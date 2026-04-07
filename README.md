Handwritten Digit Classifier (From Scratch)
This project implements a neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset. The goal of this project is to understand the core concepts of deep learning without relying on high-level machine learning frameworks.

Features
- Built neural network without TensorFlow or PyTorch
- Implemented forward propagation and backpropagation manually
- Used ReLU activation for hidden layer
- Applied Softmax activation for multi-class classification
- Optimized using cross-entropy loss and gradient descent
- Trained using mini-batch gradient descent
- Visualized training loss and accuracy

Model Architecture
- Input Layer: 784 neurons (28×28 image pixels)
- Hidden Layer: 128 neurons (ReLU activation)
- Output Layer: 10 neurons (Softmax activation)

Dataset
- MNIST Handwritten Digits
- Training: 60,000 samples
- Testing: 10,000 samples

Technologies Used
- Python
- NumPy
- Matplotlib
- Scikit-learn

Results
- Achieved approximately 96% test accuracy on MNIST dataset
- Training loss decreases over epochs
- Accuracy improves with training

How to Run
1. Clone the repository
2. Install dependencies: numpy, matplotlib, scikit-learn
3. Run the Python script

Learning Outcomes
- Understanding neural network fundamentals
- Implementation of backpropagation
- Knowledge of activation and loss functions
- Model training and evaluation

Future Improvements
- Add more hidden layers
- Try different activation functions
- Improve accuracy with tuning
- Use more complex datasets

Author
Heer Madhu
