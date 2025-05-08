"use client"

import type React from "react"

// Types for code execution
export type ExecutionResult = {
  output: string | React.ReactNode
  error?: string
  type?: "text" | "plot" | "table" | "error"
}

// Realistic code execution patterns
const EXECUTION_PATTERNS = {
  // Python imports
  imports: {
    numpy: "NumPy imported successfully.",
    pandas: "Pandas imported successfully.",
    matplotlib: "Matplotlib imported successfully.",
    sklearn: "Scikit-learn imported successfully.",
    tensorflow: "TensorFlow 2.15.0 imported successfully.",
    keras: "Keras imported successfully.",
    torch: "PyTorch 2.2.0 imported successfully.",
    seaborn: "Seaborn imported successfully.",
  },
  
  // Model creation
  modelCreation: {
    linearRegression: `Linear regression model created:
Model: LinearRegression()
Parameters:
  - fit_intercept: True
  - copy_X: True
  - n_jobs: None
  - positive: False`,
    
    decisionTree: `Decision tree classifier created:
Model: DecisionTreeClassifier()
Parameters:
  - criterion: gini
  - splitter: best
  - max_depth: None
  - min_samples_split: 2`,
    
    svm: `Support Vector Machine classifier created:
Model: SVC()
Parameters:
  - C: 1.0
  - kernel: 'rbf'
  - degree: 3
  - gamma: 'scale'`,
    
    cnn: `Convolutional Neural Network created:
Model: Sequential()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)          0         
flatten (Flatten)            (None, 1600)              0         
dense (Dense)                (None, 128)               204928    
dropout (Dropout)            (None, 128)               0         
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 225,034
Trainable params: 225,034
Non-trainable params: 0`,
    
    rnn: `Recurrent Neural Network created:
Model: Sequential()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 100, 32)           160000    
lstm (LSTM)                  (None, 100, 64)           24832     
lstm_1 (LSTM)                (None, 32)                12416     
dense (Dense)                (None, 1)                 33        
=================================================================
Total params: 197,281
Trainable params: 197,281
Non-trainable params: 0`,
  },
  
  // Model training
  modelTraining: {
    linearRegression: `Training linear regression model...
Model fitted successfully in 0.034s
R² on training set: 0.912
Mean squared error: 0.234`,
    
    decisionTree: `Training decision tree...
Model fitted successfully in 0.087s
Accuracy on training set: 0.982
Feature importances: [0.12, 0.08, 0.35, 0.15, 0.30]`,

    svm: `Training SVM...
Model fitted successfully in 0.156s
Accuracy on training set: 0.984
Number of support vectors: 42`,

    cnn: `Training CNN...
Epoch 1/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.2153 - accuracy: 0.9348 - val_loss: 0.0762 - val_accuracy: 0.9764
Epoch 2/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0852 - accuracy: 0.9744 - val_loss: 0.0612 - val_accuracy: 0.9812
Epoch 3/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0634 - accuracy: 0.9803 - val_loss: 0.0481 - val_accuracy: 0.9846
Epoch 4/5
1875/1875 [==============================] - 9s 5ms/step
}`
