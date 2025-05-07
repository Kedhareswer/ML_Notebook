"use client"

import { useState } from "react"
import Link from "next/link"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, ArrowRight, BookOpen, Code, BarChart } from "lucide-react"
import NotebookCell from "@/components/notebook-cell"
import ModelVisualization from "@/components/model-visualization"

export default function MLPPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  // MLP visualization function
  const renderNeuralNetwork = (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => {
    const { hiddenLayers, neuronsPerLayer, learningRate } = params

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Set up network dimensions
    const margin = 40
    const networkWidth = width - 2 * margin
    const networkHeight = height - 2 * margin

    // Define layer positions
    const layers = []
    const inputNeurons = 4
    const outputNeurons = 2

    // Input layer
    layers.push({
      x: margin,
      neurons: inputNeurons,
      type: "input",
    })

    // Hidden layers
    const hiddenLayerWidth = networkWidth / (hiddenLayers + 1)
    for (let i = 0; i < hiddenLayers; i++) {
      layers.push({
        x: margin + hiddenLayerWidth * (i + 1),
        neurons: neuronsPerLayer,
        type: "hidden",
      })
    }

    // Output layer
    layers.push({
      x: width - margin,
      neurons: outputNeurons,
      type: "output",
    })

    // Calculate neuron positions for each layer
    layers.forEach((layer) => {
      layer.neuronPositions = []
      const neuronSpacing = Math.min(networkHeight / (layer.neurons + 1), 50)
      const startY = (height - (layer.neurons - 1) * neuronSpacing) / 2

      for (let i = 0; i < layer.neurons; i++) {
        layer.neuronPositions.push({
          x: layer.x,
          y: startY + i * neuronSpacing,
        })
      }
    })

    // Draw connections between layers
    ctx.lineWidth = 0.5

    for (let i = 0; i < layers.length - 1; i++) {
      const currentLayer = layers[i]
      const nextLayer = layers[i + 1]

      // Draw connections between all neurons in adjacent layers
      for (let j = 0; j < currentLayer.neurons; j++) {
        for (let k = 0; k < nextLayer.neurons; k++) {
          const start = currentLayer.neuronPositions[j]
          const end = nextLayer.neuronPositions[k]

          // Calculate connection strength (simulated)
          const connectionStrength = Math.random() * 2 - 1 // Random value between -1 and 1

          // Set connection color based on weight
          if (connectionStrength > 0) {
            const alpha = Math.min(0.8, (Math.abs(connectionStrength) * learningRate) / 5)
            ctx.strokeStyle = `rgba(65, 105, 225, ${alpha})`
          } else {
            const alpha = Math.min(0.8, (Math.abs(connectionStrength) * learningRate) / 5)
            ctx.strokeStyle = `rgba(220, 20, 60, ${alpha})`
          }

          // Draw the connection
          ctx.beginPath()
          ctx.moveTo(start.x, start.y)
          ctx.lineTo(end.x, end.y)
          ctx.stroke()
        }
      }
    }

    // Draw neurons
    layers.forEach((layer) => {
      layer.neuronPositions.forEach((pos) => {
        ctx.beginPath()

        // Different colors for different layer types
        if (layer.type === "input") {
          ctx.fillStyle = "#4CAF50" // Green for input
        } else if (layer.type === "output") {
          ctx.fillStyle = "#F44336" // Red for output
        } else {
          ctx.fillStyle = "#2196F3" // Blue for hidden
        }

        ctx.arc(pos.x, pos.y, 10, 0, Math.PI * 2)
        ctx.fill()

        // Add neuron border
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1
        ctx.stroke()
      })
    })

    // Add legend
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "left"

    // Position legend in top-left for better visibility on all screen sizes
    const legendX = margin + 10
    const legendY = margin + 20
    const legendSpacing = 20

    ctx.fillStyle = "#0000FF"
    ctx.beginPath()
    ctx.arc(legendX, legendY, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = "#000"
    ctx.fillText("Class 0", legendX + 20, legendY + 5)

    ctx.fillStyle = "#FF0000"
    ctx.beginPath()
    ctx.arc(legendX, legendY + 25, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = "#000"
    ctx.fillText("Class 1", legendX + 20, legendY + 30)

    // Positive weights
    ctx.beginPath()
    ctx.moveTo(legendX - 10, legendY + 3 * legendSpacing)
    ctx.lineTo(legendX + 10, legendY + 3 * legendSpacing)
    ctx.strokeStyle = "rgba(65, 105, 225, 0.8)"
    ctx.lineWidth = 2
    ctx.stroke()
    ctx.fillText("Positive Weights", legendX + 15, legendY + 3 * legendSpacing + 4)

    // Negative weights
    ctx.beginPath()
    ctx.moveTo(legendX - 10, legendY + 4 * legendSpacing)
    ctx.lineTo(legendX + 10, legendY + 4 * legendSpacing)
    ctx.strokeStyle = "rgba(220, 20, 60, 0.8)"
    ctx.lineWidth = 2
    ctx.stroke()
    ctx.fillText("Negative Weights", legendX + 15, legendY + 4 * legendSpacing + 4)

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "center"
    ctx.fillText(
      `MLP with ${hiddenLayers} Hidden Layers, ${neuronsPerLayer} Neurons/Layer, LR=${learningRate.toFixed(2)}`,
      width / 2,
      height - 10,
    )
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Epoch 1/10 25/25 [==============================] - 0s 2ms/step - loss: 0.6931 - accuracy: 0.5120 - val_loss:
          0.6928 - val_accuracy: 0.5200 Epoch 2/10 25/25 [==============================] - 0s 2ms/step - loss: 0.6914 -
          accuracy: 0.5440 - val_loss: 0.6910 - val_accuracy: 0.5600 Epoch 3/10 25/25 [==============================] -
          0s 2ms/step - loss: 0.6895 - accuracy: 0.5680 - val_loss: 0.6891 - val_accuracy: 0.5800 Epoch 4/10 25/25
          [==============================] - 0s 2ms/step - loss: 0.6873 - accuracy: 0.6000 - val_loss: 0.6869 -
          val_accuracy: 0.6200 Epoch 5/10 25/25 [==============================] - 0s 2ms/step - loss: 0.6848 -
          accuracy: 0.6320 - val_loss: 0.6844 - val_accuracy: 0.6400 Epoch 6/10 25/25 [==============================] -
          0s 2ms/step - loss: 0.6819 - accuracy: 0.6640 - val_loss: 0.6815 - val_accuracy: 0.6800 Epoch 7/10 25/25
          [==============================] - 0s 2ms/step - loss: 0.6786 - accuracy: 0.6960 - val_loss: 0.6782 -
          val_accuracy: 0.7000 Epoch 8/10 25/25 [==============================] - 0s 2ms/step - loss: 0.6748 -
          accuracy: 0.7280 - val_loss: 0.6744 - val_accuracy: 0.7400 Epoch 9/10 25/25 [==============================] -
          0s 2ms/step - loss: 0.6705 - accuracy: 0.7600 - val_loss: 0.6701 - val_accuracy: 0.7800 Epoch 10/10 25/25
          [==============================] - 0s 2ms/step - loss: 0.6657 - accuracy: 0.7920 - val_loss: 0.6653 -
          val_accuracy: 0.8200 Test accuracy: 0.8400
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">MLP Decision Boundaries:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">MLP decision boundaries plot</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Architecture Comparison:
          <br />
          10 Neurons, 1 Hidden Layer: 0.8200 accuracy
          <br />
          100 Neurons, 1 Hidden Layer: 0.8600 accuracy
          <br />
          10 Neurons, 2 Hidden Layers: 0.8400 accuracy
          <br />
          100 Neurons, 2 Hidden Layers: 0.8800 accuracy
          <br />
          100-50-25 Neurons, 3 Hidden Layers: 0.9000 accuracy
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Multilayer Perceptron (MLP)</h1>
          <p className="text-neutral-700 mt-2">The fundamental building block of deep learning networks</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models/pca">
              <ArrowLeft className="mr-2 h-4 w-4" /> Principal Component Analysis
            </Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/cnn">
              Next: Convolutional Neural Networks <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      <Tabs defaultValue="explanation" value={activeTab} onValueChange={setActiveTab} className="space-y-8">
        <TabsList className="grid w-full grid-cols-3 bg-neutral-100 text-neutral-900">
          <TabsTrigger value="explanation" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BookOpen className="h-4 w-4" />
            <span>Overview</span>
          </TabsTrigger>
          <TabsTrigger value="visualization" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BarChart className="h-4 w-4" />
            <span>Interactive Demo</span>
          </TabsTrigger>
          <TabsTrigger value="notebook" className="flex items-center gap-2 data-[state=active]:bg-white">
            <Code className="h-4 w-4" />
            <span>Code Implementation</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="explanation" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">What is a Multilayer Perceptron?</CardTitle>
              <CardDescription className="text-neutral-600">
                A class of feedforward artificial neural network with multiple layers
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                A Multilayer Perceptron (MLP) is a class of feedforward artificial neural network that consists of at
                least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Each node
                in one layer connects with a certain weight to every node in the following layer. MLPs are fully
                connected networks where each neuron in one layer is connected to all neurons in the next layer.
              </p>

              <p className="text-neutral-700">
                MLPs use a supervised learning technique called backpropagation for training. The key characteristic
                that gives neural networks their power is the ability to learn non-linear relationships through
                activation functions applied at each neuron.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">Key Concepts in MLPs</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Neurons & Layers</strong>: The basic computational units
                    (neurons) are organized into layers. The input layer receives the data, hidden layers perform
                    computations, and the output layer produces the final result.
                  </li>
                  <li>
                    <strong className="text-neutral-900">Activation Functions</strong>: Non-linear functions like ReLU,
                    sigmoid, or tanh that introduce non-linearity into the network, allowing it to learn complex
                    patterns.
                  </li>
                  <li>
                    <strong className="text-neutral-900">Backpropagation</strong>: The algorithm used to train MLPs,
                    which calculates the gradient of the loss function with respect to the weights and biases.
                  </li>
                  <li>
                    <strong className="text-neutral-900">Weight Initialization</strong>: The process of setting initial
                    values for the network weights, which is crucial for proper training.
                  </li>
                  <li>
                    <strong className="text-neutral-900">Forward Pass</strong>: The process of computing the output of
                    the network given an input by propagating values through the layers.
                  </li>
                  <li>
                    <strong className="text-neutral-900">Loss Function</strong>: A function that measures the difference
                    between the network's predictions and the actual target values.
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">How MLPs Work</h3>
              <p className="text-neutral-700 mb-4">The MLP learning process follows these steps:</p>
              <ol className="list-decimal list-inside space-y-4 text-neutral-700">
                <li>
                  <strong className="text-neutral-900">Forward Propagation</strong>: Input data is fed through the
                  network, with each neuron computing a weighted sum of its inputs, applying an activation function, and
                  passing the result to the next layer.
                </li>
                <li>
                  <strong className="text-neutral-900">Loss Calculation</strong>: The difference between the network's
                  output and the expected output is calculated using a loss function.
                </li>
                <li>
                  <strong className="text-neutral-900">Backward Propagation</strong>: The gradient of the loss function
                  with respect to each weight is calculated, starting from the output layer and moving backward.
                </li>
                <li>
                  <strong className="text-neutral-900">Weight Update</strong>: The weights are updated using an
                  optimization algorithm (like gradient descent) to minimize the loss function.
                </li>
                <li>
                  <strong className="text-neutral-900">Iteration</strong>: Steps 1-4 are repeated for multiple epochs
                  until the network converges to a satisfactory solution.
                </li>
              </ol>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Advantages and Limitations</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Advantages</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Can learn non-linear relationships</li>
                      <li>Adaptable to various problem types</li>
                      <li>Robust to noisy data</li>
                      <li>Capable of parallel processing</li>
                      <li>Can model complex patterns</li>
                      <li>Foundation for more complex neural networks</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Limitations</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Prone to overfitting with small datasets</li>
                      <li>Computationally intensive to train</li>
                      <li>Requires careful hyperparameter tuning</li>
                      <li>Black-box nature limits interpretability</li>
                      <li>Sensitive to feature scaling</li>
                      <li>May get stuck in local minima</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualization" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Interactive MLP Architecture</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how they affect the MLP structure and connections
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="Multilayer Perceptron Visualization"
                parameters={[
                  {
                    name: "hiddenLayers",
                    min: 1,
                    max: 5,
                    step: 1,
                    default: 2,
                    label: "Hidden Layers",
                  },
                  {
                    name: "neuronsPerLayer",
                    min: 2,
                    max: 10,
                    step: 1,
                    default: 5,
                    label: "Neurons per Layer",
                  },
                  {
                    name: "learningRate",
                    min: 0.1,
                    max: 2.0,
                    step: 0.1,
                    default: 1.0,
                    label: "Learning Rate",
                  },
                ]}
                renderVisualization={renderNeuralNetwork}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Hidden Layers</h3>
                <p className="text-neutral-700">
                  This parameter controls the depth of the neural network. More hidden layers allow the network to learn
                  more complex patterns and hierarchical representations. However, deeper networks are more difficult to
                  train and may require more data. Notice how adding more layers creates a deeper network structure.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Neurons per Layer</h3>
                <p className="text-neutral-700">
                  This parameter determines the width of each hidden layer. More neurons increase the network's capacity
                  to learn complex patterns but also increase the risk of overfitting and computational cost. Observe
                  how increasing the number of neurons creates wider layers with more connections.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Learning Rate</h3>
                <p className="text-neutral-700">
                  This parameter controls how much the weights are updated during training. A higher learning rate means
                  larger weight updates, which can lead to faster convergence but may also cause the network to
                  overshoot the optimal solution. In this visualization, the learning rate affects the intensity of the
                  connection colors, representing the magnitude of weight updates.
                </p>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Interpreting the Visualization</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong>Green nodes</strong> represent input neurons, which receive the raw data
                  </li>
                  <li>
                    <strong>Blue nodes</strong> represent hidden neurons, which perform intermediate computations
                  </li>
                  <li>
                    <strong>Red nodes</strong> represent output neurons, which produce the final predictions
                  </li>
                  <li>
                    <strong>Blue lines</strong> represent positive weights (excitatory connections)
                  </li>
                  <li>
                    <strong>Red lines</strong> represent negative weights (inhibitory connections)
                  </li>
                  <li>
                    The <strong>intensity of the lines</strong> represents the strength of the connections, influenced
                    by the learning rate
                  </li>
                  <li>
                    Notice how the network becomes more complex with more layers and neurons, increasing its capacity to
                    model complex functions
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">MLP Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement Multilayer Perceptrons using TensorFlow/Keras. Execute each
                cell to see the results.
              </p>
            </div>

            <div className="space-y-6">
              <NotebookCell
                cellId="cell0"
                executionCount={1}
                initialCode="import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 1: Create and train a basic MLP</p>
                <p>Let's create a synthetic dataset and train a simple MLP with two hidden layers.</p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode="# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f'Test accuracy: {test_acc:.4f}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Visualize decision boundaries</p>
                <p>Let's create a 2D dataset to visualize how the MLP creates non-linear decision boundaries.</p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Create a 2D dataset for visualization
X_2d, y_2d = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=2,
    class_sep=1.0,
    random_state=42
)

# Split the data
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_2d, test_size=0.2, random_state=42
)

# Standardize
scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)

# Build a simple MLP for the 2D data
model_2d = Sequential([
    Dense(32, activation='relu', input_shape=(2,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model_2d.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history_2d = model_2d.fit(
    X_train_2d_scaled, y_train_2d,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Function to plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    # Set up the mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Scale the mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler_2d.transform(mesh_points)
    
    # Predict on the mesh
    Z = model.predict(mesh_points_scaled)
    Z = (Z > 0.5).astype(int).reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='viridis', edgecolors='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

# Plot the decision boundary
plot_decision_boundary(
    model_2d, 
    X_2d, 
    y_2d, 
    'MLP Decision Boundary'
)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Compare different architectures</p>
                <p>Let's compare how different MLP architectures perform on the same dataset.</p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Define different architectures to compare
architectures = [
    ([10], '10 Neurons, 1 Hidden Layer'),
    ([100], '100 Neurons, 1 Hidden Layer'),
    ([10, 10], '10 Neurons, 2 Hidden Layers'),
    ([100, 100], '100 Neurons, 2 Hidden Layers'),
    ([100, 50, 25], '100-50-25 Neurons, 3 Hidden Layers')
]

# Train and evaluate each architecture
results = []

for layers, name in architectures:
    # Build model with this architecture
    model = Sequential()
    
    # Add input layer
    model.add(Dense(layers[0], activation='relu', input_shape=(X_train_2d_scaled.shape[1],)))
    
    # Add additional hidden layers
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    model.fit(
        X_train_2d_scaled, y_train_2d,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate
    _, accuracy = model.evaluate(X_test_2d_scaled, y_test_2d, verbose=0)
    results.append((name, accuracy))

# Print results
print('Architecture Comparison:')
for name, accuracy in results:
    print(f'{name}: {accuracy:.4f} accuracy')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of MLPs:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>Try different activation functions (ReLU, tanh, sigmoid) and compare their performance</li>
                  <li>Experiment with different optimizers (SGD, RMSprop, Adam) and learning rates</li>
                  <li>Add regularization techniques like L1/L2 regularization or dropout to prevent overfitting</li>
                  <li>Implement a regression MLP for predicting continuous values</li>
                  <li>Visualize the training process with learning curves (loss and accuracy over epochs)</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
