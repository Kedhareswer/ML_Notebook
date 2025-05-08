"use client"

import { useState } from "react"
import Link from "next/link"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, ArrowRight, BookOpen, BarChart } from "lucide-react"
import ModelVisualization from "@/components/model-visualization"

export default function MLPPage() {
  const [activeTab, setActiveTab] = useState("explanation")

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
              <ArrowLeft className="mr-2 h-4 w-4" /> Previous: Principal Component Analysis
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
        <TabsList className="grid w-full grid-cols-2 bg-neutral-100 text-neutral-900">
          <TabsTrigger value="explanation" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BookOpen className="h-4 w-4" />
            <span>Overview</span>
          </TabsTrigger>
          <TabsTrigger value="visualization" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BarChart className="h-4 w-4" />
            <span>Interactive Demo</span>
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
        </TabsContent>
      </Tabs>
    </div>
  )
}
