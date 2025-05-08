"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Shuffle, Download, Play, Pause } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface NeuralNetworkVisualizationProps {
  width?: number
  height?: number
}

export default function NeuralNetworkVisualization({ width = 600, height = 400 }: NeuralNetworkVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [activeTab, setActiveTab] = useState("feedforward")
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationStep, setAnimationStep] = useState(0)

  // Parameters for feedforward network
  const [layersCount, setLayersCount] = useState(4)
  const [neuronsPerLayer, setNeuronsPerLayer] = useState(5)
  const [activationFunction, setActivationFunction] = useState("relu")

  // Parameters for gradient descent
  const [learningRate, setLearningRate] = useState(0.1)
  const [epochs, setEpochs] = useState(5)
  const [batchSize, setBatchSize] = useState(32)

  // Parameters for backpropagation
  const [error, setError] = useState(0.5)
  const [weightUpdateMagnitude, setWeightUpdateMagnitude] = useState(0.3)

  // Generate random data
  const generateRandomData = () => {
    if (activeTab === "feedforward") {
      setLayersCount(Math.floor(Math.random() * 3) + 3)
      setNeuronsPerLayer(Math.floor(Math.random() * 5) + 3)
      const activations = ["relu", "sigmoid", "tanh", "linear"]
      setActivationFunction(activations[Math.floor(Math.random() * activations.length)])
    } else if (activeTab === "gradient") {
      setLearningRate(Math.random() * 0.3 + 0.01)
      setEpochs(Math.floor(Math.random() * 10) + 1)
      setBatchSize(Math.pow(2, Math.floor(Math.random() * 5) + 3)) // 8, 16, 32, 64, 128
    } else if (activeTab === "backprop") {
      setError(Math.random() * 0.8 + 0.1)
      setWeightUpdateMagnitude(Math.random() * 0.8)
    }
  }

  // Toggle animation
  const toggleAnimation = () => {
    setIsAnimating(!isAnimating)
  }

  // Animation loop
  useEffect(() => {
    if (isAnimating) {
      const animate = () => {
        setAnimationStep((prev) => (prev + 1) % 120) // 120 frames for a complete cycle
        animationRef.current = requestAnimationFrame(animate)
      }
      animationRef.current = requestAnimationFrame(animate)
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isAnimating])

  // Render feedforward network
  const renderFeedforwardNetwork = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Define network dimensions
    const neuronRadius = 15
    const layerSpacing = width / (layersCount + 1)
    const maxNeuronsInLayer = neuronsPerLayer
    const neuronSpacing = height / (maxNeuronsInLayer + 1)

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "18px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    ctx.fillText("Feedforward Neural Network", width / 2, 20)

    // Draw layers
    for (let layer = 0; layer < layersCount; layer++) {
      const x = (layer + 1) * layerSpacing
      const neuronsInThisLayer =
        layer === 0 || layer === layersCount - 1 ? Math.max(2, Math.floor(neuronsPerLayer / 2)) : neuronsPerLayer

      // Draw layer label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "bottom"

      const layerLabel =
        layer === 0 ? "Input Layer" : layer === layersCount - 1 ? "Output Layer" : `Hidden Layer ${layer}`

      ctx.fillText(layerLabel, x, height - 10)

      // Draw neurons for this layer
      for (let neuron = 0; neuron < neuronsInThisLayer; neuron++) {
        const y = (neuron + 1) * neuronSpacing

        // Draw neuron
        ctx.beginPath()
        ctx.arc(x, y, neuronRadius, 0, Math.PI * 2)

        // Set neuron color based on layer type
        if (layer === 0) {
          ctx.fillStyle = "#e0e0e0" // Light gray for input
        } else if (layer === layersCount - 1) {
          ctx.fillStyle = "#c0c0c0" // Medium gray for output
        } else {
          ctx.fillStyle = "#f5f5f5" // Very light gray for hidden
        }

        // Highlight based on animation
        if (isAnimating) {
          const neuronActivationStep = (animationStep + layer * 10) % 120
          if (neuronActivationStep < 10) {
            // Brief activation highlight
            ctx.fillStyle = "#909090" // Gray for activation
          }
        }

        ctx.fill()
        ctx.strokeStyle = "#666"
        ctx.lineWidth = 1
        ctx.stroke()

        // Add activation function indicator for hidden layers
        if (layer > 0 && layer < layersCount - 1) {
          let symbol = ""

          switch (activationFunction) {
            case "relu":
              symbol = "ReLU"
              break
            case "sigmoid":
              symbol = "σ"
              break
            case "tanh":
              symbol = "tanh"
              break
            case "linear":
              symbol = "f(x)=x"
              break
          }

          ctx.fillStyle = "#333"
          ctx.font = "10px sans-serif"
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.fillText(symbol, x, y)
        }

        // Draw connections to the previous layer
        if (layer > 0) {
          const prevX = layer * layerSpacing
          const neuronsInPrevLayer = layer === 1 ? Math.max(2, Math.floor(neuronsPerLayer / 2)) : neuronsPerLayer

          for (let prevNeuron = 0; prevNeuron < neuronsInPrevLayer; prevNeuron++) {
            const prevY = (prevNeuron + 1) * neuronSpacing

            // Draw connection
            ctx.beginPath()
            ctx.moveTo(prevX + neuronRadius, prevY)
            ctx.lineTo(x - neuronRadius, y)

            // Animation effect for weights
            if (isAnimating) {
              const connectionActivationStep = (animationStep + layer * 10 + neuron * 3 + prevNeuron) % 120
              if (connectionActivationStep < 20) {
                ctx.strokeStyle = "#ffa000" // Orange for activation flow
                ctx.lineWidth = 2
              } else {
                ctx.strokeStyle = "#ccc"
                ctx.lineWidth = 0.5
              }
            } else {
              ctx.strokeStyle = "#ccc"
              ctx.lineWidth = 0.5
            }

            ctx.stroke()
          }
        }
      }
    }

    // Draw activation function info
    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "top"

    let activationDescription = ""
    switch (activationFunction) {
      case "relu":
        activationDescription = "ReLU: f(x) = max(0, x)"
        break
      case "sigmoid":
        activationDescription = "Sigmoid: f(x) = 1 / (1 + e^(-x))"
        break
      case "tanh":
        activationDescription = "Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))"
        break
      case "linear":
        activationDescription = "Linear: f(x) = x"
        break
    }

    ctx.fillText(`Activation: ${activationDescription}`, 20, height - 40)
  }

  // Render gradient descent
  const renderGradientDescent = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Define gradient descent dimensions
    const graphWidth = width - 100
    const graphHeight = height - 100
    const graphStartX = 50
    const graphStartY = 50

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "18px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    ctx.fillText("Gradient Descent Optimization", width / 2, 20)

    // Draw axes
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2

    // X-axis
    ctx.beginPath()
    ctx.moveTo(graphStartX, graphStartY + graphHeight)
    ctx.lineTo(graphStartX + graphWidth, graphStartY + graphHeight)
    ctx.stroke()

    // Y-axis
    ctx.beginPath()
    ctx.moveTo(graphStartX, graphStartY)
    ctx.lineTo(graphStartX, graphStartY + graphHeight)
    ctx.stroke()

    // Draw axis labels
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Weight", graphStartX + graphWidth / 2, graphStartY + graphHeight + 25)

    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    ctx.fillText("Loss", graphStartX - 10, graphStartY + graphHeight / 2)

    // Draw parabola (loss function)
    ctx.beginPath()
    const centerX = graphStartX + graphWidth / 2
    const amplitude = graphHeight * 0.8
    const parabolaWidth = learningRate < 0.1 ? 0.8 : learningRate < 0.2 ? 0.5 : 0.3

    for (let x = 0; x <= graphWidth; x++) {
      const normalizedX = (x / graphWidth - 0.5) / parabolaWidth
      const y = normalizedX * normalizedX * amplitude

      if (x === 0) {
        ctx.moveTo(graphStartX + x, graphStartY + graphHeight - y)
      } else {
        ctx.lineTo(graphStartX + x, graphStartY + graphHeight - y)
      }
    }

    ctx.strokeStyle = "#666"
    ctx.lineWidth = 2
    ctx.stroke()

    // Draw optimization path
    const numSteps = isAnimating ? Math.floor(animationStep / 10) + 1 : epochs
    const stepSize = (graphWidth * 0.7) / (numSteps + 1)
    let currentX = graphStartX + graphWidth * 0.15

    ctx.beginPath()

    for (let step = 0; step <= numSteps; step++) {
      const normalizedX = ((currentX - graphStartX) / graphWidth - 0.5) / parabolaWidth
      const y = normalizedX * normalizedX * amplitude

      // Plot point
      if (step === 0) {
        ctx.moveTo(currentX, graphStartY + graphHeight - y)
      } else {
        ctx.lineTo(currentX, graphStartY + graphHeight - y)
      }

      // Calculate "gradient" (derivative of the parabola)
      const gradient = (((2 * normalizedX) / parabolaWidth) * amplitude) / graphWidth

      // Update position using gradient descent
      const actualLearningRate = learningRate * 50 // Scale for visualization
      currentX = currentX - gradient * actualLearningRate
    }

    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.stroke()

    // Draw points at each step
    currentX = graphStartX + graphWidth * 0.15

    for (let step = 0; step <= numSteps; step++) {
      const normalizedX = ((currentX - graphStartX) / graphWidth - 0.5) / parabolaWidth
      const y = normalizedX * normalizedX * amplitude

      ctx.beginPath()
      ctx.arc(currentX, graphStartY + graphHeight - y, 4, 0, Math.PI * 2)

      ctx.fillStyle = step === numSteps ? "#444" : "#666"

      ctx.fill()
      ctx.strokeStyle = "#fff"
      ctx.lineWidth = 1
      ctx.stroke()

      // Calculate "gradient" (derivative of the parabola)
      const gradient = (((2 * normalizedX) / parabolaWidth) * amplitude) / graphWidth

      // Update position using gradient descent
      const actualLearningRate = learningRate * 50 // Scale for visualization
      currentX = currentX - gradient * actualLearningRate
    }

    // Draw optimization info
    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "top"
    ctx.fillText(`Learning Rate: ${learningRate.toFixed(3)}`, 20, height - 80)
    ctx.fillText(`Epochs: ${epochs}`, 20, height - 60)
    ctx.fillText(`Batch Size: ${batchSize}`, 20, height - 40)
  }

  // Render backpropagation
  const renderBackpropagation = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Define network dimensions for backpropagation visualization
    const neuronRadius = 15
    const layerSpacing = width / 5
    const neuronsPerLayer = [3, 4, 3, 2]
    const layersCount = neuronsPerLayer.length
    const maxNeuronsInLayer = Math.max(...neuronsPerLayer)
    const neuronSpacing = height / (maxNeuronsInLayer + 1)

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "18px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    ctx.fillText("Backpropagation", width / 2, 20)

    // Draw network layers
    for (let layer = 0; layer < layersCount; layer++) {
      const x = (layer + 1) * layerSpacing
      const neuronsInThisLayer = neuronsPerLayer[layer]

      // Draw layer label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "bottom"

      const layerLabel =
        layer === 0 ? "Input Layer" : layer === layersCount - 1 ? "Output Layer" : `Hidden Layer ${layer}`

      ctx.fillText(layerLabel, x, height - 10)

      // Draw neurons for this layer
      for (let neuron = 0; neuron < neuronsInThisLayer; neuron++) {
        const y = (neuron + 1) * neuronSpacing * (maxNeuronsInLayer / Math.max(neuronsInThisLayer, 1))

        // Draw neuron
        ctx.beginPath()
        ctx.arc(x, y, neuronRadius, 0, Math.PI * 2)

        // Set neuron color based on layer type
        if (layer === 0) {
          ctx.fillStyle = "#e3f2fd" // Light blue for input
        } else if (layer === layersCount - 1) {
          ctx.fillStyle = "#e8f5e9" // Light green for output
        } else {
          ctx.fillStyle = "#f5f5f5" // Light gray for hidden
        }

        // Highlight based on backward pass animation
        if (isAnimating && layer > 0) {
          const neuronActivationStep = (120 - animationStep + layer * 15) % 120
          if (neuronActivationStep < 15) {
            // Brief activation highlight for backprop
            ctx.fillStyle = "#ffccbc" // Orange/red for error signal
          }
        }

        ctx.fill()
        ctx.strokeStyle = "#666"
        ctx.lineWidth = 1
        ctx.stroke()

        // Draw error value for output layer
        if (layer === layersCount - 1) {
          const errorValue = ((error * (neuron + 1)) / neuronsInThisLayer).toFixed(2)
          ctx.fillStyle = "#d32f2f"
          ctx.font = "10px sans-serif"
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.fillText(`e: ${errorValue}`, x, y)
        }

        // Draw connections to the previous layer
        if (layer > 0) {
          const prevX = layer * layerSpacing
          const prevNeuronsInLayer = neuronsPerLayer[layer - 1]

          for (let prevNeuron = 0; prevNeuron < prevNeuronsInLayer; prevNeuron++) {
            const prevY = (prevNeuron + 1) * neuronSpacing * (maxNeuronsInLayer / Math.max(prevNeuronsInLayer, 1))

            // Draw connection
            ctx.beginPath()
            ctx.moveTo(prevX + neuronRadius, prevY)
            ctx.lineTo(x - neuronRadius, y)

            // Forward pass connections (light gray)
            ctx.strokeStyle = "#ddd"
            ctx.lineWidth = 0.5
            ctx.stroke()

            // Backward pass with animation
            if (isAnimating) {
              const backwardStep = (120 - animationStep + layer * 15) % 120
              if (backwardStep < 20) {
                // Draw backward connection (red/orange for error propagation)
                ctx.beginPath()
                ctx.moveTo(x - neuronRadius, y)
                ctx.lineTo(prevX + neuronRadius, prevY)
                ctx.strokeStyle = "#ff5722"
                ctx.lineWidth = 2
                ctx.stroke()

                // Weight update visualization
                if (backwardStep > 5 && backwardStep < 15) {
                  // Draw weight update symbol
                  const midX = (prevX + x) / 2
                  const midY = (prevY + y) / 2

                  ctx.fillStyle = "#ff5722"
                  ctx.font = "12px sans-serif"
                  ctx.textAlign = "center"
                  ctx.textBaseline = "middle"
                  ctx.fillText("Δw", midX, midY - 8)
                }
              }
            }
          }
        }
      }
    }

    // Draw backpropagation info
    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "top"
    ctx.fillText(`Error Magnitude: ${error.toFixed(2)}`, 20, height - 60)
    ctx.fillText(`Weight Update Magnitude: ${weightUpdateMagnitude.toFixed(2)}`, 20, height - 40)

    // Draw backpropagation algorithm phases
    if (isAnimating) {
      const phase = Math.floor(animationStep / 30) % 4

      ctx.fillStyle = "#000"
      ctx.font = "16px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"

      let phaseText = ""
      switch (phase) {
        case 0:
          phaseText = "1. Forward Pass"
          break
        case 1:
          phaseText = "2. Calculate Output Error"
          break
        case 2:
          phaseText = "3. Backpropagate Error"
          break
        case 3:
          phaseText = "4. Update Weights"
          break
      }

      ctx.fillText(phaseText, width / 2, height - 60)
    }
  }

  // Update visualization when parameters change
  useEffect(() => {
    if (activeTab === "feedforward") {
      renderFeedforwardNetwork()
    } else if (activeTab === "gradient") {
      renderGradientDescent()
    } else if (activeTab === "backprop") {
      renderBackpropagation()
    }
  }, [
    activeTab,
    layersCount,
    neuronsPerLayer,
    activationFunction,
    learningRate,
    epochs,
    batchSize,
    error,
    weightUpdateMagnitude,
    animationStep,
    isAnimating,
  ])

  return (
    <Card className="w-full border-neutral-300 bg-white">
      <CardContent className="pt-6">
        <div className="flex justify-between items-center mb-4">
          <div className="text-lg font-medium text-neutral-900">Neural Network Visualization</div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={generateRandomData} className="flex items-center gap-1">
              <Shuffle className="h-4 w-4" /> Random Data
            </Button>
            <Button variant="outline" size="sm" onClick={toggleAnimation} className="flex items-center gap-1">
              {isAnimating ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {isAnimating ? "Pause" : "Animate"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-1"
              onClick={() => {
                const canvas = canvasRef.current
                if (canvas) {
                  const link = document.createElement("a")
                  link.download = `neural-network-${activeTab}.png`
                  link.href = canvas.toDataURL("image/png")
                  link.click()
                }
              }}
            >
              <Download className="h-4 w-4" /> Save Image
            </Button>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-4">
          <TabsList className="grid w-full grid-cols-3 bg-neutral-100 text-neutral-900">
            <TabsTrigger value="feedforward" className="data-[state=active]:bg-white">
              Feedforward
            </TabsTrigger>
            <TabsTrigger value="gradient" className="data-[state=active]:bg-white">
              Gradient Descent
            </TabsTrigger>
            <TabsTrigger value="backprop" className="data-[state=active]:bg-white">
              Backpropagation
            </TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1 order-2 lg:order-1">
            <canvas
              ref={canvasRef}
              width={width}
              height={height}
              className="w-full h-auto bg-white border border-neutral-300 rounded-md"
            />
          </div>

          <div className="w-full lg:w-64 space-y-6 order-1 lg:order-2">
            {activeTab === "feedforward" && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="layersCount" className="text-neutral-900">
                      Number of Layers
                    </Label>
                    <span className="text-sm text-neutral-600">{layersCount}</span>
                  </div>
                  <Slider
                    id="layersCount"
                    min={3}
                    max={6}
                    step={1}
                    value={[layersCount]}
                    onValueChange={(value) => setLayersCount(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="neuronsPerLayer" className="text-neutral-900">
                      Neurons Per Layer
                    </Label>
                    <span className="text-sm text-neutral-600">{neuronsPerLayer}</span>
                  </div>
                  <Slider
                    id="neuronsPerLayer"
                    min={2}
                    max={8}
                    step={1}
                    value={[neuronsPerLayer]}
                    onValueChange={(value) => setNeuronsPerLayer(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="activationFunction" className="text-neutral-900">
                    Activation Function
                  </Label>
                  <Select value={activationFunction} onValueChange={setActivationFunction}>
                    <SelectTrigger id="activationFunction">
                      <SelectValue placeholder="Select activation function" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="relu">ReLU</SelectItem>
                      <SelectItem value="sigmoid">Sigmoid</SelectItem>
                      <SelectItem value="tanh">Tanh</SelectItem>
                      <SelectItem value="linear">Linear</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </>
            )}

            {activeTab === "gradient" && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="learningRate" className="text-neutral-900">
                      Learning Rate
                    </Label>
                    <span className="text-sm text-neutral-600">{learningRate.toFixed(3)}</span>
                  </div>
                  <Slider
                    id="learningRate"
                    min={0.01}
                    max={0.3}
                    step={0.01}
                    value={[learningRate]}
                    onValueChange={(value) => setLearningRate(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="epochs" className="text-neutral-900">
                      Epochs
                    </Label>
                    <span className="text-sm text-neutral-600">{epochs}</span>
                  </div>
                  <Slider
                    id="epochs"
                    min={1}
                    max={20}
                    step={1}
                    value={[epochs]}
                    onValueChange={(value) => setEpochs(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="batchSize" className="text-neutral-900">
                      Batch Size
                    </Label>
                    <span className="text-sm text-neutral-600">{batchSize}</span>
                  </div>
                  <Slider
                    id="batchSize"
                    min={8}
                    max={128}
                    step={8}
                    value={[batchSize]}
                    onValueChange={(value) => setBatchSize(value[0])}
                    className="notebook-slider"
                  />
                </div>
              </>
            )}

            {activeTab === "backprop" && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="error" className="text-neutral-900">
                      Error Magnitude
                    </Label>
                    <span className="text-sm text-neutral-600">{error.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="error"
                    min={0.1}
                    max={1}
                    step={0.05}
                    value={[error]}
                    onValueChange={(value) => setError(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="weightUpdateMagnitude" className="text-neutral-900">
                      Weight Update Magnitude
                    </Label>
                    <span className="text-sm text-neutral-600">{weightUpdateMagnitude.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="weightUpdateMagnitude"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={[weightUpdateMagnitude]}
                    onValueChange={(value) => setWeightUpdateMagnitude(value[0])}
                    className="notebook-slider"
                  />
                </div>
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
