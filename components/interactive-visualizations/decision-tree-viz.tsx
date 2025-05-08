"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Shuffle, Download, Play, Pause, RotateCcw } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"

interface DecisionTreeVisualizationProps {
  width?: number
  height?: number
}

// Define dataset types
type DatasetType = "classification" | "regression"

// Define sample datasets
const sampleDatasets = {
  classification: [
    { name: "Binary Classification", features: ["Age", "Income"], classes: ["Yes", "No"] },
    { name: "Multi-class", features: ["Feature X", "Feature Y"], classes: ["Class A", "Class B", "Class C"] },
    { name: "Iris-like", features: ["Sepal Length", "Petal Width"], classes: ["Setosa", "Versicolor", "Virginica"] },
  ],
  regression: [
    { name: "House Prices", features: ["Size (sqft)", "Age (years)"], target: "Price ($)" },
    { name: "Salary Prediction", features: ["Experience (years)", "Education Level"], target: "Salary ($)" },
  ],
}

export default function DecisionTreeVisualization({ width = 800, height = 500 }: DecisionTreeVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [activeTab, setActiveTab] = useState("tree")
  const [datasetType, setDatasetType] = useState<DatasetType>("classification")
  const [selectedDataset, setSelectedDataset] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)
  const [showLabels, setShowLabels] = useState(true)
  const animationRef = useRef<number | null>(null)

  // Parameters for tree visualization
  const [maxDepth, setMaxDepth] = useState(3)
  const [minSamplesSplit, setMinSamplesSplit] = useState(5)
  const [randomness, setRandomness] = useState(0.3)

  // Parameters for decision boundary visualization
  const [complexity, setComplexity] = useState(5)
  const [noiseLevel, setNoiseLevel] = useState(0.2)
  const [featureImportance, setFeatureImportance] = useState(0.7)
  const [numPoints, setNumPoints] = useState(100)
  const [animationSpeed, setAnimationSpeed] = useState(0.5)

  // Animation state
  const [animationProgress, setAnimationProgress] = useState(0)

  // Generate random data
  const generateRandomData = () => {
    if (activeTab === "tree") {
      setMaxDepth(Math.floor(Math.random() * 4) + 2)
      setMinSamplesSplit(Math.floor(Math.random() * 8) + 2)
      setRandomness(Math.random() * 0.6)
    } else if (activeTab === "boundary") {
      setComplexity(Math.floor(Math.random() * 8) + 2)
      setNoiseLevel(Math.random() * 0.4)
      setFeatureImportance(Math.random() * 0.6 + 0.2)
      setNumPoints(Math.floor(Math.random() * 150) + 50)
    }
  }

  // Toggle animation
  const toggleAnimation = () => {
    setIsAnimating(!isAnimating)
  }

  // Reset animation
  const resetAnimation = () => {
    setAnimationProgress(0)
    setIsAnimating(false)
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }
  }

  // Animation loop with improved handling
  useEffect(() => {
    let lastTime = 0
    const animate = (time: number) => {
      if (!lastTime) lastTime = time
      const deltaTime = time - lastTime
      lastTime = time

      // Update animation progress based on speed
      setAnimationProgress((prev) => {
        const newProgress = prev + deltaTime * 0.0001 * animationSpeed
        if (newProgress >= 1) {
          // Schedule stopping animation in next frame to avoid state update conflicts
          setTimeout(() => setIsAnimating(false), 0)
          return 1
        }
        return newProgress
      })

      // Continue animation if still animating
      if (isAnimating) {
        animationRef.current = requestAnimationFrame(animate)
      }
    }

    if (isAnimating) {
      animationRef.current = requestAnimationFrame(animate)
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isAnimating, animationSpeed]) // Remove animationProgress from dependencies to avoid re-triggering

  // Render decision tree
  const renderDecisionTree = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Calculate tree structure based on parameters
    const depth = Math.floor(maxDepth)
    const nodeSize = 30
    const levelHeight = height / (depth + 1)

    // Get current dataset
    const dataset = sampleDatasets[datasetType][selectedDataset]
    const features = dataset.features
    const classes = datasetType === "classification" ? dataset.classes : null
    const targetName = datasetType === "regression" ? (dataset as any).target : null

    // Function to draw a node
    const drawNode = (
      x: number,
      y: number,
      level: number,
      parentX?: number,
      parentY?: number,
      isLeft?: boolean,
      pathProbability = 1,
    ) => {
      // Calculate if this node should be visible based on animation progress
      const nodeDepthRatio = level / depth
      const shouldDrawNode = animationProgress >= nodeDepthRatio

      if (!shouldDrawNode) return

      // Draw connection to parent
      if (parentX !== undefined && parentY !== undefined) {
        const connectionProgress = Math.min(1, (animationProgress - nodeDepthRatio + 0.2) / 0.2)

        if (connectionProgress > 0) {
          // Calculate intermediate points for animation
          const midX = parentX + (x - parentX) * connectionProgress
          const midY = parentY + (y - parentY) * connectionProgress

          ctx.beginPath()
          ctx.moveTo(parentX, parentY)
          ctx.lineTo(midX, midY)
          ctx.strokeStyle = "#666"
          ctx.lineWidth = 2
          ctx.stroke()

          // Add split condition text if animation is complete for this connection
          if (connectionProgress >= 1 && showLabels) {
            const textX = (parentX + x) / 2
            const textY = (parentY + y) / 2
            ctx.fillStyle = "#666"
            ctx.font = "12px sans-serif"
            ctx.textAlign = "center"

            // Get feature name for this level
            const featureIndex = (level - 1) % features.length
            const featureName = features[featureIndex]

            // Draw split condition
            ctx.fillText(
              isLeft
                ? `${featureName} ≤ ${(Math.random() * 10).toFixed(1)}`
                : `${featureName} > ${(Math.random() * 10).toFixed(1)}`,
              textX,
              textY - 10,
            )
          }
        }
      }

      // Only draw the node if it should be visible based on animation progress
      if (animationProgress >= nodeDepthRatio) {
        // Draw node
        ctx.beginPath()
        ctx.arc(x, y, nodeSize, 0, Math.PI * 2)

        // Color based on level (leaf nodes are different)
        if (level >= depth) {
          // Leaf nodes - use different shades of gray
          if (datasetType === "classification") {
            const classIndex = Math.floor(Math.random() * (classes?.length || 2))
            const grayValue = 200 - classIndex * 40
            ctx.fillStyle = `rgb(${grayValue}, ${grayValue}, ${grayValue})`
          } else {
            // For regression, use a gradient of grays
            const predictedValue = Math.random() * 100
            const normalizedValue = predictedValue / 100
            // Create a gradient from light to dark gray
            const grayValue = Math.floor(255 - normalizedValue * 200)
            ctx.fillStyle = `rgb(${grayValue}, ${grayValue}, ${grayValue})`
          }
        } else {
          ctx.fillStyle = "#555" // Decision nodes
        }
        ctx.fill()
        ctx.strokeStyle = "#333"
        ctx.lineWidth = 2
        ctx.stroke()

        // Add text
        if (showLabels) {
          ctx.fillStyle = "#fff"
          ctx.textAlign = "center"
          ctx.textBaseline = "middle"
          ctx.font = "12px sans-serif"

          if (level >= depth) {
            // For leaf nodes, show class and probability
            if (datasetType === "classification") {
              const classIndex = Math.floor(Math.random() * (classes?.length || 2))
              const className = classes?.[classIndex] || "Class " + classIndex
              const probability = (0.5 + Math.random() * 0.4).toFixed(2)
              ctx.fillText(`${className}`, x, y - 5)
              ctx.fillText(`p: ${probability}`, x, y + 10)
            } else {
              // For regression, show predicted value
              const predictedValue = Math.floor(Math.random() * 100)
              ctx.fillText(`${targetName}:`, x, y - 5)
              ctx.fillText(`${predictedValue}`, x, y + 10)
            }
          } else {
            // For decision nodes, show the node ID and samples
            const samples = Math.floor(100 * pathProbability)
            ctx.fillText(`Node ${level + 1}`, x, y - 5)
            ctx.fillText(`n=${samples}`, x, y + 10)
          }
        }
      }

      // Stop recursion at max depth or based on minSamplesSplit
      if (level >= depth) return
      if (pathProbability * 100 < minSamplesSplit && level > 0) return

      // Add randomness to the split position
      const splitRatio = 0.5 + (Math.random() - 0.5) * randomness * 0.3

      // Calculate child positions
      const nextLevel = level + 1
      const childSpacing = width / Math.pow(2, nextLevel + 1)
      const leftX = x - childSpacing
      const rightX = x + childSpacing
      const childY = y + levelHeight

      // Calculate path probabilities for children
      const leftProb = pathProbability * splitRatio
      const rightProb = pathProbability * (1 - splitRatio)

      // Draw children recursively
      drawNode(leftX, childY, nextLevel, x, y, true, leftProb)
      drawNode(rightX, childY, nextLevel, x, y, false, rightProb)
    }

    // Start drawing from the root
    drawNode(width / 2, levelHeight, 0)
  }

  // Render decision boundary
  const renderDecisionBoundary = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Set up coordinate system
    const margin = 50
    const plotWidth = width - 2 * margin
    const plotHeight = height - 2 * margin

    // Get current dataset
    const dataset = sampleDatasets[datasetType][selectedDataset]
    const features = dataset.features
    const classes = datasetType === "classification" ? dataset.classes : null
    const targetName = datasetType === "regression" ? (dataset as any).target : null

    // Draw axes
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1

    // X-axis
    ctx.beginPath()
    ctx.moveTo(margin, height - margin)
    ctx.lineTo(width - margin, height - margin)
    ctx.stroke()

    // Y-axis
    ctx.beginPath()
    ctx.moveTo(margin, margin)
    ctx.lineTo(margin, height - margin)
    ctx.stroke()

    // Axis labels
    if (showLabels) {
      ctx.fillStyle = "#333"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.fillText(features[0], width / 2, height - 15)
      ctx.textAlign = "right"
      ctx.textBaseline = "middle"
      ctx.fillText(features[1], 15, height / 2)
    }

    // Generate decision boundaries
    // We'll create a grid and assign each cell a class
    const gridSize = Math.floor(50 * animationProgress)
    if (gridSize <= 0) return

    const cellWidth = plotWidth / gridSize
    const cellHeight = plotHeight / gridSize

    // Function to determine the class of a point based on decision tree logic
    const getClass = (x: number, y: number) => {
      // Normalize coordinates to [0, 1]
      const normX = x / plotWidth
      const normY = y / plotHeight

      // Apply feature importance - one feature might be more important
      const weightedX = normX * featureImportance
      const weightedY = normY * (1 - featureImportance)

      // Add noise
      const noise = (Math.random() - 0.5) * noiseLevel

      // Create complex decision boundaries based on complexity parameter
      let result: number | string = 0

      // Simple threshold for low complexity
      if (complexity <= 3) {
        if (weightedX + weightedY + noise > 0.5) {
          result = datasetType === "classification" ? 1 : 70 + Math.random() * 30
        } else {
          result = datasetType === "classification" ? 0 : Math.random() * 30
        }
      }
      // Checkerboard pattern for medium complexity
      else if (complexity <= 6) {
        const gridX = Math.floor(normX * complexity)
        const gridY = Math.floor(normY * complexity)
        if ((gridX + gridY) % 2 === 0) {
          result = datasetType === "classification" ? 1 : 70 + Math.random() * 30
        } else {
          result = datasetType === "classification" ? 0 : Math.random() * 30
        }
      }
      // Circular pattern for high complexity
      else {
        const centerX = 0.5
        const centerY = 0.5
        const distance = Math.sqrt(Math.pow(normX - centerX, 2) + Math.pow(normY - centerY, 2))
        if (distance < 0.3 || (distance > 0.4 && distance < 0.5)) {
          result = datasetType === "classification" ? 1 : 70 + Math.random() * 30
        } else {
          result = datasetType === "classification" ? 0 : Math.random() * 30
        }
      }

      // For multi-class classification, add more classes
      if (datasetType === "classification" && (classes?.length || 0) > 2) {
        // Add a third class in a specific region
        if (normX > 0.7 && normY > 0.7) {
          result = 2
        }
      }

      return result
    }

    // Draw the decision boundary grid
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = margin + i * cellWidth
        const y = margin + j * cellHeight

        const classValue = getClass(i * cellWidth, j * cellHeight)

        if (datasetType === "classification") {
          // For classification, use shades of gray
          const grayShade = 230 - (classValue as number) * 50
          ctx.fillStyle = `rgba(${grayShade}, ${grayShade}, ${grayShade}, 0.2)`
        } else {
          // For regression, use a grayscale gradient
          const normalizedValue = (classValue as number) / 100
          const grayValue = Math.floor(255 - normalizedValue * 200)
          ctx.fillStyle = `rgba(${grayValue}, ${grayValue}, ${grayValue}, 0.2)`
        }

        ctx.fillRect(x, y, cellWidth, cellHeight)
      }
    }

    // Generate and draw data points
    const pointsToShow = Math.floor(numPoints * animationProgress)
    const points: { x: number; y: number; classValue: number | string }[] = []

    // Generate all points first
    for (let i = 0; i < numPoints; i++) {
      const x = Math.random() * plotWidth
      const y = Math.random() * plotHeight
      const classValue = getClass(x, y)
      points.push({ x, y, classValue })
    }

    // Draw visible points
    for (let i = 0; i < pointsToShow; i++) {
      const { x, y, classValue } = points[i]

      ctx.beginPath()
      ctx.arc(margin + x, margin + y, 5, 0, Math.PI * 2)

      if (datasetType === "classification") {
        // For classification, use distinct colors for classes
        const colors = ["#2196F3", "#4CAF50", "#FFC107"]
        ctx.fillStyle = colors[(classValue as number) % colors.length]
        ctx.strokeStyle = "#333"
      } else {
        // For regression, use a color gradient
        const normalizedValue = (classValue as number) / 100
        const r = Math.floor(normalizedValue * 255)
        const b = Math.floor((1 - normalizedValue) * 255)
        ctx.fillStyle = `rgb(${r}, 100, ${b})`
        ctx.strokeStyle = "#333"
      }

      ctx.fill()
      ctx.lineWidth = 1
      ctx.stroke()
    }

    // Draw legend
    if (showLabels) {
      ctx.font = "12px sans-serif"
      ctx.textAlign = "left"
      ctx.textBaseline = "middle"

      if (datasetType === "classification") {
        // Draw legend for classification
        const legendClasses = classes || ["Class 0", "Class 1"]
        const colors = ["#2196F3", "#4CAF50", "#FFC107", "#9C27B0", "#F44336"]

        legendClasses.forEach((className, idx) => {
          const y = 30 + idx * 25

          ctx.beginPath()
          ctx.arc(width - 120, y, 5, 0, Math.PI * 2)
          ctx.fillStyle = colors[idx % colors.length]
          ctx.fill()
          ctx.strokeStyle = "#333"
          ctx.lineWidth = 1
          ctx.stroke()

          ctx.fillStyle = "#333"
          ctx.fillText(className, width - 105, y)
        })
      } else {
        // Draw legend for regression
        ctx.fillStyle = "#333"
        ctx.fillText(`${targetName} Value:`, width - 120, 20)

        // Draw gradient bar
        const gradientHeight = 100
        const gradientWidth = 20

        for (let i = 0; i < gradientHeight; i++) {
          const normalizedValue = 1 - i / gradientHeight
          const r = Math.floor(normalizedValue * 255)
          const b = Math.floor((1 - normalizedValue) * 255)
          ctx.fillStyle = `rgb(${r}, 100, ${b})`
          ctx.fillRect(width - 120, 30 + i, gradientWidth, 1)
        }

        // Add min/max labels
        ctx.fillStyle = "#333"
        ctx.textAlign = "left"
        ctx.fillText("100", width - 95, 30)
        ctx.fillText("0", width - 95, 130)
      }
    }
  }

  // Update visualization when parameters change
  useEffect(() => {
    if (activeTab === "tree") {
      renderDecisionTree()
    } else if (activeTab === "boundary") {
      renderDecisionBoundary()
    }
  }, [
    activeTab,
    maxDepth,
    minSamplesSplit,
    randomness,
    complexity,
    noiseLevel,
    featureImportance,
    numPoints,
    animationProgress,
    datasetType,
    selectedDataset,
    showLabels,
  ])

  // Handle dataset type change
  const handleDatasetTypeChange = (value: string) => {
    setDatasetType(value as DatasetType)
    setSelectedDataset(0) // Reset to first dataset of the new type
    resetAnimation()
  }

  return (
    <Card className="w-full border-neutral-300 bg-white">
      <CardContent className="pt-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4 gap-4">
          <div className="text-lg font-medium text-neutral-900">Decision Tree Visualization</div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={generateRandomData} className="flex items-center gap-1">
              <Shuffle className="h-4 w-4" /> Random Data
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-1"
              onClick={() => {
                const canvas = canvasRef.current
                if (canvas) {
                  const link = document.createElement("a")
                  link.download = `decision-tree-${activeTab}.png`
                  link.href = canvas.toDataURL("image/png")
                  link.click()
                }
              }}
            >
              <Download className="h-4 w-4" /> Save Image
            </Button>
          </div>
        </div>

        <div className="flex flex-col md:flex-row gap-4 mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Label htmlFor="datasetType">Dataset Type:</Label>
              <Select value={datasetType} onValueChange={handleDatasetTypeChange}>
                <SelectTrigger id="datasetType" className="w-[180px]">
                  <SelectValue placeholder="Select type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="classification">Classification</SelectItem>
                  <SelectItem value="regression">Regression</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Label htmlFor="dataset">Dataset:</Label>
              <Select
                value={selectedDataset.toString()}
                onValueChange={(value) => {
                  setSelectedDataset(Number.parseInt(value))
                  resetAnimation()
                }}
              >
                <SelectTrigger id="dataset" className="w-[180px]">
                  <SelectValue placeholder="Select dataset" />
                </SelectTrigger>
                <SelectContent>
                  {sampleDatasets[datasetType].map((dataset, index) => (
                    <SelectItem key={index} value={index.toString()}>
                      {dataset.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Label htmlFor="showLabels" className="text-sm">
              Show Labels:
            </Label>
            <Switch id="showLabels" checked={showLabels} onCheckedChange={setShowLabels} />
          </div>
        </div>

        <Tabs
          value={activeTab}
          onValueChange={(value) => {
            setActiveTab(value)
            resetAnimation()
          }}
          className="mb-4"
        >
          <TabsList className="grid w-full grid-cols-2 bg-neutral-100 text-neutral-900">
            <TabsTrigger value="tree" className="data-[state=active]:bg-white">
              Tree Structure
            </TabsTrigger>
            <TabsTrigger value="boundary" className="data-[state=active]:bg-white">
              Decision Boundary
            </TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1 order-2 lg:order-1">
            <div className="relative">
              <canvas
                ref={canvasRef}
                width={width}
                height={height}
                className="w-full h-auto bg-white border border-neutral-300 rounded-md"
              />
              <div className="absolute bottom-4 right-4 flex gap-2">
                <Button variant="outline" size="sm" onClick={toggleAnimation} className="bg-white">
                  {isAnimating ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>
                <Button variant="outline" size="sm" onClick={resetAnimation} className="bg-white">
                  <RotateCcw className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <div className="mt-2 bg-neutral-100 rounded-full h-2 w-full">
              <div
                className="bg-neutral-700 h-2 rounded-full transition-all duration-300"
                style={{ width: `${animationProgress * 100}%` }}
              ></div>
            </div>
          </div>

          <div className="w-full lg:w-64 space-y-6 order-1 lg:order-2">
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="animationSpeed" className="text-neutral-900">
                  Animation Speed
                </Label>
                <span className="text-sm text-neutral-600">{animationSpeed.toFixed(1)}x</span>
              </div>
              <Slider
                id="animationSpeed"
                min={0.1}
                max={2}
                step={0.1}
                value={[animationSpeed]}
                onValueChange={(value) => setAnimationSpeed(value[0])}
                className="notebook-slider"
              />
            </div>

            {activeTab === "tree" && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="maxDepth" className="text-neutral-900">
                      Max Depth
                    </Label>
                    <span className="text-sm text-neutral-600">{maxDepth}</span>
                  </div>
                  <Slider
                    id="maxDepth"
                    min={1}
                    max={5}
                    step={1}
                    value={[maxDepth]}
                    onValueChange={(value) => {
                      setMaxDepth(value[0])
                      resetAnimation()
                    }}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="minSamplesSplit" className="text-neutral-900">
                      Min Samples Split
                    </Label>
                    <span className="text-sm text-neutral-600">{minSamplesSplit}</span>
                  </div>
                  <Slider
                    id="minSamplesSplit"
                    min={2}
                    max={20}
                    step={1}
                    value={[minSamplesSplit]}
                    onValueChange={(value) => {
                      setMinSamplesSplit(value[0])
                      resetAnimation()
                    }}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="randomness" className="text-neutral-900">
                      Tree Randomness
                    </Label>
                    <span className="text-sm text-neutral-600">{randomness.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="randomness"
                    min={0}
                    max={1}
                    step={0.05}
                    value={[randomness]}
                    onValueChange={(value) => {
                      setRandomness(value[0])
                      resetAnimation()
                    }}
                    className="notebook-slider"
                  />
                </div>
              </>
            )}

            {activeTab === "boundary" && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="complexity" className="text-neutral-900">
                      Boundary Complexity
                    </Label>
                    <span className="text-sm text-neutral-600">{complexity}</span>
                  </div>
                  <Slider
                    id="complexity"
                    min={1}
                    max={10}
                    step={1}
                    value={[complexity]}
                    onValueChange={(value) => {
                      setComplexity(value[0])
                      resetAnimation()
                    }}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="noiseLevel" className="text-neutral-900">
                      Noise Level
                    </Label>
                    <span className="text-sm text-neutral-600">{noiseLevel.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="noiseLevel"
                    min={0}
                    max={0.5}
                    step={0.05}
                    value={[noiseLevel]}
                    onValueChange={(value) => {
                      setNoiseLevel(value[0])
                      resetAnimation()
                    }}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="featureImportance" className="text-neutral-900">
                      Feature 1 Importance
                    </Label>
                    <span className="text-sm text-neutral-600">{featureImportance.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="featureImportance"
                    min={0}
                    max={1}
                    step={0.05}
                    value={[featureImportance]}
                    onValueChange={(value) => {
                      setFeatureImportance(value[0])
                      resetAnimation()
                    }}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="numPoints" className="text-neutral-900">
                      Number of Points
                    </Label>
                    <span className="text-sm text-neutral-600">{numPoints}</span>
                  </div>
                  <Slider
                    id="numPoints"
                    min={20}
                    max={200}
                    step={10}
                    value={[numPoints]}
                    onValueChange={(value) => {
                      setNumPoints(value[0])
                      resetAnimation()
                    }}
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
