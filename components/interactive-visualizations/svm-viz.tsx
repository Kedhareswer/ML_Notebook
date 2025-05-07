"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Shuffle, Download } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface SVMVisualizationProps {
  width?: number
  height?: number
}

export default function SVMVisualization({ width = 600, height = 400 }: SVMVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [activeTab, setActiveTab] = useState("linear")

  // Parameters for linear SVM
  const [c, setC] = useState(1.0)
  const [noise, setNoise] = useState(0.3)
  const [margin, setMargin] = useState(0.5)

  // Parameters for non-linear SVM
  const [kernel, setKernel] = useState("rbf")
  const [gamma, setGamma] = useState(0.5)
  const [complexity, setComplexity] = useState(3)

  // Generate random data
  const generateRandomData = () => {
    if (activeTab === "linear") {
      setC(Math.random() * 2 + 0.1)
      setNoise(Math.random() * 0.5)
      setMargin(Math.random() * 0.8 + 0.2)
    } else if (activeTab === "nonlinear") {
      setGamma(Math.random() * 1.5 + 0.1)
      setComplexity(Math.floor(Math.random() * 5) + 1)
      const kernels = ["rbf", "poly", "sigmoid"]
      setKernel(kernels[Math.floor(Math.random() * kernels.length)])
    }
  }

  // Render linear SVM
  const renderLinearSVM = () => {
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
    ctx.fillStyle = "#333"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Feature 1", width / 2, height - 15)
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    ctx.fillText("Feature 2", 15, height / 2)

    // Generate two classes of points
    const points = []
    const numPoints = 50

    // Calculate margin width based on C parameter
    const marginWidth = 20 + (1 / c) * 40

    // Generate points for class 1 (above the line)
    for (let i = 0; i < numPoints / 2; i++) {
      const x = margin + Math.random() * plotWidth
      const y = margin + Math.random() * plotHeight

      // Determine if point is above or below the decision boundary
      // y = -x + height (simplified linear boundary)
      const boundaryY = -x + height

      // Add noise to create some overlap
      const noiseValue = (Math.random() - 0.5) * noise * 200

      if (y < boundaryY - marginWidth + noiseValue) {
        points.push({ x, y, class: 1 })
      }
    }

    // Generate points for class 2 (below the line)
    for (let i = 0; i < numPoints / 2; i++) {
      const x = margin + Math.random() * plotWidth
      const y = margin + Math.random() * plotHeight

      // Determine if point is above or below the decision boundary
      const boundaryY = -x + height

      // Add noise to create some overlap
      const noiseValue = (Math.random() - 0.5) * noise * 200

      if (y > boundaryY + marginWidth + noiseValue) {
        points.push({ x, y, class: 2 })
      }
    }

    // Draw decision boundary
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(margin, height - margin)
    ctx.lineTo(width - margin, margin)
    ctx.stroke()

    // Draw margin boundaries
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1
    ctx.setLineDash([5, 3])

    // Upper margin line
    ctx.beginPath()
    ctx.moveTo(margin, height - margin - marginWidth)
    ctx.lineTo(width - margin - marginWidth, margin)
    ctx.stroke()

    // Lower margin line
    ctx.beginPath()
    ctx.moveTo(margin + marginWidth, height - margin)
    ctx.lineTo(width - margin, margin + marginWidth)
    ctx.stroke()

    ctx.setLineDash([])

    // Draw points
    points.forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 6, 0, Math.PI * 2)

      if (point.class === 1) {
        ctx.fillStyle = "#000"
      } else {
        ctx.fillStyle = "#fff"
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1
        ctx.stroke()
      }

      ctx.fill()
    })

    // Identify support vectors (points near the margin)
    const supportVectors = points.filter((p) => {
      const boundaryY = -p.x + height
      const distance = Math.abs(p.y - boundaryY)
      return distance < marginWidth * 1.2
    })

    // Draw support vectors
    supportVectors.forEach((sv) => {
      ctx.beginPath()
      ctx.arc(sv.x, sv.y, 10, 0, Math.PI * 2)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 2
      ctx.stroke()
    })

    // Draw legend
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"

    // Class 1
    ctx.beginPath()
    ctx.arc(width - 100, 30, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillText("Class 1", width - 85, 30)

    // Class 2
    ctx.beginPath()
    ctx.arc(width - 100, 50, 4, 0, Math.PI * 2)
    ctx.fillStyle = "#fff"
    ctx.fill()
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 1
    ctx.stroke()
    ctx.fillStyle = "#000"
    ctx.fillText("Class 2", width - 85, 50)

    // Support Vector
    ctx.beginPath()
    ctx.arc(width - 100, 70, 4, 0, Math.PI * 2)
    ctx.fillStyle = "#000"
    ctx.fill()
    ctx.beginPath()
    ctx.arc(width - 100, 70, 8, 0, Math.PI * 2)
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.stroke()
    ctx.fillStyle = "#000"
    ctx.fillText("Support Vector", width - 85, 70)
  }

  // Render non-linear SVM
  const renderNonLinearSVM = () => {
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
    ctx.fillStyle = "#333"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Feature 1", width / 2, height - 15)
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    ctx.fillText("Feature 2", 15, height / 2)

    // Generate decision boundary based on kernel
    const gridSize = 50
    const cellWidth = plotWidth / gridSize
    const cellHeight = plotHeight / gridSize

    // Function to determine the class of a point based on kernel
    const getClass = (x: number, y: number) => {
      // Normalize coordinates to [-1, 1]
      const normX = (x / plotWidth) * 2 - 1
      const normY = (y / plotHeight) * 2 - 1

      let result = 0

      if (kernel === "rbf") {
        // RBF kernel creates circular boundaries
        const centerX = 0
        const centerY = 0
        const distance = Math.sqrt(Math.pow(normX - centerX, 2) + Math.pow(normY - centerY, 2))
        if (distance < 0.5 * gamma) result = 1
      } else if (kernel === "poly") {
        // Polynomial kernel creates more complex boundaries
        const polyValue = Math.pow(normX * normY + 0.5, complexity)
        if (polyValue > 0.5) result = 1
      } else if (kernel === "sigmoid") {
        // Sigmoid kernel
        const sigmoidValue = Math.tanh(gamma * normX * normY + 0.5)
        if (sigmoidValue > 0) result = 1
      }

      return result
    }

    // Draw decision boundary grid with smoother transitions
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = i * cellWidth
        const y = j * cellHeight

        const classValue = getClass(x, y)

        // Use more distinct colors with better opacity for visibility
        ctx.fillStyle = classValue === 1 ? "rgba(33, 150, 243, 0.2)" : "rgba(244, 67, 54, 0.2)"
        ctx.fillRect(margin + x, margin + y, cellWidth, cellHeight)
      }
    }

    // Generate and draw data points
    const numPoints = 100
    const points = []

    for (let i = 0; i < numPoints; i++) {
      const x = Math.random() * plotWidth
      const y = Math.random() * plotHeight

      const classValue = getClass(x, y)

      points.push({
        x: margin + x,
        y: margin + y,
        class: classValue,
      })
    }

    // Draw points
    points.forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)

      if (point.class === 1) {
        ctx.fillStyle = "#000"
      } else {
        ctx.fillStyle = "#fff"
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1
        ctx.stroke()
      }

      ctx.fill()
    })

    // Draw legend
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"

    // Class 1
    ctx.beginPath()
    ctx.arc(width - 100, 30, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillText("Class 1", width - 85, 30)

    // Class 2
    ctx.beginPath()
    ctx.arc(width - 100, 50, 4, 0, Math.PI * 2)
    ctx.fillStyle = "#fff"
    ctx.fill()
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 1
    ctx.stroke()
    ctx.fillStyle = "#000"
    ctx.fillText("Class 2", width - 85, 50)

    // Kernel info
    ctx.fillText(`Kernel: ${kernel}`, width - 150, 80)
    ctx.fillText(`Gamma: ${gamma.toFixed(2)}`, width - 150, 100)
    if (kernel === "poly") {
      ctx.fillText(`Degree: ${complexity}`, width - 150, 120)
    }
  }

  // Update visualization when parameters change
  useEffect(() => {
    if (activeTab === "linear") {
      renderLinearSVM()
    } else if (activeTab === "nonlinear") {
      renderNonLinearSVM()
    }
  }, [activeTab, c, noise, margin, kernel, gamma, complexity])

  return (
    <Card className="w-full border-neutral-300 bg-white">
      <CardContent className="pt-6">
        <div className="flex justify-between items-center mb-4">
          <div className="text-lg font-medium text-neutral-900">SVM Visualization</div>
          <div className="flex gap-2">
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
                  link.download = `svm-${activeTab}.png`
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
          <TabsList className="grid w-full grid-cols-2 bg-neutral-100 text-neutral-900">
            <TabsTrigger value="linear" className="data-[state=active]:bg-white">
              Linear SVM
            </TabsTrigger>
            <TabsTrigger value="nonlinear" className="data-[state=active]:bg-white">
              Non-Linear SVM
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
            {activeTab === "linear" && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="c" className="text-neutral-900">
                      C (Regularization)
                    </Label>
                    <span className="text-sm text-neutral-600">{c.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="c"
                    min={0.1}
                    max={2}
                    step={0.1}
                    value={[c]}
                    onValueChange={(value) => setC(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="noise" className="text-neutral-900">
                      Data Noise
                    </Label>
                    <span className="text-sm text-neutral-600">{noise.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="noise"
                    min={0}
                    max={1}
                    step={0.05}
                    value={[noise]}
                    onValueChange={(value) => setNoise(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="margin" className="text-neutral-900">
                      Margin Width
                    </Label>
                    <span className="text-sm text-neutral-600">{margin.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="margin"
                    min={0.1}
                    max={1}
                    step={0.05}
                    value={[margin]}
                    onValueChange={(value) => setMargin(value[0])}
                    className="notebook-slider"
                  />
                </div>
              </>
            )}

            {activeTab === "nonlinear" && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="kernel" className="text-neutral-900">
                    Kernel Type
                  </Label>
                  <Select value={kernel} onValueChange={setKernel}>
                    <SelectTrigger id="kernel">
                      <SelectValue placeholder="Select kernel" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="rbf">RBF (Radial Basis Function)</SelectItem>
                      <SelectItem value="poly">Polynomial</SelectItem>
                      <SelectItem value="sigmoid">Sigmoid</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="gamma" className="text-neutral-900">
                      Gamma
                    </Label>
                    <span className="text-sm text-neutral-600">{gamma.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="gamma"
                    min={0.1}
                    max={2}
                    step={0.1}
                    value={[gamma]}
                    onValueChange={(value) => setGamma(value[0])}
                    className="notebook-slider"
                  />
                </div>

                {kernel === "poly" && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="complexity" className="text-neutral-900">
                        Polynomial Degree
                      </Label>
                      <span className="text-sm text-neutral-600">{complexity}</span>
                    </div>
                    <Slider
                      id="complexity"
                      min={1}
                      max={5}
                      step={1}
                      value={[complexity]}
                      onValueChange={(value) => setComplexity(value[0])}
                      className="notebook-slider"
                    />
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
