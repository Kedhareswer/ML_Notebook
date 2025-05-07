"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Shuffle, Download } from "lucide-react"

interface LinearRegressionVisualizationProps {
  width?: number
  height?: number
}

export default function LinearRegressionVisualization({
  width = 600,
  height = 400,
}: LinearRegressionVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [activeTab, setActiveTab] = useState("2d")

  // Parameters for 2D visualization
  const [slope, setSlope] = useState(0.5)
  const [intercept, setIntercept] = useState(2)
  const [noise, setNoise] = useState(0.5)
  const [pointCount, setPointCount] = useState(30)

  // Parameters for 3D visualization
  const [slope1, setSlope1] = useState(0.5)
  const [slope2, setSlope2] = useState(0.3)
  const [intercept3d, setIntercept3d] = useState(2)
  const [noise3d, setNoise3d] = useState(0.3)
  const [rotation, setRotation] = useState(30)

  // Parameters for polynomial regression
  const [degree, setDegree] = useState(2)
  const [polyNoise, setPolyNoise] = useState(0.3)

  // Generate random data
  const generateRandomData = () => {
    if (activeTab === "2d") {
      setSlope(Math.random() * 2 - 1)
      setIntercept(Math.random() * 4)
      setNoise(Math.random() * 0.8)
      setPointCount(Math.floor(Math.random() * 30) + 20)
    } else if (activeTab === "3d") {
      setSlope1(Math.random() * 2 - 1)
      setSlope2(Math.random() * 2 - 1)
      setIntercept3d(Math.random() * 4)
      setNoise3d(Math.random() * 0.5)
      setRotation(Math.random() * 60)
    } else if (activeTab === "polynomial") {
      setDegree(Math.floor(Math.random() * 4) + 2)
      setPolyNoise(Math.random() * 0.5)
    }
  }

  // Render 2D linear regression
  const render2DLinearRegression = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw coordinate system
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1

    // Draw x-axis
    ctx.beginPath()
    ctx.moveTo(50, height - 50)
    ctx.lineTo(width - 50, height - 50)
    ctx.stroke()

    // Draw y-axis
    ctx.beginPath()
    ctx.moveTo(50, 50)
    ctx.lineTo(50, height - 50)
    ctx.stroke()

    // Draw axis labels
    ctx.fillStyle = "#333"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("X", width - 30, height - 50)
    ctx.textAlign = "right"
    ctx.fillText("Y", 50, 30)

    // Generate data points
    const points = []
    const xScale = (width - 100) / 10
    const yScale = (height - 100) / 10

    for (let i = 0; i < pointCount; i++) {
      const x = i * ((width - 100) / pointCount) + 50
      const xVal = (x - 50) / xScale

      // Calculate y based on linear equation y = mx + b + noise
      const noiseValue = (Math.random() - 0.5) * noise * 2
      const yVal = slope * xVal + intercept + noiseValue
      const y = height - 50 - yVal * yScale

      points.push({ x, y })
    }

    // Draw data points
    ctx.fillStyle = "#000"
    points.forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 4, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw regression line
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.beginPath()

    const x1 = 50
    const y1 = height - 50 - intercept * yScale
    const x2 = width - 50
    const xVal2 = (x2 - 50) / xScale
    const y2 = height - 50 - (slope * xVal2 + intercept) * yScale

    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()

    // Draw equation
    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.fillText(`y = ${slope.toFixed(2)}x + ${intercept.toFixed(2)}`, 70, 70)
  }

  // Render 3D multiple linear regression
  const render3DLinearRegression = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Set up 3D projection (simplified)
    const centerX = width / 2
    const centerY = height / 2
    const scale = 40
    const rotationRad = (rotation * Math.PI) / 180

    // Draw coordinate system
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1

    // Function to project 3D to 2D
    const project = (x: number, y: number, z: number) => {
      // Apply rotation around y-axis
      const rotatedX = x * Math.cos(rotationRad) - z * Math.sin(rotationRad)
      const rotatedZ = x * Math.sin(rotationRad) + z * Math.cos(rotationRad)

      // Project to 2D
      const projectedX = centerX + scale * rotatedX
      const projectedY = centerY - scale * y + scale * rotatedZ * 0.3

      return { x: projectedX, y: projectedY }
    }

    // Draw axes
    const origin = project(0, 0, 0)
    const xEnd = project(5, 0, 0)
    const yEnd = project(0, 5, 0)
    const zEnd = project(0, 0, 5)

    ctx.beginPath()
    ctx.moveTo(origin.x, origin.y)
    ctx.lineTo(xEnd.x, xEnd.y)
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(origin.x, origin.y)
    ctx.lineTo(yEnd.x, yEnd.y)
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(origin.x, origin.y)
    ctx.lineTo(zEnd.x, zEnd.y)
    ctx.stroke()

    // Label axes
    ctx.fillStyle = "#333"
    ctx.font = "12px sans-serif"
    ctx.fillText("X", xEnd.x + 5, xEnd.y)
    ctx.fillText("Y", yEnd.x, yEnd.y - 5)
    ctx.fillText("Z", zEnd.x + 5, zEnd.y)

    // Generate data points
    const points = []

    for (let i = 0; i < 50; i++) {
      const x = Math.random() * 4 - 2
      const z = Math.random() * 4 - 2

      // Calculate y based on multiple linear regression: y = b + m1*x + m2*z + noise
      const noiseValue = (Math.random() - 0.5) * noise3d * 2
      const y = intercept3d + slope1 * x + slope2 * z + noiseValue

      points.push({ x, y, z })
    }

    // Draw data points
    ctx.fillStyle = "#000"
    points.forEach((point) => {
      const projected = project(point.x, point.y, point.z)
      ctx.beginPath()
      ctx.arc(projected.x, projected.y, 3, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw regression plane (as a grid)
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 0.5

    const gridSize = 10
    const gridStep = 4 / gridSize

    for (let i = 0; i <= gridSize; i++) {
      const x = -2 + i * gridStep

      // Draw lines along z-axis
      ctx.beginPath()
      for (let j = 0; j <= gridSize; j++) {
        const z = -2 + j * gridStep
        const y = intercept3d + slope1 * x + slope2 * z
        const point = project(x, y, z)

        if (j === 0) {
          ctx.moveTo(point.x, point.y)
        } else {
          ctx.lineTo(point.x, point.y)
        }
      }
      ctx.stroke()

      // Draw lines along x-axis
      ctx.beginPath()
      for (let j = 0; j <= gridSize; j++) {
        const z = -2 + i * gridStep
        const x = -2 + j * gridStep
        const y = intercept3d + slope1 * x + slope2 * z
        const point = project(x, y, z)

        if (j === 0) {
          ctx.moveTo(point.x, point.y)
        } else {
          ctx.lineTo(point.x, point.y)
        }
      }
      ctx.stroke()
    }

    // Draw equation
    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.fillText(`y = ${intercept3d.toFixed(2)} + ${slope1.toFixed(2)}x₁ + ${slope2.toFixed(2)}x₂`, 70, 70)
  }

  // Render polynomial regression
  const renderPolynomialRegression = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw coordinate system
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1

    // Draw x-axis
    ctx.beginPath()
    ctx.moveTo(50, height - 50)
    ctx.lineTo(width - 50, height - 50)
    ctx.stroke()

    // Draw y-axis
    ctx.beginPath()
    ctx.moveTo(50, 50)
    ctx.lineTo(50, height - 50)
    ctx.stroke()

    // Draw axis labels
    ctx.fillStyle = "#333"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("X", width - 30, height - 50)
    ctx.textAlign = "right"
    ctx.fillText("Y", 50, 30)

    // Generate data points
    const points = []
    const xScale = (width - 100) / 10
    const yScale = (height - 100) / 10

    // Generate polynomial coefficients
    const coefficients = []
    for (let i = 0; i <= degree; i++) {
      // Generate coefficients that decrease with higher powers
      coefficients.push((Math.random() - 0.5) * (1 / (i + 1)) * 2)
    }

    // Function to calculate polynomial value
    const polyValue = (x: number) => {
      let result = 0
      for (let i = 0; i <= degree; i++) {
        result += coefficients[i] * Math.pow(x, i)
      }
      return result
    }

    for (let i = 0; i < 40; i++) {
      const x = i * ((width - 100) / 40) + 50
      const xVal = (x - 50) / xScale

      // Calculate y based on polynomial equation
      const noiseValue = (Math.random() - 0.5) * polyNoise * 4
      const yVal = polyValue(xVal) + noiseValue
      const y = height - 50 - (yVal + 5) * yScale

      points.push({ x, y })
    }

    // Draw data points
    ctx.fillStyle = "#000"
    points.forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 3, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw polynomial curve
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.beginPath()

    // Draw the polynomial curve with many points for smoothness
    const curvePoints = 200
    for (let i = 0; i <= curvePoints; i++) {
      const x = 50 + i * ((width - 100) / curvePoints)
      const xVal = (x - 50) / xScale
      const yVal = polyValue(xVal)
      const y = height - 50 - (yVal + 5) * yScale

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    }
    ctx.stroke()

    // Draw equation
    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"

    let equation = "y = "
    for (let i = degree; i >= 0; i--) {
      const coef = coefficients[i].toFixed(2)
      if (i > 1) {
        equation += `${coef}x^${i} + `
      } else if (i === 1) {
        equation += `${coef}x + `
      } else {
        equation += `${coef}`
      }
    }
    ctx.fillText(equation, 70, 70)
  }

  // Update visualization when parameters change
  useEffect(() => {
    if (activeTab === "2d") {
      render2DLinearRegression()
    } else if (activeTab === "3d") {
      render3DLinearRegression()
    } else if (activeTab === "polynomial") {
      renderPolynomialRegression()
    }
  }, [
    activeTab,
    slope,
    intercept,
    noise,
    pointCount,
    slope1,
    slope2,
    intercept3d,
    noise3d,
    rotation,
    degree,
    polyNoise,
  ])

  return (
    <Card className="w-full border-neutral-300 bg-white">
      <CardContent className="pt-6">
        <div className="flex justify-between items-center mb-4">
          <div className="text-lg font-medium text-neutral-900">Linear Regression Visualization</div>
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
                  link.download = `linear-regression-${activeTab}.png`
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
            <TabsTrigger value="2d" className="data-[state=active]:bg-white">
              Simple Linear
            </TabsTrigger>
            <TabsTrigger value="3d" className="data-[state=active]:bg-white">
              Multiple Linear
            </TabsTrigger>
            <TabsTrigger value="polynomial" className="data-[state=active]:bg-white">
              Polynomial
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
            {activeTab === "2d" && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="slope" className="text-neutral-900">
                      Slope (m)
                    </Label>
                    <span className="text-sm text-neutral-600">{slope.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="slope"
                    min={-2}
                    max={2}
                    step={0.1}
                    value={[slope]}
                    onValueChange={(value) => setSlope(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="intercept" className="text-neutral-900">
                      Y-Intercept (b)
                    </Label>
                    <span className="text-sm text-neutral-600">{intercept.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="intercept"
                    min={-2}
                    max={6}
                    step={0.1}
                    value={[intercept]}
                    onValueChange={(value) => setIntercept(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="noise" className="text-neutral-900">
                      Noise Level
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
                    <Label htmlFor="pointCount" className="text-neutral-900">
                      Number of Points
                    </Label>
                    <span className="text-sm text-neutral-600">{pointCount}</span>
                  </div>
                  <Slider
                    id="pointCount"
                    min={10}
                    max={100}
                    step={1}
                    value={[pointCount]}
                    onValueChange={(value) => setPointCount(value[0])}
                    className="notebook-slider"
                  />
                </div>
              </>
            )}

            {activeTab === "3d" && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="slope1" className="text-neutral-900">
                      Slope X₁
                    </Label>
                    <span className="text-sm text-neutral-600">{slope1.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="slope1"
                    min={-1}
                    max={1}
                    step={0.1}
                    value={[slope1]}
                    onValueChange={(value) => setSlope1(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="slope2" className="text-neutral-900">
                      Slope X₂
                    </Label>
                    <span className="text-sm text-neutral-600">{slope2.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="slope2"
                    min={-1}
                    max={1}
                    step={0.1}
                    value={[slope2]}
                    onValueChange={(value) => setSlope2(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="intercept3d" className="text-neutral-900">
                      Intercept
                    </Label>
                    <span className="text-sm text-neutral-600">{intercept3d.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="intercept3d"
                    min={0}
                    max={4}
                    step={0.1}
                    value={[intercept3d]}
                    onValueChange={(value) => setIntercept3d(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="noise3d" className="text-neutral-900">
                      Noise Level
                    </Label>
                    <span className="text-sm text-neutral-600">{noise3d.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="noise3d"
                    min={0}
                    max={1}
                    step={0.05}
                    value={[noise3d]}
                    onValueChange={(value) => setNoise3d(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="rotation" className="text-neutral-900">
                      Rotation (°)
                    </Label>
                    <span className="text-sm text-neutral-600">{rotation.toFixed(0)}°</span>
                  </div>
                  <Slider
                    id="rotation"
                    min={0}
                    max={90}
                    step={1}
                    value={[rotation]}
                    onValueChange={(value) => setRotation(value[0])}
                    className="notebook-slider"
                  />
                </div>
              </>
            )}

            {activeTab === "polynomial" && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="degree" className="text-neutral-900">
                      Polynomial Degree
                    </Label>
                    <span className="text-sm text-neutral-600">{degree}</span>
                  </div>
                  <Slider
                    id="degree"
                    min={1}
                    max={5}
                    step={1}
                    value={[degree]}
                    onValueChange={(value) => setDegree(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="polyNoise" className="text-neutral-900">
                      Noise Level
                    </Label>
                    <span className="text-sm text-neutral-600">{polyNoise.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="polyNoise"
                    min={0}
                    max={1}
                    step={0.05}
                    value={[polyNoise]}
                    onValueChange={(value) => setPolyNoise(value[0])}
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
