"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const RegularizedRegressionViz: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [alphaRidge, setAlphaRidge] = useState<number>(0.1)
  const [alphaLasso, setAlphaLasso] = useState<number>(0.1)
  const [complexity, setComplexity] = useState<number>(5)
  const [noiseLevel, setNoiseLevel] = useState<number>(0.5)
  const [activeTab, setActiveTab] = useState<string>("ridge")
  const [datasetType, setDatasetType] = useState<string>("polynomial")

  // Generate synthetic data
  const generateData = (n = 50, complexity = 5, noise = 0.5, type = "polynomial") => {
    const data = []
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2 - 1 // x between -1 and 1

      let y
      if (type === "polynomial") {
        // Polynomial relationship
        y = Array(complexity)
          .fill(0)
          .reduce((acc, _, j) => acc + Math.pow(x, j + 1) * (Math.random() * 2 - 1), 0)
      } else if (type === "sinusoidal") {
        // Sinusoidal relationship
        y = Math.sin(x * complexity) + (Math.random() * 2 - 1) * 0.5
      } else {
        // Exponential relationship
        y = Math.exp((x * complexity) / 5) - 1 + (Math.random() * 2 - 1) * 0.5
      }

      // Add noise
      y += (Math.random() * 2 - 1) * noise

      data.push({ x, y })
    }
    return data
  }

  // Fit models
  const fitModels = (data: { x: number; y: number }[], complexity: number, alphaRidge: number, alphaLasso: number) => {
    // Prepare design matrix X
    const X = data.map((point) => {
      return Array(complexity)
        .fill(0)
        .map((_, i) => Math.pow(point.x, i + 1))
    })

    const y = data.map((point) => point.y)

    // Ordinary Least Squares
    const olsCoefficients = solveOLS(X, y)

    // Ridge Regression
    const ridgeCoefficients = solveRidge(X, y, alphaRidge)

    // Lasso Regression (approximation using soft thresholding)
    const lassoCoefficients = solveLasso(X, y, alphaLasso, 1000)

    return { ols: olsCoefficients, ridge: ridgeCoefficients, lasso: lassoCoefficients }
  }

  // Solve OLS using normal equation
  const solveOLS = (X: number[][], y: number[]) => {
    // Simplified implementation for visualization purposes
    // X'X coefficients = X'y
    const coefficients = Array(X[0].length).fill(0)

    // Gradient descent for simplicity
    const learningRate = 0.01
    const iterations = 1000

    for (let iter = 0; iter < iterations; iter++) {
      const predictions = X.map((row, i) => row.reduce((sum, val, j) => sum + val * coefficients[j], 0))

      const errors = predictions.map((pred, i) => pred - y[i])

      // Update coefficients
      for (let j = 0; j < coefficients.length; j++) {
        const gradient = X.reduce((sum, row, i) => sum + row[j] * errors[i], 0) / X.length
        coefficients[j] -= learningRate * gradient
      }
    }

    return coefficients
  }

  // Solve Ridge Regression
  const solveRidge = (X: number[][], y: number[], alpha: number) => {
    const coefficients = Array(X[0].length).fill(0)

    // Gradient descent with L2 regularization
    const learningRate = 0.01
    const iterations = 1000

    for (let iter = 0; iter < iterations; iter++) {
      const predictions = X.map((row, i) => row.reduce((sum, val, j) => sum + val * coefficients[j], 0))

      const errors = predictions.map((pred, i) => pred - y[i])

      // Update coefficients with L2 regularization
      for (let j = 0; j < coefficients.length; j++) {
        const gradient = X.reduce((sum, row, i) => sum + row[j] * errors[i], 0) / X.length
        const regularization = alpha * coefficients[j]
        coefficients[j] -= learningRate * (gradient + regularization)
      }
    }

    return coefficients
  }

  // Solve Lasso Regression using coordinate descent
  const solveLasso = (X: number[][], y: number[], alpha: number, iterations: number) => {
    const coefficients = Array(X[0].length).fill(0)

    // Coordinate descent with soft thresholding
    for (let iter = 0; iter < iterations; iter++) {
      for (let j = 0; j < coefficients.length; j++) {
        // Calculate residual without current feature
        const residuals = y.map((yi, i) => {
          let pred = 0
          for (let k = 0; k < coefficients.length; k++) {
            if (k !== j) pred += X[i][k] * coefficients[k]
          }
          return yi - pred
        })

        // Calculate correlation of residual with current feature
        const correlation = X.reduce((sum, row, i) => sum + row[j] * residuals[i], 0)

        // Soft thresholding
        if (correlation > alpha) {
          coefficients[j] = (correlation - alpha) / X.length
        } else if (correlation < -alpha) {
          coefficients[j] = (correlation + alpha) / X.length
        } else {
          coefficients[j] = 0
        }
      }
    }

    return coefficients
  }

  // Predict using model
  const predict = (x: number, coefficients: number[]) => {
    return coefficients.reduce((sum, coef, i) => sum + coef * Math.pow(x, i + 1), 0)
  }

  // Draw visualization
  const drawVisualization = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Generate data
    const data = generateData(50, complexity, noiseLevel, datasetType)

    // Fit models
    const models = fitModels(data, complexity, alphaRidge, alphaLasso)

    // Scale for visualization
    const margin = 30
    const width = canvas.width - 2 * margin
    const height = canvas.height - 2 * margin

    // Find data range
    const xMin = Math.min(...data.map((d) => d.x))
    const xMax = Math.max(...data.map((d) => d.x))
    const yMin = Math.min(...data.map((d) => d.y))
    const yMax = Math.max(...data.map((d) => d.y))

    const xRange = xMax - xMin
    const yRange = yMax - yMin

    // Scale functions
    const scaleX = (x: number) => margin + ((x - xMin) / xRange) * width
    const scaleY = (y: number) => canvas.height - (margin + ((y - yMin) / yRange) * height)

    // Draw axes
    ctx.strokeStyle = "#ccc"
    ctx.lineWidth = 1

    // X-axis
    ctx.beginPath()
    ctx.moveTo(margin, canvas.height - margin)
    ctx.lineTo(canvas.width - margin, canvas.height - margin)
    ctx.stroke()

    // Y-axis
    ctx.beginPath()
    ctx.moveTo(margin, margin)
    ctx.lineTo(margin, canvas.height - margin)
    ctx.stroke()

    // Draw data points
    ctx.fillStyle = "rgba(0, 0, 255, 0.5)"
    data.forEach((point) => {
      ctx.beginPath()
      ctx.arc(scaleX(point.x), scaleY(point.y), 4, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw model predictions
    const drawModel = (coefficients: number[], color: string, label: string) => {
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()

      // Draw curve
      for (let i = 0; i <= 100; i++) {
        const x = xMin + (i / 100) * xRange
        const y = predict(x, coefficients)

        if (i === 0) {
          ctx.moveTo(scaleX(x), scaleY(y))
        } else {
          ctx.lineTo(scaleX(x), scaleY(y))
        }
      }

      ctx.stroke()

      // Draw label
      ctx.fillStyle = color
      ctx.font = "14px Arial"
      ctx.fillText(label, canvas.width - 100, 20 + (label === "OLS" ? 0 : label === "Ridge" ? 20 : 40))
    }

    // Draw OLS model
    drawModel(models.ols, "rgba(255, 0, 0, 0.8)", "OLS")

    // Draw active model based on tab
    if (activeTab === "ridge") {
      drawModel(models.ridge, "rgba(0, 128, 0, 0.8)", "Ridge")
    } else {
      drawModel(models.lasso, "rgba(128, 0, 128, 0.8)", "Lasso")
    }

    // Draw coefficient comparison
    drawCoefficientComparison(ctx, models, activeTab)
  }

  // Draw coefficient comparison
  const drawCoefficientComparison = (
    ctx: CanvasRenderingContext2D,
    models: { ols: number[]; ridge: number[]; lasso: number[] },
    activeTab: string,
  ) => {
    const barWidth = 15
    const spacing = 5
    const startX = 50
    const startY = 350
    const maxHeight = 80

    // Find max coefficient for scaling
    const allCoefs = [...models.ols, ...models.ridge, ...models.lasso]
    const maxCoef = Math.max(...allCoefs.map(Math.abs))

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.fillText("Coefficient Comparison", startX, startY - 10)

    // Draw bars for each coefficient
    for (let i = 0; i < models.ols.length; i++) {
      const x = startX + i * (barWidth * 2 + spacing)

      // OLS coefficient
      const olsHeight = (Math.abs(models.ols[i]) / maxCoef) * maxHeight
      ctx.fillStyle = "rgba(255, 0, 0, 0.8)"
      ctx.fillRect(x, startY - olsHeight * Math.sign(models.ols[i]), barWidth, olsHeight)

      // Ridge or Lasso coefficient
      const regCoef = activeTab === "ridge" ? models.ridge[i] : models.lasso[i]
      const regHeight = (Math.abs(regCoef) / maxCoef) * maxHeight
      ctx.fillStyle = activeTab === "ridge" ? "rgba(0, 128, 0, 0.8)" : "rgba(128, 0, 128, 0.8)"
      ctx.fillRect(x + barWidth, startY - regHeight * Math.sign(regCoef), barWidth, regHeight)

      // Label
      ctx.fillStyle = "#000"
      ctx.font = "12px Arial"
      ctx.fillText(`x^${i + 1}`, x, startY + 15)
    }

    // Legend
    ctx.fillStyle = "rgba(255, 0, 0, 0.8)"
    ctx.fillRect(startX, startY + 30, 15, 15)
    ctx.fillStyle = "#000"
    ctx.fillText("OLS", startX + 20, startY + 42)

    ctx.fillStyle = activeTab === "ridge" ? "rgba(0, 128, 0, 0.8)" : "rgba(128, 0, 128, 0.8)"
    ctx.fillRect(startX + 60, startY + 30, 15, 15)
    ctx.fillStyle = "#000"
    ctx.fillText(activeTab === "ridge" ? "Ridge" : "Lasso", startX + 80, startY + 42)
  }

  // Effect to draw visualization when parameters change
  useEffect(() => {
    drawVisualization()
  }, [alphaRidge, alphaLasso, complexity, noiseLevel, activeTab, datasetType])

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-4">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="ridge">Ridge Regression</TabsTrigger>
            <TabsTrigger value="lasso">Lasso Regression</TabsTrigger>
          </TabsList>

          <TabsContent value="ridge" className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Ridge Alpha: {alphaRidge.toFixed(2)}</span>
              </div>
              <Slider
                value={[alphaRidge]}
                min={0}
                max={2}
                step={0.01}
                onValueChange={(value) => setAlphaRidge(value[0])}
              />
              <p className="text-sm text-gray-500">
                Ridge regression adds an L2 penalty (sum of squared coefficients) to reduce overfitting. Higher alpha
                values increase regularization strength.
              </p>
            </div>
          </TabsContent>

          <TabsContent value="lasso" className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Lasso Alpha: {alphaLasso.toFixed(2)}</span>
              </div>
              <Slider
                value={[alphaLasso]}
                min={0}
                max={2}
                step={0.01}
                onValueChange={(value) => setAlphaLasso(value[0])}
              />
              <p className="text-sm text-gray-500">
                Lasso regression adds an L1 penalty (sum of absolute coefficients) to reduce overfitting. It tends to
                produce sparse models by setting some coefficients to exactly zero.
              </p>
            </div>
          </TabsContent>
        </Tabs>

        <div className="space-y-4 mt-4">
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Model Complexity: {complexity}</span>
            </div>
            <Slider value={[complexity]} min={1} max={10} step={1} onValueChange={(value) => setComplexity(value[0])} />
            <p className="text-sm text-gray-500">
              Controls the highest polynomial degree in the model. Higher values can lead to overfitting without
              regularization.
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Noise Level: {noiseLevel.toFixed(2)}</span>
            </div>
            <Slider
              value={[noiseLevel]}
              min={0}
              max={2}
              step={0.01}
              onValueChange={(value) => setNoiseLevel(value[0])}
            />
          </div>

          <div className="space-y-2">
            <span>Dataset Type:</span>
            <Select value={datasetType} onValueChange={setDatasetType}>
              <SelectTrigger>
                <SelectValue placeholder="Select dataset type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="polynomial">Polynomial</SelectItem>
                <SelectItem value="sinusoidal">Sinusoidal</SelectItem>
                <SelectItem value="exponential">Exponential</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button onClick={() => drawVisualization()} className="w-full">
            Generate New Data
          </Button>
        </div>

        <div className="mt-6 border rounded-lg overflow-hidden">
          <canvas ref={canvasRef} width={600} height={450} className="w-full h-auto" />
        </div>

        <div className="mt-4 text-sm">
          <h3 className="font-semibold">Key Insights:</h3>
          <ul className="list-disc pl-5 space-y-1 mt-2">
            <li>Ridge regression shrinks all coefficients toward zero, but rarely to exactly zero.</li>
            <li>Lasso regression can produce sparse models by setting some coefficients to exactly zero.</li>
            <li>Both methods help prevent overfitting, especially with high-dimensional data.</li>
            <li>
              The optimal regularization strength (alpha) depends on the noise level and complexity of the true
              relationship.
            </li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}

export default RegularizedRegressionViz
