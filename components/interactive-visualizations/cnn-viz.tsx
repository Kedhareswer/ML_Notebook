"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Shuffle, Play, Pause } from "lucide-react"

interface CNNVisualizationProps {
  width?: number
  height?: number
}

export default function CNNVisualization({ width = 600, height = 400 }: CNNVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationStep, setAnimationStep] = useState(0)

  // CNN parameters
  const [inputSize, setInputSize] = useState(28)
  const [filterSize, setFilterSize] = useState(3)
  const [numFilters, setNumFilters] = useState(16)
  const [stride, setStride] = useState(1)
  const [padding, setPadding] = useState(0)
  const [poolSize, setPoolSize] = useState(2)
  const [activeFilter, setActiveFilter] = useState(0)

  // Toggle animation
  const toggleAnimation = () => {
    setIsAnimating(!isAnimating)
  }

  // Generate random data
  const generateRandomData = () => {
    setFilterSize([3, 5, 7][Math.floor(Math.random() * 3)])
    setNumFilters(Math.floor(Math.random() * 24) + 8)
    setStride(Math.random() > 0.7 ? 2 : 1)
    setPadding(Math.random() > 0.7 ? 1 : 0)
    setPoolSize(Math.random() > 0.3 ? 2 : 3)
    setActiveFilter(Math.floor(Math.random() * numFilters))
  }

  // Animation loop
  useEffect(() => {
    if (isAnimating) {
      const animate = () => {
        setAnimationStep((prev) => (prev + 1) % 60) // 60 frames for a complete cycle
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

  // Render CNN visualization
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Calculate dimensions
    const outputSize = Math.floor((inputSize + 2 * padding - filterSize) / stride) + 1
    const pooledSize = Math.floor(outputSize / poolSize)

    // Define positions
    const startX = 50
    const startY = 50
    const layerSpacing = 150
    const maxLayerHeight = 200

    // Draw input layer
    const inputLayerX = startX
    const inputLayerY = startY + maxLayerHeight / 2 - (inputSize * 3) / 2
    const inputPixelSize = Math.min(3, Math.max(1, Math.floor(maxLayerHeight / inputSize)))

    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Input", inputLayerX, startY - 20)

    // Draw input image (grayscale noise)
    for (let i = 0; i < inputSize; i++) {
      for (let j = 0; j < inputSize; j++) {
        const gray = Math.floor(Math.random() * 200) + 55
        ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`
        ctx.fillRect(
          inputLayerX - (inputSize * inputPixelSize) / 2 + j * inputPixelSize,
          inputLayerY + i * inputPixelSize,
          inputPixelSize,
          inputPixelSize,
        )
      }
    }

    // Draw convolution layer
    const convLayerX = startX + layerSpacing
    const convLayerY = startY + maxLayerHeight / 2 - (outputSize * 3) / 2
    const convPixelSize = Math.min(3, Math.max(1, Math.floor(maxLayerHeight / outputSize)))

    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Convolution", convLayerX, startY - 20)
    ctx.fillText(`${numFilters} filters`, convLayerX, startY - 5)

    // Draw feature maps (only show the active one)
    const featureMapX = convLayerX - (outputSize * convPixelSize) / 2
    const featureMapY = convLayerY

    // Draw feature map outline
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.strokeRect(featureMapX - 2, featureMapY - 2, outputSize * convPixelSize + 4, outputSize * convPixelSize + 4)

    // Draw feature map content
    for (let i = 0; i < outputSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        // Generate a pattern based on filter and position
        const val = (Math.sin(i * j * (activeFilter + 1) * 0.1 + animationStep * 0.1) + 1) / 2
        const color = Math.floor(val * 200) + 55
        ctx.fillStyle = `rgb(${color}, ${color}, ${color})`
        ctx.fillRect(featureMapX + j * convPixelSize, featureMapY + i * convPixelSize, convPixelSize, convPixelSize)
      }
    }

    // Draw filter selector
    const filterSelectorY = convLayerY + outputSize * convPixelSize + 20
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(`Filter: ${activeFilter + 1}/${numFilters}`, convLayerX, filterSelectorY)

    // Draw pooling layer
    const poolLayerX = startX + 2 * layerSpacing
    const poolLayerY = startY + maxLayerHeight / 2 - (pooledSize * 4) / 2
    const poolPixelSize = Math.min(4, Math.max(2, Math.floor(maxLayerHeight / pooledSize)))

    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Max Pooling", poolLayerX, startY - 20)
    ctx.fillText(`${poolSize}x${poolSize}`, poolLayerX, startY - 5)

    // Draw pooled feature map
    const pooledMapX = poolLayerX - (pooledSize * poolPixelSize) / 2
    const pooledMapY = poolLayerY

    // Draw pooled map outline
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.strokeRect(pooledMapX - 2, pooledMapY - 2, pooledSize * poolPixelSize + 4, pooledSize * poolPixelSize + 4)

    // Draw pooled map content
    for (let i = 0; i < pooledSize; i++) {
      for (let j = 0; j < pooledSize; j++) {
        // Generate a pattern based on filter and position, but more pronounced
        const val = (Math.sin(i * j * (activeFilter + 1) * 0.2 + animationStep * 0.1) + 1) / 2
        const color = Math.floor(val * 200) + 55
        ctx.fillStyle = `rgb(${color}, ${color}, ${color})`
        ctx.fillRect(pooledMapX + j * poolPixelSize, pooledMapY + i * poolPixelSize, poolPixelSize, poolPixelSize)
      }
    }

    // Draw fully connected layer
    const fcLayerX = startX + 3 * layerSpacing
    const fcLayerY = startY + maxLayerHeight / 2
    const fcRadius = 5
    const fcRows = 10
    const fcCols = 3
    const fcSpacingY = 15
    const fcSpacingX = 20

    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText("Fully Connected", fcLayerX, startY - 20)

    // Draw neurons
    for (let col = 0; col < fcCols; col++) {
      const numNeurons = col === 0 ? fcRows : col === 1 ? 6 : 3
      for (let i = 0; i < numNeurons; i++) {
        const neuronX = fcLayerX + col * fcSpacingX
        const neuronY = fcLayerY - ((numNeurons - 1) * fcSpacingY) / 2 + i * fcSpacingY

        // Draw neuron
        ctx.beginPath()
        ctx.arc(neuronX, neuronY, fcRadius, 0, Math.PI * 2)
        ctx.fillStyle = "#f0f0f0"
        ctx.fill()
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1
        ctx.stroke()

        // Draw connections to previous layer if not first column
        if (col > 0) {
          const prevNumNeurons = col === 1 ? fcRows : 6
          for (let j = 0; j < prevNumNeurons; j++) {
            const prevNeuronX = fcLayerX + (col - 1) * fcSpacingX
            const prevNeuronY = fcLayerY - ((prevNumNeurons - 1) * fcSpacingY) / 2 + j * fcSpacingY

            ctx.beginPath()
            ctx.moveTo(prevNeuronX + fcRadius, prevNeuronY)
            ctx.lineTo(neuronX - fcRadius, neuronY)
            ctx.strokeStyle = "rgba(0, 0, 0, 0.2)"
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        }
      }
    }

    // Draw connection from pooled layer to FC layer
    ctx.beginPath()
    ctx.moveTo(pooledMapX + pooledSize * poolPixelSize + 2, poolLayerY + (pooledSize * poolPixelSize) / 2)
    ctx.lineTo(fcLayerX - fcRadius, fcLayerY)
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw arrows between layers
    // Input to Conv
    drawArrow(
      ctx,
      inputLayerX + (inputSize * inputPixelSize) / 2 + 10,
      inputLayerY + (inputSize * inputPixelSize) / 2,
      convLayerX - (outputSize * convPixelSize) / 2 - 10,
      convLayerY + (outputSize * convPixelSize) / 2,
    )

    // Conv to Pool
    drawArrow(
      ctx,
      convLayerX + (outputSize * convPixelSize) / 2 + 10,
      convLayerY + (outputSize * convPixelSize) / 2,
      poolLayerX - (pooledSize * poolPixelSize) / 2 - 10,
      poolLayerY + (pooledSize * poolPixelSize) / 2,
    )

    // Draw filter visualization
    const filterVisX = convLayerX
    const filterVisY = convLayerY + outputSize * convPixelSize + 40
    const filterVisSize = 30

    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "center"
    ctx.fillText(`Filter ${activeFilter + 1} Weights`, filterVisX, filterVisY - 5)

    // Draw filter weights
    const filterPixelSize = filterVisSize / filterSize
    for (let i = 0; i < filterSize; i++) {
      for (let j = 0; j < filterSize; j++) {
        // Generate a pattern based on filter and position
        const val = (Math.sin((i + j) * (activeFilter + 1) * 0.5) + 1) / 2
        const r = Math.floor(val * 200)
        const g = Math.floor((1 - val) * 200)
        const b = Math.floor(Math.abs(0.5 - val) * 400)
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
        ctx.fillRect(
          filterVisX - filterVisSize / 2 + j * filterPixelSize,
          filterVisY + i * filterPixelSize,
          filterPixelSize,
          filterPixelSize,
        )
      }
    }

    // Draw filter outline
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 1
    ctx.strokeRect(filterVisX - filterVisSize / 2, filterVisY, filterVisSize, filterVisSize)

    // Draw parameters
    const paramsY = height - 40
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "left"
    ctx.fillText(
      `Parameters: Filter Size: ${filterSize}x${filterSize}, Stride: ${stride}, Padding: ${padding}, Pool Size: ${poolSize}x${poolSize}`,
      20,
      paramsY,
    )
  }, [inputSize, filterSize, numFilters, stride, padding, poolSize, activeFilter, animationStep, isAnimating])

  // Helper function to draw arrows
  const drawArrow = (ctx: CanvasRenderingContext2D, fromX: number, fromY: number, toX: number, toY: number) => {
    const headLength = 10
    const angle = Math.atan2(toY - fromY, toX - fromX)

    // Draw line
    ctx.beginPath()
    ctx.moveTo(fromX, fromY)
    ctx.lineTo(toX, toY)
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw arrowhead
    ctx.beginPath()
    ctx.moveTo(toX, toY)
    ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), toY - headLength * Math.sin(angle - Math.PI / 6))
    ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), toY - headLength * Math.sin(angle + Math.PI / 6))
    ctx.closePath()
    ctx.fillStyle = "#000"
    ctx.fill()
  }

  return (
    <Card className="w-full border-neutral-300 bg-white">
      <CardContent className="pt-6">
        <div className="flex justify-between items-center mb-4">
          <div className="text-lg font-medium text-neutral-900">CNN Visualization</div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={generateRandomData} className="flex items-center gap-1">
              <Shuffle className="h-4 w-4" /> Random Parameters
            </Button>
            <Button variant="outline" size="sm" onClick={toggleAnimation} className="flex items-center gap-1">
              {isAnimating ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              {isAnimating ? "Pause" : "Animate"}
            </Button>
          </div>
        </div>

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
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="filterSize" className="text-neutral-900">
                  Filter Size
                </Label>
                <span className="text-sm text-neutral-600">
                  {filterSize}x{filterSize}
                </span>
              </div>
              <Slider
                id="filterSize"
                min={1}
                max={7}
                step={2}
                value={[filterSize]}
                onValueChange={(value) => setFilterSize(value[0])}
                className="notebook-slider"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="numFilters" className="text-neutral-900">
                  Number of Filters
                </Label>
                <span className="text-sm text-neutral-600">{numFilters}</span>
              </div>
              <Slider
                id="numFilters"
                min={1}
                max={32}
                step={1}
                value={[numFilters]}
                onValueChange={(value) => setNumFilters(value[0])}
                className="notebook-slider"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="stride" className="text-neutral-900">
                  Stride
                </Label>
                <span className="text-sm text-neutral-600">{stride}</span>
              </div>
              <Slider
                id="stride"
                min={1}
                max={3}
                step={1}
                value={[stride]}
                onValueChange={(value) => setStride(value[0])}
                className="notebook-slider"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="padding" className="text-neutral-900">
                  Padding
                </Label>
                <span className="text-sm text-neutral-600">{padding}</span>
              </div>
              <Slider
                id="padding"
                min={0}
                max={2}
                step={1}
                value={[padding]}
                onValueChange={(value) => setPadding(value[0])}
                className="notebook-slider"
              />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="activeFilter" className="text-neutral-900">
                  Active Filter
                </Label>
                <span className="text-sm text-neutral-600">
                  {activeFilter + 1}/{numFilters}
                </span>
              </div>
              <Slider
                id="activeFilter"
                min={0}
                max={numFilters - 1}
                step={1}
                value={[activeFilter]}
                onValueChange={(value) => setActiveFilter(value[0])}
                className="notebook-slider"
              />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
