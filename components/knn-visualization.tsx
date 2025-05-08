"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

export default function KNNVisualization() {
  const [k, setK] = useState(3)
  const [dataPoints, setDataPoints] = useState<Array<{ x: number; y: number; class: string }>>([])
  const [testPoint, setTestPoint] = useState<{ x: number; y: number } | null>(null)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [datasetType, setDatasetType] = useState("linear")
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Generate initial dataset
  useEffect(() => {
    generateDataset(datasetType)
  }, [datasetType])

  // Draw canvas whenever data changes
  useEffect(() => {
    drawCanvas()
  }, [dataPoints, testPoint, k, prediction])

  // Generate dataset based on type
  const generateDataset = (type: string) => {
    const newDataPoints = []

    // Generate 100 random data points
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * 400
      const y = Math.random() * 400
      let pointClass = ""

      if (type === "linear") {
        // Linear separation
        pointClass = y > x ? "A" : "B"
      } else if (type === "circular") {
        // Circular separation
        const centerX = 200
        const centerY = 200
        const distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2))
        pointClass = distance < 150 ? "A" : "B"
      } else if (type === "clusters") {
        // Multiple clusters
        if (x < 150 && y < 150) pointClass = "A"
        else if (x > 250 && y < 150) pointClass = "B"
        else if (x < 150 && y > 250) pointClass = "C"
        else if (x > 250 && y > 250) pointClass = "D"
        else pointClass = ["A", "B", "C", "D"][Math.floor(Math.random() * 4)]
      }

      newDataPoints.push({ x, y, class: pointClass })
    }

    setDataPoints(newDataPoints)
    setTestPoint(null)
    setPrediction(null)
  }

  // Draw the canvas with all data points
  const drawCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw data points
    dataPoints.forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)

      // Set color based on class
      if (point.class === "A")
        ctx.fillStyle = "#1f1f1f" // Dark gray for class A
      else if (point.class === "B")
        ctx.fillStyle = "#7f7f7f" // Medium gray for class B
      else if (point.class === "C")
        ctx.fillStyle = "#bfbfbf" // Light gray for class C
      else if (point.class === "D") ctx.fillStyle = "#e5e5e5" // Very light gray for class D

      ctx.fill()
    })

    // Draw test point if exists
    if (testPoint) {
      // Draw the test point
      ctx.beginPath()
      ctx.arc(testPoint.x, testPoint.y, 8, 0, Math.PI * 2)
      ctx.fillStyle = prediction
        ? prediction === "A"
          ? "#1f1f1f"
          : prediction === "B"
            ? "#7f7f7f"
            : prediction === "C"
              ? "#bfbfbf"
              : "#e5e5e5"
        : "#9ca3af"
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 2
      ctx.stroke()

      // Draw lines to k nearest neighbors
      if (k > 0) {
        // Calculate distances
        const distances = dataPoints.map((point) => ({
          point,
          distance: Math.sqrt(Math.pow(point.x - testPoint.x, 2) + Math.pow(point.y - testPoint.y, 2)),
        }))

        // Sort by distance
        distances.sort((a, b) => a.distance - b.distance)

        // Draw lines to k nearest neighbors
        for (let i = 0; i < Math.min(k, distances.length); i++) {
          const neighbor = distances[i].point
          ctx.beginPath()
          ctx.moveTo(testPoint.x, testPoint.y)
          ctx.lineTo(neighbor.x, neighbor.y)
          ctx.strokeStyle = "#666666" // Medium gray
          ctx.lineWidth = 1
          ctx.stroke()
        }
      }
    }
  }

  // Handle canvas click to set test point
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    setTestPoint({ x, y })

    // Make prediction
    if (dataPoints.length > 0 && k > 0) {
      // Calculate distances
      const distances = dataPoints.map((point) => ({
        point,
        distance: Math.sqrt(Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2)),
      }))

      // Sort by distance
      distances.sort((a, b) => a.distance - b.distance)

      // Count classes of k nearest neighbors
      const classCounts: Record<string, number> = {}
      for (let i = 0; i < Math.min(k, distances.length); i++) {
        const neighborClass = distances[i].point.class
        classCounts[neighborClass] = (classCounts[neighborClass] || 0) + 1
      }

      // Find the most common class
      let maxCount = 0
      let predictedClass = ""
      for (const [className, count] of Object.entries(classCounts)) {
        if (count > maxCount) {
          maxCount = count
          predictedClass = className
        }
      }

      setPrediction(predictedClass)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col md:flex-row gap-4 mb-4">
        <div className="flex-1">
          <label className="block text-sm font-medium mb-2">K Value: {k}</label>
          <Slider value={[k]} min={1} max={15} step={1} onValueChange={(value) => setK(value[0])} />
        </div>

        <div className="w-full md:w-64">
          <label className="block text-sm font-medium mb-2">Dataset Type</label>
          <Select value={datasetType} onValueChange={setDatasetType}>
            <SelectTrigger>
              <SelectValue placeholder="Select dataset type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="linear">Linear Boundary</SelectItem>
              <SelectItem value="circular">Circular Boundary</SelectItem>
              <SelectItem value="clusters">Multiple Clusters</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-end">
          <Button
            onClick={() => generateDataset(datasetType)}
            className="bg-neutral-800 text-white hover:bg-neutral-700"
          >
            Regenerate Data
          </Button>
        </div>
      </div>

      <Card className="border-neutral-300">
        <CardContent className="p-4">
          <canvas
            ref={canvasRef}
            width={400}
            height={400}
            onClick={handleCanvasClick}
            className="border border-neutral-300 rounded-md cursor-crosshair"
          />
        </CardContent>
      </Card>

      <div className="bg-neutral-100 p-4 rounded-md">
        <h3 className="font-medium mb-2">Instructions:</h3>
        <ul className="list-disc pl-5 space-y-1">
          <li>Click anywhere on the canvas to place a test point</li>
          <li>Adjust the K value to see how it affects the classification</li>
          <li>Try different dataset types to see how KNN performs on various data distributions</li>
        </ul>

        {testPoint && prediction && (
          <div className="mt-4">
            <p className="font-medium">Prediction: Class {prediction}</p>
            <p className="text-sm text-neutral-600">Based on the {k} nearest neighbors</p>
          </div>
        )}
      </div>
    </div>
  )
}
