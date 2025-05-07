"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"

interface Parameter {
  name: string
  min: number
  max: number
  step: number
  defaultValue: number
  label: string
}

interface ModelVisualizationProps {
  title: string
  parameters: Parameter[]
  renderVisualization: (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => void
  width?: number
  height?: number
}

export default function ModelVisualization({
  title,
  parameters,
  renderVisualization,
  width = 600,
  height = 400,
}: ModelVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [params, setParams] = useState<Record<string, number>>({})

  // Initialize parameters with default values
  useEffect(() => {
    const initialParams: Record<string, number> = {}
    parameters.forEach((param) => {
      initialParams[param.name] = param.defaultValue
    })
    setParams(initialParams)
  }, [parameters])

  // Update visualization when parameters change
  useEffect(() => {
    if (Object.keys(params).length === 0) return

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Render visualization
    renderVisualization(ctx, params, canvas.width, canvas.height)
  }, [params, renderVisualization])

  const handleParamChange = (name: string, value: number[]) => {
    setParams((prev) => ({
      ...prev,
      [name]: value[0],
    }))
  }

  return (
    <Card className="w-full border-neutral-300 bg-white">
      <CardContent className="pt-6">
        <div className="mb-4 text-lg font-medium text-neutral-900">{title}</div>
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
            {parameters.map((param) => (
              <div key={param.name} className="space-y-2">
                <div className="flex justify-between">
                  <Label htmlFor={param.name} className="text-neutral-900">
                    {param.label}
                  </Label>
                  <span className="text-sm text-neutral-600">
                    {params[param.name]?.toFixed(2) || param.defaultValue.toFixed(2)}
                  </span>
                </div>
                <Slider
                  id={param.name}
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  value={[params[param.name] || param.defaultValue]}
                  onValueChange={(value) => handleParamChange(param.name, value)}
                  className="notebook-slider"
                />
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
