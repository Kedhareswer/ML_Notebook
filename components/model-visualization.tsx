"use client"

import { useState, useEffect, useRef } from "react"
import { Slider } from "@/components/ui/slider"

interface ModelVisualizationProps {
  title: string
  description?: string
  parameters: {
    name: string
    min: number
    max: number
    step: number
    default: number
    label: string
  }[]
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
  description,
  parameters,
  renderVisualization,
  width = 600,
  height = 400,
}: ModelVisualizationProps) {
  const [params, setParams] = useState<Record<string, number>>({})
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const initialParams: Record<string, number> = {}
    parameters.forEach((param) => {
      initialParams[param.name] = param.default
    })
    setParams(initialParams)
  }, [parameters])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    renderVisualization(ctx, params, width, height)
  }, [params, renderVisualization, width, height])

  const handleParamChange = (name: string, value: number[]) => {
    setParams((prev) => ({
      ...prev,
      [name]: value[0],
    }))
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h3 className="text-lg font-medium">{title}</h3>
        {description && <p className="text-sm text-neutral-500">{description}</p>}
      </div>

      <div className="border rounded-lg overflow-hidden bg-neutral-50">
        <canvas ref={canvasRef} width={width} height={height} />
      </div>

      <div className="space-y-4">
        {parameters.map((param) => (
          <div key={param.name} className="space-y-2">
            <div className="flex justify-between">
              <label htmlFor={param.name} className="text-sm font-medium">
                {param.label}
              </label>
              <span className="text-sm text-neutral-500">{params[param.name] ?? param.default}</span>
            </div>
            <Slider
              id={param.name}
              min={param.min}
              max={param.max}
              step={param.step}
              value={[params[param.name] ?? param.default]}
              onValueChange={(value) => handleParamChange(param.name, value)}
            />
          </div>
        ))}
      </div>
    </div>
  )
}
