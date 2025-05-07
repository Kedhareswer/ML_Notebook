"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"

export default function KNNVisualization() {
  const [k, setK] = useState(5)
  const [noise, setNoise] = useState(0.3)

  return (
    <Card className="w-full border-neutral-300 bg-white">
      <CardContent className="pt-6">
        <div className="mb-4 text-lg font-medium text-neutral-900">K-Nearest Neighbors Visualization</div>
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1 order-2 lg:order-1">
            <div className="w-full h-[400px] bg-neutral-50 border border-neutral-300 rounded-md flex items-center justify-center">
              <div className="text-center p-4">
                <p className="text-neutral-500 mb-2">
                  KNN visualization with k={k} and noise={noise.toFixed(2)}
                </p>
                <p className="text-sm text-neutral-400">
                  This interactive visualization demonstrates how KNN classifies points in a 2D space.
                </p>
              </div>
            </div>
          </div>
          <div className="w-full lg:w-64 space-y-6 order-1 lg:order-2">
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="k-value" className="text-neutral-900">
                  Number of Neighbors (k)
                </Label>
                <span className="text-sm text-neutral-600">{k}</span>
              </div>
              <Slider
                id="k-value"
                min={1}
                max={20}
                step={1}
                value={[k]}
                onValueChange={(value) => setK(value[0])}
                className="notebook-slider"
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="noise-value" className="text-neutral-900">
                  Data Noise
                </Label>
                <span className="text-sm text-neutral-600">{noise.toFixed(2)}</span>
              </div>
              <Slider
                id="noise-value"
                min={0}
                max={1}
                step={0.05}
                value={[noise]}
                onValueChange={(value) => setNoise(value[0])}
                className="notebook-slider"
              />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
