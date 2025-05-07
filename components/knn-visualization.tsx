"use client"

import { useState } from "react"
import { Slider } from "@/components/ui/slider"

export default function KNNVisualization() {
  const [k, setK] = useState(5)
  const [noise, setNoise] = useState(0.3)

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h3 className="text-lg font-medium">K-Nearest Neighbors Visualization</h3>
        <p className="text-sm text-neutral-500">
          This visualization shows how KNN classifies points in a 2D space. You can adjust the number of neighbors (k)
          and the noise in the data to see how the decision boundaries change.
        </p>
      </div>

      <div className="border rounded-lg overflow-hidden bg-neutral-50">
        <div className="flex items-center justify-center bg-neutral-100 h-[400px]">
          <div className="text-center p-6">
            <div className="mb-4 text-neutral-700 font-medium">KNN Decision Boundaries</div>
            <div className="w-full h-[300px] bg-white rounded-lg border border-neutral-200 flex items-center justify-center">
              <div className="text-neutral-500">
                KNN visualization with k={k} and noise={noise.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <div className="flex justify-between">
            <label htmlFor="k-value" className="text-sm font-medium">
              Number of Neighbors (k)
            </label>
            <span className="text-sm text-neutral-500">{k}</span>
          </div>
          <Slider id="k-value" min={1} max={20} step={1} value={[k]} onValueChange={(value) => setK(value[0])} />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <label htmlFor="noise-value" className="text-sm font-medium">
              Data Noise
            </label>
            <span className="text-sm text-neutral-500">{noise.toFixed(2)}</span>
          </div>
          <Slider
            id="noise-value"
            min={0}
            max={1}
            step={0.05}
            value={[noise]}
            onValueChange={(value) => setNoise(value[0])}
          />
        </div>
      </div>
    </div>
  )
}
