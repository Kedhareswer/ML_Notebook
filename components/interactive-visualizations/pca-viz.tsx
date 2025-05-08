"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

const PCAViz: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [components, setComponents] = useState<number>(2)
  const [dataPoints, setDataPoints] = useState<number>(100)
  const [variance, setVariance] = useState<number>(0.8)
  const [rotation, setRotation] = useState<number>(45)
  const [datasetType, setDatasetType] = useState<string>("gaussian")
  const [activeTab, setActiveTab] = useState<string>("visualization")

  // Generate synthetic data
  const generateData = (n = 100, variance = 0.8, rotation = 45, type = "gaussian") => {
    const data = []
    const angle = (rotation * Math.PI) / 180
    const cos = Math.cos(angle)
    const sin = Math.sin(angle)

    // Set variance for each dimension
    const varX = variance
    const varY = 1 - variance

    for (let i = 0; i < n; i++) {
      let x, y

      if (type === "gaussian") {
        // Gaussian distribution
        x = randomGaussian() * Math.sqrt(varX)
        y = randomGaussian() * Math.sqrt(varY)
      } else if (type === "uniform") {
        // Uniform distribution
        x = (Math.random() * 2 - 1) * Math.sqrt(varX * 3)
        y = (Math.random() * 2 - 1) * Math.sqrt(varY * 3)
      } else if (type === "clustered") {
        // Clustered data (3 clusters)
        const cluster = Math.floor(Math.random() * 3)
        const clusterCenters = [
          { x: -0.5, y: -0.5 },
          { x: 0.5, y: 0.5 },
          { x: 0, y: 0 },
        ]

        x = clusterCenters[cluster].x + randomGaussian() * 0.2 * Math.sqrt(varX)
        y = clusterCenters[cluster].y + randomGaussian() * 0.2 * Math.sqrt(varY)
      }

      // Rotate the point
      const rotatedX = x * cos - y * sin
      const rotatedY = x * sin + y * cos

      data.push({ x: rotatedX, y: rotatedY })
    }

    return data
  }

  // Box-Muller transform for Gaussian random numbers
  const randomGaussian = () => {
    let u = 0,
      v = 0
    while (u === 0) u = Math.random()
    while (v === 0) v = Math.random()
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
  }

  // Perform PCA
  const performPCA = (data: { x: number; y: number }[]) => {
    // Calculate mean
    const meanX = data.reduce((sum, p) => sum + p.x, 0) / data.length
    const meanY = data.reduce((sum, p) => sum + p.y, 0) / data.length

    // Center the data
    const centeredData = data.map((p) => ({ x: p.x - meanX, y: p.y - meanY }))

    // Calculate covariance matrix
    let covXX = 0,
      covXY = 0,
      covYY = 0

    centeredData.forEach((p) => {
      covXX += p.x * p.x
      covXY += p.x * p.y
      covYY += p.y * p.y
    })

    covXX /= data.length
    covXY /= data.length
    covYY /= data.length

    // Calculate eigenvalues and eigenvectors
    const trace = covXX + covYY
    const det = covXX * covYY - covXY * covXY

    const lambda1 = (trace + Math.sqrt(trace * trace - 4 * det)) / 2
    const lambda2 = (trace - Math.sqrt(trace * trace - 4 * det)) / 2

    // First eigenvector
    let v1x, v1y
    if (covXY !== 0) {
      v1x = lambda1 - covYY
      v1y = covXY
    } else {
      v1x = 1
      v1y = 0
    }

    // Normalize
    const mag1 = Math.sqrt(v1x * v1x + v1y * v1y)
    v1x /= mag1
    v1y /= mag1

    // Second eigenvector (perpendicular to first)
    const v2x = -v1y
    const v2y = v1x

    // Calculate explained variance
    const totalVariance = lambda1 + lambda2
    const explainedVariance1 = lambda1 / totalVariance
    const explainedVariance2 = lambda2 / totalVariance

    // Project data onto principal components
    const projectedData = centeredData.map((p) => {
      const pc1 = p.x * v1x + p.y * v1y
      const pc2 = p.x * v2x + p.y * v2y
      return { pc1, pc2 }
    })

    return {
      eigenvalues: [lambda1, lambda2],
      eigenvectors: [
        { x: v1x, y: v1y },
        { x: v2x, y: v2y },
      ],
      explainedVariance: [explainedVariance1, explainedVariance2],
      projectedData,
      meanX,
      meanY,
    }
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
    const data = generateData(dataPoints, variance, rotation, datasetType)

    // Perform PCA
    const pca = performPCA(data)

    // Scale for visualization
    const scale = 100
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2

    // Draw coordinate axes
    ctx.strokeStyle = "#ccc"
    ctx.lineWidth = 1

    // X-axis
    ctx.beginPath()
    ctx.moveTo(0, centerY)
    ctx.lineTo(canvas.width, centerY)
    ctx.stroke()

    // Y-axis
    ctx.beginPath()
    ctx.moveTo(centerX, 0)
    ctx.lineTo(centerX, canvas.height)
    ctx.stroke()

    // Draw data points
    if (activeTab === "visualization") {
      ctx.fillStyle = "rgba(0, 0, 0, 0.5)"
      data.forEach((point) => {
        ctx.beginPath()
        ctx.arc(centerX + point.x * scale, centerY - point.y * scale, 4, 0, Math.PI * 2)
        ctx.fill()
      })

      // Draw principal components
      const drawEigenvector = (vector: { x: number; y: number }, eigenvalue: number, color: string, label: string) => {
        const length = Math.sqrt(eigenvalue) * scale * 2

        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(centerX - vector.x * length, centerY + vector.y * length)
        ctx.lineTo(centerX + vector.x * length, centerY - vector.y * length)
        ctx.stroke()

        // Draw arrowhead
        const arrowSize = 10
        const angle = Math.atan2(-vector.y, vector.x)

        ctx.beginPath()
        ctx.moveTo(centerX + vector.x * length, centerY - vector.y * length)
        ctx.lineTo(
          centerX + vector.x * length - arrowSize * Math.cos(angle - Math.PI / 6),
          centerY - vector.y * length + arrowSize * Math.sin(angle - Math.PI / 6),
        )
        ctx.lineTo(
          centerX + vector.x * length - arrowSize * Math.cos(angle + Math.PI / 6),
          centerY - vector.y * length + arrowSize * Math.sin(angle + Math.PI / 6),
        )
        ctx.closePath()
        ctx.fillStyle = color
        ctx.fill()

        // Draw label
        ctx.fillStyle = color
        ctx.font = "14px Arial"
        ctx.fillText(
          `${label} (${(pca.explainedVariance[label === "PC1" ? 0 : 1] * 100).toFixed(1)}%)`,
          centerX + vector.x * length * 1.1,
          centerY - vector.y * length * 1.1,
        )
      }

      // Draw eigenvectors
      drawEigenvector(pca.eigenvectors[0], pca.eigenvalues[0], "rgba(50, 50, 50, 0.8)", "PC1")
      if (components > 1) {
        drawEigenvector(pca.eigenvectors[1], pca.eigenvalues[1], "rgba(110, 110, 110, 0.8)", "PC2")
      }
    } else if (activeTab === "projection") {
      // Draw original data
      ctx.fillStyle = "rgba(0, 0, 255, 0.3)"
      data.forEach((point) => {
        ctx.beginPath()
        ctx.arc(centerX + point.x * scale, centerY - point.y * scale, 4, 0, Math.PI * 2)
        ctx.fill()
      })

      // Draw projected data
      ctx.fillStyle = "rgba(120, 120, 120, 0.7)"

      // Project onto first principal component only
      const pc1 = pca.eigenvectors[0]

      data.forEach((point, i) => {
        // Center the point
        const centeredX = point.x - pca.meanX
        const centeredY = point.y - pca.meanY

        // Project onto PC1
        const projection = centeredX * pc1.x + centeredY * pc1.y

        // Reconstruct in original space
        const reconstructedX = projection * pc1.x + pca.meanX
        const reconstructedY = projection * pc1.y + pca.meanY

        // Draw projected point
        ctx.beginPath()
        ctx.arc(centerX + reconstructedX * scale, centerY - reconstructedY * scale, 4, 0, Math.PI * 2)
        ctx.fill()

        // Draw line from original to projection
        ctx.strokeStyle = "rgba(0, 0, 0, 0.2)"
        ctx.beginPath()
        ctx.moveTo(centerX + point.x * scale, centerY - point.y * scale)
        ctx.lineTo(centerX + reconstructedX * scale, centerY - reconstructedY * scale)
        ctx.stroke()
      })

      // Draw PC1 axis
      ctx.strokeStyle = "rgba(255, 0, 0, 0.8)"
      ctx.lineWidth = 2
      const length = Math.sqrt(pca.eigenvalues[0]) * scale * 2

      ctx.beginPath()
      ctx.moveTo(centerX - pc1.x * length, centerY + pc1.y * length)
      ctx.lineTo(centerX + pc1.x * length, centerY - pc1.y * length)
      ctx.stroke()
    } else if (activeTab === "reconstruction") {
      // Draw original data
      ctx.fillStyle = "rgba(0, 0, 255, 0.3)"
      data.forEach((point) => {
        ctx.beginPath()
        ctx.arc(centerX + point.x * scale, centerY - point.y * scale, 4, 0, Math.PI * 2)
        ctx.fill()
      })

      // Draw reconstructed data
      ctx.fillStyle = "rgba(255, 0, 0, 0.7)"

      // Get principal components
      const pc1 = pca.eigenvectors[0]
      const pc2 = pca.eigenvectors[1]

      data.forEach((point, i) => {
        // Center the point
        const centeredX = point.x - pca.meanX
        const centeredY = point.y - pca.meanY

        // Project onto principal components
        const projection1 = centeredX * pc1.x + centeredY * pc1.y
        let projection2 = 0

        if (components > 1) {
          projection2 = centeredX * pc2.x + centeredY * pc2.y
        }

        // Reconstruct in original space
        const reconstructedX = projection1 * pc1.x + projection2 * pc2.x + pca.meanX
        const reconstructedY = projection1 * pc1.y + projection2 * pc2.y + pca.meanY

        // Draw reconstructed point
        ctx.beginPath()
        ctx.arc(centerX + reconstructedX * scale, centerY - reconstructedY * scale, 4, 0, Math.PI * 2)
        ctx.fill()

        // Draw line from original to reconstruction
        ctx.strokeStyle = "rgba(0, 0, 0, 0.2)"
        ctx.beginPath()
        ctx.moveTo(centerX + point.x * scale, centerY - point.y * scale)
        ctx.lineTo(centerX + reconstructedX * scale, centerY - reconstructedY * scale)
        ctx.stroke()
      })
    }
  }

  // Update visualization when parameters change
  useEffect(() => {
    drawVisualization()
  }, [components, dataPoints, variance, rotation, datasetType, activeTab])

  // Resize canvas when window resizes
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current
      if (!canvas) return

      canvas.width = canvas.offsetWidth
      canvas.height = canvas.offsetHeight

      drawVisualization()
    }

    window.addEventListener("resize", handleResize)
    handleResize()

    return () => window.removeEventListener("resize", handleResize)
  }, [])

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Principal Component Analysis (PCA) Visualization</CardTitle>
          <CardDescription>
            Explore how PCA works by adjusting parameters and seeing the effects on dimensionality reduction
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="visualization">Visualization</TabsTrigger>
                <TabsTrigger value="projection">Projection</TabsTrigger>
                <TabsTrigger value="reconstruction">Reconstruction</TabsTrigger>
              </TabsList>
              <TabsContent value="visualization" className="mt-4">
                <p className="text-sm text-muted-foreground mb-4">
                  This view shows the original data points and the principal components as vectors. The length of each
                  vector represents the amount of variance explained.
                </p>
              </TabsContent>
              <TabsContent value="projection" className="mt-4">
                <p className="text-sm text-muted-foreground mb-4">
                  This view shows the original data points (blue) and their projections onto the first principal
                  component (red).
                </p>
              </TabsContent>
              <TabsContent value="reconstruction" className="mt-4">
                <p className="text-sm text-muted-foreground mb-4">
                  This view shows the original data points (blue) and their reconstructions (red) after projecting onto
                  the principal components.
                </p>
              </TabsContent>
            </Tabs>

            <div className="relative aspect-[4/3] w-full bg-muted/20 rounded-lg overflow-hidden">
              <canvas ref={canvasRef} className="w-full h-full" />
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="grid gap-2">
                <label className="text-sm font-medium">Dataset Type</label>
                <Select value={datasetType} onValueChange={setDatasetType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select dataset type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gaussian">Gaussian</SelectItem>
                    <SelectItem value="uniform">Uniform</SelectItem>
                    <SelectItem value="clustered">Clustered</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid gap-2">
                <label className="text-sm font-medium">Number of Components: {components}</label>
                <Slider
                  value={[components]}
                  min={1}
                  max={2}
                  step={1}
                  onValueChange={(value) => setComponents(value[0])}
                />
              </div>

              <div className="grid gap-2">
                <label className="text-sm font-medium">Data Points: {dataPoints}</label>
                <Slider
                  value={[dataPoints]}
                  min={10}
                  max={500}
                  step={10}
                  onValueChange={(value) => setDataPoints(value[0])}
                />
              </div>

              <div className="grid gap-2">
                <label className="text-sm font-medium">Variance Ratio: {variance.toFixed(2)}</label>
                <Slider
                  value={[variance]}
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  onValueChange={(value) => setVariance(value[0])}
                />
              </div>

              <div className="grid gap-2">
                <label className="text-sm font-medium">Rotation: {rotation}°</label>
                <Slider
                  value={[rotation]}
                  min={0}
                  max={180}
                  step={5}
                  onValueChange={(value) => setRotation(value[0])}
                />
              </div>

              <div className="flex items-end">
                <Button onClick={() => drawVisualization()}>Regenerate Data</Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>How PCA Works</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p>
              Principal Component Analysis (PCA) is a dimensionality reduction technique that finds the directions
              (principal components) along which the data varies the most. These directions are orthogonal to each other
              and are ordered by the amount of variance they explain.
            </p>

            <ol className="list-decimal list-inside space-y-2">
              <li>Center the data by subtracting the mean</li>
              <li>Compute the covariance matrix</li>
              <li>Find the eigenvalues and eigenvectors of the covariance matrix</li>
              <li>Sort the eigenvectors by their eigenvalues in descending order</li>
              <li>Project the data onto the selected principal components</li>
            </ol>

            <p>
              The interactive visualization above demonstrates these concepts. You can adjust parameters to see how they
              affect the principal components and the projection of data onto these components.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default PCAViz
