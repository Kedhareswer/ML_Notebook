"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight, BookOpen, PieChart } from "lucide-react"
import Link from "next/link"
import ModelVisualization from "@/components/model-visualization"

export default function KMeansPage() {
  const [activeTab, setActiveTab] = useState("explanation")

  // K-Means visualization function
  const renderKMeans = (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => {
    const { clusters, iterations, noise } = params

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Set up margins
    const margin = 30
    const plotWidth = width - 2 * margin
    const plotHeight = height - 2 * margin

    // Draw background
    ctx.fillStyle = "#f8f9fa"
    ctx.fillRect(margin, margin, plotWidth, plotHeight)

    // Generate random data points with clusters
    const points = []
    const k = clusters
    const numPoints = 150

    // Create random cluster centers
    const centers = []
    for (let i = 0; i < k; i++) {
      centers.push({
        x: margin + margin + Math.random() * (plotWidth - 2 * margin),
        y: margin + margin + Math.random() * (plotHeight - 2 * margin),
      })
    }

    // Generate points around centers
    for (let i = 0; i < numPoints; i++) {
      const centerIdx = i % k
      const center = centers[centerIdx]
      const noiseLevel = noise * 50
      const x = center.x + (Math.random() - 0.5) * noiseLevel
      const y = center.y + (Math.random() - 0.5) * noiseLevel
      points.push({ x, y, cluster: -1 }) // Initially no cluster assignment
    }

    // Colors for different clusters with improved visibility
    const colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#33FFF5", "#F533FF", "#FF3333", "#33FF33", "#3333FF"]

    // Save initial state of points for animation
    const initialPoints = JSON.parse(JSON.stringify(points))

    // Show initial state first
    // Draw all points in gray to show initial state
    for (let i = 0; i < points.length; i++) {
      const point = initialPoints[i]
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
      ctx.fillStyle = "#AAAAAA"
      ctx.fill()
    }

    // Initialize centroids randomly
    const centroids = []
    const usedIndexes = new Set()

    while (centroids.length < k) {
      const idx = Math.floor(Math.random() * points.length)
      if (!usedIndexes.has(idx)) {
        usedIndexes.add(idx)
        centroids.push({ x: points[idx].x, y: points[idx].y })
      }
    }

    // Visualize centroid initialization
    for (let i = 0; i < centroids.length; i++) {
      ctx.beginPath()
      ctx.arc(centroids[i].x, centroids[i].y, 8, 0, Math.PI * 2)
      ctx.fillStyle = "#000000"
      ctx.fill()
      ctx.strokeStyle = colors[i % colors.length]
      ctx.lineWidth = 3
      ctx.stroke()
    }

    // Store centroid movement for visualization
    const centroidHistory = [JSON.parse(JSON.stringify(centroids))]

    // Run iterations
    for (let iter = 0; iter < iterations; iter++) {
      // Assign points to clusters
      for (let i = 0; i < points.length; i++) {
        let minDist = Number.POSITIVE_INFINITY
        let closestCentroid = 0

        for (let j = 0; j < centroids.length; j++) {
          const dist = Math.sqrt(Math.pow(points[i].x - centroids[j].x, 2) + Math.pow(points[i].y - centroids[j].y, 2))

          if (dist < minDist) {
            minDist = dist
            closestCentroid = j
          }
        }

        points[i].cluster = closestCentroid
      }

      // Update centroids
      const newCentroids = Array(k)
        .fill(null)
        .map(() => ({ x: 0, y: 0, count: 0 }))

      for (let i = 0; i < points.length; i++) {
        const cluster = points[i].cluster
        newCentroids[cluster].x += points[i].x
        newCentroids[cluster].y += points[i].y
        newCentroids[cluster].count += 1
      }

      for (let i = 0; i < k; i++) {
        if (newCentroids[i].count > 0) {
          centroids[i] = {
            x: newCentroids[i].x / newCentroids[i].count,
            y: newCentroids[i].y / newCentroids[i].count,
          }
        }
      }

      // Save centroid positions for animation
      centroidHistory.push(JSON.parse(JSON.stringify(centroids)))
    }

    // Draw voronoi-like decision boundaries
    const gridSize = 10
    for (let x = margin; x <= width - margin; x += gridSize) {
      for (let y = margin; y <= height - margin; y += gridSize) {
        let minDist = Number.POSITIVE_INFINITY
        let closestCentroid = 0

        for (let j = 0; j < centroids.length; j++) {
          const dist = Math.sqrt(Math.pow(x - centroids[j].x, 2) + Math.pow(y - centroids[j].y, 2))

          if (dist < minDist) {
            minDist = dist
            closestCentroid = j
          }
        }

        ctx.fillStyle = `${colors[closestCentroid % colors.length]}22`
        ctx.fillRect(x - gridSize / 2, y - gridSize / 2, gridSize, gridSize)
      }
    }

    // Visualize convergence paths for centroids
    for (let i = 0; i < centroids.length; i++) {
      const path = centroidHistory.map((history) => history[i])

      ctx.strokeStyle = colors[i % colors.length]
      ctx.lineWidth = 2
      ctx.setLineDash([3, 3])

      ctx.beginPath()
      ctx.moveTo(path[0].x, path[0].y)

      for (let j = 1; j < path.length; j++) {
        ctx.lineTo(path[j].x, path[j].y)
      }

      ctx.stroke()
      ctx.setLineDash([])
    }

    // Draw the final state of points
    for (let i = 0; i < points.length; i++) {
      const point = points[i]
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
      ctx.fillStyle = colors[point.cluster % colors.length]
      ctx.fill()
    }

    // Draw final centroids
    for (let i = 0; i < centroids.length; i++) {
      ctx.beginPath()
      ctx.arc(centroids[i].x, centroids[i].y, 8, 0, Math.PI * 2)
      ctx.fillStyle = "#000000"
      ctx.fill()
      ctx.strokeStyle = colors[i % colors.length]
      ctx.lineWidth = 3
      ctx.stroke()
    }

    // Draw enhanced legend
    const legendX = width - 160
    const legendY = 50
    const legendSpacing = 25
    const legendBoxSize = 15

    // Draw legend background
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)"
    ctx.fillRect(legendX - 10, legendY - 10, 150, (k + 3) * legendSpacing + 10)
    ctx.strokeStyle = "#ddd"
    ctx.strokeRect(legendX - 10, legendY - 10, 150, (k + 3) * legendSpacing + 10)

    // Draw legend title
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "left"
    ctx.fillText("K-Means Legend", legendX, legendY)

    // Draw cluster legend items
    for (let i = 0; i < k; i++) {
      // Cluster circle
      ctx.beginPath()
      ctx.arc(
        legendX + legendBoxSize / 2,
        legendY + (i + 1) * legendSpacing + legendBoxSize / 2,
        legendBoxSize / 2,
        0,
        Math.PI * 2,
      )
      ctx.fillStyle = colors[i % colors.length]
      ctx.fill()

      // Label
      ctx.fillStyle = "#000"
      ctx.textAlign = "left"
      ctx.fillText(
        `Cluster ${i + 1}`,
        legendX + legendBoxSize + 10,
        legendY + (i + 1) * legendSpacing + legendBoxSize / 2 + 4,
      )
    }

    // Centroid legend
    ctx.beginPath()
    ctx.arc(
      legendX + legendBoxSize / 2,
      legendY + (k + 1) * legendSpacing + legendBoxSize / 2,
      legendBoxSize / 2,
      0,
      Math.PI * 2,
    )
    ctx.fillStyle = "#000"
    ctx.fill()
    ctx.strokeStyle = "#FF5733"
    ctx.lineWidth = 2
    ctx.stroke()

    ctx.fillStyle = "#000"
    ctx.textAlign = "left"
    ctx.fillText("Centroid", legendX + legendBoxSize + 10, legendY + (k + 1) * legendSpacing + legendBoxSize / 2 + 4)

    // Path legend
    ctx.strokeStyle = "#FF5733"
    ctx.lineWidth = 2
    ctx.setLineDash([3, 3])

    ctx.beginPath()
    ctx.moveTo(legendX, legendY + (k + 2) * legendSpacing + legendBoxSize / 2)
    ctx.lineTo(legendX + legendBoxSize, legendY + (k + 2) * legendSpacing + legendBoxSize / 2)
    ctx.stroke()
    ctx.setLineDash([])

    ctx.fillStyle = "#000"
    ctx.textAlign = "left"
    ctx.fillText(
      "Convergence Path",
      legendX + legendBoxSize + 10,
      legendY + (k + 2) * legendSpacing + legendBoxSize / 2 + 4,
    )

    // Add algorithm iteration info
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "left"
    ctx.fillText(`K = ${clusters}, Iterations = ${iterations}, Noise = ${noise}`, margin, height - 10)

    // Inertia calculation (sum of squared distances from points to their centroids)
    let inertia = 0
    for (let i = 0; i < points.length; i++) {
      const point = points[i]
      const centroid = centroids[point.cluster]
      const dist = Math.sqrt(Math.pow(point.x - centroid.x, 2) + Math.pow(point.y - centroid.y, 2))
      inertia += dist * dist
    }

    // Show inertia value (quality metric)
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "left"
    ctx.fillText(`Inertia: ${Math.round(inertia)}`, margin, height - 30)
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">K-Means Clustering</h1>
          <p className="text-neutral-700 mt-2">
            A simple yet powerful unsupervised learning algorithm for partitioning data into clusters
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models">All Models</Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/hierarchical-clustering">
              Next: Hierarchical Clustering <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      <Tabs defaultValue="explanation" value={activeTab} onValueChange={setActiveTab} className="space-y-8">
        <TabsList className="grid w-full grid-cols-2 bg-neutral-100 text-neutral-900">
          <TabsTrigger value="explanation" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BookOpen className="h-4 w-4" />
            <span>Explanation</span>
          </TabsTrigger>
          <TabsTrigger value="visualization" className="flex items-center gap-2 data-[state=active]:bg-white">
            <PieChart className="h-4 w-4" />
            <span>Visualization</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="explanation" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">What is K-Means Clustering?</CardTitle>
              <CardDescription className="text-neutral-600">
                An unsupervised learning algorithm that groups similar data points into clusters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                K-Means is one of the simplest and most popular unsupervised machine learning algorithms used for
                clustering analysis. The algorithm works by partitioning n observations into k clusters where each
                observation belongs to the cluster with the nearest mean (cluster centroid), serving as a prototype of
                the cluster.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">The K-Means Algorithm</h3>
                <p className="text-neutral-700 mb-3">
                  The standard K-Means algorithm follows an iterative refinement approach:
                </p>
                <ol className="list-decimal list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Initialization</strong>: Randomly select k points as initial
                    centroids
                  </li>
                  <li>
                    <strong className="text-neutral-900">Assignment</strong>: Assign each data point to the nearest
                    centroid, forming k clusters
                  </li>
                  <li>
                    <strong className="text-neutral-900">Update</strong>: Recalculate the centroids as the mean of all
                    points in each cluster
                  </li>
                  <li>
                    <strong className="text-neutral-900">Repeat</strong>: Repeat steps 2 and 3 until convergence
                    (centroids no longer change significantly or a maximum number of iterations is reached)
                  </li>
                </ol>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">Key Concepts</h3>
              <div className="grid gap-6 md:grid-cols-2">
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Centroids</h4>
                  <p className="text-neutral-700">
                    The mean point of all samples in a cluster, representing the cluster's center. The goal of K-Means
                    is to find centroids that minimize the distance to points in their respective clusters.
                  </p>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Number of Clusters (k)</h4>
                  <p className="text-neutral-700">
                    A user-defined parameter that specifies how many clusters to partition the data into. Selecting the
                    optimal k is crucial and often determined using techniques like the elbow method.
                  </p>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Inertia</h4>
                  <p className="text-neutral-700">
                    The sum of squared distances of samples to their closest centroid, measuring how internally coherent
                    clusters are. Lower inertia indicates better-defined clusters.
                  </p>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Distance Metric</h4>
                  <p className="text-neutral-700">
                    Typically Euclidean distance is used, but other metrics like Manhattan or cosine distance can be
                    employed depending on the problem domain and data characteristics.
                  </p>
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Advantages and Limitations</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Advantages</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Simple to understand and implement</li>
                      <li>Scales well to large datasets</li>
                      <li>Guarantees convergence</li>
                      <li>Easily adapts to new examples</li>
                      <li>Generalizes to clusters of different shapes and sizes</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Limitations</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Requires number of clusters (k) to be specified</li>
                      <li>Sensitive to initial placement of centroids</li>
                      <li>May converge to local optima</li>
                      <li>Assumes clusters are spherical and equally sized</li>
                      <li>Sensitive to outliers</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualization" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Interactive K-Means Clustering</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how K-Means clustering works
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="K-Means Clustering Visualization"
                parameters={[
                  {
                    name: "clusters",
                    min: 2,
                    max: 8,
                    step: 1,
                    default: 4,
                    label: "Number of Clusters (k)",
                  },
                  {
                    name: "iterations",
                    min: 1,
                    max: 10,
                    step: 1,
                    default: 3,
                    label: "Iterations",
                  },
                  {
                    name: "noise",
                    min: 0,
                    max: 2,
                    step: 0.1,
                    default: 1,
                    label: "Noise Level",
                  },
                ]}
                renderVisualization={renderKMeans}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Number of Clusters (k)</h3>
                <p className="text-neutral-700">
                  This parameter defines how many clusters the algorithm will create. Choosing the right value for k is
                  crucial for effective clustering. Too few clusters might group dissimilar data points together, while
                  too many clusters might fragment natural groupings. In practice, techniques like the elbow method or
                  silhouette analysis help determine the optimal k.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Iterations</h3>
                <p className="text-neutral-700">
                  This controls how many times the algorithm will repeat the assignment and update steps. More
                  iterations typically lead to better convergence, but the algorithm may stabilize after just a few
                  iterations. In practice, K-means usually runs until the centroids stop moving significantly or a
                  maximum iteration count is reached.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Noise Level</h3>
                <p className="text-neutral-700">
                  This controls how spread out the data points are around their true centroids. Higher noise makes the
                  clusters less distinct and more challenging to separate. This illustrates how K-means performs on data
                  with varying degrees of cluster overlap and separation.
                </p>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Interpreting the Visualization</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong>Colored points</strong> represent data points assigned to different clusters
                  </li>
                  <li>
                    <strong>Black circles</strong> represent the centroids of each cluster
                  </li>
                  <li>
                    <strong>Colored backgrounds</strong> show the decision boundaries between clusters
                  </li>
                  <li>
                    Observe how the centroids move with each iteration to find the center of their assigned points
                  </li>
                  <li>Notice how increasing noise makes the clusters less distinct and more challenging to separate</li>
                </ul>
              </div>
            </CardContent>
          </Card>
          <div className="bg-neutral-50 p-4 rounded-lg border border-neutral-200 mt-6">
            <h3 className="font-medium text-neutral-900 mb-2">Visualization Features</h3>
            <ul className="list-disc list-inside space-y-1 text-neutral-700">
              <li>
                <strong>Cluster boundaries</strong>: The colored regions show the decision boundaries of each cluster
              </li>
              <li>
                <strong>Centroids</strong>: The black circles represent the centroid (mean) of each cluster
              </li>
              <li>
                <strong>Convergence paths</strong>: The dashed lines show how centroids move during the iterations
              </li>
              <li>
                <strong>Inertia</strong>: A metric of clustering quality (lower is better) - represents the sum of
                squared distances
              </li>
              <li>
                <strong>Interactive parameters</strong>: Adjust clusters, iterations, and noise to see their effects
              </li>
            </ul>
            <p className="mt-3 text-neutral-700">
              The visualization shows the complete K-means algorithm process from initialization to convergence.
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
