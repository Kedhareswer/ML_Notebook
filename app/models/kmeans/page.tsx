"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight, BookOpen, Code, PieChart } from "lucide-react"
import Link from "next/link"
import NotebookCell from "@/components/notebook-cell"
import ModelVisualization from "@/components/model-visualization"

export default function KMeansPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

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

    // Generate random data points with clusters
    const points = []
    const k = clusters
    const numPoints = 150

    // Create random cluster centers
    const centers = []
    for (let i = 0; i < k; i++) {
      centers.push({
        x: margin + Math.random() * plotWidth,
        y: margin + Math.random() * plotHeight,
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

    // Colors for different clusters
    const colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#33FFF5", "#F533FF", "#FF3333", "#33FF33", "#3333FF"]

    // Run K-means algorithm
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
    }

    // Draw points
    for (let i = 0; i < points.length; i++) {
      const point = points[i]
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
      ctx.fillStyle = colors[point.cluster % colors.length]
      ctx.fill()
    }

    // Draw centroids
    for (let i = 0; i < centroids.length; i++) {
      ctx.beginPath()
      ctx.arc(centroids[i].x, centroids[i].y, 8, 0, Math.PI * 2)
      ctx.fillStyle = "#000000"
      ctx.fill()
      ctx.strokeStyle = colors[i % colors.length]
      ctx.lineWidth = 3
      ctx.stroke()
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

    // Add legend
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "left"
    ctx.fillText(`K = ${clusters}, Iterations = ${iterations}, Noise = ${noise}`, margin, height - 10)
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Inertia: 307.25
          <br />
          Silhouette Score: 0.6824
          <br />
          Number of iterations: 8
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">K-Means Clusters and Centroids:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">K-Means clustering plot</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Optimal number of clusters: 4
          <br />
          Inertia values for k=1 to k=10:
          <br />
          k=1: 1432.56
          <br />
          k=2: 681.93
          <br />
          k=3: 452.18
          <br />
          k=4: 307.25
          <br />
          k=5: 253.87
          <br />
          k=6: 220.43
          <br />
          k=7: 193.21
          <br />
          k=8: 175.68
          <br />
          k=9: 160.24
          <br />
          k=10: 148.56
        </div>
      )
    }

    return "Executed successfully"
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
        <TabsList className="grid w-full grid-cols-3 bg-neutral-100 text-neutral-900">
          <TabsTrigger value="explanation" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BookOpen className="h-4 w-4" />
            <span>Explanation</span>
          </TabsTrigger>
          <TabsTrigger value="visualization" className="flex items-center gap-2 data-[state=active]:bg-white">
            <PieChart className="h-4 w-4" />
            <span>Visualization</span>
          </TabsTrigger>
          <TabsTrigger value="notebook" className="flex items-center gap-2 data-[state=active]:bg-white">
            <Code className="h-4 w-4" />
            <span>Notebook</span>
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

              <div className="bg-neutral-100 p-4 rounded-lg mt-6">
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
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">K-Means Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement K-Means clustering using Python and scikit-learn. Execute
                each cell to see the results.
              </p>
            </div>

            <div className="space-y-6">
              <NotebookCell
                cellId="cell0"
                executionCount={1}
                initialCode="import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap

# Set random seed for reproducibility
np.random.seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 1: Generate and cluster synthetic data</p>
                <p>Let's create a synthetic dataset with clearly defined clusters and apply K-means clustering.</p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode="# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Create and fit the K-means model
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X)

# Get the cluster assignments and centroids
y_kmeans = kmeans.predict(X)
centroids = kmeans.cluster_centers_

# Calculate inertia and silhouette score
inertia = kmeans.inertia_
silhouette = silhouette_score(X, y_kmeans)

print(f'Inertia: {inertia:.2f}')
print(f'Silhouette Score: {silhouette:.4f}')
print(f'Number of iterations: {kmeans.n_iter_}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Visualize the clusters</p>
                <p>Let's visualize the clusters and centroids from our K-means model.</p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Plot the clusters
plt.figure(figsize=(10, 8))
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

# Plot data points with cluster colors
for i in range(len(np.unique(y_kmeans))):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')

plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Find the optimal number of clusters</p>
                <p>Let's use the Elbow Method to determine the optimal number of clusters for our data.</p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Calculate inertia for different values of k
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Find the elbow point
# (A simple heuristic: look for the point where the rate of decrease sharply changes)
differences = np.diff(inertias)
differences_of_differences = np.diff(differences)
elbow_idx = np.argmin(differences_of_differences) + 1
optimal_k = k_range[elbow_idx]

print('Optimal number of clusters:', optimal_k)
print('Inertia values for k=1 to k=10:')
for i, k in enumerate(k_range):
    print('k=' + str(k) + ': ' + str(round(inertias[i], 2)))"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of K-Means clustering:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>Try different initialization methods ('random', 'k-means++') and compare results</li>
                  <li>Generate clusters with different variances and see how K-means performs</li>
                  <li>Implement the silhouette method for finding the optimal k</li>
                  <li>Apply K-means to a real-world dataset (e.g., Iris dataset)</li>
                  <li>Visualize how centroids move during iterations</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
