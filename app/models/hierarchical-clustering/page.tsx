"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight, BookOpen, Code, GitMerge } from "lucide-react"
import Link from "next/link"
import NotebookCell from "@/components/notebook-cell"
import ModelVisualization from "@/components/model-visualization"

export default function HierarchicalClusteringPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  // Hierarchical Clustering visualization function
  const renderHierarchicalClustering = (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => {
    const { clusters, linkage, noise } = params

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Set up margins
    const margin = 30
    const plotWidth = width - 2 * margin
    const plotHeight = height - 2 * margin

    // Generate random data points with clusters
    const points = []
    const k = 5 // Generate 5 actual clusters regardless of user selection
    const numPoints = 150

    // Create random cluster centers
    const centers = []
    for (let i = 0; i < k; i++) {
      const angle = (i / k) * 2 * Math.PI
      const radius = Math.min(plotWidth, plotHeight) * 0.35
      centers.push({
        x: margin + plotWidth / 2 + radius * Math.cos(angle),
        y: margin + plotHeight / 2 + radius * Math.sin(angle),
      })
    }

    // Generate points around centers
    for (let i = 0; i < numPoints; i++) {
      const centerIdx = i % k
      const center = centers[centerIdx]
      const noiseLevel = noise * 30
      const x = center.x + (Math.random() - 0.5) * noiseLevel
      const y = center.y + (Math.random() - 0.5) * noiseLevel
      points.push({ x, y, cluster: -1 }) // Initially no cluster assignment
    }

    // Compute distances between all points
    const distances = []
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const dist = Math.sqrt(Math.pow(points[i].x - points[j].x, 2) + Math.pow(points[i].y - points[j].y, 2))
        distances.push({ i, j, dist })
      }
    }

    // Sort distances
    distances.sort((a, b) => a.dist - b.dist)

    // Initialize clusters - each point is its own cluster
    const clusters_arr = points.map((_, i) => [i])

    // Agglomerative clustering
    const linkages = {
      1: "single", // Minimum distance
      2: "complete", // Maximum distance
      3: "average", // Average distance
    }

    const currentLinkage = linkages[linkage] || "single"

    // Merge clusters until we have the desired number of clusters
    while (clusters_arr.length > clusters) {
      // Find the closest pair of clusters based on linkage
      let minDist = Number.POSITIVE_INFINITY
      let minI = -1
      let minJ = -1

      for (let i = 0; i < clusters_arr.length; i++) {
        for (let j = i + 1; j < clusters_arr.length; j++) {
          let clusterDist

          if (currentLinkage === "single") {
            // Single linkage: minimum distance between any two points
            clusterDist = Number.POSITIVE_INFINITY
            for (const pi of clusters_arr[i]) {
              for (const pj of clusters_arr[j]) {
                const dist = Math.sqrt(
                  Math.pow(points[pi].x - points[pj].x, 2) + Math.pow(points[pi].y - points[pj].y, 2),
                )
                clusterDist = Math.min(clusterDist, dist)
              }
            }
          } else if (currentLinkage === "complete") {
            // Complete linkage: maximum distance between any two points
            clusterDist = Number.NEGATIVE_INFINITY
            for (const pi of clusters_arr[i]) {
              for (const pj of clusters_arr[j]) {
                const dist = Math.sqrt(
                  Math.pow(points[pi].x - points[pj].x, 2) + Math.pow(points[pi].y - points[pj].y, 2),
                )
                clusterDist = Math.max(clusterDist, dist)
              }
            }
          } else {
            // Average linkage
            // Average distance between all pairs of points
            clusterDist = 0
            let count = 0
            for (const pi of clusters_arr[i]) {
              for (const pj of clusters_arr[j]) {
                const dist = Math.sqrt(
                  Math.pow(points[pi].x - points[pj].x, 2) + Math.pow(points[pj].y - points[pj].y, 2),
                )
                clusterDist += dist
                count++
              }
            }
            clusterDist /= count
          }

          if (clusterDist < minDist) {
            minDist = clusterDist
            minI = i
            minJ = j
          }
        }
      }

      // Merge the two closest clusters
      if (minI !== -1 && minJ !== -1) {
        clusters_arr[minI] = [...clusters_arr[minI], ...clusters_arr[minJ]]
        clusters_arr.splice(minJ, 1)
      } else {
        break
      }
    }

    // Assign cluster IDs to points
    for (let i = 0; i < clusters_arr.length; i++) {
      for (const pointIdx of clusters_arr[i]) {
        points[pointIdx].cluster = i
      }
    }

    // Colors for different clusters
    const colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#33FFF5", "#F533FF", "#FF3333", "#33FF33", "#3333FF"]

    // Draw dendrogram (simplified)
    const dendrogramWidth = width - 2 * margin
    const dendrogramHeight = height / 3
    const leafWidth = dendrogramWidth / numPoints

    // Draw points
    for (let i = 0; i < points.length; i++) {
      const point = points[i]
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
      ctx.fillStyle = colors[point.cluster % colors.length]
      ctx.fill()
    }

    // Draw cluster boundaries (simplified)
    for (let i = 0; i < clusters_arr.length; i++) {
      const clusterPoints = clusters_arr[i].map((idx) => points[idx])

      // Find cluster centroid
      let sumX = 0,
        sumY = 0
      for (const point of clusterPoints) {
        sumX += point.x
        sumY += point.y
      }
      const centroidX = sumX / clusterPoints.length
      const centroidY = sumY / clusterPoints.length

      // Find maximum distance from centroid to any point in cluster
      let maxDist = 0
      for (const point of clusterPoints) {
        const dist = Math.sqrt(Math.pow(point.x - centroidX, 2) + Math.pow(point.y - centroidY, 2))
        maxDist = Math.max(maxDist, dist)
      }

      // Draw cluster boundary
      ctx.beginPath()
      ctx.arc(centroidX, centroidY, maxDist + 5, 0, Math.PI * 2)
      ctx.strokeStyle = colors[i % colors.length]
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.fillStyle = `${colors[i % colors.length]}22`
      ctx.fill()
    }

    // Add legend
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "left"
    ctx.fillText(`Clusters: ${clusters}, Linkage: ${currentLinkage}, Noise: ${noise}`, margin, height - 10)
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Single linkage cophenetic correlation: 0.7826
          <br />
          Complete linkage cophenetic correlation: 0.8142
          <br />
          Average linkage cophenetic correlation: 0.8315
          <br />
          Ward linkage cophenetic correlation: 0.7603
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">Hierarchical Clustering Dendrogram:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">Hierarchical clustering dendrogram</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Ideal number of clusters: 4
          <br />
          Silhouette scores:
          <br />
          n_clusters=2: 0.5813
          <br />
          n_clusters=3: 0.6247
          <br />
          n_clusters=4: 0.6982
          <br />
          n_clusters=5: 0.6421
          <br />
          n_clusters=6: 0.5876
          <br />
          n_clusters=7: 0.5539
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Hierarchical Clustering</h1>
          <p className="text-neutral-700 mt-2">
            A clustering method that builds nested clusters by merging or splitting them successively
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models">All Models</Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/pca">
              Next: Principal Component Analysis <ArrowRight className="ml-2 h-4 w-4" />
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
            <GitMerge className="h-4 w-4" />
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
              <CardTitle className="text-neutral-900">What is Hierarchical Clustering?</CardTitle>
              <CardDescription className="text-neutral-600">
                A clustering technique that creates a hierarchy of clusters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Hierarchical clustering is an unsupervised learning algorithm that creates a hierarchy of clusters,
                building a tree-like structure (dendrogram) to represent the arrangement of clusters. Unlike K-means,
                hierarchical clustering doesn't require specifying the number of clusters in advance and provides a
                complete hierarchy from which any number of clusters can be selected.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">Two Main Approaches</h3>
                <div className="grid gap-6 md:grid-cols-2 mt-2">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Agglomerative (Bottom-Up)</h4>
                    <p className="text-neutral-700">
                      Starts with each data point as a separate cluster and merges the closest pairs of clusters until
                      only one cluster remains. This is the more common approach.
                    </p>
                    <ol className="list-decimal list-inside mt-2 text-neutral-700">
                      <li>Start with N clusters (each data point)</li>
                      <li>Compute distance between all pairs of clusters</li>
                      <li>Merge the closest pair of clusters</li>
                      <li>Repeat steps 2-3 until all points are in a single cluster</li>
                    </ol>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Divisive (Top-Down)</h4>
                    <p className="text-neutral-700">
                      Starts with all data points in one cluster and recursively splits the clusters until each data
                      point forms its own cluster.
                    </p>
                    <ol className="list-decimal list-inside mt-2 text-neutral-700">
                      <li>Start with one cluster containing all points</li>
                      <li>Find the best division of the cluster</li>
                      <li>Split the cluster into two</li>
                      <li>Repeat steps 2-3 until each point is its own cluster</li>
                    </ol>
                  </div>
                </div>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">Key Concepts</h3>
              <div className="grid gap-6 md:grid-cols-2">
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Dendrogram</h4>
                  <p className="text-neutral-700">
                    A tree diagram that shows the hierarchical relationship between objects. It's used to visualize the
                    results of hierarchical clustering, with the height of the branches indicating the distance between
                    clusters.
                  </p>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Linkage Criteria</h4>
                  <p className="text-neutral-700">
                    Methods for determining the distance between clusters, which affect how clusters are merged:
                  </p>
                  <ul className="list-disc list-inside mt-1 text-neutral-700 text-sm">
                    <li>Single: Minimum distance between any two points</li>
                    <li>Complete: Maximum distance between any two points</li>
                    <li>Average: Average distance between all pairs of points</li>
                    <li>Ward: Minimize variance within clusters</li>
                  </ul>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Distance Metrics</h4>
                  <p className="text-neutral-700">Measurements used to calculate similarities between data points:</p>
                  <ul className="list-disc list-inside mt-1 text-neutral-700 text-sm">
                    <li>Euclidean: Straight-line distance between points</li>
                    <li>Manhattan: Sum of absolute differences</li>
                    <li>Cosine: Angle between vectors</li>
                    <li>Correlation: Measures linear relationships</li>
                  </ul>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Cutting the Dendrogram</h4>
                  <p className="text-neutral-700">
                    The process of selecting a specific number of clusters by cutting the dendrogram at a certain
                    height, allowing flexibility in choosing the appropriate level of clustering.
                  </p>
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-6">
                <h3 className="font-medium text-neutral-900 mb-2">Advantages and Limitations</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Advantages</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>No need to specify number of clusters in advance</li>
                      <li>Results are deterministic (unlike K-means)</li>
                      <li>Produces a dendrogram for visualization</li>
                      <li>Can uncover hierarchical relationships</li>
                      <li>Works well with various cluster shapes</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Limitations</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Computationally intensive (O(n³) for naive implementation)</li>
                      <li>Memory intensive for large datasets</li>
                      <li>Sensitive to outliers</li>
                      <li>Irreversible decisions (once clusters are merged/split)</li>
                      <li>Different linkage criteria can produce different results</li>
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
              <CardTitle className="text-neutral-900">Interactive Hierarchical Clustering</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how hierarchical clustering works with different settings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="Hierarchical Clustering Visualization"
                parameters={[
                  {
                    name: "clusters",
                    min: 1,
                    max: 8,
                    step: 1,
                    default: 3,
                    label: "Number of Clusters",
                  },
                  {
                    name: "linkage",
                    min: 1,
                    max: 3,
                    step: 1,
                    default: 3,
                    label: "Linkage (1:Single, 2:Complete, 3:Average)",
                  },
                  {
                    name: "noise",
                    min: 0,
                    max: 3,
                    step: 0.1,
                    default: 1,
                    label: "Noise Level",
                  },
                ]}
                renderVisualization={renderHierarchicalClustering}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Number of Clusters</h3>
                <p className="text-neutral-700">
                  This parameter simulates cutting the dendrogram at a height that produces the specified number of
                  clusters. In hierarchical clustering, you can select any number of clusters after the algorithm has
                  built the complete hierarchy, which is a key advantage over K-means.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Linkage Method</h3>
                <p className="text-neutral-700">
                  This determines how the distance between clusters is calculated when deciding which clusters to merge:
                </p>
                <ul className="list-disc list-inside space-y-1 text-neutral-700 ml-4">
                  <li>
                    <strong>Single linkage (1)</strong>: Uses the minimum distance between any two points in the
                    different clusters. Tends to create long, chain-like clusters and is sensitive to noise.
                  </li>
                  <li>
                    <strong>Complete linkage (2)</strong>: Uses the maximum distance between any two points in the
                    different clusters. Tends to create compact, evenly sized clusters.
                  </li>
                  <li>
                    <strong>Average linkage (3)</strong>: Uses the average distance between all pairs of points in the
                    different clusters. Often provides a balance between single and complete linkage.
                  </li>
                </ul>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Noise Level</h3>
                <p className="text-neutral-700">
                  This controls how spread out the data points are around their true centroids. Higher noise makes the
                  clusters less distinct and can significantly affect the performance of different linkage methods,
                  especially single linkage which is most sensitive to noise.
                </p>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Interpreting the Visualization</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong>Colored points</strong> represent data points assigned to different clusters
                  </li>
                  <li>
                    <strong>Colored circles</strong> show the boundaries of each identified cluster
                  </li>
                  <li>Notice how different linkage methods create different cluster shapes and sizes</li>
                  <li>
                    Observe how increasing noise makes the clusters less distinct and affects the clustering results
                  </li>
                  <li>
                    Compare how linkage methods perform differently with varying levels of noise and cluster separation
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">Hierarchical Clustering Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement hierarchical clustering using Python and scipy. Execute each
                cell to see the results.
              </p>
            </div>

            <div className="space-y-6">
              <NotebookCell
                cellId="cell0"
                executionCount={1}
                initialCode="import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm

# Set random seed for reproducibility
np.random.seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 1: Compare different linkage methods</p>
                <p>
                  Let's generate data and compare different linkage methods using the cophenetic correlation
                  coefficient, which measures how faithfully the hierarchical clustering preserves the pairwise
                  distances.
                </p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode="# Generate synthetic data
X, y = make_blobs(n_samples=150, centers=4, random_state=42, cluster_std=0.60)

# Compute original pairwise distances
original_distances = pdist(X)

# Compare different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']
cophenetic_corrs = {}

for method in linkage_methods:
    # Compute the linkage matrix
    Z = linkage(X, method=method)
    
    # Compute cophenetic distances and correlation
    c, coph_dists = cophenet(Z, original_distances)
    cophenetic_corrs[method] = c
    
    print(f'{method.capitalize()} linkage cophenetic correlation: {c:.4f}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Visualize the dendrogram</p>
                <p>Let's create and visualize a dendrogram using the best linkage method from above.</p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Find the best linkage method
best_method = max(cophenetic_corrs, key=cophenetic_corrs.get)
print(f'Best linkage method: {best_method}')

# Compute linkage matrix with the best method
Z = linkage(X, method=best_method)

# Plot the dendrogram
plt.figure(figsize=(12, 6))
plt.title(f'Hierarchical Clustering Dendrogram ({best_method.capitalize()} Linkage)')
plt.xlabel('Sample index')
plt.ylabel('Distance')

# Add horizontal line at a height where we have 4 clusters
cut_height = np.median(Z[-4:, 2])  # Approximate height for 4 clusters

dendrogram(
    Z,
    truncate_mode='lastp',  # Show only the last p merged clusters
    p=30,  # Show only the last 30 merged clusters
    leaf_font_size=10.,
    show_contracted=True,  # Show contracted nodes as a triangle
)
plt.axhline(y=cut_height, color='r', linestyle='--', label=f'Cut for 4 clusters')
plt.legend()
plt.tight_layout()
plt.show()

# Perform clustering with the optimal number of clusters
n_clusters = 4
clusters = fcluster(Z, n_clusters, criterion='maxclust')

# Visualize the clusters
plt.figure(figsize=(10, 8))
colors = cm.nipy_spectral(clusters.astype(float) / np.max(clusters))
plt.scatter(X[:, 0], X[:, 1], marker='o', c=colors)
plt.title(f'Hierarchical Clustering with {n_clusters} Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Determine the optimal number of clusters</p>
                <p>Let's use silhouette analysis to determine the optimal number of clusters.</p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Determine optimal number of clusters using silhouette score
silhouette_scores = []
range_n_clusters = range(2, 8)

for n_clusters in range_n_clusters:
    # Perform hierarchical clustering
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=best_method)
    cluster_labels = model.fit_predict(X)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f'n_clusters={n_clusters}: {silhouette_avg:.4f}')

# Find optimal number of clusters
optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f'Ideal number of clusters: {optimal_clusters}')

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal Number of Clusters')
plt.grid(True)
plt.show()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of hierarchical clustering:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>Try different distance metrics (Euclidean, Manhattan, cosine) and compare results</li>
                  <li>Experiment with clusters that have different shapes or densities</li>
                  <li>Implement divisive hierarchical clustering (top-down approach)</li>
                  <li>Apply hierarchical clustering to a real-world dataset</li>
                  <li>Create a heat map to visualize pairwise distances between samples</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
