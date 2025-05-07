"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight, BookOpen, Code, Component } from "lucide-react"
import Link from "next/link"
import NotebookCell from "@/components/notebook-cell"
import ModelVisualization from "@/components/model-visualization"

export default function PCAPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  // PCA visualization function
  const renderPCA = (ctx: CanvasRenderingContext2D, params: Record<string, number>, width: number, height: number) => {
    const { components, variance, rotation } = params

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Set up margins
    const margin = 50
    const plotWidth = width - 2 * margin
    const plotHeight = height - 2 * margin

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "16px Arial"
    ctx.textAlign = "center"
    ctx.fillText(`Principal Component Analysis: ${components} Component${components > 1 ? "s" : ""}`, width / 2, 25)

    // Draw coordinate axes
    ctx.strokeStyle = "#aaa"
    ctx.lineWidth = 1.5

    // Draw x-axis
    ctx.beginPath()
    ctx.moveTo(margin, height - margin)
    ctx.lineTo(width - margin, height - margin)
    ctx.stroke()

    // Draw y-axis
    ctx.beginPath()
    ctx.moveTo(margin, margin)
    ctx.lineTo(margin, height - margin)
    ctx.stroke()

    // Add axis labels
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "center"
    ctx.fillText("Principal Component 1", width / 2, height - 10)

    ctx.textAlign = "center"
    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText("Principal Component 2", 0, 0)
    ctx.restore()

    // Draw grid lines
    ctx.strokeStyle = "#e0e0e0"
    ctx.lineWidth = 0.5

    // Vertical grid lines
    for (let x = margin + 50; x < width - margin; x += 50) {
      ctx.beginPath()
      ctx.moveTo(x, margin)
      ctx.lineTo(x, height - margin)
      ctx.stroke()
    }

    // Horizontal grid lines
    for (let y = margin + 50; y < height - margin; y += 50) {
      ctx.beginPath()
      ctx.moveTo(margin, y)
      ctx.lineTo(width - margin, y)
      ctx.stroke()
    }

    // Origin of vectors (center of plot)
    const originX = margin + plotWidth / 2
    const originY = height - margin - plotHeight / 2

    // Generate high-dimensional data
    const numPoints = 200
    const numDimensions = 10
    const points = []

    // Generate points along a plane in high-dimensional space
    // with some variance along the principal components
    const rotationRad = (rotation * Math.PI) / 180

    // Create an elliptical distribution to better visualize variance
    for (let i = 0; i < numPoints; i++) {
      // Generate angle for elliptical distribution
      const angle = Math.random() * 2 * Math.PI

      // Generate distance from center with some randomness
      const distance = Math.random() * 0.8 + 0.2

      // Calculate position on ellipse
      const pc1 = Math.cos(angle) * distance * plotWidth * 0.4 // First principal component
      const pc2 = Math.sin(angle) * distance * plotHeight * 0.4 * variance // Second principal component (scaled by variance)

      // Add some noise in other dimensions
      const otherDims = Array(numDimensions - 2)
        .fill(0)
        .map(() => (Math.random() - 0.5) * 10 * (1 - variance))

      // Apply rotation to simulate changing principal components
      const rotatedPC1 = pc1 * Math.cos(rotationRad) - pc2 * Math.sin(rotationRad)
      const rotatedPC2 = pc1 * Math.sin(rotationRad) + pc2 * Math.cos(rotationRad)

      // Add point
      points.push({
        original: [rotatedPC1, rotatedPC2, ...otherDims],
        projected: [], // Will store PCA-projected coordinates
      })
    }

    // Simulate PCA by "projecting" onto the principal components
    // Since we already generated data with the principal components,
    // we'll just use those directly but limit by the number of components
    for (const point of points) {
      if (components >= 1) point.projected.push(point.original[0])
      if (components >= 2) point.projected.push(point.original[1])
    }

    // Calculate scaling factors to fit the points in the plot area
    let minX = Number.POSITIVE_INFINITY,
      maxX = Number.NEGATIVE_INFINITY
    let minY = Number.POSITIVE_INFINITY,
      maxY = Number.NEGATIVE_INFINITY

    for (const point of points) {
      if (point.projected.length > 0) {
        minX = Math.min(minX, point.projected[0])
        maxX = Math.max(maxX, point.projected[0])
      }
      if (point.projected.length > 1) {
        minY = Math.min(minY, point.projected[1])
        maxY = Math.max(maxY, point.projected[1])
      }
    }

    const xScale = plotWidth / (maxX - minX || 1)
    const yScale = plotHeight / (maxY - minY || 1)

    // Draw the original data distribution (faded)
    if (components < 2) {
      ctx.fillStyle = "rgba(200, 200, 200, 0.3)"
      for (const point of points) {
        const x = margin + (point.original[0] - minX) * xScale
        const y = height - margin - (point.original[1] - minY) * yScale

        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    // Draw the projected points
    ctx.fillStyle = components >= 2 ? "#1a5fb4" : "#e67e22"
    for (const point of points) {
      const x = margin + (point.projected[0] - minX) * xScale

      // For 1D projection, all points are on the x-axis
      const y =
        components >= 2 ? height - margin - (point.projected[1] - minY) * yScale : height - margin - plotHeight / 2

      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fill()
    }

    // Draw principal component vectors
    // First principal component vector
    const pc1Length = 100
    ctx.strokeStyle = "#c01c28"
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(originX, originY)
    ctx.lineTo(originX + pc1Length * Math.cos(rotationRad), originY - pc1Length * Math.sin(rotationRad))
    ctx.stroke()

    // Add arrowhead
    const arrowSize = 10
    const arrowAngle = Math.atan2(-pc1Length * Math.sin(rotationRad), pc1Length * Math.cos(rotationRad))
    ctx.beginPath()
    ctx.moveTo(originX + pc1Length * Math.cos(rotationRad), originY - pc1Length * Math.sin(rotationRad))
    ctx.lineTo(
      originX + pc1Length * Math.cos(rotationRad) - arrowSize * Math.cos(arrowAngle - Math.PI / 6),
      originY - pc1Length * Math.sin(rotationRad) + arrowSize * Math.sin(arrowAngle - Math.PI / 6),
    )
    ctx.lineTo(
      originX + pc1Length * Math.cos(rotationRad) - arrowSize * Math.cos(arrowAngle + Math.PI / 6),
      originY - pc1Length * Math.sin(rotationRad) + arrowSize * Math.sin(arrowAngle + Math.PI / 6),
    )
    ctx.closePath()
    ctx.fillStyle = "#c01c28"
    ctx.fill()

    // Label PC1
    ctx.fillStyle = "#c01c28"
    ctx.font = "14px Arial"
    ctx.textAlign = "center"
    ctx.fillText(
      "PC1",
      originX + (pc1Length + 20) * Math.cos(rotationRad),
      originY - (pc1Length + 20) * Math.sin(rotationRad),
    )

    // Second principal component vector (if using 2 components)
    if (components >= 2) {
      const pc2Length = 100 * variance
      ctx.strokeStyle = "#2980b9"
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(originX, originY)
      ctx.lineTo(
        originX + pc2Length * Math.cos(rotationRad + Math.PI / 2),
        originY - pc2Length * Math.sin(rotationRad + Math.PI / 2),
      )
      ctx.stroke()

      // Add arrowhead for PC2
      const arrowAngle2 = Math.atan2(
        -pc2Length * Math.sin(rotationRad + Math.PI / 2),
        pc2Length * Math.cos(rotationRad + Math.PI / 2),
      )
      ctx.beginPath()
      ctx.moveTo(
        originX + pc2Length * Math.cos(rotationRad + Math.PI / 2),
        originY - pc2Length * Math.sin(rotationRad + Math.PI / 2),
      )
      ctx.lineTo(
        originX + pc2Length * Math.cos(rotationRad + Math.PI / 2) - arrowSize * Math.cos(arrowAngle2 - Math.PI / 6),
        originY - pc2Length * Math.sin(rotationRad + Math.PI / 2) + arrowSize * Math.sin(arrowAngle2 - Math.PI / 6),
      )
      ctx.lineTo(
        originX + pc2Length * Math.cos(rotationRad + Math.PI / 2) - arrowSize * Math.cos(arrowAngle2 + Math.PI / 6),
        originY - pc2Length * Math.sin(rotationRad + Math.PI / 2) + arrowSize * Math.sin(arrowAngle2 + Math.PI / 6),
      )
      ctx.closePath()
      ctx.fillStyle = "#2980b9"
      ctx.fill()

      // Label PC2
      ctx.fillStyle = "#2980b9"
      ctx.font = "14px Arial"
      ctx.textAlign = "center"
      ctx.fillText(
        "PC2",
        originX + (pc2Length + 20) * Math.cos(rotationRad + Math.PI / 2),
        originY - (pc2Length + 20) * Math.sin(rotationRad + Math.PI / 2),
      )
    }

    // Draw variance explained indicators
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "left"

    const varPC1 = 0.7 + 0.2 * (1 - variance)
    const varPC2 = variance * 0.7

    ctx.fillText(`PC1: ${(varPC1 * 100).toFixed(1)}% variance`, margin, 50)
    if (components >= 2) {
      ctx.fillText(`PC2: ${(varPC2 * 100).toFixed(1)}% variance`, margin, 70)
      ctx.fillText(`Total: ${((varPC1 + varPC2) * 100).toFixed(1)}% variance`, margin, 90)
    }

    // Draw a box around the legend
    ctx.strokeStyle = "#ddd"
    ctx.lineWidth = 1
    ctx.strokeRect(margin - 5, 35, 200, components >= 2 ? 65 : 25)

    // Draw projection lines for a few selected points if in 1D mode
    if (components === 1) {
      ctx.strokeStyle = "rgba(230, 126, 34, 0.3)"
      ctx.lineWidth = 1

      // Only draw projection lines for some points to avoid clutter
      for (let i = 0; i < points.length; i += 10) {
        const point = points[i]
        const projX = margin + (point.projected[0] - minX) * xScale
        const projY = height - margin - plotHeight / 2
        const origX = margin + (point.original[0] - minX) * xScale
        const origY = height - margin - (point.original[1] - minY) * yScale

        ctx.beginPath()
        ctx.moveTo(origX, origY)
        ctx.lineTo(projX, projY)
        ctx.stroke()
      }

      // Add a label explaining the projection
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
      ctx.font = "12px Arial"
      ctx.textAlign = "center"
      ctx.fillText("Projection onto PC1", width / 2, height - margin - plotHeight / 2 - 15)
    }
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Explained variance ratio:
          <br />
          [0.725, 0.227, 0.041, 0.007]
          <br />
          Cumulative explained variance:
          <br />
          [0.725, 0.952, 0.993, 1.000]
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">PCA Visualization:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">PCA visualization plot</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Original image shape: (8, 8)
          <br />
          Compressed image shape with 2 components: (8, 2)
          <br />
          Reconstruction error: 9.81
          <br />
          Data compression ratio: 75.0%
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Principal Component Analysis</h1>
          <p className="text-neutral-700 mt-2">
            A dimensionality reduction technique that transforms high-dimensional data while preserving variance
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models">All Models</Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/mlp">
              Next: Multilayer Perceptron <ArrowRight className="ml-2 h-4 w-4" />
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
            <Component className="h-4 w-4" />
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
              <CardTitle className="text-neutral-900">What is Principal Component Analysis?</CardTitle>
              <CardDescription className="text-neutral-600">
                A dimensionality reduction technique that finds the directions of maximum variance
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique that transforms
                a dataset with potentially correlated variables into a set of linearly uncorrelated variables called
                principal components. These principal components are ordered so that the first component explains the
                largest possible variance in the data, and each succeeding component explains the highest variance
                possible while being orthogonal to the preceding components.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">The PCA Algorithm</h3>
                <ol className="list-decimal list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Standardize the data</strong>: Center the data around the mean
                    and scale it to have unit variance
                  </li>
                  <li>
                    <strong className="text-neutral-900">Compute the covariance matrix</strong>: Calculate how each
                    variable relates to each other variable
                  </li>
                  <li>
                    <strong className="text-neutral-900">Calculate eigenvectors and eigenvalues</strong>: Find the
                    principal directions (eigenvectors) and their importance (eigenvalues)
                  </li>
                  <li>
                    <strong className="text-neutral-900">Sort eigenvectors</strong>: Order them by decreasing
                    eigenvalues to get principal components in order of importance
                  </li>
                  <li>
                    <strong className="text-neutral-900">Select top k eigenvectors</strong>: Choose how many dimensions
                    to keep based on explained variance
                  </li>
                  <li>
                    <strong className="text-neutral-900">Project the data</strong>: Transform the original data onto the
                    new subspace defined by the principal components
                  </li>
                </ol>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">Key Concepts</h3>
              <div className="grid gap-6 md:grid-cols-2">
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Principal Components</h4>
                  <p className="text-neutral-700">
                    The orthogonal axes that capture the directions of maximum variance in the data. Each principal
                    component is a linear combination of the original features.
                  </p>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Eigenvalues & Eigenvectors</h4>
                  <p className="text-neutral-700">
                    Eigenvectors of the covariance matrix define the principal components, while eigenvalues represent
                    the amount of variance explained by each principal component.
                  </p>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Explained Variance Ratio</h4>
                  <p className="text-neutral-700">
                    The proportion of the dataset's variance explained by each principal component, which helps
                    determine how many components to retain.
                  </p>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h4 className="text-lg font-medium text-neutral-800 mb-2">Dimensionality Reduction</h4>
                  <p className="text-neutral-700">
                    By keeping only the top k principal components, we can represent high-dimensional data in a
                    lower-dimensional space while preserving most of the information.
                  </p>
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-6">
                <h3 className="font-medium text-neutral-900 mb-2">Advantages and Limitations</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Advantages</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Reduces dimensionality without losing much information</li>
                      <li>Removes correlated features and reduces redundancy</li>
                      <li>Helps visualize high-dimensional data</li>
                      <li>Mitigates the curse of dimensionality</li>
                      <li>Can improve performance of machine learning models</li>
                      <li>Useful for noise reduction and data compression</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Limitations</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Assumes linear relationships between variables</li>
                      <li>Sensitive to the scale of the features</li>
                      <li>May not work well for non-linear data</li>
                      <li>Principal components may be hard to interpret</li>
                      <li>May lose information if too few components are retained</li>
                      <li>Not suitable when variance doesn't represent information</li>
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
              <CardTitle className="text-neutral-900">Interactive PCA Visualization</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how PCA transforms high-dimensional data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="PCA Visualization"
                parameters={[
                  {
                    name: "components",
                    min: 1,
                    max: 2,
                    step: 1,
                    default: 2,
                    label: "Number of Components",
                  },
                  {
                    name: "variance",
                    min: 0.1,
                    max: 1,
                    step: 0.1,
                    default: 0.5,
                    label: "Variance Ratio (PC2/PC1)",
                  },
                  {
                    name: "rotation",
                    min: 0,
                    max: 360,
                    step: 10,
                    default: 45,
                    label: "Rotation Angle (°)",
                  },
                ]}
                renderVisualization={renderPCA}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Number of Components</h3>
                <p className="text-neutral-700">
                  This parameter controls how many principal components are used for the reduced representation. With 1
                  component, the data is projected onto a single axis (the direction of maximum variance). With 2
                  components, the data is projected onto a plane defined by the first two principal components, allowing
                  for a richer representation.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Variance Ratio</h3>
                <p className="text-neutral-700">
                  This parameter simulates the ratio of variance explained by the second principal component relative to
                  the first. In real datasets, each successive principal component explains less variance than the
                  previous one. A higher value means the second component captures more information, making the data
                  more spread out along that axis.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Rotation Angle</h3>
                <p className="text-neutral-700">
                  This parameter controls the orientation of the principal components in the original feature space. As
                  you change the rotation, you can see how PCA finds the directions of maximum variance regardless of
                  the original coordinate system. This demonstrates PCA's ability to discover the intrinsic structure of
                  the data.
                </p>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Interpreting the Visualization</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong>Blue points</strong> represent the data after projection onto the principal components
                  </li>
                  <li>
                    <strong>Red lines</strong> show the directions of the principal components
                  </li>
                  <li>
                    The <strong>length of each line</strong> indicates the amount of variance explained by that
                    component
                  </li>
                  <li>
                    With 1 component, all points are projected onto a single axis, losing information in other
                    directions
                  </li>
                  <li>
                    With 2 components, the data is projected onto a plane, preserving more of the original structure
                  </li>
                  <li>
                    Notice how changing the rotation angle changes the orientation of the principal components but
                    preserves the overall shape of the projected data
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">PCA Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement Principal Component Analysis using Python and scikit-learn.
                Execute each cell to see the results.
              </p>
            </div>

            <div className="space-y-6">
              <NotebookCell
                cellId="cell0"
                executionCount={1}
                initialCode="import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 1: Load and preprocess data</p>
                <p>Let's load the Iris dataset, which has 4 features, and apply PCA to reduce it to 2 dimensions.</p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode="# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Look at explained variance ratio
print('Explained variance ratio:')
print(pca.explained_variance_ratio_)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print('Cumulative explained variance:')
print(cumulative_variance)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Visualize the PCA results</p>
                <p>Let's visualize the data projected onto the first two principal components.</p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Create PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for easier plotting
pca_df = pd.DataFrame(
    data=X_pca, 
    columns=['PC1', 'PC2']
)
pca_df['species'] = pd.Categorical.from_codes(y, target_names)

# Plot the results
plt.figure(figsize=(10, 8))
colors = ['#FF9999', '#66B2FF', '#99FF99']
markers = ['o', 's', '^']

for i, species in enumerate(target_names):
    plt.scatter(
        pca_df.loc[pca_df['species'] == species, 'PC1'],
        pca_df.loc[pca_df['species'] == species, 'PC2'],
        c=colors[i],
        marker=markers[i],
        s=70,
        label=species,
        alpha=0.8
    )

# Add feature vectors
coeff = pca.components_
n_features = len(feature_names)

# Scale the feature vectors for visualization
scale = 5
for i in range(n_features):
    plt.arrow(
        0, 0,
        coeff[0, i] * scale,
        coeff[1, i] * scale,
        color='k',
        alpha=0.5
    )
    plt.text(
        coeff[0, i] * scale * 1.2,
        coeff[1, i] * scale * 1.2,
        feature_names[i],
        color='k',
        ha='center',
        va='center'
    )

# Customize the plot
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Iris Dataset - PCA')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Image compression with PCA</p>
                <p>
                  Let's demonstrate how PCA can be used for dimensionality reduction and compression using the digits
                  dataset.
                </p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Load the digits dataset (8x8 images)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# Take a single digit image for demonstration
digit_image = digits.images[0]
flattened_digit = digit_image.flatten()

# Apply PCA for compression
n_components = 2  # Using only 2 components for demonstration
pca = PCA(n_components=n_components)
pca.fit(X_digits)

# Compress and then reconstruct the image
compressed = pca.transform([flattened_digit])
reconstructed = pca.inverse_transform(compressed)

# Calculate reconstruction error
error = np.mean((flattened_digit - reconstructed[0]) ** 2)
compression_ratio = (1 - n_components / len(flattened_digit)) * 100

print(f'Original image shape: {digit_image.shape}')
print(f'Compressed image shape with {n_components} components: ({digit_image.shape[0]}, {n_components})')
print(f'Reconstruction error: {error:.2f}')
print(f'Data compression ratio: {compression_ratio:.1f}%')

# Plot original and reconstructed images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(digit_image, cmap='binary')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed[0].reshape(8, 8), cmap='binary')
plt.title(f'Reconstructed Image\n({n_components} components)')
plt.axis('off')

plt.tight_layout()
plt.show()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of PCA:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>Try different numbers of components and observe the effect on variance explained</li>
                  <li>Apply PCA to a different dataset and visualize the results</li>
                  <li>Experiment with image compression using different numbers of components</li>
                  <li>Compare PCA with other dimensionality reduction techniques like t-SNE or UMAP</li>
                  <li>Use PCA as a preprocessing step for a classification model and evaluate performance</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
