"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import ModelVisualization from "@/components/model-visualization"
import { ArrowLeft, ArrowRight, BookOpen, Code, BarChart } from "lucide-react"
import Link from "next/link"
import NotebookCell from "@/components/notebook-cell"

export default function SVMPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  // SVM visualization function
  const renderSVM = (ctx: CanvasRenderingContext2D, params: Record<string, number>, width: number, height: number) => {
    const { c, gamma, noise } = params

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw coordinate axes
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, height / 2)
    ctx.lineTo(width, height / 2)
    ctx.moveTo(width / 2, 0)
    ctx.lineTo(width / 2, height)
    ctx.stroke()

    // Generate two clusters of points
    const points = []
    const numPoints = 30
    const centerOffset = 80 * (1 - gamma * 0.5) // Adjust separation based on gamma
    const noiseLevel = noise * 30

    // Class 1 points (upper right and lower left)
    for (let i = 0; i < numPoints / 2; i++) {
      const x = width / 2 + centerOffset + (Math.random() - 0.5) * noiseLevel
      const y = height / 2 - centerOffset + (Math.random() - 0.5) * noiseLevel
      points.push({ x, y, class: 1 })
    }
    for (let i = 0; i < numPoints / 2; i++) {
      const x = width / 2 - centerOffset + (Math.random() - 0.5) * noiseLevel
      const y = height / 2 + centerOffset + (Math.random() - 0.5) * noiseLevel
      points.push({ x, y, class: 1 })
    }

    // Class 2 points (upper left and lower right)
    for (let i = 0; i < numPoints / 2; i++) {
      const x = width / 2 - centerOffset + (Math.random() - 0.5) * noiseLevel
      const y = height / 2 - centerOffset + (Math.random() - 0.5) * noiseLevel
      points.push({ x, y, class: 2 })
    }
    for (let i = 0; i < numPoints / 2; i++) {
      const x = width / 2 + centerOffset + (Math.random() - 0.5) * noiseLevel
      const y = height / 2 + centerOffset + (Math.random() - 0.5) * noiseLevel
      points.push({ x, y, class: 2 })
    }

    // Draw decision boundary (simplified representation)
    const margin = 10 + (1 / c) * 40 // Margin width based on C parameter

    // Draw diagonal decision boundary
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, height)
    ctx.lineTo(width, 0)
    ctx.stroke()

    // Draw margin boundaries
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1
    ctx.setLineDash([5, 3])

    // Upper margin line
    ctx.beginPath()
    ctx.moveTo(0, height - margin)
    ctx.lineTo(width - margin, 0)
    ctx.stroke()

    // Lower margin line
    ctx.beginPath()
    ctx.moveTo(margin, height)
    ctx.lineTo(width, margin)
    ctx.stroke()

    ctx.setLineDash([])

    // Draw points
    points.forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 6, 0, Math.PI * 2)
      if (point.class === 1) {
        ctx.fillStyle = "#000"
      } else {
        ctx.fillStyle = "#fff"
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1.5
        ctx.stroke()
      }
      ctx.fill()
    })

    // Draw support vectors (simplified)
    const supportVectors = points.filter((p) => Math.abs(p.x + p.y - (width + height) / 2) < margin * 1.5)

    supportVectors.forEach((sv) => {
      ctx.beginPath()
      ctx.arc(sv.x, sv.y, 10, 0, Math.PI * 2)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 2
      ctx.stroke()
    })
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Linear SVM accuracy: 0.8267
          <br />
          RBF kernel SVM accuracy: 0.9467
          <br />
          Polynomial kernel SVM accuracy: 0.9067
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">Decision boundaries visualization:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">SVM decision boundaries plot</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Best parameters: {"C': 10, 'gamma': 0.1, 'kernel': 'rbf"}
          <br />
          Best cross-validation score: 0.9533
          <br />
          Test accuracy with best parameters: 0.9600
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Support Vector Machines</h1>
          <p className="text-neutral-700 mt-2">
            Understanding SVMs and their implementation for classification and regression
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models/decision-trees">
              <ArrowLeft className="mr-2 h-4 w-4" /> Previous: Decision Trees
            </Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/cnn">
              Next: CNNs <ArrowRight className="ml-2 h-4 w-4" />
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
            <BarChart className="h-4 w-4" />
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
              <CardTitle className="text-neutral-900">What are Support Vector Machines?</CardTitle>
              <CardDescription className="text-neutral-600">
                A powerful supervised learning algorithm for classification and regression
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Support Vector Machines (SVMs) are a set of supervised learning methods used for classification,
                regression, and outlier detection. The objective of an SVM is to find a hyperplane in an N-dimensional
                space that distinctly classifies the data points.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">Key Concepts in SVMs</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Hyperplane</strong>: A decision boundary that separates
                    different classes
                  </li>
                  <li>
                    <strong className="text-neutral-900">Support Vectors</strong>: Data points closest to the hyperplane
                    that influence its position and orientation
                  </li>
                  <li>
                    <strong className="text-neutral-900">Margin</strong>: The distance between the hyperplane and the
                    closest data points (support vectors)
                  </li>
                  <li>
                    <strong className="text-neutral-900">Kernel Trick</strong>: A method to transform the input space to
                    a higher-dimensional space where a linear separator might exist
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">How SVMs Work</h3>
              <p className="text-neutral-700 mb-4">
                SVMs work by finding the hyperplane that maximizes the margin between classes. The algorithm follows
                these steps:
              </p>
              <ol className="list-decimal list-inside space-y-4 text-neutral-700">
                <li>
                  <strong className="text-neutral-900">Map data to a high-dimensional feature space</strong> (implicitly
                  using kernels)
                </li>
                <li>
                  <strong className="text-neutral-900">Find the optimal hyperplane</strong> that maximizes the margin
                  between classes
                </li>
                <li>
                  <strong className="text-neutral-900">Identify support vectors</strong> (points that lie closest to the
                  hyperplane)
                </li>
                <li>
                  <strong className="text-neutral-900">Use the support vectors</strong> to define the decision boundary
                </li>
              </ol>

              <div className="bg-neutral-100 p-4 rounded-lg mt-6">
                <h3 className="font-medium text-neutral-900 mb-2">SVM Kernels</h3>
                <p className="mb-2 text-neutral-700">
                  Kernels allow SVMs to handle non-linearly separable data by transforming it into a higher-dimensional
                  space:
                </p>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Linear Kernel</strong>: K(x, y) = x · y (dot product)
                  </li>
                  <li>
                    <strong className="text-neutral-900">Polynomial Kernel</strong>: K(x, y) = (γx · y + r)^d
                  </li>
                  <li>
                    <strong className="text-neutral-900">Radial Basis Function (RBF) Kernel</strong>: K(x, y) =
                    exp(-γ||x - y||²)
                  </li>
                  <li>
                    <strong className="text-neutral-900">Sigmoid Kernel</strong>: K(x, y) = tanh(γx · y + r)
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">SVM Parameters and Tuning</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-neutral-700">The performance of an SVM model depends on several key parameters:</p>

              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-neutral-900">C Parameter (Regularization)</h3>
                  <p className="text-neutral-700">
                    Controls the trade-off between having a smooth decision boundary and classifying training points
                    correctly. A small C makes the decision surface smooth but may lead to training errors. A large C
                    aims to classify all training examples correctly but may lead to overfitting.
                  </p>
                </div>

                <div>
                  <h3 className="font-medium text-neutral-900">Gamma Parameter</h3>
                  <p className="text-neutral-700">
                    Defines how far the influence of a single training example reaches. Low gamma means a point has a
                    far reach, while high gamma means the reach is limited to close points. High gamma can lead to
                    overfitting.
                  </p>
                </div>

                <div>
                  <h3 className="font-medium text-neutral-900">Kernel Selection</h3>
                  <p className="text-neutral-700">
                    The choice of kernel depends on the data. Linear kernels work well for linearly separable data,
                    while RBF kernels are versatile for non-linear data. Polynomial kernels can capture more complex
                    relationships.
                  </p>
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Advantages and Limitations</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Advantages</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Effective in high-dimensional spaces</li>
                      <li>Memory efficient (uses subset of training points)</li>
                      <li>Versatile through different kernel functions</li>
                      <li>Robust against overfitting</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Limitations</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Computationally intensive for large datasets</li>
                      <li>Requires careful parameter tuning</li>
                      <li>Difficult to interpret</li>
                      <li>Not directly suitable for multi-class problems</li>
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
              <CardTitle className="text-neutral-900">Interactive SVM Model</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how they affect the SVM decision boundary
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="SVM Visualization"
                parameters={[
                  {
                    name: "c",
                    min: 0.1,
                    max: 10,
                    step: 0.1,
                    default: 1,
                    label: "C (Regularization)",
                  },
                  {
                    name: "gamma",
                    min: 0.1,
                    max: 2,
                    step: 0.1,
                    default: 1,
                    label: "Gamma",
                  },
                  {
                    name: "noise",
                    min: 0,
                    max: 1,
                    step: 0.1,
                    default: 0.3,
                    label: "Data Noise",
                  },
                ]}
                renderVisualization={renderSVM}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">C (Regularization)</h3>
                <p className="text-neutral-700">
                  The C parameter controls the trade-off between achieving a low training error and a low testing error.
                  A lower C value creates a wider margin that may include some misclassifications, while a higher C
                  value creates a narrower margin that tries to classify all training examples correctly.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Gamma</h3>
                <p className="text-neutral-700">
                  Gamma defines how far the influence of a single training example reaches. Low values mean 'far' reach,
                  while high values mean 'close' reach. With a high gamma, the model focuses more on points close to the
                  decision boundary, potentially creating a more complex boundary.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Data Noise</h3>
                <p className="text-neutral-700">
                  This parameter simulates noise in the data. Higher values create more overlap between classes, making
                  the classification task more challenging and demonstrating how SVMs handle noisy data.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">SVM Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement Support Vector Machines using Python and scikit-learn.
                Execute each cell to see the results.
              </p>
            </div>

            <div className="space-y-6">
              <NotebookCell
                cellId="cell0"
                executionCount={1}
                initialCode="import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 1: Prepare data and train different SVM models</p>
                <p>Let's create a synthetic dataset and compare different SVM kernels to see how they perform.</p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode="# Generate a synthetic dataset
X, y = make_classification(
    n_samples=300, 
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train different SVM models
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_rbf = SVC(kernel='rbf', gamma=0.1, C=1.0, random_state=42)
svm_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42)

# Fit models
svm_linear.fit(X_train_scaled, y_train)
svm_rbf.fit(X_train_scaled, y_train)
svm_poly.fit(X_train_scaled, y_train)

# Evaluate models
linear_acc = accuracy_score(y_test, svm_linear.predict(X_test_scaled))
rbf_acc = accuracy_score(y_test, svm_rbf.predict(X_test_scaled))
poly_acc = accuracy_score(y_test, svm_poly.predict(X_test_scaled))

print(f'Linear SVM accuracy: {linear_acc:.4f}')
print(f'RBF kernel SVM accuracy: {rbf_acc:.4f}')
print(f'Polynomial kernel SVM accuracy: {poly_acc:.4f}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Visualize decision boundaries</p>
                <p>Let's visualize how different kernels create different decision boundaries.</p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Function to plot decision boundaries
def plot_decision_boundaries(X, y, models, titles, scaler):
    # Set up the figure
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    
    # Determine plot boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Plot each model
    for i, (model, title) in enumerate(zip(models, titles)):
        # Scale the mesh grid points
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = scaler.transform(mesh_points)
        
        # Predict using the model
        Z = model.predict(mesh_points_scaled)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and points
        axes[i].contourf(xx, yy, Z, alpha=0.3)
        axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        axes[i].set_title(title)
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()

# Plot decision boundaries
plot_decision_boundaries(
    X, y, 
    [svm_linear, svm_rbf, svm_poly],
    ['Linear Kernel', 'RBF Kernel', 'Polynomial Kernel'],
    scaler
)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Parameter tuning with GridSearchCV</p>
                <p>Let's use grid search to find the optimal parameters for our SVM model.</p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'linear']
}

# Create grid search
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)

# Fit grid search
grid_search.fit(X_train_scaled, y_train)

# Get best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Evaluate on test set
best_model = grid_search.best_estimator_
test_accuracy = accuracy_score(y_test, best_model.predict(X_test_scaled))

print(f'Best parameters: {best_params}')
print(f'Best cross-validation score: {best_score:.4f}')
print(f'Test accuracy with best parameters: {test_accuracy:.4f}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of SVMs:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>Try different kernels and parameters</li>
                  <li>Experiment with class weights for imbalanced datasets</li>
                  <li>Implement SVM for regression (SVR) instead of classification</li>
                  <li>Use SVM with a custom kernel function</li>
                  <li>Compare SVM performance with other classification algorithms</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
