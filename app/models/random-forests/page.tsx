"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import ModelVisualization from "@/components/model-visualization"
import { ArrowRight, BookOpen, Code, BarChart } from "lucide-react"
import Link from "next/link"
import NotebookCell from "@/components/notebook-cell"

export default function RandomForestsPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  // Random Forest visualization function
  const renderRandomForest = (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => {
    const { trees, maxDepth, sampleSize } = params

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Set up coordinate system
    const margin = 30
    const plotWidth = width - 2 * margin
    const plotHeight = height - 2 * margin

    // Draw grid for decision boundaries
    const gridSize = 10
    const gridCells: { x: number; y: number; votes: Record<number, number> }[] = []

    // Create a grid of cells
    for (let x = margin; x <= width - margin; x += gridSize) {
      for (let y = margin; y <= height - margin; y += gridSize) {
        gridCells.push({ x, y, votes: { 0: 0, 1: 0 } })
      }
    }

    // Generate synthetic data points
    const points: { x: number; y: number; class: number }[] = []
    const numPoints = 100

    // Create a spiral pattern for class 0
    for (let i = 0; i < numPoints / 2; i++) {
      const angle = (i / (numPoints / 2)) * Math.PI * 4
      const radius = 10 + (i / (numPoints / 2)) * 100
      const x = margin + plotWidth / 2 + Math.cos(angle) * radius
      const y = margin + plotHeight / 2 + Math.sin(angle) * radius
      if (x >= margin && x <= width - margin && y >= margin && y <= height - margin) {
        points.push({ x, y, class: 0 })
      }
    }

    // Create a spiral pattern for class 1
    for (let i = 0; i < numPoints / 2; i++) {
      const angle = (i / (numPoints / 2)) * Math.PI * 4 + Math.PI
      const radius = 10 + (i / (numPoints / 2)) * 100
      const x = margin + plotWidth / 2 + Math.cos(angle) * radius
      const y = margin + plotHeight / 2 + Math.sin(angle) * radius
      if (x >= margin && x <= width - margin && y >= margin && y <= height - margin) {
        points.push({ x, y, class: 1 })
      }
    }

    // Simulate multiple decision trees in a random forest
    for (let t = 0; t < trees; t++) {
      // Randomly sample points (bagging)
      const sampleCount = Math.floor(points.length * (sampleSize / 100))
      const sampledIndices = new Set<number>()
      while (sampledIndices.size < sampleCount) {
        sampledIndices.add(Math.floor(Math.random() * points.length))
      }

      const sampledPoints = Array.from(sampledIndices).map((i) => points[i])

      // Simulate a decision tree with recursive binary splits
      const simulateTree = (
        depth: number,
        x1: number,
        y1: number,
        x2: number,
        y2: number,
        points: { x: number; y: number; class: number }[],
      ) => {
        if (depth >= maxDepth || points.length <= 3) {
          // Determine majority class in this region
          const classCounts = points.reduce((counts: Record<number, number>, p) => {
            counts[p.class] = (counts[p.class] || 0) + 1
            return counts
          }, {})

          let majorityClass = 0
          let maxCount = 0
          for (const cls in classCounts) {
            if (classCounts[cls] > maxCount) {
              maxCount = classCounts[cls]
              majorityClass = Number(cls)
            }
          }

          // Vote for all grid cells in this region
          gridCells.forEach((cell) => {
            if (cell.x >= x1 && cell.x <= x2 && cell.y >= y1 && cell.y <= y2) {
              cell.votes[majorityClass] = (cell.votes[majorityClass] || 0) + 1
            }
          })

          return
        }

        // Choose a random split (simplified)
        const splitVertical = Math.random() > 0.5
        const splitPos = splitVertical ? x1 + Math.random() * (x2 - x1) : y1 + Math.random() * (y2 - y1)

        // Split points
        const leftPoints: { x: number; y: number; class: number }[] = []
        const rightPoints: { x: number; y: number; class: number }[] = []

        points.forEach((p) => {
          if (splitVertical) {
            if (p.x < splitPos) leftPoints.push(p)
            else rightPoints.push(p)
          } else {
            if (p.y < splitPos) leftPoints.push(p)
            else rightPoints.push(p)
          }
        })

        // Recurse on both splits
        if (leftPoints.length > 0) {
          simulateTree(depth + 1, x1, y1, splitVertical ? splitPos : x2, splitVertical ? y2 : splitPos, leftPoints)
        }

        if (rightPoints.length > 0) {
          simulateTree(depth + 1, splitVertical ? splitPos : x1, splitVertical ? y1 : splitPos, x2, y2, rightPoints)
        }
      }

      // Start the tree simulation
      simulateTree(0, margin, margin, width - margin, height - margin, sampledPoints)
    }

    // Determine final class for each grid cell based on majority voting
    gridCells.forEach((cell) => {
      const class0Votes = cell.votes[0] || 0
      const class1Votes = cell.votes[1] || 0
      const majorityClass = class0Votes > class1Votes ? 0 : 1
      const confidence = Math.max(class0Votes, class1Votes) / trees

      ctx.beginPath()
      ctx.rect(cell.x - gridSize / 2, cell.y - gridSize / 2, gridSize, gridSize)

      if (majorityClass === 0) {
        ctx.fillStyle = `rgba(100, 100, 255, ${confidence * 0.7})`
      } else {
        ctx.fillStyle = `rgba(255, 100, 100, ${confidence * 0.7})`
      }

      ctx.fill()
    })

    // Draw points
    points.forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 4, 0, Math.PI * 2)
      if (point.class === 0) {
        ctx.fillStyle = "#0000FF"
      } else {
        ctx.fillStyle = "#FF0000"
      }
      ctx.fill()
    })

    // Add legend
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "left"

    ctx.fillStyle = "#0000FF"
    ctx.beginPath()
    ctx.arc(width - 100, margin + 10, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = "#000"
    ctx.fillText("Class 0", width - 80, margin + 15)

    ctx.fillStyle = "#FF0000"
    ctx.beginPath()
    ctx.arc(width - 100, margin + 35, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = "#000"
    ctx.fillText("Class 1", width - 80, margin + 40)

    // Draw title showing parameters
    ctx.fillStyle = "#000"
    ctx.font = "12px Arial"
    ctx.textAlign = "left"
    ctx.fillText(`Trees: ${trees}, Max Depth: ${maxDepth}, Sample Size: ${sampleSize}%`, margin, height - 10)
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Decision Tree Accuracy: 0.8800
          <br />
          Random Forest Accuracy: 0.9600
          <br />
          Feature Importances:
          <br />
          Feature 0: 0.3421
          <br />
          Feature 1: 0.6579
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">Random Forest Decision Boundaries:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">Random Forest decision boundaries plot</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Best parameters: {"{"}'max_depth': 10, 'n_estimators': 100{"}"}
          <br />
          Best cross-validation score: 0.9533
          <br />
          Test accuracy with best parameters: 0.9600
          <br />
          OOB Score: 0.9467
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Random Forests</h1>
          <p className="text-neutral-700 mt-2">Understanding Random Forests for classification and regression tasks</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models">All Models</Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/svm">
              Next: Support Vector Machines <ArrowRight className="ml-2 h-4 w-4" />
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
              <CardTitle className="text-neutral-900">What is Random Forest?</CardTitle>
              <CardDescription className="text-neutral-600">
                An ensemble learning method that combines multiple decision trees
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Random Forest is an ensemble learning method that builds multiple decision trees during training and
                outputs the class that is the mode of the classes (for classification) or mean prediction (for
                regression) of the individual trees. It was developed by Leo Breiman and Adele Cutler, and combines the
                concepts of bagging and random feature selection to create a powerful and robust model.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">Key Concepts in Random Forest</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Ensemble Learning</strong>: Combining multiple models to
                    improve performance
                  </li>
                  <li>
                    <strong className="text-neutral-900">Bagging (Bootstrap Aggregating)</strong>: Training each tree on
                    a random subset of the data
                  </li>
                  <li>
                    <strong className="text-neutral-900">Feature Randomness</strong>: Each tree considers only a random
                    subset of features at each split
                  </li>
                  <li>
                    <strong className="text-neutral-900">Majority Voting</strong>: Final prediction is based on the
                    majority vote of all trees
                  </li>
                  <li>
                    <strong className="text-neutral-900">Out-of-Bag (OOB) Error</strong>: Error estimate using samples
                    not used in training individual trees
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">How Random Forest Works</h3>
              <p className="text-neutral-700 mb-4">The Random Forest algorithm follows these steps:</p>
              <ol className="list-decimal list-inside space-y-4 text-neutral-700">
                <li>
                  <strong className="text-neutral-900">Bootstrap Sampling</strong>: Create multiple datasets by randomly
                  sampling with replacement from the original dataset
                </li>
                <li>
                  <strong className="text-neutral-900">Build Decision Trees</strong>: For each bootstrap sample, grow a
                  decision tree with the following modification:
                  <ul className="list-disc list-inside ml-6 mt-2">
                    <li>
                      At each node, randomly select a subset of features (typically sqrt(n) for classification or n/3
                      for regression, where n is the total number of features)
                    </li>
                    <li>
                      Choose the best feature/split from this subset using criteria like Gini impurity or information
                      gain
                    </li>
                    <li>Split the node and continue recursively until stopping criteria are met</li>
                  </ul>
                </li>
                <li>
                  <strong className="text-neutral-900">Make Predictions</strong>: For a new instance, each tree makes a
                  prediction, and the final prediction is:
                  <ul className="list-disc list-inside ml-6 mt-2">
                    <li>For classification: the majority vote (most common class predicted by individual trees)</li>
                    <li>For regression: the average of all tree predictions</li>
                  </ul>
                </li>
              </ol>

              <div className="bg-neutral-100 p-4 rounded-lg mt-6">
                <h3 className="font-medium text-neutral-900 mb-2">Feature Importance</h3>
                <p className="text-neutral-700">Random Forests provide a natural way to measure feature importance:</p>
                <ul className="list-disc list-inside space-y-2 text-neutral-700 mt-2">
                  <li>
                    For each feature, calculate how much the prediction error increases when that feature's values are
                    permuted
                  </li>
                  <li>Features that lead to larger increases in error are more important</li>
                  <li>This helps identify which features are most influential in making predictions</li>
                </ul>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Advantages and Limitations</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Advantages</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Robust against overfitting</li>
                      <li>Handles large datasets with high dimensionality</li>
                      <li>Provides feature importance measures</li>
                      <li>Handles missing values and maintains accuracy</li>
                      <li>Requires minimal hyperparameter tuning</li>
                      <li>Built-in validation through OOB error</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Limitations</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Less interpretable than single decision trees</li>
                      <li>Computationally intensive for very large datasets</li>
                      <li>May overfit on noisy datasets</li>
                      <li>Not as effective for regression as for classification</li>
                      <li>Biased in favor of features with more levels (in categorical variables)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Applications of Random Forests</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-neutral-700">
                Random Forests are used in various fields for classification, regression, and feature selection:
              </p>
              <ul className="list-disc list-inside space-y-2 text-neutral-700">
                <li>
                  <strong className="text-neutral-900">Finance</strong>: Credit scoring, fraud detection, stock price
                  prediction
                </li>
                <li>
                  <strong className="text-neutral-900">Healthcare</strong>: Disease prediction, patient risk
                  stratification, genomics
                </li>
                <li>
                  <strong className="text-neutral-900">Marketing</strong>: Customer segmentation, churn prediction,
                  recommendation systems
                </li>
                <li>
                  <strong className="text-neutral-900">Computer Vision</strong>: Object detection, image classification
                </li>
                <li>
                  <strong className="text-neutral-900">Ecology</strong>: Species distribution modeling, land cover
                  classification
                </li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualization" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Interactive Random Forest Model</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how they affect the Random Forest decision boundaries
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="Random Forest Visualization"
                parameters={[
                  {
                    name: "trees",
                    min: 1,
                    max: 50,
                    step: 1,
                    default: 10,
                    label: "Number of Trees",
                  },
                  {
                    name: "maxDepth",
                    min: 1,
                    max: 10,
                    step: 1,
                    default: 3,
                    label: "Max Tree Depth",
                  },
                  {
                    name: "sampleSize",
                    min: 10,
                    max: 100,
                    step: 5,
                    default: 70,
                    label: "Sample Size (%)",
                  },
                ]}
                renderVisualization={renderRandomForest}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Number of Trees</h3>
                <p className="text-neutral-700">
                  This parameter controls how many decision trees are included in the forest. More trees generally lead
                  to better performance but increase computation time. The benefit of additional trees typically
                  diminishes after a certain point. Notice how the decision boundary becomes smoother and more stable as
                  you increase the number of trees.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Max Tree Depth</h3>
                <p className="text-neutral-700">
                  This parameter limits how deep each individual decision tree can grow. Deeper trees can capture more
                  complex patterns but may lead to overfitting. Shallower trees create simpler, more generalized models.
                  Observe how increasing the max depth creates more complex decision boundaries that more closely fit
                  the training data.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Sample Size (%)</h3>
                <p className="text-neutral-700">
                  This parameter determines what percentage of the training data is randomly sampled (with replacement)
                  to train each tree. This is the "bagging" aspect of Random Forests. Smaller sample sizes create more
                  diverse trees, which can help reduce overfitting but might miss important patterns if too small.
                </p>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Interpreting the Visualization</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    The <strong>colored regions</strong> represent the decision boundaries for each class
                  </li>
                  <li>
                    The <strong>color intensity</strong> indicates the confidence of the prediction (darker = more trees
                    agree)
                  </li>
                  <li>
                    <strong>Blue points</strong> represent instances of class 0
                  </li>
                  <li>
                    <strong>Red points</strong> represent instances of class 1
                  </li>
                  <li>Notice how the decision boundary becomes more complex with deeper trees</li>
                  <li>With more trees, the boundary becomes smoother and more stable</li>
                  <li>
                    The sample size affects how well the model captures the overall pattern vs. specific instances
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">Random Forest Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement Random Forests using Python and scikit-learn. Execute each
                cell to see the results.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">
                  Step 1: Compare a single Decision Tree with a Random Forest
                </p>
                <p>
                  Let's create a synthetic dataset and compare the performance of a single decision tree versus a random
                  forest.
                </p>
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
    n_clusters_per_class=2,
    class_sep=1.0,
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train a single decision tree
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
tree_pred = tree_clf.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)

# Train a random forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Get feature importances from the random forest
importances = rf_clf.feature_importances_

print(f'Decision Tree Accuracy: {tree_accuracy:.4f}')
print(f'Random Forest Accuracy: {rf_accuracy:.4f}')
print('Feature Importances:')
for i, importance in enumerate(importances):
    print(f'Feature {i}: {importance:.4f}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Visualize decision boundaries</p>
                <p>
                  Let's visualize how the Random Forest creates more stable decision boundaries compared to a single
                  Decision Tree.
                </p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Function to plot decision boundaries
def plot_decision_boundaries(X, y, models, titles, figsize=(12, 5)):
    # Set up the figure
    fig, axes = plt.subplots(1, len(models), figsize=figsize)
    
    # Determine plot boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Plot each model
    for i, (model, title) in enumerate(zip(models, titles)):
        # Predict using the model
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
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
    [tree_clf, rf_clf],
    ['Decision Tree', 'Random Forest (100 trees)']
)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Hyperparameter tuning and OOB score</p>
                <p>Let's use grid search to find the optimal hyperparameters and examine the out-of-bag (OOB) score.</p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Define parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a smaller grid for demonstration
small_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

# Create grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    small_param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Get best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train a model with the best parameters and compute OOB score
best_rf = RandomForestClassifier(
    **best_params,
    oob_score=True,  # Enable out-of-bag scoring
    random_state=42
)
best_rf.fit(X_train, y_train)

# Evaluate on test set
test_accuracy = accuracy_score(y_test, best_rf.predict(X_test))
oob_score = best_rf.oob_score_

print(f'Best parameters: {best_params}')
print(f'Best cross-validation score: {best_score:.4f}')
print(f'Test accuracy with best parameters: {test_accuracy:.4f}')
print(f'OOB Score: {oob_score:.4f}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of Random Forests:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>Try different numbers of trees and observe the effect on performance</li>
                  <li>Experiment with feature selection by varying max_features</li>
                  <li>Implement Random Forest for regression instead of classification</li>
                  <li>Visualize feature importances using a bar chart</li>
                  <li>Compare Random Forest with other ensemble methods like Gradient Boosting</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
