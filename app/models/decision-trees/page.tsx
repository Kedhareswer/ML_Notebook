"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import ModelVisualization from "@/components/model-visualization"
import { ArrowLeft, ArrowRight, BookOpen, Code, GitBranch } from "lucide-react"
import Link from "next/link"
import NotebookCell from "@/components/notebook-cell"

export default function DecisionTreesPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  // Decision tree visualization function
  const renderDecisionTree = (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => {
    const { maxDepth, minSamplesSplit, randomness } = params

    // Calculate tree structure based on parameters
    const depth = Math.floor(maxDepth)
    const nodeSize = 30
    const levelHeight = height / (depth + 1)

    // Draw the tree recursively
    const drawNode = (x: number, y: number, level: number, parentX?: number, parentY?: number) => {
      // Draw connection to parent
      if (parentX !== undefined && parentY !== undefined) {
        ctx.beginPath()
        ctx.moveTo(parentX, parentY)
        ctx.lineTo(x, y)
        ctx.strokeStyle = "#666"
        ctx.stroke()
      }

      // Draw node
      ctx.beginPath()
      ctx.arc(x, y, nodeSize, 0, Math.PI * 2)

      // Color based on level (leaf nodes are different)
      if (level >= depth) {
        ctx.fillStyle = "#333" // Dark gray for leaf nodes
      } else {
        ctx.fillStyle = "#000" // Black for decision nodes
      }
      ctx.fill()

      // Add some text
      ctx.fillStyle = "#fff"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.font = "12px sans-serif"

      if (level >= depth) {
        ctx.fillText("Leaf", x, y)
      } else {
        ctx.fillText(`X${level + 1}`, x, y)
      }

      // Stop recursion at max depth or based on minSamplesSplit
      if (level >= depth) return
      if (Math.random() * 10 < minSamplesSplit) return

      // Add randomness to the split position
      const splitRatio = 0.5 + (Math.random() - 0.5) * randomness * 0.3

      // Calculate child positions
      const nextLevel = level + 1
      const childSpacing = width / Math.pow(2, nextLevel + 1)
      const leftX = x - childSpacing
      const rightX = x + childSpacing
      const childY = y + levelHeight

      // Draw children recursively
      drawNode(leftX, childY, nextLevel, x, y)
      drawNode(rightX, childY, nextLevel, x, y)
    }

    // Start drawing from the root
    drawNode(width / 2, levelHeight, 0)
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Accuracy: 0.8733
          <br />
          <br />
          Decision tree with 7 nodes and max depth 3<br />
          Number of features: 2<br />
          Number of classes: 2
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">Tree visualization:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">Decision tree visualization</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">Decision boundaries:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">Decision boundaries visualization</p>
          </div>
        </div>
      )
    } else if (cellId === "cell4") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Feature importances:
          <br />
          Feature 0: 0.7621
          <br />
          Feature 1: 0.2379
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Decision Trees</h1>
          <p className="text-neutral-700 mt-2">
            Understanding decision trees and their implementation for classification and regression
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models/polynomial-regression">
              <ArrowLeft className="mr-2 h-4 w-4" /> Polynomial Regression
            </Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/random-forests">
              Next: Random Forests <ArrowRight className="ml-2 h-4 w-4" />
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
            <GitBranch className="h-4 w-4" />
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
              <CardTitle className="text-neutral-900">What are Decision Trees?</CardTitle>
              <CardDescription className="text-neutral-600">
                A versatile machine learning algorithm for classification and regression tasks
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Decision trees are a popular supervised learning method used for both classification and regression
                tasks. They work by creating a model that predicts the value of a target variable by learning simple
                decision rules inferred from the data features.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">How Decision Trees Work</h3>
                <p className="mb-4 text-neutral-700">A decision tree is a flowchart-like structure where:</p>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Internal nodes</strong> represent a "test" on an attribute
                    (e.g., whether a feature is greater than a certain value)
                  </li>
                  <li>
                    <strong className="text-neutral-900">Branches</strong> represent the outcome of the test
                  </li>
                  <li>
                    <strong className="text-neutral-900">Leaf nodes</strong> represent class labels or continuous values
                    (for classification or regression)
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">Decision Tree Learning Process</h3>
              <ol className="list-decimal list-inside space-y-4 text-neutral-700">
                <li>
                  <strong className="text-neutral-900">Start at the root node</strong> with the entire dataset
                </li>
                <li>
                  <strong className="text-neutral-900">Find the best feature and threshold</strong> to split the data
                  that maximizes information gain
                </li>
                <li>
                  <strong className="text-neutral-900">Create child nodes</strong> based on the split
                </li>
                <li>
                  <strong className="text-neutral-900">Recursively repeat</strong> the process for each child node until
                  stopping criteria are met
                </li>
                <li>
                  <strong className="text-neutral-900">Assign class labels or values</strong> to the leaf nodes
                </li>
              </ol>

              <div className="bg-neutral-100 p-4 rounded-lg mt-6">
                <h3 className="font-medium text-neutral-900 mb-2">Splitting Criteria</h3>
                <p className="mb-2 text-neutral-700">
                  Decision trees use different metrics to determine the best split:
                </p>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Gini Impurity</strong>: Measures the probability of incorrect
                    classification
                  </li>
                  <li>
                    <strong className="text-neutral-900">Entropy</strong>: Measures the level of disorder or uncertainty
                  </li>
                  <li>
                    <strong className="text-neutral-900">Information Gain</strong>: The reduction in entropy after a
                    dataset is split
                  </li>
                  <li>
                    <strong className="text-neutral-900">Mean Squared Error</strong>: Used for regression trees to
                    minimize prediction error
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Advantages and Limitations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="font-medium text-neutral-900">Advantages</h3>
                  <ul className="list-disc list-inside space-y-2 text-neutral-700">
                    <li>Easy to understand and interpret</li>
                    <li>Requires little data preprocessing</li>
                    <li>Can handle both numerical and categorical data</li>
                    <li>Can handle multi-output problems</li>
                    <li>Implicitly performs feature selection</li>
                    <li>Non-parametric (no assumptions about data distribution)</li>
                  </ul>
                </div>
                <div className="space-y-4">
                  <h3 className="font-medium text-neutral-900">Limitations</h3>
                  <ul className="list-disc list-inside space-y-2 text-neutral-700">
                    <li>Can create overly complex trees that don't generalize well</li>
                    <li>Prone to overfitting, especially with deep trees</li>
                    <li>Can be unstable (small variations in data can result in different trees)</li>
                    <li>Biased toward features with more levels</li>
                    <li>Not optimal for continuous variables</li>
                    <li>May struggle with imbalanced datasets</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualization" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Interactive Decision Tree Model</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how they affect the decision tree structure
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="Decision Tree Visualization"
                parameters={[
                  {
                    name: "maxDepth",
                    min: 1,
                    max: 5,
                    step: 1,
                    default: 3,
                    label: "Max Depth",
                  },
                  {
                    name: "minSamplesSplit",
                    min: 0,
                    max: 10,
                    step: 1,
                    default: 2,
                    label: "Min Samples Split",
                  },
                  {
                    name: "randomness",
                    min: 0,
                    max: 1,
                    step: 0.1,
                    default: 0.3,
                    label: "Data Randomness",
                  },
                ]}
                renderVisualization={renderDecisionTree}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Max Depth</h3>
                <p className="text-neutral-700">
                  The maximum depth of the tree. This limits how many levels of decisions the tree can make. A deeper
                  tree can capture more complex patterns but is more prone to overfitting.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Min Samples Split</h3>
                <p className="text-neutral-700">
                  The minimum number of samples required to split an internal node. Higher values prevent the tree from
                  creating splits that only affect a small number of samples, which helps prevent overfitting.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Data Randomness</h3>
                <p className="text-neutral-700">
                  This parameter simulates the randomness in the data. Higher values create more irregular splits,
                  representing noisier data that's harder to classify perfectly.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">Decision Tree Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement decision trees using Python and scikit-learn. Execute each
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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 1: Generate and prepare data</p>
                <p>Let's create a synthetic classification dataset and split it into training and testing sets.</p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode="# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train the decision tree classifier
tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    random_state=42
)
tree.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print()
print(f'Decision tree with {tree.tree_.node_count} nodes and max depth {tree.tree_.max_depth}')
print(f'Number of features: {tree.n_features_in_}')
print(f'Number of classes: {len(tree.classes_)}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Visualize the decision tree</p>
                <p>Let's visualize the structure of our trained decision tree.</p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(
    tree, 
    filled=True, 
    feature_names=[f'Feature {i}' for i in range(X.shape[1])],
    class_names=['Class 0', 'Class 1']
)
plt.title('Decision Tree Visualization')
plt.show()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Visualize the decision boundaries</p>
                <p>Now let's see how the decision tree separates the feature space into regions.</p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Visualize the decision boundaries
plt.figure(figsize=(10, 6))
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Decision Boundaries')
plt.show()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 4: Analyze feature importance</p>
                <p>Let's examine which features are most important for the decision tree's predictions.</p>
              </div>

              <NotebookCell
                cellId="cell4"
                executionCount={5}
                initialCode="# Get feature importances
importances = tree.feature_importances_
print('Feature importances:')
for i, importance in enumerate(importances):
    print(f'Feature {i}: {importance:.4f}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of decision trees:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>Change the max_depth parameter to see how it affects model complexity</li>
                  <li>Try different criteria (gini or entropy) for the split quality</li>
                  <li>Implement pruning techniques to reduce overfitting</li>
                  <li>Use cross-validation to find optimal hyperparameters</li>
                  <li>Try a regression tree instead of a classification tree</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
