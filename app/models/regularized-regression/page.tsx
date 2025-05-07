"use client"

import { useState } from "react"
import Link from "next/link"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, ArrowRight, BookOpen, Code, BarChart } from "lucide-react"
import NotebookCell from "@/components/notebook-cell"
import ModelVisualization from "@/components/model-visualization"

export default function RegularizedRegressionPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  // Ridge and Lasso visualization function
  const renderRegularization = (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => {
    const { alphaRidge, alphaLasso, complexity } = params

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Set up coordinate system
    const margin = 50
    const plotWidth = width - 2 * margin
    const plotHeight = height - 2 * margin
    const xScale = plotWidth / 10
    const yScale = plotHeight / 10

    // Draw axes
    ctx.beginPath()
    ctx.moveTo(margin, height - margin)
    ctx.lineTo(width - margin, height - margin) // x-axis
    ctx.moveTo(margin, height - margin)
    ctx.lineTo(margin, margin) // y-axis
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 1.5
    ctx.stroke()

    // Draw axis labels
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "center"
    ctx.fillText("Model Complexity", width / 2, height - 10)

    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = "center"
    ctx.fillText("Error", 0, 0)
    ctx.restore()

    // Draw axis ticks
    for (let x = 0; x <= 10; x += 2) {
      ctx.beginPath()
      ctx.moveTo(margin + x * xScale, height - margin)
      ctx.lineTo(margin + x * xScale, height - margin + 5)
      ctx.stroke()
      ctx.fillText(x.toString(), margin + x * xScale, height - margin + 20)
    }

    for (let y = 0; y <= 10; y += 2) {
      ctx.beginPath()
      ctx.moveTo(margin, height - margin - y * yScale)
      ctx.lineTo(margin - 5, height - margin - y * yScale)
      ctx.stroke()
      ctx.textAlign = "right"
      ctx.fillText(y.toString(), margin - 10, height - margin - y * yScale + 5)
    }

    // Generate data points for a typical bias-variance tradeoff curve
    const complexityFactor = complexity / 5

    // Generate data for training error (decreases with complexity)
    const trainingError = []
    for (let x = 0; x <= 10; x += 0.1) {
      const error = 5 - 4 * Math.tanh(x * complexityFactor) + 0.1 * Math.random()
      trainingError.push({ x, y: error })
    }

    // Generate data for test error (U-shaped curve)
    const testError = []
    for (let x = 0; x <= 10; x += 0.1) {
      const error =
        5 -
        3 * Math.tanh(x * 0.5 * complexityFactor) +
        0.5 * Math.pow((x * complexityFactor) / 2, 2) +
        0.1 * Math.random()
      testError.push({ x, y: error })
    }

    // Generate data for ridge regression error
    const ridgeError = []
    for (let x = 0; x <= 10; x += 0.1) {
      // Ridge reduces error more uniformly
      const error =
        5 -
        3 * Math.tanh(x * 0.5 * complexityFactor) +
        (0.5 - 0.4 * (alphaRidge / 10)) * Math.pow((x * complexityFactor) / 2, 2) +
        0.1 * Math.random()
      ridgeError.push({ x, y: error })
    }

    // Generate data for lasso regression error
    const lassoError = []
    for (let x = 0; x <= 10; x += 0.1) {
      // Lasso has a sharper reduction in error at higher complexity
      const error =
        5 -
        3 * Math.tanh(x * 0.5 * complexityFactor) +
        (0.5 - 0.45 * (alphaLasso / 10)) * Math.pow((x * complexityFactor) / 2, 1.8) +
        0.1 * Math.random()
      lassoError.push({ x, y: error })
    }

    // Draw grid lines
    ctx.strokeStyle = "#e0e0e0"
    ctx.lineWidth = 0.5
    for (let x = 0; x <= 10; x += 2) {
      ctx.beginPath()
      ctx.moveTo(margin + x * xScale, margin)
      ctx.lineTo(margin + x * xScale, height - margin)
      ctx.stroke()
    }
    for (let y = 0; y <= 10; y += 2) {
      ctx.beginPath()
      ctx.moveTo(margin, height - margin - y * yScale)
      ctx.lineTo(width - margin, height - margin - y * yScale)
      ctx.stroke()
    }

    // Draw optimal complexity region
    const optimalX = 3 + (1 - alphaRidge / 10) * 2
    ctx.fillStyle = "rgba(200, 200, 200, 0.2)"
    ctx.beginPath()
    ctx.rect(margin + (optimalX - 0.5) * xScale, margin, xScale, height - 2 * margin)
    ctx.fill()
    ctx.strokeStyle = "#888"
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(margin + optimalX * xScale, margin)
    ctx.lineTo(margin + optimalX * xScale, height - margin)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = "#888"
    ctx.textAlign = "center"
    ctx.fillText("Optimal", margin + optimalX * xScale, margin - 10)

    // Draw training error curve
    ctx.beginPath()
    ctx.moveTo(margin + trainingError[0].x * xScale, height - margin - trainingError[0].y * yScale)
    for (let i = 1; i < trainingError.length; i++) {
      ctx.lineTo(margin + trainingError[i].x * xScale, height - margin - trainingError[i].y * yScale)
    }
    ctx.strokeStyle = "#3498db" // blue
    ctx.lineWidth = 3
    ctx.stroke()

    // Draw test error curve
    ctx.beginPath()
    ctx.moveTo(margin + testError[0].x * xScale, height - margin - testError[0].y * yScale)
    for (let i = 1; i < testError.length; i++) {
      ctx.lineTo(margin + testError[i].x * xScale, height - margin - testError[i].y * yScale)
    }
    ctx.strokeStyle = "#e74c3c" // red
    ctx.lineWidth = 3
    ctx.stroke()

    // Draw ridge error curve
    ctx.beginPath()
    ctx.moveTo(margin + ridgeError[0].x * xScale, height - margin - ridgeError[0].y * yScale)
    for (let i = 1; i < ridgeError.length; i++) {
      ctx.lineTo(margin + ridgeError[i].x * xScale, height - margin - ridgeError[i].y * yScale)
    }
    ctx.strokeStyle = "#9b59b6" // purple
    ctx.lineWidth = 3
    ctx.stroke()

    // Draw lasso error curve
    ctx.beginPath()
    ctx.moveTo(margin + lassoError[0].x * xScale, height - margin - lassoError[0].y * yScale)
    for (let i = 1; i < lassoError.length; i++) {
      ctx.lineTo(margin + lassoError[i].x * xScale, height - margin - lassoError[i].y * yScale)
    }
    ctx.strokeStyle = "#2ecc71" // green
    ctx.lineWidth = 3
    ctx.stroke()

    // Add legend
    const legendX = width - 180
    const legendY = 40
    const legendSpacing = 25
    const legendBoxSize = 15

    // Training error
    ctx.fillStyle = "#3498db"
    ctx.fillRect(legendX, legendY, legendBoxSize, legendBoxSize)
    ctx.fillStyle = "#000"
    ctx.textAlign = "left"
    ctx.fillText("Training Error", legendX + 25, legendY + 12)

    // Test error
    ctx.fillStyle = "#e74c3c"
    ctx.fillRect(legendX, legendY + legendSpacing, legendBoxSize, legendBoxSize)
    ctx.fillStyle = "#000"
    ctx.fillText("Test Error", legendX + 25, legendY + legendSpacing + 12)

    // Ridge error
    ctx.fillStyle = "#9b59b6"
    ctx.fillRect(legendX, legendY + 2 * legendSpacing, legendBoxSize, legendBoxSize)
    ctx.fillStyle = "#000"
    ctx.fillText(`Ridge (α=${alphaRidge.toFixed(1)})`, legendX + 25, legendY + 2 * legendSpacing + 12)

    // Lasso error
    ctx.fillStyle = "#2ecc71"
    ctx.fillRect(legendX, legendY + 3 * legendSpacing, legendBoxSize, legendBoxSize)
    ctx.fillStyle = "#000"
    ctx.fillText(`Lasso (α=${alphaLasso.toFixed(1)})`, legendX + 25, legendY + 3 * legendSpacing + 12)

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "16px Arial"
    ctx.textAlign = "center"
    ctx.fillText("Regularization Effect on Model Error", width / 2, 20)

    // Draw annotations
    ctx.font = "12px Arial"
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
    ctx.textAlign = "left"
    ctx.fillText("Underfitting", margin + xScale, height - margin - 8 * yScale)
    ctx.textAlign = "right"
    ctx.fillText("Overfitting", margin + 9 * xScale, height - margin - 6 * yScale)

    // Draw arrows for annotations
    ctx.beginPath()
    ctx.moveTo(margin + xScale, height - margin - 7.8 * yScale)
    ctx.lineTo(margin + 2 * xScale, height - margin - 6 * yScale)
    ctx.strokeStyle = "rgba(0, 0, 0, 0.7)"
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(margin + 9 * xScale, height - margin - 5.8 * yScale)
    ctx.lineTo(margin + 8 * xScale, height - margin - 4 * yScale)
    ctx.stroke()
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Linear Regression:
          <br />
          Train MSE: 1023.45
          <br />
          Test MSE: 1245.67
          <br />
          Non-zero coefficients: 50 out of 50
          <br />
          <br />
          Ridge (α=1.0):
          <br />
          Train MSE: 1056.78
          <br />
          Test MSE: 1089.23
          <br />
          Non-zero coefficients: 50 out of 50
          <br />
          <br />
          Ridge (α=10.0):
          <br />
          Train MSE: 1102.34
          <br />
          Test MSE: 1045.67
          <br />
          Non-zero coefficients: 50 out of 50
          <br />
          <br />
          Lasso (α=0.1):
          <br />
          Train MSE: 1078.90
          <br />
          Test MSE: 1067.45
          <br />
          Non-zero coefficients: 32 out of 50
          <br />
          <br />
          Lasso (α=1.0):
          <br />
          Train MSE: 1156.78
          <br />
          Test MSE: 1034.56
          <br />
          Non-zero coefficients: 15 out of 50
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">Coefficient Values:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">Coefficient values plot</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Best Ridge parameters: {"{"}'alpha': 5.0{"}"}
          <br />
          Best Ridge score: 0.8765
          <br />
          <br />
          Best Lasso parameters: {"{"}'alpha': 0.01{"}"}
          <br />
          Best Lasso score: 0.8654
          <br />
          <br />
          Best Elastic Net parameters: {"{"}'alpha': 0.1, 'l1_ratio': 0.5{"}"}
          <br />
          Best Elastic Net score: 0.8821
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Ridge & Lasso Regression</h1>
          <p className="text-neutral-700 mt-2">Regularization techniques to prevent overfitting in linear models</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models/polynomial-regression">
              <ArrowLeft className="mr-2 h-4 w-4" /> Polynomial Regression
            </Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/logistic-regression">
              Next: Logistic Regression <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      <Tabs defaultValue="explanation" value={activeTab} onValueChange={setActiveTab} className="space-y-8">
        <TabsList className="grid w-full grid-cols-3 bg-neutral-100 text-neutral-900">
          <TabsTrigger value="explanation" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BookOpen className="h-4 w-4" />
            <span>Overview</span>
          </TabsTrigger>
          <TabsTrigger value="visualization" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BarChart className="h-4 w-4" />
            <span>Interactive Demo</span>
          </TabsTrigger>
          <TabsTrigger value="notebook" className="flex items-center gap-2 data-[state=active]:bg-white">
            <Code className="h-4 w-4" />
            <span>Code Implementation</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="explanation" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">What are Ridge and Lasso Regression?</CardTitle>
              <CardDescription className="text-neutral-600">
                Regularization techniques that extend linear regression to prevent overfitting
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Ridge and Lasso regression are regularization techniques that extend linear regression to address
                overfitting, particularly when dealing with many features or multicollinearity. Both methods add a
                penalty term to the linear regression cost function, but they differ in the type of penalty applied.
              </p>

              <div className="grid gap-6 md:grid-cols-2 mt-6">
                <div className="bg-white p-6 rounded-lg border border-neutral-300">
                  <h3 className="text-xl font-medium text-neutral-800 mb-2">Ridge Regression (L2 Regularization)</h3>
                  <p className="text-neutral-700 mb-3">
                    Adds a penalty equal to the square of the magnitude of coefficients.
                  </p>
                  <div className="py-3 px-4 bg-neutral-100 rounded-lg font-mono text-sm text-neutral-800">
                    Cost = RSS + α * Σ(β_j²)
                  </div>
                  <p className="text-neutral-700 mt-3">
                    Ridge regression shrinks coefficients toward zero but rarely eliminates them completely.
                  </p>
                </div>
                <div className="bg-white p-6 rounded-lg border border-neutral-300">
                  <h3 className="text-xl font-medium text-neutral-800 mb-2">Lasso Regression (L1 Regularization)</h3>
                  <p className="text-neutral-700 mb-3">Adds a penalty equal to the absolute value of coefficients.</p>
                  <div className="py-3 px-4 bg-neutral-100 rounded-lg font-mono text-sm text-neutral-800">
                    Cost = RSS + α * Σ|β_j|
                  </div>
                  <p className="text-neutral-700 mt-3">
                    Lasso regression can reduce coefficients exactly to zero, effectively performing feature selection.
                  </p>
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">Key Concepts in Regularized Regression</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Regularization Parameter (α)</strong>: Controls the strength of
                    the penalty; higher values of α apply stronger regularization
                  </li>
                  <li>
                    <strong className="text-neutral-900">Feature Selection</strong>: Lasso can completely eliminate less
                    important features by setting their coefficients to zero
                  </li>
                  <li>
                    <strong className="text-neutral-900">Bias-Variance Tradeoff</strong>: Regularization increases bias
                    but reduces variance, which can lead to better generalization
                  </li>
                  <li>
                    <strong className="text-neutral-900">Elastic Net</strong>: A hybrid approach that combines L1 and L2
                    penalties, offering a middle ground between Ridge and Lasso
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">When to Use Regularized Regression</h3>
              <ul className="list-disc list-inside space-y-2 text-neutral-700">
                <li>When dealing with datasets with many features relative to the number of observations</li>
                <li>When there is multicollinearity among features</li>
                <li>To prevent overfitting in complex models</li>
                <li>Use Ridge when you believe most features contribute to the outcome</li>
                <li>Use Lasso when you suspect only a subset of features are relevant</li>
                <li>Use Elastic Net when you have groups of correlated features</li>
              </ul>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Advantages and Limitations</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Advantages</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Reduces overfitting in high-dimensional data</li>
                      <li>Handles multicollinearity effectively</li>
                      <li>Lasso provides built-in feature selection</li>
                      <li>Ridge works well when all features are relevant</li>
                      <li>Improves model generalization</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Limitations</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Requires tuning of the regularization parameter</li>
                      <li>May introduce bias in coefficient estimates</li>
                      <li>Lasso may be unstable with highly correlated features</li>
                      <li>Ridge doesn't perform feature selection</li>
                      <li>Performance depends on proper feature scaling</li>
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
              <CardTitle className="text-neutral-900">Interactive Regularization Demo</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how regularization affects model error and complexity
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="Regularization Effect Visualization"
                parameters={[
                  {
                    name: "alphaRidge",
                    min: 0,
                    max: 10,
                    step: 0.1,
                    default: 1.0,
                    label: "Ridge Alpha (α)",
                  },
                  {
                    name: "alphaLasso",
                    min: 0,
                    max: 10,
                    step: 0.1,
                    default: 1.0,
                    label: "Lasso Alpha (α)",
                  },
                  {
                    name: "complexity",
                    min: 1,
                    max: 10,
                    step: 0.5,
                    default: 5,
                    label: "Model Complexity",
                  },
                ]}
                renderVisualization={renderRegularization}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Ridge Alpha (α)</h3>
                <p className="text-neutral-700">
                  This parameter controls the strength of the L2 regularization in Ridge regression. Higher values of α
                  apply stronger regularization, shrinking all coefficients more toward zero. This reduces model
                  complexity and variance but may increase bias. In the visualization, notice how increasing Ridge α
                  (purple line) flattens the error curve at high complexity levels, preventing overfitting.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Lasso Alpha (α)</h3>
                <p className="text-neutral-700">
                  This parameter controls the strength of the L1 regularization in Lasso regression. Higher values of α
                  apply stronger regularization, potentially setting some coefficients exactly to zero. This performs
                  feature selection and can significantly reduce model complexity. In the visualization, observe how
                  increasing Lasso α (green line) affects the error curve, especially at high complexity levels, often
                  resulting in a sharper decrease in test error compared to Ridge.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Model Complexity</h3>
                <p className="text-neutral-700">
                  This parameter simulates increasing the complexity of the underlying model, such as adding more
                  features or using higher-degree polynomial terms. As complexity increases, the training error (blue)
                  continues to decrease, but the test error (red) follows a U-shaped curve due to overfitting. The
                  shaded region indicates the optimal complexity zone where the model generalizes best. Notice how
                  regularization methods help mitigate overfitting at high complexity levels.
                </p>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Interpreting the Visualization</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    The <strong className="text-[#3498db]">blue line</strong> represents training error, which decreases
                    as model complexity increases, eventually approaching zero (perfect fit on training data)
                  </li>
                  <li>
                    The <strong className="text-[#e74c3c]">red line</strong> represents test error, which follows a
                    U-shaped curve (the classic bias-variance tradeoff) with a minimum at the optimal complexity
                  </li>
                  <li>
                    The <strong className="text-[#9b59b6]">purple line</strong> shows how Ridge regression affects test
                    error by shrinking all coefficients proportionally, smoothing the error curve
                  </li>
                  <li>
                    The <strong className="text-[#2ecc71]">green line</strong> shows how Lasso regression affects test
                    error through feature selection, often resulting in a sharper decrease in error
                  </li>
                  <li>
                    The <strong className="text-neutral-500">shaded region</strong> indicates the optimal complexity
                    zone where models generalize best
                  </li>
                  <li>
                    Notice how both regularization methods help prevent the sharp increase in test error at high
                    complexity levels, with their effectiveness depending on the regularization strength (α)
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">Ridge and Lasso Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement Ridge and Lasso regression using Python and scikit-learn.
                Execute each cell to see the results.
              </p>
            </div>

            <div className="space-y-6">
              <NotebookCell
                cellId="cell0"
                executionCount={1}
                initialCode="import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 1: Compare Linear Regression with Ridge and Lasso</p>
                <p>
                  Let's create a synthetic dataset with many features and compare the performance of standard linear
                  regression versus Ridge and Lasso regression.
                </p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode="# Generate synthetic data with many features
X, y = make_regression(n_samples=100, n_features=50, n_informative=10, 
                      noise=20, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Ridge (α=10.0)': Ridge(alpha=10.0),
    'Lasso (α=0.1)': Lasso(alpha=0.1),
    'Lasso (α=1.0)': Lasso(alpha=1.0)
}

results = {}
coefficients = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predict and evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    results[name] = {'train_mse': train_mse, 'test_mse': test_mse}
    coefficients[name] = model.coef_

# Output results
for name, metrics in results.items():
    non_zero_coefs = np.sum(np.abs(coefficients[name]) > 1e-10)
    print(name + ':')
    print('  Train MSE: {:.2f}'.format(metrics['train_mse']))
    print('  Test MSE: {:.2f}'.format(metrics['test_mse']))
    print('  Non-zero coefficients: {} out of {}'.format(non_zero_coefs, X.shape[1]))
    print()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Visualize coefficient values</p>
                <p>
                  Let's visualize how Ridge and Lasso regression affect the coefficient values compared to standard
                  linear regression.
                </p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Plot coefficient values
plt.figure(figsize=(14, 10))
for i, (name, coefs) in enumerate(coefficients.items()):
    plt.subplot(len(models), 1, i+1)
    plt.stem(range(len(coefs)), coefs)
    plt.title(name + ' - Coefficient Values')
    plt.ylabel('Coefficient')
    plt.xlim([-1, X.shape[1]])

plt.tight_layout()
plt.show()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Hyperparameter tuning</p>
                <p>
                  Let's use grid search to find the optimal regularization parameter (α) for Ridge, Lasso, and Elastic
                  Net regression.
                </p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Define parameter grids
ridge_params = {'alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 100.0]}
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
elastic_net_params = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Create grid searches
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=5, scoring='neg_mean_squared_error')
elastic_net_grid = GridSearch  lasso_params, cv=5, scoring='neg_mean_squared_error')
elastic_net_grid = GridSearchCV(ElasticNet(), elastic_net_params, cv=5, scoring='neg_mean_squared_error')

# Fit grid searches
ridge_grid.fit(X_train_scaled, y_train)
lasso_grid.fit(X_train_scaled, y_train)
elastic_net_grid.fit(X_train_scaled, y_train)

# Get best parameters and scores
best_ridge = ridge_grid.best_params_
best_ridge_score = -ridge_grid.best_score_  # Convert back from negative MSE

best_lasso = lasso_grid.best_params_
best_lasso_score = -lasso_grid.best_score_

best_elastic_net = elastic_net_grid.best_params_
best_elastic_net_score = -elastic_net_grid.best_score_

# Print results
print('Best Ridge parameters: ' + str(best_ridge))
print('Best Ridge score: {:.4f}'.format(best_ridge_score))
print()
print('Best Lasso parameters: ' + str(best_lasso))
print('Best Lasso score: {:.4f}'.format(best_lasso_score))
print()
print('Best Elastic Net parameters: ' + str(best_elastic_net))
print('Best Elastic Net score: {:.4f}'.format(best_elastic_net_score))"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of regularized regression:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>Try different values of α and observe the effect on coefficient values</li>
                  <li>Create a dataset with multicollinearity and see how regularization helps</li>
                  <li>Implement cross-validation to find the optimal regularization parameter</li>
                  <li>Compare the performance of Ridge, Lasso, and Elastic Net on real-world datasets</li>
                  <li>Visualize the regularization path (how coefficients change with different α values)</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
