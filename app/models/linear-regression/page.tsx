"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import ModelVisualization from "@/components/model-visualization"
import { ArrowRight, BookOpen, Code, LineChart } from "lucide-react"
import Link from "next/link"
import NotebookCell from "@/components/notebook-cell"

export default function LinearRegressionPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  // Linear regression visualization function
  const renderLinearRegression = (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => {
    const { slope, intercept, noise } = params

    // Generate data points
    const points = []
    for (let i = 0; i < 20; i++) {
      const x = i * (width / 20)
      // Calculate y based on linear equation y = mx + b + noise
      const noiseValue = (Math.random() - 0.5) * noise * 50
      const y = height - (slope * x + intercept * 50 + noiseValue)
      points.push({ x, y })
    }

    // Draw axes
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, height)
    ctx.lineTo(width, height) // x-axis
    ctx.moveTo(0, 0)
    ctx.lineTo(0, height) // y-axis
    ctx.stroke()

    // Draw data points
    ctx.fillStyle = "#000"
    points.forEach((point) => {
      ctx.beginPath()
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw regression line
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, height - intercept * 50)
    ctx.lineTo(width, height - (slope * width + intercept * 50))
    ctx.stroke()
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Intercept: 4.21
          <br />
          Slope: 2.94
          <br />
          R² score: 0.768
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">Model trained successfully!</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">Plot visualization</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Predictions for new data points:
          <br />
          [4.21, 10.09]
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Linear Regression</h1>
          <p className="text-neutral-700 mt-2">
            Understanding the fundamentals of linear regression and its implementation
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models">All Models</Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/polynomial-regression">
              Next: Polynomial Regression <ArrowRight className="ml-2 h-4 w-4" />
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
            <LineChart className="h-4 w-4" />
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
              <CardTitle className="text-neutral-900">What is Linear Regression?</CardTitle>
              <CardDescription className="text-neutral-600">
                A fundamental supervised learning algorithm for predicting continuous values
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Linear regression is one of the simplest and most widely used statistical models for predictive
                analysis. It attempts to model the relationship between a dependent variable and one or more independent
                variables by fitting a linear equation to the observed data.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">The Linear Equation</h3>
                <p className="font-mono text-center my-4 text-neutral-900">y = mx + b</p>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">y</strong>: The dependent variable (what we're trying to
                    predict)
                  </li>
                  <li>
                    <strong className="text-neutral-900">x</strong>: The independent variable (our input feature)
                  </li>
                  <li>
                    <strong className="text-neutral-900">m</strong>: The slope (how much y changes when x changes)
                  </li>
                  <li>
                    <strong className="text-neutral-900">b</strong>: The y-intercept (the value of y when x = 0)
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">How Linear Regression Works</h3>
              <ol className="list-decimal list-inside space-y-4 text-neutral-700">
                <li>
                  <strong className="text-neutral-900">Collect data</strong>: Gather pairs of input (x) and output (y)
                  values.
                </li>
                <li>
                  <strong className="text-neutral-900">Find the best-fitting line</strong>: Use a method called
                  "Ordinary Least Squares" to find the line that minimizes the sum of squared differences between
                  observed and predicted values.
                </li>
                <li>
                  <strong className="text-neutral-900">Evaluate the model</strong>: Assess how well the line fits the
                  data using metrics like R-squared or Mean Squared Error.
                </li>
                <li>
                  <strong className="text-neutral-900">Make predictions</strong>: Use the fitted line to predict y
                  values for new x inputs.
                </li>
              </ol>

              <div className="bg-neutral-100 p-4 rounded-lg mt-6">
                <h3 className="font-medium text-neutral-900 mb-2">Cost Function: Mean Squared Error</h3>
                <p className="text-neutral-700">
                  Linear regression finds the best line by minimizing the Mean Squared Error (MSE):
                </p>
                <p className="font-mono text-center my-4 text-neutral-900">MSE = (1/n) * Σ(y_actual - y_predicted)²</p>
                <p className="text-neutral-700">
                  This measures the average squared difference between the actual values and the predicted values.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Applications of Linear Regression</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-neutral-700">
                Linear regression is used in various fields for prediction and analysis:
              </p>
              <ul className="list-disc list-inside space-y-2 text-neutral-700">
                <li>
                  <strong className="text-neutral-900">Economics</strong>: Predicting sales, pricing, and economic
                  trends
                </li>
                <li>
                  <strong className="text-neutral-900">Finance</strong>: Risk assessment, stock price prediction
                </li>
                <li>
                  <strong className="text-neutral-900">Healthcare</strong>: Predicting patient outcomes based on
                  treatment variables
                </li>
                <li>
                  <strong className="text-neutral-900">Real Estate</strong>: Estimating property values based on
                  features
                </li>
                <li>
                  <strong className="text-neutral-900">Marketing</strong>: Analyzing the relationship between
                  advertising spend and sales
                </li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualization" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Interactive Linear Regression Model</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how they affect the regression line
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="Linear Regression Visualization"
                parameters={[
                  {
                    name: "slope",
                    min: -2,
                    max: 2,
                    step: 0.1,
                    default: 0.5,
                    label: "Slope (m)",
                  },
                  {
                    name: "intercept",
                    min: 0,
                    max: 5,
                    step: 0.1,
                    default: 2,
                    label: "Y-Intercept (b)",
                  },
                  {
                    name: "noise",
                    min: 0,
                    max: 2,
                    step: 0.1,
                    default: 0.5,
                    label: "Noise Level",
                  },
                ]}
                renderVisualization={renderLinearRegression}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Slope (m)</h3>
                <p className="text-neutral-700">
                  The slope determines how steep the line is. A positive slope means the line goes up as x increases,
                  while a negative slope means the line goes down. The steeper the slope, the more y changes for each
                  unit change in x.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Y-Intercept (b)</h3>
                <p className="text-neutral-700">
                  The y-intercept is where the line crosses the y-axis (when x = 0). It represents the baseline value of
                  y when none of the x factors are present.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Noise Level</h3>
                <p className="text-neutral-700">
                  In real-world data, there's always some randomness or "noise" that can't be explained by the model.
                  Increasing the noise level shows how random variation affects the data points and makes it harder to
                  fit a perfect line.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">Linear Regression Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement linear regression using Python and scikit-learn. Execute
                each cell to see the results.
              </p>
            </div>

            <div className="space-y-6">
              <NotebookCell
                cellId="cell0"
                executionCount={1}
                initialCode="import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set random seed for reproducibility
np.random.seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 1: Generate sample data</p>
                <p>First, let's create some synthetic data with a known relationship (y = 3x + 4 + noise).</p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode="# Generate sample data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Print model parameters
print(f'Intercept: {model.intercept_[0]:.2f}')
print(f'Slope: {model.coef_[0][0]:.2f}')
print(f'R² score: {model.score(X, y):.3f}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Visualize the data and regression line</p>
                <p>Now let's plot our data points and the fitted regression line.</p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Make predictions for plotting
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7)
plt.plot(X_new, y_pred, 'k-', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.grid(True)
plt.show()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Make predictions with the model</p>
                <p>Let's use our trained model to make predictions for new data points.</p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Make predictions for new data
X_test = np.array([[0], [2]])
y_pred = model.predict(X_test)
print('Predictions for new data points:')
print(y_pred.flatten())"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of linear regression:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>Change the data generation parameters</li>
                  <li>Add more features for multiple linear regression</li>
                  <li>Implement polynomial regression by adding polynomial features</li>
                  <li>Split the data into training and testing sets</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
