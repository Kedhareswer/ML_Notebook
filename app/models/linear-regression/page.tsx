"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import ModelVisualization from "@/components/model-visualization"
import { ArrowRight, BookOpen, LineChart, ArrowLeft } from "lucide-react"
import Link from "next/link"

export default function LinearRegressionPage() {
  const [activeTab, setActiveTab] = useState("explanation")

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

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-black">Linear Regression</h1>
          <p className="text-gray-700 mt-2">
            Understanding the fundamentals of linear regression and its implementation
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild className="border-gray-300 hover:bg-gray-100 hover:text-black">
            <Link href="/models/knn">
              <ArrowLeft className="mr-2 h-4 w-4" /> Previous: KNN
            </Link>
          </Button>
          <Button asChild variant="notebook" className="bg-black text-white hover:bg-gray-800">
            <Link href="/models/polynomial-regression">
              Next: Polynomial Regression <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      <Tabs defaultValue="explanation" value={activeTab} onValueChange={setActiveTab} className="space-y-8">
        <TabsList className="grid w-full grid-cols-2 bg-gray-100 text-black">
          <TabsTrigger value="explanation" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BookOpen className="h-4 w-4" />
            <span>Explanation</span>
          </TabsTrigger>
          <TabsTrigger value="visualization" className="flex items-center gap-2 data-[state=active]:bg-white">
            <LineChart className="h-4 w-4" />
            <span>Visualization</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="explanation" className="space-y-8">
          <Card className="border-gray-300 bg-white">
            <CardHeader>
              <CardTitle className="text-black">What is Linear Regression?</CardTitle>
              <CardDescription className="text-gray-600">
                A fundamental supervised learning algorithm for predicting continuous values
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-gray-700">
                Linear regression is one of the simplest and most widely used statistical models for predictive
                analysis. It attempts to model the relationship between a dependent variable and one or more independent
                variables by fitting a linear equation to the observed data.
              </p>

              <div className="bg-gray-100 p-4 rounded-lg">
                <h3 className="font-medium text-black mb-2">The Linear Equation</h3>
                <p className="font-mono text-center my-4 text-black">y = mx + b</p>
                <ul className="list-disc list-inside space-y-2 text-gray-700">
                  <li>
                    <strong className="text-black">y</strong>: The dependent variable (what we're trying to predict)
                  </li>
                  <li>
                    <strong className="text-black">x</strong>: The independent variable (our input feature)
                  </li>
                  <li>
                    <strong className="text-black">m</strong>: The slope (how much y changes when x changes)
                  </li>
                  <li>
                    <strong className="text-black">b</strong>: The y-intercept (the value of y when x = 0)
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-black mt-6">How Linear Regression Works</h3>
              <ol className="list-decimal list-inside space-y-4 text-gray-700">
                <li>
                  <strong className="text-black">Collect data</strong>: Gather pairs of input (x) and output (y) values.
                </li>
                <li>
                  <strong className="text-black">Find the best-fitting line</strong>: Use a method called "Ordinary
                  Least Squares" to find the line that minimizes the sum of squared differences between observed and
                  predicted values.
                </li>
                <li>
                  <strong className="text-black">Evaluate the model</strong>: Assess how well the line fits the data
                  using metrics like R-squared or Mean Squared Error.
                </li>
                <li>
                  <strong className="text-black">Make predictions</strong>: Use the fitted line to predict y values for
                  new x inputs.
                </li>
              </ol>

              <div className="bg-gray-100 p-4 rounded-lg mt-6">
                <h3 className="font-medium text-black mb-2">Cost Function: Mean Squared Error</h3>
                <p className="text-gray-700">
                  Linear regression finds the best line by minimizing the Mean Squared Error (MSE):
                </p>
                <p className="font-mono text-center my-4 text-black">MSE = (1/n) * Σ(y_actual - y_predicted)²</p>
                <p className="text-gray-700">
                  This measures the average squared difference between the actual values and the predicted values.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-gray-300 bg-white">
            <CardHeader>
              <CardTitle className="text-black">Applications of Linear Regression</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-gray-700">Linear regression is used in various fields for prediction and analysis:</p>
              <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li>
                  <strong className="text-black">Economics</strong>: Predicting sales, pricing, and economic trends
                </li>
                <li>
                  <strong className="text-black">Finance</strong>: Risk assessment, stock price prediction
                </li>
                <li>
                  <strong className="text-black">Healthcare</strong>: Predicting patient outcomes based on treatment
                  variables
                </li>
                <li>
                  <strong className="text-black">Real Estate</strong>: Estimating property values based on features
                </li>
                <li>
                  <strong className="text-black">Marketing</strong>: Analyzing the relationship between advertising
                  spend and sales
                </li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualization" className="space-y-8">
          <Card className="border-gray-300 bg-white">
            <CardHeader>
              <CardTitle className="text-black">Interactive Linear Regression Model</CardTitle>
              <CardDescription className="text-gray-600">
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

          <Card className="border-gray-300 bg-white">
            <CardHeader>
              <CardTitle className="text-black">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-black">Slope (m)</h3>
                <p className="text-gray-700">
                  The slope determines how steep the line is. A positive slope means the line goes up as x increases,
                  while a negative slope means the line goes down. The steeper the slope, the more y changes for each
                  unit change in x.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-black">Y-Intercept (b)</h3>
                <p className="text-gray-700">
                  The y-intercept is where the line crosses the y-axis (when x = 0). It represents the baseline value of
                  y when none of the x factors are present.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-black">Noise Level</h3>
                <p className="text-gray-700">
                  In real-world data, there's always some randomness or "noise" that can't be explained by the model.
                  Increasing the noise level shows how random variation affects the data points and makes it harder to
                  fit a perfect line.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
