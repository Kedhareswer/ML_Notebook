"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, ArrowRight, BookOpen, BarChart } from "lucide-react"
import Link from "next/link"
import ModelVisualization from "@/components/model-visualization"

export default function LogisticRegressionPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  // Logistic Regression visualization function
  const renderLogisticRegression = (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => {
    const { slope, intercept, noise } = params

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Set up coordinate system
    const margin = 40
    const plotWidth = width - 2 * margin
    const plotHeight = height - 2 * margin

    // Draw axes
    ctx.strokeStyle = "#666"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(margin, margin)
    ctx.lineTo(margin, height - margin)
    ctx.lineTo(width - margin, height - margin)
    ctx.stroke()

    // Add axis labels
    ctx.fillStyle = "#666"
    ctx.font = "12px Arial"
    ctx.textAlign = "center"
    ctx.fillText("Feature X", width / 2, height - 10)
    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText("Probability", 0, 0)
    ctx.restore()

    // Generate data points
    const numPoints = 50
    const points = []

    for (let i = 0; i < numPoints; i++) {
      const x = margin + (i / (numPoints - 1)) * plotWidth

      // Convert to domain coordinates (0 to 1)
      const xDomain = i / (numPoints - 1)

      // Calculate logistic function: 1 / (1 + e^-(slope*x + intercept))
      const z = slope * (xDomain * 10 - 5) + intercept
      const probability = 1 / (1 + Math.exp(-z))

      // Add some noise
      const noiseFactor = (Math.random() - 0.5) * noise * 0.3
      const y = height - margin - probability * plotHeight + noiseFactor * plotHeight

      points.push({ x, y, probability })
    }

    // Draw logistic curve
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.beginPath()
    points.forEach((point, i) => {
      if (i === 0) {
        ctx.moveTo(point.x, point.y)
      } else {
        ctx.lineTo(point.x, point.y)
      }
    })
    ctx.stroke()

    // Draw decision boundary at probability = 0.5
    const decisionY = height - margin - 0.5 * plotHeight
    ctx.strokeStyle = "#ff0000"
    ctx.setLineDash([5, 3])
    ctx.beginPath()
    ctx.moveTo(margin, decisionY)
    ctx.lineTo(width - margin, decisionY)
    ctx.stroke()
    ctx.setLineDash([])

    // Draw data points
    const classPoints = []
    for (let i = 0; i < 40; i++) {
      const xDomain = Math.random()
      const x = margin + xDomain * plotWidth

      // Calculate true probability
      const z = slope * (xDomain * 10 - 5) + intercept
      const probability = 1 / (1 + Math.exp(-z))

      // Determine class based on probability
      const classValue = Math.random() < probability ? 1 : 0

      // Add some noise to y position for visualization
      const yNoise = (Math.random() - 0.5) * noise * 0.2 * plotHeight
      const y =
        classValue === 1 ? height - margin - plotHeight * 0.8 + yNoise : height - margin - plotHeight * 0.2 + yNoise

      classPoints.push({ x, y, class: classValue })
    }

    // Draw class points
    classPoints.forEach((point) => {
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

    // Add labels
    ctx.fillStyle = "#000"
    ctx.font = "14px Arial"
    ctx.textAlign = "left"
    ctx.fillText("Class 1", width - margin - 80, height - margin - plotHeight * 0.8)
    ctx.fillText("Class 0", width - margin - 80, height - margin - plotHeight * 0.2)

    // Add decision boundary label
    ctx.fillStyle = "#ff0000"
    ctx.textAlign = "right"
    ctx.fillText("Decision Boundary (p=0.5)", width - margin - 10, decisionY - 10)
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Accuracy: 0.8533
          <br />
          Precision: 0.8421
          <br />
          Recall: 0.8889
          <br />
          F1 Score: 0.8649
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">Logistic Regression Decision Boundary:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">Decision boundary visualization</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Best parameters: {"{"}'C': 1.0, 'penalty': 'l2'{"}"}
          <br />
          Best cross-validation score: 0.8667
          <br />
          Test accuracy with best parameters: 0.8800
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Logistic Regression</h1>
          <p className="text-neutral-700 mt-2">
            Understanding logistic regression for binary and multi-class classification
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models/classification">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back to Classification
            </Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/svm">
              Next: SVMs <ArrowRight className="ml-2 h-4 w-4" />
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
            <BarChart className="h-4 w-4" />
            <span>Visualization</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="explanation" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">What is Logistic Regression?</CardTitle>
              <CardDescription className="text-neutral-600">
                A statistical method for binary and multi-class classification
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Despite its name, logistic regression is a classification algorithm, not a regression algorithm. It's
                used to predict the probability that an instance belongs to a particular class. If the probability is
                greater than a threshold (typically 0.5), the model predicts that class.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">Key Concepts in Logistic Regression</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Logistic Function (Sigmoid)</strong>: Transforms linear
                    predictions to probabilities between 0 and 1
                  </li>
                  <li>
                    <strong className="text-neutral-900">Decision Boundary</strong>: The threshold that separates
                    different classes
                  </li>
                  <li>
                    <strong className="text-neutral-900">Maximum Likelihood Estimation</strong>: The method used to find
                    the best coefficients
                  </li>
                  <li>
                    <strong className="text-neutral-900">Regularization</strong>: Techniques to prevent overfitting (L1
                    and L2)
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">How Logistic Regression Works</h3>
              <p className="text-neutral-700 mb-4">
                Logistic regression uses the logistic function to model the probability of a certain class:
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg text-center">
                <p className="font-mono text-neutral-900">P(y=1) = 1 / (1 + e^-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ))</p>
              </div>

              <p className="text-neutral-700 mt-4">Where:</p>
              <ul className="list-disc list-inside space-y-1 text-neutral-700">
                <li>P(y=1) is the probability that the instance belongs to class 1</li>
                <li>β₀, β₁, ..., βₙ are the model parameters (coefficients)</li>
                <li>x₁, x₂, ..., xₙ are the feature values</li>
              </ul>

              <p className="text-neutral-700 mt-4">
                The model makes a prediction based on whether the calculated probability is above or below a threshold
                (typically 0.5):
              </p>
              <ul className="list-disc list-inside space-y-1 text-neutral-700">
                <li>If P(y=1) ≥ 0.5, predict class 1</li>
                <li>If P(y=1) &lt; 0.5, predict class 0</li>
              </ul>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Types of Logistic Regression</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-neutral-900">Binary Logistic Regression</h3>
                  <p className="text-neutral-700">
                    Used when the target variable has two possible outcomes (e.g., spam/not spam, disease/no disease).
                    This is the most common form of logistic regression.
                  </p>
                </div>

                <div>
                  <h3 className="font-medium text-neutral-900">Multinomial Logistic Regression</h3>
                  <p className="text-neutral-700">
                    Used when the target variable has three or more unordered categories (e.g., predicting types of
                    cuisine: Italian, Chinese, Mexican).
                  </p>
                </div>

                <div>
                  <h3 className="font-medium text-neutral-900">Ordinal Logistic Regression</h3>
                  <p className="text-neutral-700">
                    Used when the target variable has three or more ordered categories (e.g., movie ratings from 1 to 5
                    stars).
                  </p>
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Common Use Cases</h3>
                <ul className="list-disc list-inside space-y-1 text-neutral-700">
                  <li>Email spam detection</li>
                  <li>Disease diagnosis</li>
                  <li>Credit risk assessment</li>
                  <li>Customer churn prediction</li>
                  <li>Marketing campaign response prediction</li>
                </ul>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Advantages and Limitations</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Advantages</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Simple to implement and interpret</li>
                      <li>Efficient training process</li>
                      <li>Less prone to overfitting in high-dimensional spaces</li>
                      <li>Outputs well-calibrated probabilities</li>
                      <li>Works well for linearly separable classes</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Limitations</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Assumes linear relationship between features and log-odds</li>
                      <li>May underperform with complex non-linear relationships</li>
                      <li>Sensitive to outliers</li>
                      <li>Requires feature engineering for best results</li>
                      <li>Assumes independence of features</li>
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
              <CardTitle className="text-neutral-900">Interactive Logistic Regression Model</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how they affect the logistic regression decision boundary
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="Logistic Regression Visualization"
                parameters={[
                  {
                    name: "slope",
                    min: -5,
                    max: 5,
                    step: 0.1,
                    default: 1,
                    label: "Slope (β₁)",
                  },
                  {
                    name: "intercept",
                    min: -5,
                    max: 5,
                    step: 0.1,
                    default: 0,
                    label: "Intercept (β₀)",
                  },
                  {
                    name: "noise",
                    min: 0,
                    max: 1,
                    step: 0.05,
                    default: 0.3,
                    label: "Data Noise",
                  },
                ]}
                renderVisualization={renderLogisticRegression}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Slope (β₁)</h3>
                <p className="text-neutral-700">
                  The slope parameter determines how quickly the probability changes as the feature value increases. A
                  higher absolute value creates a steeper curve and a more abrupt transition between classes. The sign
                  of the slope determines the direction of the relationship.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Intercept (β₀)</h3>
                <p className="text-neutral-700">
                  The intercept shifts the entire logistic curve left or right. It determines the probability when the
                  feature value is zero. Changing the intercept moves the decision boundary without changing its slope.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Data Noise</h3>
                <p className="text-neutral-700">
                  This parameter simulates noise in the data. Higher values create more overlap between classes, making
                  the classification task more difficult. This demonstrates how logistic regression performs with
                  increasingly ambiguous data.
                </p>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Interpreting the Visualization</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    The <strong>S-shaped curve</strong> represents the logistic function, showing how probability
                    changes with the feature value
                  </li>
                  <li>
                    The <strong>red horizontal line</strong> at 0.5 probability is the decision boundary
                  </li>
                  <li>
                    <strong>Black points</strong> represent instances of class 1
                  </li>
                  <li>
                    <strong>White points</strong> represent instances of class 0
                  </li>
                  <li>Points that appear on the "wrong" side of the boundary are misclassifications</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
