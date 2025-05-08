"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, ArrowRight, BookOpen, BarChart } from "lucide-react"
import Link from "next/link"
import ModelVisualization from "@/components/model-visualization"

export default function NaiveBayesPage() {
  const [activeTab, setActiveTab] = useState("explanation")

  // Naive Bayes visualization function
  const renderNaiveBayes = (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => {
    const { variance, priorRatio } = params

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
    ctx.moveTo(margin, height - margin)
    ctx.lineTo(width - margin, height - margin)
    ctx.moveTo(margin, height - margin)
    ctx.lineTo(margin, margin)
    ctx.stroke()

    // Add axis labels
    ctx.fillStyle = "#666"
    ctx.font = "12px Arial"
    ctx.textAlign = "center"
    ctx.fillText("Feature Value", width / 2, height - 10)
    ctx.save()
    ctx.translate(15, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText("Probability Density", 0, 0)
    ctx.restore()

    // Define class means and calculate standard deviation from variance
    const mean1 = width / 2 - plotWidth / 4
    const mean2 = width / 2 + plotWidth / 4
    const stdDev = Math.sqrt(variance) * 30

    // Calculate prior probabilities based on priorRatio
    const prior1 = 1 / (1 + priorRatio)
    const prior2 = priorRatio / (1 + priorRatio)

    // Function to calculate Gaussian probability density
    const gaussian = (x: number, mean: number, stdDev: number) => {
      return (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2))
    }

    // Draw probability density functions
    const drawPDF = (mean: number, stdDev: number, color: string, prior: number) => {
      const scaleFactor = plotHeight * 0.8 * prior * 2 // Scale to fit in plot area

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()

      for (let x = margin; x <= width - margin; x += 2) {
        const y = height - margin - gaussian(x, mean, stdDev) * scaleFactor
        if (x === margin) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      }

      ctx.stroke()

      // Fill area under curve
      ctx.fillStyle = color.replace(")", ", 0.2)")
      ctx.beginPath()
      ctx.moveTo(margin, height - margin)

      for (let x = margin; x <= width - margin; x += 2) {
        const y = height - margin - gaussian(x, mean, stdDev) * scaleFactor
        ctx.lineTo(x, y)
      }

      ctx.lineTo(width - margin, height - margin)
      ctx.closePath()
      ctx.fill()

      // Mark mean with vertical line
      ctx.strokeStyle = color
      ctx.setLineDash([5, 3])
      ctx.beginPath()
      ctx.moveTo(mean, height - margin)
      ctx.lineTo(mean, margin)
      ctx.stroke()
      ctx.setLineDash([])

      // Label mean
      ctx.fillStyle = color.replace(", 0.2)", ")")
      ctx.textAlign = "center"
      ctx.fillText(`μ = ${((mean - width / 2) / (plotWidth / 4)).toFixed(1)}`, mean, margin - 10)
    }

    // Draw PDFs for both classes
    drawPDF(mean1, stdDev, "rgba(0, 0, 255, 1)", prior1)
    drawPDF(mean2, stdDev, "rgba(255, 0, 0, 1)", prior2)

    // Calculate and draw decision boundary
    // In Naive Bayes, the decision boundary is where posterior probabilities are equal
    // For Gaussian Naive Bayes with equal variance, this is a vertical line
    const decisionBoundary = (mean1 * prior1 + mean2 * prior2) / (prior1 + prior2)

    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(decisionBoundary, height - margin)
    ctx.lineTo(decisionBoundary, margin)
    ctx.stroke()
    ctx.setLineDash([])

    // Label decision boundary
    ctx.fillStyle = "#000"
    ctx.textAlign = "center"
    ctx.fillText("Decision Boundary", decisionBoundary, height - margin + 20)

    // Add legend
    const legendX = width - margin - 120
    const legendY = margin + 20

    ctx.fillStyle = "rgba(0, 0, 255, 1)"
    ctx.fillRect(legendX, legendY, 15, 15)
    ctx.fillStyle = "#000"
    ctx.textAlign = "left"
    ctx.fillText(`Class 1 (Prior: ${prior1.toFixed(2)})`, legendX + 20, legendY + 12)

    ctx.fillStyle = "rgba(255, 0, 0, 1)"
    ctx.fillRect(legendX, legendY + 25, 15, 15)
    ctx.fillStyle = "#000"
    ctx.fillText(`Class 2 (Prior: ${prior2.toFixed(2)})`, legendX + 20, legendY + 37)

    // Add variance label
    ctx.fillStyle = "#000"
    ctx.textAlign = "left"
    ctx.fillText(`σ² = ${variance.toFixed(2)}`, margin + 10, margin + 20)
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Naive Bayes</h1>
          <p className="text-neutral-700 mt-2">
            Understanding Naive Bayes classifiers and their probabilistic foundation
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models/classification">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back to Classification
            </Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/gradient-boosting">
              Next: Gradient Boosting <ArrowRight className="ml-2 h-4 w-4" />
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
              <CardTitle className="text-neutral-900">What is Naive Bayes?</CardTitle>
              <CardDescription className="text-neutral-600">
                A family of probabilistic classifiers based on Bayes' theorem
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Naive Bayes is a family of simple probabilistic classifiers based on applying Bayes' theorem with strong
                (naive) independence assumptions between the features. Despite its simplicity, Naive Bayes often
                performs surprisingly well and is widely used for text classification, spam filtering, sentiment
                analysis, and recommendation systems.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">Key Concepts in Naive Bayes</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Bayes' Theorem</strong>: Calculates the probability of a
                    hypothesis given prior knowledge
                  </li>
                  <li>
                    <strong className="text-neutral-900">Conditional Independence</strong>: Features are assumed to be
                    independent given the class
                  </li>
                  <li>
                    <strong className="text-neutral-900">Prior Probability</strong>: Initial belief about class
                    distribution before seeing the data
                  </li>
                  <li>
                    <strong className="text-neutral-900">Likelihood</strong>: Probability of observing the features
                    given the class
                  </li>
                  <li>
                    <strong className="text-neutral-900">Posterior Probability</strong>: Updated belief about class
                    membership after considering the evidence
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">Bayes' Theorem</h3>
              <p className="text-neutral-700 mb-4">
                At the heart of Naive Bayes is Bayes' theorem, which provides a way to calculate the probability of a
                hypothesis given prior knowledge:
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg text-center">
                <p className="font-mono text-neutral-900">P(y|X) = P(X|y) × P(y) / P(X)</p>
              </div>

              <p className="text-neutral-700 mt-4">Where:</p>
              <ul className="list-disc list-inside space-y-1 text-neutral-700">
                <li>P(y|X) is the posterior probability of class y given features X</li>
                <li>P(X|y) is the likelihood of features X given class y</li>
                <li>P(y) is the prior probability of class y</li>
                <li>P(X) is the evidence (probability of features X)</li>
              </ul>

              <p className="text-neutral-700 mt-4">
                The "naive" assumption is that features are conditionally independent given the class:
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg text-center">
                <p className="font-mono text-neutral-900">P(X|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)</p>
              </div>

              <p className="text-neutral-700 mt-4">
                This simplifies the calculation and makes Naive Bayes computationally efficient.
              </p>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Types of Naive Bayes Classifiers</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-neutral-900">Gaussian Naive Bayes</h3>
                  <p className="text-neutral-700">
                    Used for continuous data, assuming features follow a normal distribution. For each class y and
                    feature x, we calculate the mean and variance of x for all samples belonging to class y. The
                    likelihood is then calculated using the normal probability density function.
                  </p>
                  <div className="bg-neutral-100 p-2 rounded-lg text-center mt-2">
                    <p className="font-mono text-neutral-900 text-sm">P(x|y) = (1/√(2πσ²)) × e^(-(x-μ)²/(2σ²))</p>
                  </div>
                </div>

                <div>
                  <h3 className="font-medium text-neutral-900">Multinomial Naive Bayes</h3>
                  <p className="text-neutral-700">
                    Suitable for discrete data, particularly for text classification where features represent word
                    counts or frequencies. It assumes features follow a multinomial distribution, which is appropriate
                    for features representing counts or frequencies.
                  </p>
                </div>

                <div>
                  <h3 className="font-medium text-neutral-900">Bernoulli Naive Bayes</h3>
                  <p className="text-neutral-700">
                    Used when features are binary (0/1), such as presence or absence of a word in a document. It's
                    similar to Multinomial Naive Bayes but penalizes the non-occurrence of a feature that is an
                    indicator of a class.
                  </p>
                </div>

                <div>
                  <h3 className="font-medium text-neutral-900">Complement Naive Bayes</h3>
                  <p className="text-neutral-700">
                    A variant of Multinomial Naive Bayes designed to address the imbalanced data problem. It uses
                    statistics from the complement of each class to compute the model's weights, which helps when
                    dealing with imbalanced datasets.
                  </p>
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Common Use Cases</h3>
                <ul className="list-disc list-inside space-y-1 text-neutral-700">
                  <li>Text classification (spam detection, sentiment analysis)</li>
                  <li>Document categorization</li>
                  <li>Email filtering</li>
                  <li>Medical diagnosis</li>
                  <li>Real-time prediction (due to its computational efficiency)</li>
                </ul>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Advantages and Limitations</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Advantages</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Simple and easy to implement</li>
                      <li>Fast training and prediction</li>
                      <li>Works well with small datasets</li>
                      <li>Handles high-dimensional data efficiently</li>
                      <li>Not sensitive to irrelevant features</li>
                      <li>Performs well even with the naive assumption</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Limitations</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Assumes feature independence (often unrealistic)</li>
                      <li>Can be outperformed by more sophisticated models</li>
                      <li>Sensitive to how input data is prepared</li>
                      <li>The "zero frequency" problem (solved by smoothing)</li>
                      <li>Less accurate for numerical features</li>
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
              <CardTitle className="text-neutral-900">Interactive Gaussian Naive Bayes Model</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how they affect the Naive Bayes decision boundary
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="Gaussian Naive Bayes Visualization"
                parameters={[
                  {
                    name: "variance",
                    min: 0.1,
                    max: 2,
                    step: 0.1,
                    default: 0.5,
                    label: "Variance (σ²)",
                  },
                  {
                    name: "priorRatio",
                    min: 0.1,
                    max: 5,
                    step: 0.1,
                    default: 1,
                    label: "Prior Ratio (P(y=2)/P(y=1))",
                  },
                ]}
                renderVisualization={renderNaiveBayes}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      <div className="flex justify-between items-center mt-12 border-t border-neutral-300 pt-6">
        <Button asChild variant="outline">
          <Link href="/models/logistic-regression">
            <ArrowLeft className="mr-2 h-4 w-4" /> Previous: Logistic Regression
          </Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/models/gradient-boosting">
            Next: Gradient Boosting <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </div>
  )
}
