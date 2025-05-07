"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, ArrowRight, BookOpen, Code, BarChart } from "lucide-react"
import Link from "next/link"
import NotebookCell from "@/components/notebook-cell"
import ModelVisualization from "@/components/model-visualization"

export default function NaiveBayesPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

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

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Gaussian Naive Bayes Accuracy: 0.8400
          <br />
          Multinomial Naive Bayes Accuracy: 0.7600
          <br />
          Bernoulli Naive Bayes Accuracy: 0.7200
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div>
          <div className="font-mono text-sm mb-2">Naive Bayes Decision Boundaries:</div>
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">Naive Bayes decision boundaries plot</p>
          </div>
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Accuracy on test set: 0.8400
          <br />
          Precision: 0.8182
          <br />
          Recall: 0.9000
          <br />
          F1 Score: 0.8571
          <br />
          Log Loss: 0.4306
        </div>
      )
    }

    return "Executed successfully"
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

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Variance (σ²)</h3>
                <p className="text-neutral-700">
                  This parameter controls the spread of the probability distributions for each class. A smaller variance
                  creates narrower, more peaked distributions, while a larger variance creates wider, flatter
                  distributions. In Gaussian Naive Bayes, each feature is assumed to follow a normal distribution with a
                  class-specific mean and variance.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Prior Ratio (P(y=2)/P(y=1))</h3>
                <p className="text-neutral-700">
                  This parameter represents the ratio of prior probabilities between the two classes. A value of 1 means
                  both classes are equally likely a priori. Values greater than 1 mean class 2 is more likely, while
                  values less than 1 mean class 1 is more likely. Prior probabilities reflect our belief about class
                  distribution before seeing any features.
                </p>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Interpreting the Visualization</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    The <strong>blue curve</strong> represents the probability density function for class 1
                  </li>
                  <li>
                    The <strong>red curve</strong> represents the probability density function for class 2
                  </li>
                  <li>
                    The <strong>vertical dashed line</strong> is the decision boundary where the posterior probabilities
                    are equal
                  </li>
                  <li>Points to the left of the boundary are classified as class 1, points to the right as class 2</li>
                  <li>Notice how changing the variance affects the shape of the distributions</li>
                  <li>
                    Changing the prior ratio shifts the decision boundary, reflecting our prior belief about class
                    probabilities
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">Naive Bayes Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement different types of Naive Bayes classifiers using Python and
                scikit-learn. Execute each cell to see the results.
              </p>
            </div>

            <div className="space-y-6">
              <NotebookCell
                cellId="cell0"
                executionCount={1}
                initialCode="import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">
                  Step 1: Compare different types of Naive Bayes classifiers
                </p>
                <p>Let's create a synthetic dataset and compare the performance of different Naive Bayes variants.</p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode="# Generate a synthetic dataset
X, y = make_classification(
    n_samples=200, 
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Standardize features for Gaussian NB
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# For Multinomial and Bernoulli NB, we need non-negative features
# We'll use a simple transformation for demonstration
X_train_nonneg = X_train - X_train.min(axis=0)
X_test_nonneg = X_test - X_train.min(axis=0)

# For Bernoulli NB, we'll binarize the features
X_train_binary = (X_train_nonneg > X_train_nonneg.mean(axis=0)).astype(int)
X_test_binary = (X_test_nonneg > X_train_nonneg.mean(axis=0)).astype(int)

# Train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
gnb_pred = gnb.predict(X_test_scaled)
gnb_accuracy = accuracy_score(y_test, gnb_pred)

# Train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train_nonneg, y_train)
mnb_pred = mnb.predict(X_test_nonneg)
mnb_accuracy = accuracy_score(y_test, mnb_pred)

# Train Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_binary, y_train)
bnb_pred = bnb.predict(X_test_binary)
bnb_accuracy = accuracy_score(y_test, bnb_pred)

print(f'Gaussian Naive Bayes Accuracy: {gnb_accuracy:.4f}')
print(f'Multinomial Naive Bayes Accuracy: {mnb_accuracy:.4f}')
print(f'Bernoulli Naive Bayes Accuracy: {bnb_accuracy:.4f}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">
                  Step 2: Visualize decision boundaries for Gaussian Naive Bayes
                </p>
                <p>Let's create a 2D dataset to visualize how Gaussian Naive Bayes creates decision boundaries.</p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Generate a 2D dataset for visualization
X_2d, y_2d = make_classification(
    n_samples=300, 
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

# Split the data
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_2d, test_size=0.25, random_state=42
)

# Train Gaussian Naive Bayes
gnb_2d = GaussianNB()
gnb_2d.fit(X_train_2d, y_train_2d)

# Function to plot decision boundaries
def plot_decision_boundary(X, y, model, title):
    # Set up the figure
    plt.figure(figsize=(10, 6))
    
    # Determine plot boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict using the model
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Plot decision boundary
plot_decision_boundary(X_2d, y_2d, gnb_2d, 'Gaussian Naive Bayes Decision Boundary')

# Calculate and print accuracy
gnb_2d_accuracy = accuracy_score(y_test_2d, gnb_2d.predict(X_test_2d))
print(f'Gaussian Naive Bayes Accuracy on 2D data: {gnb_2d_accuracy:.4f}')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Detailed evaluation of Gaussian Naive Bayes</p>
                <p>Let's perform a more detailed evaluation of the Gaussian Naive Bayes model.</p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Train Gaussian Naive Bayes with prior probabilities
gnb_with_prior = GaussianNB(priors=[0.3, 0.7])  # Specifying class priors
gnb_with_prior.fit(X_train_scaled, y_train)

# Make predictions
y_pred = gnb_with_prior.predict(X_test_scaled)
y_prob = gnb_with_prior.predict_proba(X_test_scaled)  # Probability estimates

# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
logloss = log_loss(y_test, y_prob)

# Print metrics
print(f'Accuracy on test set: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Log Loss: {logloss:.4f}')

# Examine the learned parameters
print('\nLearned Parameters:')
print(f'Class Priors: {gnb_with_prior.class_prior_}')
print(f'Feature Means per Class:')
for i, class_mean in enumerate(gnb_with_prior.theta_):
    print(f'Class {i}: {class_mean[:3]}...')  # Show first 3 features
print(f'Feature Variances per Class:')
for i, class_var in enumerate(gnb_with_prior.var_):
    print(f'Class {i}: {class_var[:3]}...')  # Show first 3 features"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of Naive Bayes:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>Try different prior probabilities and observe their effect on the decision boundary</li>
                  <li>Implement text classification using Multinomial Naive Bayes</li>
                  <li>Experiment with feature selection to improve model performance</li>
                  <li>Compare Naive Bayes with other classification algorithms on the same dataset</li>
                  <li>Implement Laplace smoothing to handle the zero frequency problem</li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
