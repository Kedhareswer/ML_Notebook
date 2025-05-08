import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowLeft } from "lucide-react"
import KNNVisualization from "@/components/knn-visualization"

export default function KNNPage() {
  return (
    <main className="flex min-h-screen flex-col p-4 md:p-24">
      <div className="mb-8">
        <Button variant="outline" asChild>
          <Link href="/models">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Models
          </Link>
        </Button>
      </div>

      <h1 className="text-4xl font-bold mb-4">K-Nearest Neighbors (KNN)</h1>
      <p className="text-xl mb-8">A simple, instance-based learning algorithm for classification and regression</p>

      <Tabs defaultValue="overview">
        <TabsList className="mb-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="interactive">Interactive Demo</TabsTrigger>
          <TabsTrigger value="math">Mathematical Foundation</TabsTrigger>
          <TabsTrigger value="applications">Applications</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <Card>
            <CardHeader>
              <CardTitle>What is K-Nearest Neighbors?</CardTitle>
              <CardDescription>Understanding the fundamentals of KNN</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="mb-4">
                K-Nearest Neighbors (KNN) is one of the simplest machine learning algorithms used for both
                classification and regression. It belongs to the family of instance-based, non-parametric learning
                algorithms.
              </p>
              <p className="mb-4">
                The core idea behind KNN is that similar data points tend to have similar outputs. For a new data point,
                the algorithm finds the K closest data points (neighbors) in the training set and uses their values to
                predict the output for the new point.
              </p>
              <h3 className="text-xl font-semibold mt-6 mb-2">Key Characteristics:</h3>
              <ul className="list-disc pl-5 space-y-2">
                <li>
                  <strong>Non-parametric:</strong> KNN doesn't make assumptions about the underlying data distribution.
                </li>
                <li>
                  <strong>Lazy learning:</strong> KNN doesn't build a model during training; it simply stores the
                  training data.
                </li>
                <li>
                  <strong>Instance-based:</strong> Predictions are made based on the similarity between instances.
                </li>
                <li>
                  <strong>Versatile:</strong> Can be used for both classification and regression tasks.
                </li>
              </ul>
              <h3 className="text-xl font-semibold mt-6 mb-2">How It Works:</h3>
              <ol className="list-decimal pl-5 space-y-2">
                <li>Calculate the distance between the new point and all points in the training data.</li>
                <li>Select the K nearest points based on the calculated distances.</li>
                <li>For classification: Assign the most common class among the K neighbors.</li>
                <li>For regression: Calculate the average (or weighted average) of the K neighbors' values.</li>
              </ol>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="interactive">
          <Card>
            <CardHeader>
              <CardTitle>Interactive KNN Visualization</CardTitle>
              <CardDescription>Experiment with different parameters to see how KNN works</CardDescription>
            </CardHeader>
            <CardContent>
              <KNNVisualization />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="math">
          <Card>
            <CardHeader>
              <CardTitle>Mathematical Foundation</CardTitle>
              <CardDescription>The equations and principles behind KNN</CardDescription>
            </CardHeader>
            <CardContent>
              <h3 className="text-xl font-semibold mb-4">Distance Metrics</h3>
              <p className="mb-4">
                KNN relies on distance metrics to determine the similarity between data points. The most common distance
                metrics include:
              </p>

              <h4 className="text-lg font-medium mt-4 mb-2">Euclidean Distance</h4>
              <p className="mb-2">The straight-line distance between two points in Euclidean space:</p>
              <div className="bg-gray-100 p-4 rounded-md mb-4">
                <p>d(x, y) = √(Σ(xᵢ - yᵢ)²)</p>
              </div>

              <h4 className="text-lg font-medium mt-4 mb-2">Manhattan Distance</h4>
              <p className="mb-2">The sum of absolute differences between coordinates:</p>
              <div className="bg-gray-100 p-4 rounded-md mb-4">
                <p>d(x, y) = Σ|xᵢ - yᵢ|</p>
              </div>

              <h4 className="text-lg font-medium mt-4 mb-2">Minkowski Distance</h4>
              <p className="mb-2">A generalization of Euclidean and Manhattan distances:</p>
              <div className="bg-gray-100 p-4 rounded-md mb-4">
                <p>d(x, y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)</p>
                <p>where p = 1 gives Manhattan distance and p = 2 gives Euclidean distance</p>
              </div>

              <h3 className="text-xl font-semibold mt-6 mb-4">Classification Decision Rule</h3>
              <p className="mb-2">For classification, KNN assigns the class based on majority voting:</p>
              <div className="bg-gray-100 p-4 rounded-md mb-4">
                <p>ŷ = argmax(Σ I(y_i = c))</p>
                <p>where I is the indicator function and the sum is over the K nearest neighbors</p>
              </div>

              <h3 className="text-xl font-semibold mt-6 mb-4">Regression Prediction</h3>
              <p className="mb-2">For regression, KNN calculates the average of the K nearest neighbors:</p>
              <div className="bg-gray-100 p-4 rounded-md mb-4">
                <p>ŷ = (1/K) Σ y_i</p>
                <p>where the sum is over the K nearest neighbors</p>
              </div>

              <h3 className="text-xl font-semibold mt-6 mb-4">Weighted KNN</h3>
              <p className="mb-2">In weighted KNN, closer neighbors have more influence on the prediction:</p>
              <div className="bg-gray-100 p-4 rounded-md mb-4">
                <p>ŷ = Σ(w_i * y_i) / Σw_i</p>
                <p>where w_i = 1/d(x, x_i)² for distance-weighted KNN</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="applications">
          <Card>
            <CardHeader>
              <CardTitle>Real-World Applications</CardTitle>
              <CardDescription>How KNN is used in practice</CardDescription>
            </CardHeader>
            <CardContent>
              <h3 className="text-xl font-semibold mb-4">Common Applications of KNN</h3>

              <div className="space-y-6">
                <div>
                  <h4 className="text-lg font-medium mb-2">Recommendation Systems</h4>
                  <p>
                    KNN is used in collaborative filtering to recommend products, movies, or music based on the
                    preferences of users with similar tastes. For example, if users who liked similar movies to you also
                    enjoyed a movie you haven't seen, that movie might be recommended to you.
                  </p>
                </div>

                <div>
                  <h4 className="text-lg font-medium mb-2">Image Recognition</h4>
                  <p>
                    KNN can be used for simple image classification tasks, where images are represented as feature
                    vectors and classified based on their similarity to labeled training images.
                  </p>
                </div>

                <div>
                  <h4 className="text-lg font-medium mb-2">Medical Diagnosis</h4>
                  <p>
                    KNN can help classify patients based on their symptoms and medical history, aiding in diagnosis by
                    comparing new patients to similar cases with known diagnoses.
                  </p>
                </div>

                <div>
                  <h4 className="text-lg font-medium mb-2">Credit Scoring</h4>
                  <p>
                    Financial institutions use KNN to assess credit risk by comparing new applicants to similar
                    customers with known repayment histories.
                  </p>
                </div>

                <div>
                  <h4 className="text-lg font-medium mb-2">Anomaly Detection</h4>
                  <p>
                    KNN can identify outliers or anomalies by finding data points that are far from their nearest
                    neighbors, which is useful in fraud detection and network security.
                  </p>
                </div>
              </div>

              <h3 className="text-xl font-semibold mt-8 mb-4">Strengths and Limitations</h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-medium mb-2">Strengths</h4>
                  <ul className="list-disc pl-5 space-y-2">
                    <li>Simple to understand and implement</li>
                    <li>No training phase (lazy learning)</li>
                    <li>Naturally handles multi-class problems</li>
                    <li>Can be effective for non-linear data</li>
                    <li>Adaptable to new training data</li>
                  </ul>
                </div>

                <div>
                  <h4 className="text-lg font-medium mb-2">Limitations</h4>
                  <ul className="list-disc pl-5 space-y-2">
                    <li>Computationally expensive for large datasets</li>
                    <li>Sensitive to irrelevant features</li>
                    <li>Requires feature scaling</li>
                    <li>Struggles with high-dimensional data (curse of dimensionality)</li>
                    <li>Optimal K value selection can be challenging</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </main>
  )
}
