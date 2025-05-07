"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowLeft, BookOpen } from "lucide-react"
import Link from "next/link"

export default function ModelComparisonPage() {
  const [activeTab, setActiveTab] = useState("overview")

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Model Comparison</h1>
          <p className="text-neutral-700 mt-2">
            Compare different machine learning models across various metrics and use cases
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models">
              <ArrowLeft className="mr-2 h-4 w-4" /> All Models
            </Link>
          </Button>
        </div>
      </div>

      <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab} className="space-y-8">
        <TabsList className="grid w-full grid-cols-1 bg-neutral-100 text-neutral-900">
          <TabsTrigger value="overview" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BookOpen className="h-4 w-4" />
            <span>Overview</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Comparing Machine Learning Models</CardTitle>
              <CardDescription className="text-neutral-600">
                Understanding the strengths and weaknesses of different models
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Choosing the right machine learning model for a specific task requires understanding the tradeoffs
                between different algorithms. Models vary in their complexity, interpretability, training requirements,
                and performance characteristics. This page provides a comprehensive comparison to help you select the
                most appropriate model for your use case.
              </p>

              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-neutral-100">
                      <th className="border border-neutral-300 px-4 py-2 text-left">Model Type</th>
                      <th className="border border-neutral-300 px-4 py-2 text-left">Strengths</th>
                      <th className="border border-neutral-300 px-4 py-2 text-left">Weaknesses</th>
                      <th className="border border-neutral-300 px-4 py-2 text-left">Best Use Cases</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border border-neutral-300 px-4 py-2 font-medium">Linear/Logistic Regression</td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Simple and interpretable</li>
                          <li>Fast training and prediction</li>
                          <li>Works well with linearly separable data</li>
                          <li>Low variance</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Limited expressiveness</li>
                          <li>Cannot capture non-linear relationships</li>
                          <li>Sensitive to outliers</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Baseline models</li>
                          <li>When interpretability is crucial</li>
                          <li>Small datasets</li>
                          <li>Linear relationships</li>
                        </ul>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-neutral-300 px-4 py-2 font-medium">Decision Trees</td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Highly interpretable</li>
                          <li>Handles non-linear relationships</li>
                          <li>No feature scaling required</li>
                          <li>Handles mixed data types</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Prone to overfitting</li>
                          <li>High variance</li>
                          <li>Unstable (small changes in data can cause large changes in tree)</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>When interpretability is needed</li>
                          <li>Feature importance analysis</li>
                          <li>Rule-based decision making</li>
                        </ul>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-neutral-300 px-4 py-2 font-medium">Random Forests</td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Robust against overfitting</li>
                          <li>Handles non-linear relationships</li>
                          <li>Provides feature importance</li>
                          <li>Works well with high-dimensional data</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Less interpretable than single trees</li>
                          <li>Computationally intensive</li>
                          <li>Slower prediction time</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>General-purpose classification/regression</li>
                          <li>When accuracy is more important than interpretability</li>
                          <li>Feature selection</li>
                        </ul>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-neutral-300 px-4 py-2 font-medium">Support Vector Machines</td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Effective in high-dimensional spaces</li>
                          <li>Versatile through different kernels</li>
                          <li>Memory efficient</li>
                          <li>Works well with clear margin of separation</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Not suitable for large datasets</li>
                          <li>Sensitive to feature scaling</li>
                          <li>Difficult to interpret</li>
                          <li>Requires careful parameter tuning</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Text classification</li>
                          <li>Image classification</li>
                          <li>When data has clear boundaries</li>
                        </ul>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-neutral-300 px-4 py-2 font-medium">Neural Networks</td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Can model extremely complex relationships</li>
                          <li>Highly flexible architecture</li>
                          <li>State-of-the-art performance on many tasks</li>
                          <li>Feature learning capability</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Requires large amounts of data</li>
                          <li>Computationally intensive</li>
                          <li>Difficult to interpret</li>
                          <li>Prone to overfitting without proper regularization</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Image and speech recognition</li>
                          <li>Natural language processing</li>
                          <li>Complex pattern recognition</li>
                          <li>When performance is paramount</li>
                        </ul>
                      </td>
                    </tr>
                    <tr>
                      <td className="border border-neutral-300 px-4 py-2 font-medium">Clustering Algorithms</td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Unsupervised learning (no labels needed)</li>
                          <li>Discovers hidden patterns</li>
                          <li>Useful for data exploration</li>
                          <li>Can handle various data types</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Results can be subjective</li>
                          <li>Difficult to evaluate</li>
                          <li>Sensitive to initial conditions</li>
                          <li>May find patterns that aren't meaningful</li>
                        </ul>
                      </td>
                      <td className="border border-neutral-300 px-4 py-2">
                        <ul className="list-disc list-inside text-sm">
                          <li>Customer segmentation</li>
                          <li>Anomaly detection</li>
                          <li>Document clustering</li>
                          <li>Exploratory data analysis</li>
                        </ul>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Model Selection Guidelines</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-neutral-700">
                When selecting a machine learning model, consider the following factors:
              </p>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h3 className="text-lg font-medium text-neutral-800 mb-2">Data Characteristics</h3>
                  <ul className="list-disc list-inside space-y-1 text-neutral-700">
                    <li>
                      <strong>Size</strong>: Large datasets can benefit from complex models like neural networks
                    </li>
                    <li>
                      <strong>Dimensionality</strong>: High-dimensional data works well with tree-based models and SVMs
                    </li>
                    <li>
                      <strong>Noise</strong>: Ensemble methods like Random Forests handle noisy data better
                    </li>
                    <li>
                      <strong>Structure</strong>: Consider if relationships are linear or non-linear
                    </li>
                  </ul>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h3 className="text-lg font-medium text-neutral-800 mb-2">Problem Requirements</h3>
                  <ul className="list-disc list-inside space-y-1 text-neutral-700">
                    <li>
                      <strong>Interpretability</strong>: Linear models and decision trees offer better interpretability
                    </li>
                    <li>
                      <strong>Performance</strong>: Neural networks and ensemble methods often provide higher accuracy
                    </li>
                    <li>
                      <strong>Training time</strong>: Linear models train faster than complex models
                    </li>
                    <li>
                      <strong>Prediction speed</strong>: Consider inference time for real-time applications
                    </li>
                  </ul>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h3 className="text-lg font-medium text-neutral-800 mb-2">Practical Considerations</h3>
                  <ul className="list-disc list-inside space-y-1 text-neutral-700">
                    <li>
                      <strong>Computational resources</strong>: Complex models require more computing power
                    </li>
                    <li>
                      <strong>Maintenance</strong>: Simpler models are easier to maintain and update
                    </li>
                    <li>
                      <strong>Domain expertise</strong>: Some models benefit more from domain knowledge
                    </li>
                    <li>
                      <strong>Deployment environment</strong>: Consider where and how the model will be used
                    </li>
                  </ul>
                </div>
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <h3 className="text-lg font-medium text-neutral-800 mb-2">Best Practices</h3>
                  <ul className="list-disc list-inside space-y-1 text-neutral-700">
                    <li>
                      <strong>Start simple</strong>: Begin with simpler models as baselines
                    </li>
                    <li>
                      <strong>Iterate</strong>: Gradually increase complexity if needed
                    </li>
                    <li>
                      <strong>Ensemble</strong>: Combine multiple models for better performance
                    </li>
                    <li>
                      <strong>Cross-validate</strong>: Always validate models on multiple data splits
                    </li>
                    <li>
                      <strong>Monitor</strong>: Track model performance over time in production
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
