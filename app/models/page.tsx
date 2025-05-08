import Link from "next/link"
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"
import {
  Brain,
  GitBranch,
  LineChart,
  Network,
  BarChart,
  PieChart,
  Layers,
  Workflow,
  Sigma,
  Microscope,
  Braces,
} from "lucide-react"

export default function ModelsPage() {
  const modelCategories = [
    {
      title: "Supervised Learning",
      description: "Models that learn from labeled training data",
      subcategories: [
        {
          title: "Regression Models",
          description: "Predict continuous values",
          models: [
            {
              title: "Linear Regression",
              description: "Learn how linear regression works and how to implement it",
              icon: <LineChart className="h-8 w-8 text-black" />,
              href: "/models/linear-regression",
            },
            {
              title: "Polynomial Regression",
              description: "Extend linear models to capture non-linear relationships",
              icon: <Sigma className="h-8 w-8 text-black" />,
              href: "/models/polynomial-regression",
            },
            {
              title: "Ridge & Lasso Regression",
              description: "Regularization techniques to prevent overfitting",
              icon: <Braces className="h-8 w-8 text-black" />,
              href: "/models/regularized-regression",
            },
          ],
        },
        {
          title: "Classification Models",
          description: "Predict categorical values",
          models: [
            {
              title: "Logistic Regression",
              description: "Understand the mathematics of logistic regression",
              icon: <Sigma className="h-8 w-8 text-black" />,
              href: "/models/logistic-regression",
            },
            {
              title: "Decision Trees",
              description: "Understand decision trees and their applications",
              icon: <GitBranch className="h-8 w-8 text-black" />,
              href: "/models/decision-trees",
            },
            {
              title: "Support Vector Machines",
              description: "Explore the mathematics behind SVMs",
              icon: <BarChart className="h-8 w-8 text-black" />,
              href: "/models/svm",
            },
            {
              title: "Random Forests",
              description: "Learn how ensemble methods improve performance",
              icon: <Layers className="h-8 w-8 text-black" />,
              href: "/models/random-forests",
            },
            {
              title: "K-Nearest Neighbors (KNN)",
              description: "A simple yet effective classification algorithm",
              icon: <Microscope className="h-8 w-8 text-black" />,
              href: "/models/knn",
            },
          ],
        },
      ],
    },
    {
      title: "Unsupervised Learning",
      description: "Models that find patterns in unlabeled data",
      subcategories: [
        {
          title: "Clustering Algorithms",
          description: "Group similar data points together",
          models: [
            {
              title: "K-Means Clustering",
              description: "Explore how K-means partitions data into clusters",
              icon: <PieChart className="h-8 w-8 text-black" />,
              href: "/models/kmeans",
            },
            {
              title: "Hierarchical Clustering",
              description: "Understand how hierarchical clustering works",
              icon: <GitBranch className="h-8 w-8 text-black" />,
              href: "/models/hierarchical-clustering",
            },
          ],
        },
        {
          title: "Dimensionality Reduction",
          description: "Reduce the number of features while preserving information",
          models: [
            {
              title: "Principal Component Analysis",
              description: "Learn how PCA transforms high-dimensional data",
              icon: <Workflow className="h-8 w-8 text-black" />,
              href: "/models/pca",
            },
          ],
        },
      ],
    },
    {
      title: "Neural Networks",
      description: "Deep learning models inspired by the human brain",
      subcategories: [
        {
          title: "Basic Neural Networks",
          description: "Understand the fundamentals of neural networks",
          models: [
            {
              title: "Multilayer Perceptron",
              description: "Learn about the building blocks of deep learning",
              icon: <Network className="h-8 w-8 text-black" />,
              href: "/models/mlp",
            },
          ],
        },
        {
          title: "Specialized Neural Networks",
          description: "Neural networks designed for specific data types",
          models: [
            {
              title: "Convolutional Neural Networks",
              description: "Visualize how CNNs process images",
              icon: <Brain className="h-8 w-8 text-black" />,
              href: "/models/cnn",
            },
            {
              title: "Recurrent Neural Networks",
              description: "See how RNNs handle sequential data",
              icon: <Network className="h-8 w-8 text-black" />,
              href: "/models/rnn",
            },
            {
              title: "Transformers",
              description: "Explore the architecture behind modern NLP models",
              icon: <Microscope className="h-8 w-8 text-black" />,
              href: "/models/transformers",
            },
          ],
        },
      ],
    },
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold tracking-tight text-black mb-6">Machine Learning Models</h1>
      <p className="text-gray-700 max-w-3xl mb-12">
        Explore various machine learning algorithms through interactive visualizations and comprehensive explanations.
        Select a model category to begin your learning journey.
      </p>

      {modelCategories.map((category, index) => (
        <div key={index} className="mb-16">
          <h2 className="text-2xl font-bold text-black mb-2">{category.title}</h2>
          <p className="text-gray-700 mb-6">{category.description}</p>

          {category.subcategories.map((subcategory, subIndex) => (
            <div key={subIndex} className="mb-10">
              <h3 className="text-xl font-semibold text-black mb-4">{subcategory.title}</h3>
              <p className="text-gray-700 mb-6">{subcategory.description}</p>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {subcategory.models.map((model, modelIndex) => (
                  <Card key={modelIndex} className="border-gray-300 bg-white">
                    <CardHeader className="flex flex-row items-start gap-4 pb-2">
                      <div className="mt-1 bg-gray-100 p-2 rounded-md">{model.icon}</div>
                      <div>
                        <CardTitle className="text-xl text-black">{model.title}</CardTitle>
                        <CardDescription className="text-gray-600">{model.description}</CardDescription>
                      </div>
                    </CardHeader>
                    <CardFooter>
                      <Button
                        asChild
                        variant="outline"
                        className="w-full border-gray-300 hover:bg-gray-100 hover:text-black"
                      >
                        <Link href={model.href}>
                          Explore {model.title} <ArrowRight className="ml-2 h-4 w-4" />
                        </Link>
                      </Button>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            </div>
          ))}
        </div>
      ))}

      <div className="mt-12 bg-gray-50 p-6 rounded-lg border border-gray-200">
        <h2 className="text-2xl font-bold mb-4">Model Comparison</h2>
        <p className="mb-6">
          Not sure which model to choose? Compare different machine learning models across various metrics and use
          cases.
        </p>
        <Button asChild className="bg-black text-white hover:bg-gray-800">
          <Link href="/models/comparison">
            Compare Models <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </div>
  )
}
