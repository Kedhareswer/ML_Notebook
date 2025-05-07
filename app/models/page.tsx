import Link from "next/link"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import {
  ArrowRight,
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
              icon: <LineChart className="h-12 w-12 text-neutral-800" />,
              href: "/models/linear-regression",
              content:
                "A fundamental supervised learning algorithm for predicting continuous values based on linear relationships between variables.",
            },
            {
              title: "Polynomial Regression",
              description: "Extend linear models to capture non-linear relationships",
              icon: <Sigma className="h-12 w-12 text-neutral-800" />,
              href: "/models/polynomial-regression",
              content:
                "An extension of linear regression that can model non-linear relationships by adding polynomial terms to the regression equation.",
            },
            {
              title: "Ridge & Lasso Regression",
              description: "Regularization techniques to prevent overfitting",
              icon: <Braces className="h-12 w-12 text-neutral-800" />,
              href: "/models/regularized-regression",
              content:
                "Regularization methods that add penalty terms to the linear regression cost function to reduce model complexity and prevent overfitting.",
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
              icon: <Sigma className="h-12 w-12 text-neutral-800" />,
              href: "/models/logistic-regression",
              content:
                "A statistical model that uses a logistic function to model a binary dependent variable, commonly used for binary classification problems.",
            },
            {
              title: "Decision Trees",
              description: "Understand decision trees and their applications",
              icon: <GitBranch className="h-12 w-12 text-neutral-800" />,
              href: "/models/decision-trees",
              content:
                "A versatile machine learning algorithm that creates a flowchart-like structure for making decisions based on feature values.",
            },
            {
              title: "Support Vector Machines",
              description: "Explore the mathematics behind SVMs",
              icon: <BarChart className="h-12 w-12 text-neutral-800" />,
              href: "/models/svm",
              content:
                "A powerful classification algorithm that finds the optimal hyperplane to separate different classes with maximum margin.",
            },
            {
              title: "Random Forests",
              description: "Learn how ensemble methods improve performance",
              icon: <Layers className="h-12 w-12 text-neutral-800" />,
              href: "/models/random-forests",
              content:
                "An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.",
            },
            {
              title: "K-Nearest Neighbors (KNN)",
              description: "A simple yet effective classification algorithm.",
              icon: <Microscope className="h-12 w-12 text-neutral-800" />,
              href: "/models/knn",
              content:
                "K-Nearest Neighbors is a non-parametric, lazy learning algorithm used for classification and regression.",
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
              icon: <PieChart className="h-12 w-12 text-neutral-800" />,
              href: "/models/kmeans",
              content:
                "A clustering algorithm that partitions n observations into k clusters where each observation belongs to the cluster with the nearest mean.",
            },
            {
              title: "Hierarchical Clustering",
              description: "Understand how hierarchical clustering works",
              icon: <GitBranch className="h-12 w-12 text-neutral-800" />,
              href: "/models/hierarchical-clustering",
              content:
                "A method of cluster analysis which seeks to build a hierarchy of clusters, either using a bottom-up or top-down approach.",
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
              icon: <Workflow className="h-12 w-12 text-neutral-800" />,
              href: "/models/pca",
              content:
                "A statistical procedure that uses an orthogonal transformation to convert a set of observations into a set of linearly uncorrelated variables called principal components.",
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
              icon: <Network className="h-12 w-12 text-neutral-800" />,
              href: "/models/mlp",
              content:
                "A class of feedforward artificial neural network that consists of at least three layers of nodes: an input layer, a hidden layer and an output layer.",
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
              icon: <Brain className="h-12 w-12 text-neutral-800" />,
              href: "/models/cnn",
              content:
                "Deep learning architecture specifically designed for processing grid-like data such as images, using convolutional layers.",
            },
            {
              title: "Recurrent Neural Networks",
              description: "See how RNNs handle sequential data",
              icon: <Network className="h-12 w-12 text-neutral-800" />,
              href: "/models/rnn",
              content:
                "Neural networks designed to recognize patterns in sequences of data, such as text, time series, or speech.",
            },
            {
              title: "Transformers",
              description: "Explore the architecture behind modern NLP models",
              icon: <Microscope className="h-12 w-12 text-neutral-800" />,
              href: "/models/transformers",
              content:
                "A deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data.",
            },
          ],
        },
      ],
    },
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="mb-12 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">Machine Learning Models</h1>
        <p className="text-neutral-700 max-w-2xl mx-auto">
          Explore different machine learning and deep learning models through interactive notebook-style tutorials
        </p>
      </div>

      <div className="space-y-16">
        {modelCategories.map((category, categoryIndex) => (
          <div key={categoryIndex} className="space-y-8">
            <div className="border-b border-neutral-200 pb-2">
              <h2 className="text-2xl font-bold text-neutral-900">{category.title}</h2>
              <p className="text-neutral-700">{category.description}</p>
            </div>

            {category.subcategories.map((subcategory, subcategoryIndex) => (
              <div key={subcategoryIndex} className="space-y-6">
                <h3 className="text-xl font-semibold text-neutral-800 pl-4 border-l-4 border-neutral-300">
                  {subcategory.title}
                  <span className="block text-sm font-normal text-neutral-600 mt-1">{subcategory.description}</span>
                </h3>

                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                  {subcategory.models.map((model, modelIndex) => (
                    <Card key={modelIndex} className="border-neutral-300 bg-white hover:shadow-md transition-shadow">
                      <CardHeader className="flex flex-row items-start gap-4 pb-2">
                        <div className="mt-1 bg-neutral-100 p-2 rounded-md">{model.icon}</div>
                        <div>
                          <CardTitle className="text-xl text-neutral-900">{model.title}</CardTitle>
                          <CardDescription className="text-neutral-600">{model.description}</CardDescription>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <p className="text-neutral-700 text-sm">{model.content}</p>
                      </CardContent>
                      <CardFooter>
                        <Button asChild variant="notebook" className="w-full">
                          <Link href={model.href}>
                            Explore <ArrowRight className="ml-2 h-4 w-4" />
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
      </div>
    </div>
  )
}
