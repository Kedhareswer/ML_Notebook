import type { Metadata } from "next"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight, Brain, LineChart, BarChart, PieChart, Layers, BookOpen } from "lucide-react"

export const metadata: Metadata = {
  title: "Search - ML Notebook",
  description: "Search for machine learning models, concepts, and resources",
}

export default function SearchPage() {
  // This page is a fallback for when JavaScript is disabled
  // The actual search functionality is client-side in the search component

  const categories = [
    {
      title: "Regression Models",
      description: "Models for predicting continuous values",
      icon: <LineChart className="h-8 w-8 text-neutral-800" />,
      items: [
        { name: "Linear Regression", path: "/models/linear-regression" },
        { name: "Polynomial Regression", path: "/models/polynomial-regression" },
        { name: "Ridge & Lasso Regression", path: "/models/regularized-regression" },
      ],
    },
    {
      title: "Classification Models",
      description: "Models for predicting categorical values",
      icon: <BarChart className="h-8 w-8 text-neutral-800" />,
      items: [
        { name: "Logistic Regression", path: "/models/logistic-regression" },
        { name: "Decision Trees", path: "/models/decision-trees" },
        { name: "Support Vector Machines", path: "/models/svm" },
        { name: "Random Forests", path: "/models/random-forests" },
      ],
    },
    {
      title: "Clustering Algorithms",
      description: "Models for grouping similar data points",
      icon: <PieChart className="h-8 w-8 text-neutral-800" />,
      items: [
        { name: "K-Means Clustering", path: "/models/kmeans" },
        { name: "Hierarchical Clustering", path: "/models/hierarchical-clustering" },
      ],
    },
    {
      title: "Dimensionality Reduction",
      description: "Techniques for reducing data complexity",
      icon: <Layers className="h-8 w-8 text-neutral-800" />,
      items: [{ name: "Principal Component Analysis", path: "/models/pca" }],
    },
    {
      title: "Neural Networks",
      description: "Deep learning models for complex tasks",
      icon: <Brain className="h-8 w-8 text-neutral-800" />,
      items: [
        { name: "Multilayer Perceptron", path: "/models/mlp" },
        { name: "Convolutional Neural Networks", path: "/models/cnn" },
        { name: "Recurrent Neural Networks", path: "/models/rnn" },
        { name: "Transformers", path: "/models/transformers" },
      ],
    },
    {
      title: "Resources",
      description: "Learning materials and references",
      icon: <BookOpen className="h-8 w-8 text-neutral-800" />,
      items: [
        { name: "Learning Path", path: "/resources/learning-path" },
        { name: "Glossary", path: "/resources/glossary" },
        { name: "Cheat Sheets", path: "/resources/cheat-sheets" },
        { name: "Model Comparison", path: "/models/comparison" },
      ],
    },
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="mb-12 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">Browse Content</h1>
        <p className="text-neutral-700 max-w-2xl mx-auto">
          For a better search experience, please use the search button in the navigation bar. This page provides a
          categorized list of all available content.
        </p>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        {categories.map((category, index) => (
          <Card key={index} className="border-neutral-300 bg-white">
            <CardHeader className="flex flex-row items-start gap-4 pb-2">
              <div className="mt-1 bg-neutral-100 p-2 rounded-md">{category.icon}</div>
              <div>
                <CardTitle className="text-xl text-neutral-900">{category.title}</CardTitle>
                <p className="text-neutral-600">{category.description}</p>
              </div>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {category.items.map((item, itemIndex) => (
                  <li key={itemIndex}>
                    <Link
                      href={item.path}
                      className="text-neutral-700 hover:text-neutral-900 hover:underline flex items-center"
                    >
                      <ArrowRight className="h-3 w-3 mr-2 text-neutral-500" />
                      {item.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </CardContent>
            <CardFooter>
              <Button asChild variant="outline" className="w-full">
                <Link href={category.items[0].path}>Explore {category.title}</Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  )
}
