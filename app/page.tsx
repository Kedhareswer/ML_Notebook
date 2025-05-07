import Link from "next/link"
import { ArrowRight, BookOpen, Brain, GitBranch, LineChart, BarChart, Code, Layers } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardFooter, CardHeader, CardTitle, CardContent } from "@/components/ui/card"

export default function Home() {
  const models = [
    {
      title: "Linear Regression",
      description: "Learn how linear regression works and how to implement it",
      icon: <LineChart className="h-8 w-8 text-neutral-800" />,
      href: "/models/linear-regression",
    },
    {
      title: "Decision Trees",
      description: "Understand decision trees and their applications",
      icon: <GitBranch className="h-8 w-8 text-neutral-800" />,
      href: "/models/decision-trees",
    },
    {
      title: "Support Vector Machines",
      description: "Explore the mathematics behind SVMs",
      icon: <BarChart className="h-8 w-8 text-neutral-800" />,
      href: "/models/svm",
    },
    {
      title: "Neural Networks",
      description: "Dive into multilayer perceptrons and deep learning",
      icon: <Brain className="h-8 w-8 text-neutral-800" />,
      href: "/models/mlp",
    },
    {
      title: "Transformers",
      description: "Understand attention mechanisms and transformer architecture",
      icon: <Layers className="h-8 w-8 text-neutral-800" />,
      href: "/models/transformers",
    },
    {
      title: "Model Comparison",
      description: "Compare different models and their performance",
      icon: <Code className="h-8 w-8 text-neutral-800" />,
      href: "/models/comparison",
    },
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <section className="py-12 md:py-24 flex flex-col items-center text-center">
        <div className="space-y-4 max-w-3xl">
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tighter text-neutral-900">
            Machine Learning{" "}
            <span className="text-neutral-800 underline decoration-2 underline-offset-4">Notebook</span>
          </h1>
          <p className="text-xl text-neutral-700 max-w-[700px] mx-auto">
            Explore the inner workings of machine learning and deep learning models through interactive notebook-style
            tutorials
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-6">
            <Button asChild size="lg" variant="notebook">
              <Link href="/resources/learning-path">
                Start Learning Path <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link href="/models">Browse All Models</Link>
            </Button>
          </div>
        </div>
      </section>

      <section className="py-12">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold tracking-tight text-neutral-900">Featured Models</h2>
          <p className="text-neutral-700 mt-4 max-w-2xl mx-auto">
            Dive into different machine learning models with interactive demos, visualizations, and code implementations
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {models.map((model, index) => (
            <Card key={index} className="transition-all hover:shadow-lg border-neutral-300 bg-white">
              <CardHeader>
                <div className="mb-4">{model.icon}</div>
                <CardTitle className="text-neutral-900">{model.title}</CardTitle>
                <p className="text-neutral-600">{model.description}</p>
              </CardHeader>
              <CardFooter>
                <Button asChild variant="outline" className="w-full">
                  <Link href={model.href}>
                    Explore <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </section>

      <section className="py-12 md:py-24 bg-neutral-100 rounded-xl mt-12 px-6">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <h2 className="text-3xl font-bold tracking-tight mb-4 text-neutral-900">Consistent Learning Experience</h2>
            <p className="text-neutral-700 mb-6">
              Our platform offers a structured approach to learning machine learning concepts through:
            </p>
            <ul className="space-y-4">
              <li className="flex items-start">
                <div className="mr-4 mt-1 bg-neutral-200 p-1 rounded-full">
                  <BookOpen className="h-5 w-5 text-neutral-800" />
                </div>
                <div>
                  <h3 className="font-medium text-neutral-900">Comprehensive Overviews</h3>
                  <p className="text-neutral-700">
                    Clear explanations of each model's theory, key concepts, and applications
                  </p>
                </div>
              </li>
              <li className="flex items-start">
                <div className="mr-4 mt-1 bg-neutral-200 p-1 rounded-full">
                  <BarChart className="h-5 w-5 text-neutral-800" />
                </div>
                <div>
                  <h3 className="font-medium text-neutral-900">Interactive Demos</h3>
                  <p className="text-neutral-700">Manipulate parameters and see real-time changes in model behavior</p>
                </div>
              </li>
              <li className="flex items-start">
                <div className="mr-4 mt-1 bg-neutral-200 p-1 rounded-full">
                  <Code className="h-5 w-5 text-neutral-800" />
                </div>
                <div>
                  <h3 className="font-medium text-neutral-900">Code Implementations</h3>
                  <p className="text-neutral-700">
                    Practical examples with executable code cells to reinforce understanding
                  </p>
                </div>
              </li>
            </ul>
          </div>
          <div className="bg-white p-6 rounded-xl shadow-md border border-neutral-300">
            <div className="notebook-preview">
              <div className="notebook-cell">
                <div className="cell-input">
                  <div className="cell-prompt">In [1]:</div>
                  <div className="cell-code font-mono text-sm">
                    import numpy as np
                    <br />
                    import matplotlib.pyplot as plt
                    <br />
                    <br /># Generate sample data
                    <br />X = np.linspace(-5, 5, 100)
                    <br />y = 0.5 * X + 2 + np.random.randn(100) * 0.5
                  </div>
                </div>
              </div>
              <div className="notebook-cell">
                <div className="cell-input">
                  <div className="cell-prompt">In [2]:</div>
                  <div className="cell-code font-mono text-sm">
                    # Plot the data
                    <br />
                    plt.scatter(X, y)
                    <br />
                    plt.xlabel('X')
                    <br />
                    plt.ylabel('y')
                    <br />
                    plt.title('Sample Data')
                    <br />
                    plt.show()
                  </div>
                </div>
                <div className="cell-output">
                  <div className="output-placeholder bg-neutral-100 rounded-md h-24 flex items-center justify-center">
                    <p className="text-neutral-500">Plot output</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-12 mt-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold tracking-tight text-neutral-900">Learning Paths</h2>
          <p className="text-neutral-700 mt-4 max-w-2xl mx-auto">
            Follow structured learning paths to build your knowledge from fundamentals to advanced concepts
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Regression Models</CardTitle>
              <p className="text-neutral-600">Learn to predict continuous values</p>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-neutral-700">
                <li>• Linear Regression</li>
                <li>• Polynomial Regression</li>
                <li>• Ridge & Lasso Regression</li>
              </ul>
            </CardContent>
            <CardFooter>
              <Button asChild variant="outline" className="w-full">
                <Link href="/models/regression">
                  Start Path <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Classification Models</CardTitle>
              <p className="text-neutral-600">Master techniques for categorical prediction</p>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-neutral-700">
                <li>• Logistic Regression</li>
                <li>• Decision Trees</li>
                <li>• Support Vector Machines</li>
                <li>• Random Forests</li>
              </ul>
            </CardContent>
            <CardFooter>
              <Button asChild variant="outline" className="w-full">
                <Link href="/models/classification">
                  Start Path <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Deep Learning</CardTitle>
              <p className="text-neutral-600">Explore neural networks and beyond</p>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-neutral-700">
                <li>• Multilayer Perceptrons</li>
                <li>• Convolutional Neural Networks</li>
                <li>• Recurrent Neural Networks</li>
                <li>• Transformers</li>
              </ul>
            </CardContent>
            <CardFooter>
              <Button asChild variant="outline" className="w-full">
                <Link href="/models/neural-networks">
                  Start Path <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        </div>
      </section>
    </div>
  )
}
