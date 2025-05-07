import Link from "next/link"
import Image from "next/image"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, GitBranch, BarChart, Layers, Sigma } from "lucide-react"

export default function ClassificationModelsPage() {
  const models = [
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
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="mb-12">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">Classification Models</h1>
        <p className="text-neutral-700 max-w-3xl">
          Classification models are supervised learning algorithms that predict discrete categories or labels. They're
          used when the output variable is categorical, such as "spam" or "not spam" in email filtering.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 mb-12">
        <div>
          <h2 className="text-2xl font-bold mb-4">What is Classification?</h2>
          <p className="mb-4">
            Classification is a supervised learning approach where the algorithm learns from labeled training data to
            predict discrete class labels for new, unseen instances. The goal is to identify which category or class an
            observation belongs to based on a training set of data containing observations with known category
            memberships.
          </p>
          <p className="mb-4">
            Unlike regression models that predict continuous values, classification models predict discrete values or
            categories. These categories can be binary (two classes) or multi-class (more than two classes).
          </p>
          <h3 className="text-xl font-semibold mb-3 mt-6">Key Characteristics</h3>
          <ul className="list-disc pl-6 space-y-2">
            <li>Predicts discrete class labels or categories</li>
            <li>Requires labeled training data</li>
            <li>Can handle binary or multi-class problems</li>
            <li>Evaluated using metrics like accuracy, precision, recall, and F1-score</li>
            <li>Decision boundaries separate different classes in the feature space</li>
          </ul>
        </div>
        <div className="flex items-center justify-center bg-neutral-100 rounded-lg p-6">
          <Image
            src="/placeholder.svg?height=400&width=500"
            width={500}
            height={400}
            alt="Classification visualization showing decision boundaries between classes"
            className="rounded-md"
          />
        </div>
      </div>

      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Common Classification Algorithms</h2>
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-2">
          {models.map((model, index) => (
            <Card key={index} className="border-neutral-300 bg-white">
              <CardHeader className="flex flex-row items-start gap-4 pb-2">
                <div className="mt-1 bg-neutral-100 p-2 rounded-md">{model.icon}</div>
                <div>
                  <CardTitle className="text-xl text-neutral-900">{model.title}</CardTitle>
                  <CardDescription className="text-neutral-600">{model.description}</CardDescription>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-neutral-700">{model.content}</p>
              </CardContent>
              <CardFooter>
                <Button asChild variant="notebook" className="w-full sm:w-auto">
                  <Link href={model.href}>
                    Explore {model.title} <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>

      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-4">Common Applications</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Email Spam Detection</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Classifying emails as spam or not spam based on content, sender information, and other features.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Medical Diagnosis</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Predicting whether a patient has a particular disease based on symptoms and test results.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Image Recognition</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Identifying objects, people, or scenes in images by classifying them into predefined categories.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
        <h2 className="text-2xl font-bold mb-4">Evaluation Metrics</h2>
        <p className="mb-4">
          Classification models are evaluated using different metrics than regression models. Common evaluation metrics
          include:
        </p>
        <ul className="list-disc pl-6 space-y-2 mb-4">
          <li>
            <span className="font-semibold">Accuracy:</span> The proportion of correct predictions among the total
            number of predictions.
          </li>
          <li>
            <span className="font-semibold">Precision:</span> The proportion of true positive predictions among all
            positive predictions.
          </li>
          <li>
            <span className="font-semibold">Recall:</span> The proportion of true positive predictions among all actual
            positive instances.
          </li>
          <li>
            <span className="font-semibold">F1-Score:</span> The harmonic mean of precision and recall, providing a
            balance between the two.
          </li>
          <li>
            <span className="font-semibold">ROC Curve:</span> A graphical plot that illustrates the diagnostic ability
            of a binary classifier system.
          </li>
        </ul>
        <Button asChild variant="outline">
          <Link href="/resources/glossary">
            Learn more in the Glossary <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </div>
  )
}
