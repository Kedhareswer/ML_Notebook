import Link from "next/link"
import Image from "next/image"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, LineChart, Sigma, Braces } from "lucide-react"

export default function RegressionModelsPage() {
  const models = [
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
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="mb-12">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">Regression Models</h1>
        <p className="text-neutral-700 max-w-3xl">
          Regression models are supervised learning algorithms that predict continuous numerical values. They're used
          when the output variable is a real or continuous value, such as "price" or "weight."
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 mb-12">
        <div>
          <h2 className="text-2xl font-bold mb-4">What is Regression?</h2>
          <p className="mb-4">
            Regression is a supervised learning technique where the algorithm learns from labeled training data to
            predict a continuous output variable. The goal is to find the relationship between independent variables
            (features) and a dependent variable (target) by estimating how the target changes as the features change.
          </p>
          <p className="mb-4">
            Unlike classification models that predict discrete categories, regression models predict continuous values.
            This makes them suitable for problems where the output is a quantity rather than a category.
          </p>
          <h3 className="text-xl font-semibold mb-3 mt-6">Key Characteristics</h3>
          <ul className="list-disc pl-6 space-y-2">
            <li>Predicts continuous numerical values</li>
            <li>Requires labeled training data with numerical target values</li>
            <li>Models the relationship between independent and dependent variables</li>
            <li>Evaluated using metrics like MSE, RMSE, MAE, and R-squared</li>
            <li>Can handle simple linear relationships or complex non-linear patterns</li>
          </ul>
        </div>
        <div className="flex items-center justify-center bg-neutral-100 rounded-lg p-6">
          <Image
            src="/placeholder.svg?height=400&width=500"
            width={500}
            height={400}
            alt="Regression visualization showing a best-fit line through data points"
            className="rounded-md"
          />
        </div>
      </div>

      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Common Regression Algorithms</h2>
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
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
              <CardTitle className="text-lg">Price Prediction</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Predicting house prices, stock prices, or product prices based on various features and historical data.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Sales Forecasting</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Estimating future sales based on historical sales data, marketing spend, seasonality, and other factors.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Risk Assessment</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Predicting the likelihood of loan defaults or insurance claims based on customer attributes and
                behavior.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
        <h2 className="text-2xl font-bold mb-4">Evaluation Metrics</h2>
        <p className="mb-4">
          Regression models are evaluated using different metrics than classification models. Common evaluation metrics
          include:
        </p>
        <ul className="list-disc pl-6 space-y-2 mb-4">
          <li>
            <span className="font-semibold">Mean Squared Error (MSE):</span> The average of the squared differences
            between predicted and actual values.
          </li>
          <li>
            <span className="font-semibold">Root Mean Squared Error (RMSE):</span> The square root of MSE, which
            provides an error measure in the same units as the target variable.
          </li>
          <li>
            <span className="font-semibold">Mean Absolute Error (MAE):</span> The average of the absolute differences
            between predicted and actual values.
          </li>
          <li>
            <span className="font-semibold">R-squared (R²):</span> The proportion of the variance in the dependent
            variable that is predictable from the independent variables.
          </li>
          <li>
            <span className="font-semibold">Adjusted R-squared:</span> A modified version of R-squared that adjusts for
            the number of predictors in the model.
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
