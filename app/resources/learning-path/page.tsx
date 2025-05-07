import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"

export default function LearningPathPage() {
  return (
    <div className="container mx-auto px-4 py-12">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-6">Learning Path</h1>

        <p className="text-neutral-700 text-lg mb-8">
          This learning path provides a structured approach to understanding machine learning concepts, from
          foundational principles to advanced models. Each model page follows a consistent structure with Overview,
          Interactive Demo, and Code Implementation sections to enhance your learning experience.
        </p>

        <div className="bg-neutral-100 p-6 rounded-lg mb-12">
          <h2 className="text-xl font-bold text-neutral-900 mb-4">How to Use This Learning Path</h2>
          <p className="text-neutral-700 mb-4">
            Each model page in our platform follows a consistent three-part structure:
          </p>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white p-4 rounded-lg border border-neutral-300">
              <h3 className="font-medium text-neutral-900 mb-2">1. Overview</h3>
              <p className="text-neutral-700 text-sm">
                Comprehensive explanation of the model's theory, key concepts, and applications
              </p>
            </div>
            <div className="bg-white p-4 rounded-lg border border-neutral-300">
              <h3 className="font-medium text-neutral-900 mb-2">2. Interactive Demo</h3>
              <p className="text-neutral-700 text-sm">
                Visual demonstrations where you can adjust parameters and see real-time effects
              </p>
            </div>
            <div className="bg-white p-4 rounded-lg border border-neutral-300">
              <h3 className="font-medium text-neutral-900 mb-2">3. Code Implementation</h3>
              <p className="text-neutral-700 text-sm">
                Practical examples with executable code cells to reinforce your understanding
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-12">
          <section>
            <h2 className="text-2xl font-bold text-neutral-900 mb-4">1. Foundations</h2>
            <p className="text-neutral-700 mb-6">
              Start with these fundamental concepts to build a solid understanding of machine learning basics.
            </p>
            <div className="space-y-4">
              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Statistical Learning Theory</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Understand the theoretical foundations of machine learning, including bias-variance tradeoff,
                    overfitting, and regularization.
                  </p>
                  <Link
                    href="/resources/glossary?filter=foundations"
                    className="text-neutral-900 font-medium hover:underline"
                  >
                    Explore key concepts in the glossary →
                  </Link>
                </CardContent>
              </Card>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-neutral-900 mb-4">2. Regression Models</h2>
            <p className="text-neutral-700 mb-6">
              Begin with regression models to understand how to predict continuous values.
            </p>
            <div className="space-y-4">
              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Linear Regression</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Start with linear regression to understand the core concepts of supervised learning, model fitting,
                    and evaluation.
                  </p>
                  <Button asChild variant="outline">
                    <Link href="/models/linear-regression">
                      Explore Linear Regression <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Polynomial Regression</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Learn how to model non-linear relationships by extending linear regression with polynomial features.
                  </p>
                  <Button asChild variant="outline">
                    <Link href="/models/polynomial-regression">
                      Explore Polynomial Regression <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Ridge & Lasso Regression</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">Understand regularization techniques to prevent overfitting in linear models.</p>
                  <Button asChild variant="outline">
                    <Link href="/models/regularized-regression">
                      Explore Ridge & Lasso Regression <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-neutral-900 mb-4">3. Classification Models</h2>
            <p className="text-neutral-700 mb-6">
              After understanding regression, move on to classification problems and models.
            </p>
            <div className="space-y-4">
              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Logistic Regression</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Learn about logistic regression, a fundamental classification algorithm that predicts binary
                    outcomes.
                  </p>
                  <Button asChild variant="outline">
                    <Link href="/models/logistic-regression">
                      Explore Logistic Regression <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Decision Trees</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Learn about decision trees, a versatile and interpretable model for classification and regression.
                  </p>
                  <Button asChild variant="outline">
                    <Link href="/models/decision-trees">
                      Explore Decision Trees <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Support Vector Machines</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Explore SVMs to understand margin maximization and kernel methods for non-linear classification.
                  </p>
                  <Button asChild variant="outline">
                    <Link href="/models/svm">
                      Explore Support Vector Machines <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Random Forests</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Understand how combining multiple decision trees can create a more powerful and robust model.
                  </p>
                  <Button asChild variant="outline">
                    <Link href="/models/random-forest">
                      Explore Random Forests <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-neutral-900 mb-4">4. Deep Learning</h2>
            <p className="text-neutral-700 mb-6">
              Once you're comfortable with traditional machine learning models, advance to deep learning.
            </p>
            <div className="space-y-4">
              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Multilayer Perceptron</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Learn about neural networks, backpropagation, activation functions, and optimization algorithms.
                  </p>
                  <Button asChild variant="outline">
                    <Link href="/models/mlp">
                      Explore Multilayer Perceptron <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Convolutional Neural Networks</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">Understand CNNs for image processing and computer vision tasks.</p>
                  <Button asChild variant="outline">
                    <Link href="/models/cnn">
                      Explore Convolutional Neural Networks <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Recurrent Neural Networks</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">Learn about RNNs for sequential data processing and natural language tasks.</p>
                  <Button asChild variant="outline">
                    <Link href="/models/rnn">
                      Explore Recurrent Neural Networks <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Transformers</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">Understand the architecture behind modern NLP models like BERT and GPT.</p>
                  <Button asChild variant="outline">
                    <Link href="/models/transformers">
                      Explore Transformers <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-neutral-900 mb-4">5. Model Comparisons</h2>
            <p className="text-neutral-700 mb-6">
              Compare different models to understand their strengths, weaknesses, and appropriate use cases.
            </p>
            <div className="space-y-4">
              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Classification Models Comparison</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Compare logistic regression, decision trees, SVMs, and random forests on various classification
                    tasks.
                  </p>
                  <Button asChild variant="outline">
                    <Link href="/models/comparison?category=classification">
                      Explore Classification Comparisons <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Regression Models Comparison</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Compare linear, polynomial, ridge, and lasso regression on different regression problems.
                  </p>
                  <Button asChild variant="outline">
                    <Link href="/models/comparison?category=regression">
                      Explore Regression Comparisons <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">Neural Network Architectures Comparison</CardTitle>
                </CardHeader>
                <CardContent className="text-neutral-700">
                  <p className="mb-4">
                    Compare MLPs, CNNs, RNNs, and Transformers on various tasks to understand their strengths.
                  </p>
                  <Button asChild variant="outline">
                    <Link href="/models/comparison?category=neural-networks">
                      Explore Neural Network Comparisons <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            </div>
          </section>
        </div>

        <div className="mt-12 text-center">
          <p className="text-neutral-700 mb-6">
            Ready to start your machine learning journey with our structured, consistent learning experience?
          </p>
          <Button asChild size="lg" variant="notebook">
            <Link href="/models/linear-regression">
              Begin with Linear Regression <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  )
}
