import Link from "next/link"
import Image from "next/image"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, Brain, Network, Microscope } from "lucide-react"

export default function NeuralNetworksPage() {
  const models = [
    {
      title: "Multilayer Perceptron",
      description: "Learn about the building blocks of deep learning",
      icon: <Network className="h-12 w-12 text-neutral-800" />,
      href: "/models/mlp",
      content:
        "A class of feedforward artificial neural network that consists of at least three layers of nodes: an input layer, a hidden layer and an output layer.",
    },
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
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="mb-12">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">Neural Networks</h1>
        <p className="text-neutral-700 max-w-3xl">
          Neural networks are a class of machine learning models inspired by the human brain. They consist of
          interconnected nodes (neurons) organized in layers that can learn complex patterns from data.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 mb-12">
        <div>
          <h2 className="text-2xl font-bold mb-4">What are Neural Networks?</h2>
          <p className="mb-4">
            Neural networks are computational models inspired by the structure and function of the human brain. They
            consist of interconnected processing nodes (neurons) organized in layers that work together to learn
            patterns from data. Each connection between neurons has a weight that adjusts during learning.
          </p>
          <p className="mb-4">
            Neural networks can learn to perform tasks by considering examples, generally without being programmed with
            task-specific rules. They excel at finding patterns in complex, high-dimensional data and can be used for
            both classification and regression tasks.
          </p>
          <h3 className="text-xl font-semibold mb-3 mt-6">Key Characteristics</h3>
          <ul className="list-disc pl-6 space-y-2">
            <li>Composed of layers of interconnected neurons</li>
            <li>Learn through a process called backpropagation</li>
            <li>Can approximate any continuous function (universal approximation theorem)</li>
            <li>Require large amounts of data and computational resources</li>
            <li>Can handle complex, non-linear relationships in data</li>
            <li>Different architectures specialized for different data types (images, text, etc.)</li>
          </ul>
        </div>
        <div className="flex items-center justify-center bg-neutral-100 rounded-lg p-6">
          <Image
            src="/placeholder.svg?height=400&width=500"
            width={500}
            height={400}
            alt="Neural network architecture showing layers of neurons"
            className="rounded-md"
          />
        </div>
      </div>

      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Neural Network Architectures</h2>
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
        <h2 className="text-2xl font-bold mb-4">Key Components</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Neurons</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                The basic computational units that receive inputs, apply weights and biases, and pass the result through
                an activation function to produce an output.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Activation Functions</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Mathematical functions that determine the output of a neuron. Common examples include ReLU, Sigmoid, and
                Tanh.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Weights & Biases</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Adjustable parameters that are learned during training. Weights determine the strength of connections
                between neurons, while biases allow shifting the activation function.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Loss Functions</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Functions that measure the difference between predicted and actual outputs, guiding the learning process
                by quantifying how well the model is performing.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-4">Common Applications</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Computer Vision</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Image classification, object detection, facial recognition, and image generation using CNNs and GANs.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Natural Language Processing</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Text classification, sentiment analysis, machine translation, and text generation using RNNs and
                Transformers.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Time Series Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Stock price prediction, weather forecasting, and anomaly detection in sensor data using RNNs and LSTMs.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
        <h2 className="text-2xl font-bold mb-4">Training Neural Networks</h2>
        <p className="mb-4">Training neural networks involves several key concepts and techniques:</p>
        <ul className="list-disc pl-6 space-y-2 mb-4">
          <li>
            <span className="font-semibold">Backpropagation:</span> The algorithm used to calculate gradients of the
            loss function with respect to the weights, propagating from output to input layers.
          </li>
          <li>
            <span className="font-semibold">Gradient Descent:</span> An optimization algorithm that iteratively adjusts
            weights to minimize the loss function.
          </li>
          <li>
            <span className="font-semibold">Learning Rate:</span> A hyperparameter that controls how much to change the
            model in response to the estimated error each time the weights are updated.
          </li>
          <li>
            <span className="font-semibold">Batch Size:</span> The number of training examples used in one iteration of
            model training.
          </li>
          <li>
            <span className="font-semibold">Epochs:</span> The number of complete passes through the entire training
            dataset.
          </li>
          <li>
            <span className="font-semibold">Regularization:</span> Techniques like dropout and weight decay used to
            prevent overfitting.
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
