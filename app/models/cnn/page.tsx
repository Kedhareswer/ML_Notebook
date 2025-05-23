"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import CNNViz from "@/components/interactive-visualizations/cnn-viz"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { InfoIcon, BookOpenIcon, PlayIcon } from "lucide-react"

const CNNPage = () => {
  const [filterSize, setFilterSize] = useState(3)
  const [numFilters, setNumFilters] = useState(16)
  const [activationFunction, setActivationFunction] = useState("relu")

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-4xl font-bold mb-6">Convolutional Neural Networks (CNNs)</h1>

      <div className="mb-8">
        <p className="text-lg mb-4">
          Convolutional Neural Networks (CNNs) are specialized deep learning models designed primarily for processing
          grid-like data, such as images. They have revolutionized computer vision tasks by automatically learning
          spatial hierarchies of features.
        </p>
      </div>

      <Tabs defaultValue="overview">
        <TabsList className="mb-6">
          <TabsTrigger value="overview">
            <InfoIcon className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="interactive">
            <PlayIcon className="h-4 w-4 mr-2" />
            Interactive Demo
          </TabsTrigger>
          <TabsTrigger value="theory">
            <BookOpenIcon className="h-4 w-4 mr-2" />
            Theory
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>What are CNNs?</CardTitle>
                <CardDescription>The building blocks of modern computer vision</CardDescription>
              </CardHeader>
              <CardContent>
                <p>
                  Convolutional Neural Networks (CNNs) are deep learning models that use convolution operations to
                  process data with grid-like topology. They are particularly effective for image recognition,
                  classification, and computer vision tasks.
                </p>
                <p className="mt-4">
                  Unlike traditional neural networks, CNNs preserve spatial relationships in the input data, making them
                  ideal for processing images where the relative positions of pixels matter.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Key Components</CardTitle>
                <CardDescription>The essential layers of a CNN</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="list-disc pl-5 space-y-2">
                  <li>
                    <strong>Convolutional Layers:</strong> Apply filters to detect features
                  </li>
                  <li>
                    <strong>Pooling Layers:</strong> Reduce dimensionality while preserving important information
                  </li>
                  <li>
                    <strong>Activation Functions:</strong> Introduce non-linearity (ReLU, Sigmoid, etc.)
                  </li>
                  <li>
                    <strong>Fully Connected Layers:</strong> Perform classification based on extracted features
                  </li>
                  <li>
                    <strong>Dropout:</strong> Prevent overfitting by randomly deactivating neurons
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Applications of CNNs</CardTitle>
              <CardDescription>Real-world use cases</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Image Classification</h3>
                  <p>Identifying objects, people, or scenes in images</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Object Detection</h3>
                  <p>Locating and classifying multiple objects in images</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Image Segmentation</h3>
                  <p>Pixel-level classification for precise object boundaries</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Face Recognition</h3>
                  <p>Identifying and verifying individuals from facial features</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Medical Imaging</h3>
                  <p>Detecting abnormalities in X-rays, MRIs, and CT scans</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">Autonomous Vehicles</h3>
                  <p>Recognizing road signs, pedestrians, and other vehicles</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="interactive">
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>CNN Visualization</CardTitle>
              <CardDescription>
                Explore how convolutional neural networks process images by adjusting parameters
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="mb-8">
                <CNNViz />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Filter Size: {filterSize}x{filterSize}
                  </label>
                  <Slider
                    value={[filterSize]}
                    min={1}
                    max={7}
                    step={2}
                    onValueChange={(value) => setFilterSize(value[0])}
                    className="mb-4"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Number of Filters: {numFilters}</label>
                  <Slider
                    value={[numFilters]}
                    min={4}
                    max={64}
                    step={4}
                    onValueChange={(value) => setNumFilters(value[0])}
                    className="mb-4"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Activation Function</label>
                  <select
                    value={activationFunction}
                    onChange={(e) => setActivationFunction(e.target.value)}
                    className="w-full p-2 border rounded"
                  >
                    <option value="relu">ReLU</option>
                    <option value="sigmoid">Sigmoid</option>
                    <option value="tanh">Tanh</option>
                    <option value="leaky_relu">Leaky ReLU</option>
                  </select>
                </div>
              </div>

              <div className="flex justify-center">
                <Button>Apply Changes</Button>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Feature Maps</CardTitle>
                <CardDescription>Visualizing what each convolutional layer "sees"</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center">
                  <p className="text-gray-500">Feature map visualization</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Activation Heatmap</CardTitle>
                <CardDescription>Highlighting regions of high activation</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-gray-100 dark:bg-gray-800 rounded-lg flex items-center justify-center">
                  <p className="text-gray-500">Activation heatmap visualization</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="theory" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>The Convolution Operation</CardTitle>
              <CardDescription>The fundamental building block of CNNs</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="mb-4">
                The convolution operation is the core of CNNs. It involves sliding a filter (or kernel) over the input
                data and computing the dot product between the filter and the input at each position.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold mb-2">Mathematical Definition</h3>
                  <p className="mb-2">
                    For a 2D input image I and a 2D filter K, the convolution operation is defined as:
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg text-center">
                    <p>
                      (I * K)(i, j) = ∑<sub>m</sub>∑<sub>n</sub> I(i+m, j+n) · K(m, n)
                    </p>
                  </div>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Properties</h3>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Preserves spatial relationships</li>
                    <li>Parameter sharing reduces model complexity</li>
                    <li>Translation invariance helps recognize patterns regardless of position</li>
                    <li>Hierarchical feature extraction builds from simple to complex features</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Pooling Layers</CardTitle>
                <CardDescription>Downsampling for efficiency</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="mb-4">
                  Pooling layers reduce the spatial dimensions of the feature maps, decreasing computational load and
                  providing a form of translation invariance.
                </p>
                <div className="space-y-4">
                  <div>
                    <h3 className="font-semibold mb-1">Max Pooling</h3>
                    <p>Takes the maximum value from each window, preserving the most prominent features.</p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Average Pooling</h3>
                    <p>Takes the average of all values in each window, providing a smoother representation.</p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Global Pooling</h3>
                    <p>Reduces each feature map to a single value, often used before fully connected layers.</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Activation Functions</CardTitle>
                <CardDescription>Introducing non-linearity</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="mb-4">
                  Activation functions introduce non-linearity into the network, allowing it to learn complex patterns.
                </p>
                <div className="space-y-4">
                  <div>
                    <h3 className="font-semibold mb-1">ReLU (Rectified Linear Unit)</h3>
                    <p>f(x) = max(0, x). Simple and effective, but can suffer from "dying ReLU" problem.</p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Leaky ReLU</h3>
                    <p>f(x) = max(αx, x) where α is a small constant. Addresses the dying ReLU problem.</p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Sigmoid</h3>
                    <p>f(x) = 1/(1+e^(-x)). Maps values to [0,1], but can suffer from vanishing gradients.</p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-1">Tanh</h3>
                    <p>f(x) = (e^x - e^(-x))/(e^x + e^(-x)). Maps values to [-1,1].</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>CNN Architectures</CardTitle>
              <CardDescription>Famous CNN models and their innovations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold mb-2">LeNet-5 (1998)</h3>
                  <p>One of the earliest CNNs, designed for handwritten digit recognition.</p>
                  <ul className="list-disc pl-5 mt-2">
                    <li>7-layer network</li>
                    <li>Used convolutions, pooling, and fully connected layers</li>
                    <li>Applied to MNIST dataset</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">AlexNet (2012)</h3>
                  <p>Breakthrough architecture that won the ImageNet competition.</p>
                  <ul className="list-disc pl-5 mt-2">
                    <li>8-layer network with 60M parameters</li>
                    <li>Used ReLU activations</li>
                    <li>Implemented dropout for regularization</li>
                    <li>Trained on multiple GPUs</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">VGG-16 (2014)</h3>
                  <p>Simple but deep architecture with uniform structure.</p>
                  <ul className="list-disc pl-5 mt-2">
                    <li>16-layer network</li>
                    <li>Used small 3x3 convolution filters throughout</li>
                    <li>138M parameters</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">ResNet (2015)</h3>
                  <p>Introduced skip connections to address the vanishing gradient problem.</p>
                  <ul className="list-disc pl-5 mt-2">
                    <li>Very deep (up to 152 layers)</li>
                    <li>Used residual blocks with identity mappings</li>
                    <li>Enabled training of much deeper networks</li>
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

export default CNNPage
