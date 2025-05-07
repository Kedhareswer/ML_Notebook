import Link from "next/link"
import Image from "next/image"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, Workflow } from "lucide-react"

export default function DimensionalityReductionPage() {
  const models = [
    {
      title: "Principal Component Analysis",
      description: "Learn how PCA transforms high-dimensional data",
      icon: <Workflow className="h-12 w-12 text-neutral-800" />,
      href: "/models/pca",
      content:
        "A statistical procedure that uses an orthogonal transformation to convert a set of observations into a set of linearly uncorrelated variables called principal components.",
    },
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="mb-12">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">Dimensionality Reduction</h1>
        <p className="text-neutral-700 max-w-3xl">
          Dimensionality reduction techniques transform high-dimensional data into a lower-dimensional space while
          preserving important information and structure.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 mb-12">
        <div>
          <h2 className="text-2xl font-bold mb-4">What is Dimensionality Reduction?</h2>
          <p className="mb-4">
            Dimensionality reduction is the process of reducing the number of random variables under consideration by
            obtaining a set of principal variables. It can be divided into feature selection and feature extraction
            approaches.
          </p>
          <p className="mb-4">
            These techniques are essential when dealing with high-dimensional data, which can suffer from the "curse of
            dimensionality" - as the number of features increases, the amount of data needed to generalize accurately
            grows exponentially.
          </p>
          <h3 className="text-xl font-semibold mb-3 mt-6">Key Characteristics</h3>
          <ul className="list-disc pl-6 space-y-2">
            <li>Reduces computational complexity and storage requirements</li>
            <li>Helps mitigate the curse of dimensionality</li>
            <li>Removes noise and redundant features</li>
            <li>Enables visualization of high-dimensional data</li>
            <li>Can improve the performance of machine learning algorithms</li>
            <li>Preserves important information while discarding less relevant features</li>
          </ul>
        </div>
        <div className="flex items-center justify-center bg-neutral-100 rounded-lg p-6">
          <Image
            src="/placeholder.svg?height=400&width=500"
            width={500}
            height={400}
            alt="Dimensionality reduction visualization showing data projection from high to low dimensions"
            className="rounded-md"
          />
        </div>
      </div>

      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Common Dimensionality Reduction Techniques</h2>
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-1">
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
              <CardTitle className="text-lg">Data Visualization</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Reducing high-dimensional data to 2D or 3D for visualization and exploratory data analysis.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Image Processing</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Compressing images while preserving important features and reducing storage requirements.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Feature Engineering</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Creating more meaningful features from high-dimensional data to improve machine learning model
                performance.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
        <h2 className="text-2xl font-bold mb-4">Types of Dimensionality Reduction</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-xl font-semibold mb-3">Feature Selection</h3>
            <p className="mb-3">Selecting a subset of the original features without transformation.</p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Filter methods (statistical measures)</li>
              <li>Wrapper methods (model performance)</li>
              <li>Embedded methods (built into model training)</li>
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-3">Feature Extraction</h3>
            <p className="mb-3">Transforming the original features into a new feature space.</p>
            <ul className="list-disc pl-6 space-y-1">
              <li>Principal Component Analysis (PCA)</li>
              <li>Linear Discriminant Analysis (LDA)</li>
              <li>t-Distributed Stochastic Neighbor Embedding (t-SNE)</li>
              <li>Autoencoders</li>
            </ul>
          </div>
        </div>
        <div className="mt-4">
          <Button asChild variant="outline">
            <Link href="/resources/glossary">
              Learn more in the Glossary <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>
    </div>
  )
}
