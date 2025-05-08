import Link from "next/link"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, BookOpen, GitBranch, LineChart, Network } from "lucide-react"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <div className="flex flex-col items-center mb-12">
          <div className="flex items-center gap-3 mb-4">
            <BookOpen className="h-10 w-10 text-black" />
            <h1 className="text-4xl font-bold">ML Notebook</h1>
          </div>
          <p className="text-xl text-center text-gray-600 max-w-2xl">
            Your interactive guide to understanding machine learning algorithms through visualizations and explanations
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          <Card className="col-span-1 md:col-span-2 border-gray-300 bg-white">
            <CardHeader>
              <CardTitle>Welcome to ML Notebook</CardTitle>
              <CardDescription>Your interactive guide to understanding machine learning algorithms</CardDescription>
            </CardHeader>
            <CardContent>
              <p>
                This platform provides interactive visualizations and explanations for various machine learning
                algorithms. Explore different models, understand their mechanics, and see how they work with different
                parameters and data distributions.
              </p>
            </CardContent>
            <CardFooter>
              <Button asChild className="bg-black text-white hover:bg-gray-800">
                <Link href="/models">
                  Explore Models <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>

          <Card className="border-gray-300 bg-white">
            <CardHeader className="flex flex-row items-center gap-4">
              <LineChart className="h-8 w-8 text-black" />
              <div>
                <CardTitle>Supervised Learning</CardTitle>
                <CardDescription>Models that learn from labeled data</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <p>
                Explore algorithms like Linear Regression, Decision Trees, Support Vector Machines, and Neural Networks.
                Understand how these models learn patterns from labeled data to make predictions.
              </p>
            </CardContent>
            <CardFooter>
              <Button variant="outline" asChild className="border-gray-300 hover:bg-gray-100 hover:text-black">
                <Link href="/models/classification">
                  Classification Models <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button variant="outline" className="ml-2 border-gray-300 hover:bg-gray-100 hover:text-black" asChild>
                <Link href="/models/regression">
                  Regression Models <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>

          <Card className="border-gray-300 bg-white">
            <CardHeader className="flex flex-row items-center gap-4">
              <GitBranch className="h-8 w-8 text-black" />
              <div>
                <CardTitle>Unsupervised Learning</CardTitle>
                <CardDescription>Models that find patterns in unlabeled data</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <p>
                Discover clustering algorithms like K-Means and Hierarchical Clustering, as well as dimensionality
                reduction techniques like PCA. Learn how these models uncover hidden structures in data.
              </p>
            </CardContent>
            <CardFooter>
              <Button variant="outline" asChild className="border-gray-300 hover:bg-gray-100 hover:text-black">
                <Link href="/models/clustering">
                  Clustering Models <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button variant="outline" className="ml-2 border-gray-300 hover:bg-gray-100 hover:text-black" asChild>
                <Link href="/models/dimensionality-reduction">
                  Dimensionality Reduction <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>

          <Card className="col-span-1 md:col-span-2 border-gray-300 bg-white">
            <CardHeader className="flex flex-row items-center gap-4">
              <Network className="h-8 w-8 text-black" />
              <div>
                <CardTitle>Neural Networks</CardTitle>
                <CardDescription>Deep learning models inspired by the human brain</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <p>
                Explore the architecture and mechanics of neural networks, from simple multilayer perceptrons to
                specialized architectures like CNNs for image processing, RNNs for sequential data, and transformers for
                natural language processing.
              </p>
            </CardContent>
            <CardFooter>
              <Button variant="outline" asChild className="border-gray-300 hover:bg-gray-100 hover:text-black">
                <Link href="/models/neural-networks">
                  Neural Network Models <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>

          <Card className="col-span-1 md:col-span-2 border-gray-300 bg-white">
            <CardHeader>
              <CardTitle>Learning Resources</CardTitle>
              <CardDescription>Additional materials to enhance your understanding</CardDescription>
            </CardHeader>
            <CardContent>
              <p>
                Access glossaries, cheat sheets, and curated learning paths to deepen your knowledge of machine learning
                concepts and techniques.
              </p>
            </CardContent>
            <CardFooter>
              <Button variant="outline" asChild className="border-gray-300 hover:bg-gray-100 hover:text-black">
                <Link href="/resources">
                  Browse Resources <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        </div>
      </div>
    </main>
  )
}
