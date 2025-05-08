import Link from "next/link"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, BookOpen, LineChart, GitBranch, Network } from "lucide-react"

export default function About() {
  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-24">
      <div className="z-10 max-w-5xl w-full">
        <div className="flex items-center gap-3 mb-8">
          <BookOpen className="h-8 w-8 text-neutral-800" />
          <h1 className="text-4xl font-bold">About ML Notebook</h1>
        </div>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Our Mission</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-4">
              ML Notebook aims to make machine learning concepts accessible through interactive visualizations and clear
              explanations. We believe that understanding the fundamentals of machine learning algorithms is essential
              for anyone interested in the field.
            </p>
            <p>
              Our interactive approach allows you to see how different algorithms work with various parameters and
              datasets, providing intuitive insights into their behavior without requiring complex mathematical
              understanding or programming knowledge.
            </p>
          </CardContent>
        </Card>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Features</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc pl-5 space-y-2">
              <li>Interactive visualizations for various machine learning algorithms</li>
              <li>Comprehensive explanations of algorithm mechanics</li>
              <li>Adjustable parameters to see how algorithms respond to different settings</li>
              <li>Comparison tools to understand differences between similar models</li>
              <li>Curated learning resources and glossary of terms</li>
              <li>Mobile-friendly interface for learning on the go</li>
            </ul>
          </CardContent>
        </Card>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>How to Use This Platform</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-4">
              Navigate through the different sections using the main menu. Each algorithm page contains:
            </p>
            <ul className="list-disc pl-5 space-y-2">
              <li>
                <strong>Overview:</strong> A high-level explanation of the algorithm
              </li>
              <li>
                <strong>Interactive Demo:</strong> Visualizations that let you experiment with the algorithm
              </li>
              <li>
                <strong>Mathematical Foundation:</strong> The underlying equations and principles
              </li>
              <li>
                <strong>Applications:</strong> Real-world use cases and examples
              </li>
            </ul>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center gap-4">
              <LineChart className="h-6 w-6 text-neutral-800" />
              <CardTitle className="text-lg">Supervised Learning</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Explore models that learn from labeled data to make predictions on new, unseen data.</p>
            </CardContent>
            <CardFooter>
              <Button variant="outline" size="sm" asChild>
                <Link href="/models/regression">
                  Explore <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center gap-4">
              <GitBranch className="h-6 w-6 text-neutral-800" />
              <CardTitle className="text-lg">Unsupervised Learning</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Discover algorithms that find patterns and structures in unlabeled data.</p>
            </CardContent>
            <CardFooter>
              <Button variant="outline" size="sm" asChild>
                <Link href="/models/clustering">
                  Explore <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center gap-4">
              <Network className="h-6 w-6 text-neutral-800" />
              <CardTitle className="text-lg">Neural Networks</CardTitle>
            </CardHeader>
            <CardContent>
              <p>Learn about deep learning models inspired by the structure of the human brain.</p>
            </CardContent>
            <CardFooter>
              <Button variant="outline" size="sm" asChild>
                <Link href="/models/neural-networks">
                  Explore <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Get Started</CardTitle>
          </CardHeader>
          <CardContent>
            <p>
              Ready to explore machine learning algorithms? Start by browsing our collection of models or check out the
              learning resources for a structured approach.
            </p>
          </CardContent>
          <CardFooter className="flex gap-4">
            <Button asChild>
              <Link href="/models">
                Explore Models <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/resources/learning-path">
                Learning Path <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </CardFooter>
        </Card>
      </div>
    </main>
  )
}
