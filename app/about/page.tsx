import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { BookOpen, Code, BarChart, Lightbulb, Users } from "lucide-react"
import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function AboutPage() {
  return (
    <div className="container mx-auto px-4 py-12">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-6">About ML Notebook</h1>

        <div className="prose prose-neutral max-w-none">
          <p className="text-neutral-700 text-lg mb-8">
            ML Notebook is an interactive educational platform designed to help students, developers, and enthusiasts
            learn about machine learning and deep learning models through a hands-on, structured approach with
            consistent learning experiences across all models.
          </p>

          <h2 className="text-2xl font-bold text-neutral-900 mt-12 mb-6">Our Mission</h2>
          <p className="text-neutral-700 mb-6">
            Our mission is to make machine learning education accessible, interactive, and engaging. We believe that the
            best way to learn complex concepts is through a consistent structure that combines theory, interactive
            visualization, and practical implementation.
          </p>

          <div className="grid sm:grid-cols-3 gap-6 my-8">
            <Card className="border-neutral-300 bg-white">
              <CardHeader className="pb-2">
                <BookOpen className="h-6 w-6 text-neutral-800 mb-2" />
                <CardTitle className="text-neutral-900">Overview</CardTitle>
              </CardHeader>
              <CardContent className="text-neutral-700">
                Each model page begins with a comprehensive explanation of the theory, key concepts, and applications,
                providing a solid foundation for understanding.
              </CardContent>
            </Card>
            <Card className="border-neutral-300 bg-white">
              <CardHeader className="pb-2">
                <BarChart className="h-6 w-6 text-neutral-800 mb-2" />
                <CardTitle className="text-neutral-900">Interactive Demo</CardTitle>
              </CardHeader>
              <CardContent className="text-neutral-700">
                Interactive visualizations allow you to manipulate model parameters and see real-time changes, building
                intuition for how each model works.
              </CardContent>
            </Card>
            <Card className="border-neutral-300 bg-white">
              <CardHeader className="pb-2">
                <Code className="h-6 w-6 text-neutral-800 mb-2" />
                <CardTitle className="text-neutral-900">Code Implementation</CardTitle>
              </CardHeader>
              <CardContent className="text-neutral-700">
                Practical code examples with executable cells help you understand how to implement and use each model in
                real-world scenarios.
              </CardContent>
            </Card>
          </div>

          <h2 className="text-2xl font-bold text-neutral-900 mt-12 mb-6">Our Approach</h2>
          <p className="text-neutral-700 mb-4">
            We've designed ML Notebook with a consistent structure across all model pages to provide a seamless learning
            experience:
          </p>
          <ul className="list-disc pl-6 space-y-2 text-neutral-700 mb-8">
            <li>
              <strong>Consistent Format</strong>: Every model page follows the same three-part structure: Overview,
              Interactive Demo, and Code Implementation
            </li>
            <li>
              <strong>Progressive Learning</strong>: Models are organized into learning paths that build knowledge
              systematically
            </li>
            <li>
              <strong>Visual Understanding</strong>: Interactive visualizations help build intuition for complex
              concepts
            </li>
            <li>
              <strong>Practical Application</strong>: Code examples show how to implement models using popular libraries
              like scikit-learn and TensorFlow
            </li>
            <li>
              <strong>Model Comparisons</strong>: Dedicated comparison pages help understand the strengths and
              weaknesses of different models
            </li>
          </ul>

          <div className="bg-neutral-100 p-6 rounded-lg my-8">
            <h2 className="text-2xl font-bold text-neutral-900 mb-4">Key Features</h2>
            <div className="grid sm:grid-cols-2 gap-6">
              <div className="flex items-start">
                <div className="mr-4 mt-1 bg-neutral-200 p-1 rounded-full">
                  <Lightbulb className="h-5 w-5 text-neutral-800" />
                </div>
                <div>
                  <h3 className="font-medium text-neutral-900">Interactive Visualizations</h3>
                  <p className="text-neutral-700">
                    Manipulate model parameters and see how they affect performance and behavior in real-time
                  </p>
                </div>
              </div>
              <div className="flex items-start">
                <div className="mr-4 mt-1 bg-neutral-200 p-1 rounded-full">
                  <Code className="h-5 w-5 text-neutral-800" />
                </div>
                <div>
                  <h3 className="font-medium text-neutral-900">Executable Code Cells</h3>
                  <p className="text-neutral-700">
                    Run and modify code examples directly in your browser, similar to Jupyter notebooks
                  </p>
                </div>
              </div>
              <div className="flex items-start">
                <div className="mr-4 mt-1 bg-neutral-200 p-1 rounded-full">
                  <BookOpen className="h-5 w-5 text-neutral-800" />
                </div>
                <div>
                  <h3 className="font-medium text-neutral-900">Structured Learning Paths</h3>
                  <p className="text-neutral-700">
                    Follow guided learning paths from basic to advanced concepts across different model categories
                  </p>
                </div>
              </div>
              <div className="flex items-start">
                <div className="mr-4 mt-1 bg-neutral-200 p-1 rounded-full">
                  <Users className="h-5 w-5 text-neutral-800" />
                </div>
                <div>
                  <h3 className="font-medium text-neutral-900">Model Comparisons</h3>
                  <p className="text-neutral-700">
                    Compare different models side-by-side to understand their strengths and appropriate use cases
                  </p>
                </div>
              </div>
            </div>
          </div>

          <h2 className="text-2xl font-bold text-neutral-900 mt-12 mb-6">Who We Are</h2>
          <p className="text-neutral-700 mb-8">
            ML Notebook was created by a team of machine learning practitioners, educators, and developers who are
            passionate about making machine learning education more accessible and engaging. We combine expertise in
            machine learning theory with practical implementation experience to create learning materials that bridge
            the gap between theory and practice.
          </p>

          <div className="flex justify-center mt-12">
            <Button asChild size="lg" variant="notebook">
              <Link href="/resources/learning-path">Start Your Learning Journey</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
