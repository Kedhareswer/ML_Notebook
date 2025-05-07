import Link from "next/link"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, BookOpen, GraduationCap, Library, FileText } from "lucide-react"

export default function ResourcesPage() {
  const resources = [
    {
      title: "Learning Path",
      description: "A structured approach to learning machine learning",
      icon: <GraduationCap className="h-8 w-8 text-neutral-800" />,
      href: "/resources/learning-path",
      content:
        "Follow our recommended learning path to build a solid foundation in machine learning, from basic concepts to advanced models.",
    },
    {
      title: "Glossary",
      description: "Key terms and definitions in machine learning",
      icon: <Library className="h-8 w-8 text-neutral-800" />,
      href: "/resources/glossary",
      content:
        "A comprehensive glossary of machine learning terminology to help you understand the field's vocabulary.",
    },
    {
      title: "References",
      description: "Books, papers, and online resources",
      icon: <BookOpen className="h-8 w-8 text-neutral-800" />,
      href: "/resources/references",
      content:
        "Curated list of high-quality resources for further learning, including textbooks, research papers, and online courses.",
    },
    {
      title: "Cheat Sheets",
      description: "Quick reference guides for models and algorithms",
      icon: <FileText className="h-8 w-8 text-neutral-800" />,
      href: "/resources/cheat-sheets",
      content:
        "Downloadable cheat sheets summarizing key concepts, formulas, and implementation details for various machine learning models.",
    },
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="mb-12 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">Learning Resources</h1>
        <p className="text-neutral-700 max-w-2xl mx-auto">
          Explore our collection of resources to deepen your understanding of machine learning concepts
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {resources.map((resource, index) => (
          <Card key={index} className="border-neutral-300 bg-white">
            <CardHeader className="flex flex-row items-start gap-4 pb-2">
              <div className="mt-1 bg-neutral-100 p-2 rounded-md">{resource.icon}</div>
              <div>
                <CardTitle className="text-xl text-neutral-900">{resource.title}</CardTitle>
                <CardDescription className="text-neutral-600">{resource.description}</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">{resource.content}</p>
            </CardContent>
            <CardFooter>
              <Button asChild variant="outline" className="w-full sm:w-auto">
                <Link href={resource.href}>
                  View {resource.title} <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>

      <div className="mt-12 p-6 border border-neutral-300 rounded-lg bg-neutral-50">
        <h2 className="text-xl font-bold text-neutral-900 mb-4">Additional Learning Materials</h2>
        <p className="text-neutral-700 mb-6">
          Beyond our structured resources, we recommend these external materials to supplement your learning:
        </p>
        <ul className="space-y-3">
          <li className="flex items-start">
            <div className="mr-2 mt-1 text-neutral-800">•</div>
            <div>
              <span className="font-medium text-neutral-900">Elements of Statistical Learning</span>
              <p className="text-neutral-700 text-sm">A comprehensive textbook on machine learning theory</p>
            </div>
          </li>
          <li className="flex items-start">
            <div className="mr-2 mt-1 text-neutral-800">•</div>
            <div>
              <span className="font-medium text-neutral-900">Deep Learning by Goodfellow, Bengio, and Courville</span>
              <p className="text-neutral-700 text-sm">The definitive textbook on deep learning fundamentals</p>
            </div>
          </li>
          <li className="flex items-start">
            <div className="mr-2 mt-1 text-neutral-800">•</div>
            <div>
              <span className="font-medium text-neutral-900">Stanford CS229: Machine Learning</span>
              <p className="text-neutral-700 text-sm">Andrew Ng's course materials and lectures</p>
            </div>
          </li>
          <li className="flex items-start">
            <div className="mr-2 mt-1 text-neutral-800">•</div>
            <div>
              <span className="font-medium text-neutral-900">Fast.ai Practical Deep Learning</span>
              <p className="text-neutral-700 text-sm">A top-down, practical approach to deep learning</p>
            </div>
          </li>
        </ul>
      </div>
    </div>
  )
}
