import Link from "next/link"
import Image from "next/image"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, PieChart, GitBranch } from "lucide-react"

export default function ClusteringModelsPage() {
  const models = [
    {
      title: "K-Means Clustering",
      description: "Explore how K-means partitions data into clusters",
      icon: <PieChart className="h-12 w-12 text-neutral-800" />,
      href: "/models/kmeans",
      content:
        "A clustering algorithm that partitions n observations into k clusters where each observation belongs to the cluster with the nearest mean.",
    },
    {
      title: "Hierarchical Clustering",
      description: "Understand how hierarchical clustering works",
      icon: <GitBranch className="h-12 w-12 text-neutral-800" />,
      href: "/models/hierarchical-clustering",
      content:
        "A method of cluster analysis which seeks to build a hierarchy of clusters, either using a bottom-up or top-down approach.",
    },
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="mb-12">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">Clustering Models</h1>
        <p className="text-neutral-700 max-w-3xl">
          Clustering is an unsupervised learning technique that groups similar data points together based on their
          features, without requiring labeled training data.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8 mb-12">
        <div>
          <h2 className="text-2xl font-bold mb-4">What is Clustering?</h2>
          <p className="mb-4">
            Clustering is the task of dividing data points into groups (clusters) such that data points in the same
            group are more similar to each other than to those in other groups. It's a main task of exploratory data
            analysis and a common technique for statistical data analysis.
          </p>
          <p className="mb-4">
            Unlike supervised learning methods, clustering algorithms don't require labeled training data. Instead, they
            identify natural groupings in the data based on similarity measures.
          </p>
          <h3 className="text-xl font-semibold mb-3 mt-6">Key Characteristics</h3>
          <ul className="list-disc pl-6 space-y-2">
            <li>Unsupervised learning approach (no labeled data required)</li>
            <li>Groups data points based on similarity or distance measures</li>
            <li>Helps discover hidden patterns and structures in data</li>
            <li>Evaluated using metrics like silhouette score, Davies-Bouldin index, and inertia</li>
            <li>Used for customer segmentation, anomaly detection, and data preprocessing</li>
          </ul>
        </div>
        <div className="flex items-center justify-center bg-neutral-100 rounded-lg p-6">
          <Image
            src="/placeholder.svg?height=400&width=500"
            width={500}
            height={400}
            alt="Clustering visualization showing data points grouped into clusters"
            className="rounded-md"
          />
        </div>
      </div>

      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Common Clustering Algorithms</h2>
        <div className="grid gap-8 md:grid-cols-2">
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
                <Button asChild variant="default" className="w-full sm:w-auto">
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
              <CardTitle className="text-lg">Customer Segmentation</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Grouping customers based on purchasing behavior, demographics, and preferences to target marketing
                campaigns.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Anomaly Detection</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Identifying outliers or unusual patterns in data that don't conform to expected behavior, useful for
                fraud detection.
              </p>
            </CardContent>
          </Card>
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-lg">Image Segmentation</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-neutral-700">
                Partitioning digital images into multiple segments to simplify representation and make analysis easier.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-200">
        <h2 className="text-2xl font-bold mb-4">Evaluation Metrics</h2>
        <p className="mb-4">
          Clustering algorithms are evaluated using different metrics than supervised learning models. Common evaluation
          metrics include:
        </p>
        <ul className="list-disc pl-6 space-y-2 mb-4">
          <li>
            <span className="font-semibold">Silhouette Score:</span> Measures how similar an object is to its own
            cluster compared to other clusters.
          </li>
          <li>
            <span className="font-semibold">Davies-Bouldin Index:</span> The average similarity between each cluster and
            its most similar cluster.
          </li>
          <li>
            <span className="font-semibold">Inertia:</span> The sum of squared distances of samples to their closest
            cluster center.
          </li>
          <li>
            <span className="font-semibold">Calinski-Harabasz Index:</span> The ratio of between-cluster dispersion to
            within-cluster dispersion.
          </li>
          <li>
            <span className="font-semibold">Adjusted Rand Index:</span> Measures the similarity between the true labels
            and the clustering assignments.
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
