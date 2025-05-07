import Link from "next/link"
import { BookOpen } from "lucide-react"

export default function Footer() {
  return (
    <footer className="border-t border-neutral-300 py-8 md:py-12 bg-white">
      <div className="container flex flex-col md:flex-row justify-between items-center gap-6">
        <div className="flex flex-col items-center md:items-start gap-2">
          <div className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-neutral-900" />
            <span className="font-bold text-neutral-900">ML Notebook</span>
          </div>
          <p className="text-sm text-neutral-600 text-center md:text-left">
            Interactive machine learning education platform
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-sm">
          <div className="flex flex-col gap-2">
            <h3 className="font-medium text-neutral-900">Models</h3>
            <Link href="/models/classification" className="text-neutral-600 hover:text-neutral-900">
              Classification
            </Link>
            <Link href="/models/regression" className="text-neutral-600 hover:text-neutral-900">
              Regression
            </Link>
            <Link href="/models/neural-networks" className="text-neutral-600 hover:text-neutral-900">
              Neural Networks
            </Link>
          </div>
          <div className="flex flex-col gap-2">
            <h3 className="font-medium text-neutral-900">Algorithms</h3>
            <Link href="/models/linear-regression" className="text-neutral-600 hover:text-neutral-900">
              Linear Regression
            </Link>
            <Link href="/models/polynomial-regression" className="text-neutral-600 hover:text-neutral-900">
              Polynomial Regression
            </Link>
            <Link href="/models/decision-trees" className="text-neutral-600 hover:text-neutral-900">
              Decision Trees
            </Link>
            <Link href="/models/random-forests" className="text-neutral-600 hover:text-neutral-900">
              Random Forests
            </Link>
            <Link href="/models/pca" className="text-neutral-600 hover:text-neutral-900">
              PCA
            </Link>
          </div>
          <div className="flex flex-col gap-2">
            <h3 className="font-medium text-neutral-900">Neural Networks</h3>
            <Link href="/models/mlp" className="text-neutral-600 hover:text-neutral-900">
              Multilayer Perceptron
            </Link>
            <Link href="/models/cnn" className="text-neutral-600 hover:text-neutral-900">
              CNNs
            </Link>
            <Link href="/models/rnn" className="text-neutral-600 hover:text-neutral-900">
              RNNs
            </Link>
          </div>
          <div className="flex flex-col gap-2">
            <h3 className="font-medium text-neutral-900">Resources</h3>
            <Link href="/resources/learning-path" className="text-neutral-600 hover:text-neutral-900">
              Learning Path
            </Link>
            <Link href="/resources/glossary" className="text-neutral-600 hover:text-neutral-900">
              Glossary
            </Link>
            <Link href="/resources/references" className="text-neutral-600 hover:text-neutral-900">
              References
            </Link>
          </div>
        </div>
      </div>
      <div className="container mt-8 pt-8 border-t border-neutral-300">
        <p className="text-xs text-neutral-600 text-center">
          © {new Date().getFullYear()} ML Notebook. All rights reserved.
        </p>
      </div>
    </footer>
  )
}
