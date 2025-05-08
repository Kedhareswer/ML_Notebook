import Link from "next/link"
import { BookOpen } from "lucide-react"

export default function Footer() {
  return (
    <footer className="border-t border-gray-300 py-8 md:py-12 bg-white">
      <div className="container flex flex-col md:flex-row justify-between items-center gap-6">
        <div className="flex flex-col items-center md:items-start gap-2">
          <div className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-black" />
            <span className="font-bold text-black">ML Notebook</span>
          </div>
          <p className="text-sm text-gray-600 text-center md:text-left">
            Interactive machine learning education platform
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-sm">
          <div className="flex flex-col gap-2">
            <h3 className="font-medium text-black">Models</h3>
            <Link href="/models/classification" className="text-gray-600 hover:text-black">
              Classification
            </Link>
            <Link href="/models/regression" className="text-gray-600 hover:text-black">
              Regression
            </Link>
            <Link href="/models/neural-networks" className="text-gray-600 hover:text-black">
              Neural Networks
            </Link>
          </div>
          <div className="flex flex-col gap-2">
            <h3 className="font-medium text-black">Algorithms</h3>
            <Link href="/models/linear-regression" className="text-gray-600 hover:text-black">
              Linear Regression
            </Link>
            <Link href="/models/polynomial-regression" className="text-gray-600 hover:text-black">
              Polynomial Regression
            </Link>
            <Link href="/models/decision-trees" className="text-gray-600 hover:text-black">
              Decision Trees
            </Link>
            <Link href="/models/random-forests" className="text-gray-600 hover:text-black">
              Random Forests
            </Link>
            <Link href="/models/pca" className="text-gray-600 hover:text-black">
              PCA
            </Link>
          </div>
          <div className="flex flex-col gap-2">
            <h3 className="font-medium text-black">Neural Networks</h3>
            <Link href="/models/mlp" className="text-gray-600 hover:text-black">
              Multilayer Perceptron
            </Link>
            <Link href="/models/cnn" className="text-gray-600 hover:text-black">
              CNNs
            </Link>
            <Link href="/models/rnn" className="text-gray-600 hover:text-black">
              RNNs
            </Link>
          </div>
          <div className="flex flex-col gap-2">
            <h3 className="font-medium text-black">Resources</h3>
            <Link href="/resources/learning-path" className="text-gray-600 hover:text-black">
              Learning Path
            </Link>
            <Link href="/resources/glossary" className="text-gray-600 hover:text-black">
              Glossary
            </Link>
            <Link href="/resources/references" className="text-gray-600 hover:text-black">
              References
            </Link>
          </div>
        </div>
      </div>
      <div className="container mt-8 pt-8 border-t border-gray-300">
        <p className="text-xs text-gray-600 text-center">
          © {new Date().getFullYear()} ML Notebook. All rights reserved.
        </p>
      </div>
    </footer>
  )
}
