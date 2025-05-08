"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import {
  SearchIcon,
  X,
  ArrowRight,
  BookOpen,
  Brain,
  GitBranch,
  LineChart,
  BarChart,
  Network,
  PieChart,
  Layers,
} from "lucide-react"
import { cn } from "@/lib/utils"
import Link from "next/link"

// Define the search data structure
type SearchItem = {
  title: string
  description: string
  category: string
  path: string
  icon: JSX.Element
  keywords: string[]
}

// Search data - in a real app, this might come from an API or generated at build time
const searchData: SearchItem[] = [
  {
    title: "Linear Regression",
    description: "Learn how linear regression works and how to implement it",
    category: "Regression",
    path: "/models/linear-regression",
    icon: <LineChart className="h-4 w-4" />,
    keywords: ["regression", "linear", "continuous", "prediction", "supervised learning"],
  },
  {
    title: "Polynomial Regression",
    description: "Extend linear models to capture non-linear relationships",
    category: "Regression",
    path: "/models/polynomial-regression",
    icon: <LineChart className="h-4 w-4" />,
    keywords: ["regression", "polynomial", "non-linear", "curve fitting", "supervised learning"],
  },
  {
    title: "Ridge & Lasso Regression",
    description: "Regularization techniques to prevent overfitting",
    category: "Regression",
    path: "/models/regularized-regression",
    icon: <LineChart className="h-4 w-4" />,
    keywords: ["regression", "regularization", "ridge", "lasso", "overfitting", "supervised learning"],
  },
  {
    title: "Logistic Regression",
    description: "Understand the mathematics of logistic regression",
    category: "Classification",
    path: "/models/logistic-regression",
    icon: <BarChart className="h-4 w-4" />,
    keywords: ["classification", "logistic", "binary", "probability", "supervised learning"],
  },
  {
    title: "Decision Trees",
    description: "Understand decision trees and their applications",
    category: "Classification",
    path: "/models/decision-trees",
    icon: <GitBranch className="h-4 w-4" />,
    keywords: ["classification", "decision tree", "tree-based", "supervised learning"],
  },
  {
    title: "Support Vector Machines",
    description: "Explore the mathematics behind SVMs",
    category: "Classification",
    path: "/models/svm",
    icon: <BarChart className="h-4 w-4" />,
    keywords: ["classification", "svm", "kernel", "hyperplane", "supervised learning"],
  },
  {
    title: "Random Forests",
    description: "Learn how ensemble methods improve performance",
    category: "Classification",
    path: "/models/random-forests",
    icon: <Layers className="h-4 w-4" />,
    keywords: ["classification", "random forest", "ensemble", "bagging", "supervised learning"],
  },
  {
    title: "K-Means Clustering",
    description: "Explore how K-means partitions data into clusters",
    category: "Clustering",
    path: "/models/kmeans",
    icon: <PieChart className="h-4 w-4" />,
    keywords: ["clustering", "k-means", "unsupervised learning", "centroid"],
  },
  {
    title: "Hierarchical Clustering",
    description: "Understand how hierarchical clustering works",
    category: "Clustering",
    path: "/models/hierarchical-clustering",
    icon: <GitBranch className="h-4 w-4" />,
    keywords: ["clustering", "hierarchical", "dendrogram", "unsupervised learning"],
  },
  {
    title: "Principal Component Analysis",
    description: "Learn how PCA transforms high-dimensional data",
    category: "Dimensionality Reduction",
    path: "/models/pca",
    icon: <Layers className="h-4 w-4" />,
    keywords: ["dimensionality reduction", "pca", "principal components", "unsupervised learning"],
  },
  {
    title: "Multilayer Perceptron",
    description: "Learn about the building blocks of deep learning",
    category: "Neural Networks",
    path: "/models/mlp",
    icon: <Network className="h-4 w-4" />,
    keywords: ["neural network", "mlp", "perceptron", "deep learning", "supervised learning"],
  },
  {
    title: "Convolutional Neural Networks",
    description: "Visualize how CNNs process images",
    category: "Neural Networks",
    path: "/models/cnn",
    icon: <Brain className="h-4 w-4" />,
    keywords: ["neural network", "cnn", "convolutional", "deep learning", "computer vision", "image processing"],
  },
  {
    title: "Recurrent Neural Networks",
    description: "See how RNNs handle sequential data",
    category: "Neural Networks",
    path: "/models/rnn",
    icon: <Network className="h-4 w-4" />,
    keywords: ["neural network", "rnn", "recurrent", "deep learning", "sequence", "time series", "nlp"],
  },
  {
    title: "Transformers",
    description: "Explore the architecture behind modern NLP models",
    category: "Neural Networks",
    path: "/models/transformers",
    icon: <Brain className="h-4 w-4" />,
    keywords: ["neural network", "transformer", "attention", "deep learning", "nlp", "language model"],
  },
  {
    title: "Model Comparison",
    description: "Compare different models and their performance",
    category: "Resources",
    path: "/models/comparison",
    icon: <BarChart className="h-4 w-4" />,
    keywords: ["comparison", "performance", "metrics", "evaluation", "model selection"],
  },
  {
    title: "Learning Path",
    description: "A structured approach to learning machine learning",
    category: "Resources",
    path: "/resources/learning-path",
    icon: <BookOpen className="h-4 w-4" />,
    keywords: ["learning", "path", "guide", "roadmap", "curriculum"],
  },
  {
    title: "Glossary",
    description: "Key terms and definitions in machine learning",
    category: "Resources",
    path: "/resources/glossary",
    icon: <BookOpen className="h-4 w-4" />,
    keywords: ["glossary", "terms", "definitions", "vocabulary"],
  },
  {
    title: "Cheat Sheets",
    description: "Quick reference guides for models and algorithms",
    category: "Resources",
    path: "/resources/cheat-sheets",
    icon: <BookOpen className="h-4 w-4" />,
    keywords: ["cheat sheet", "reference", "guide", "summary"],
  },
]

export function SearchButton() {
  const [open, setOpen] = useState(false)

  // Handle keyboard shortcut (Ctrl+K or Cmd+K)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault()
        setOpen((prev) => !prev)
      }
      if (e.key === "Escape") {
        setOpen(false)
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        className="hidden md:flex items-center gap-2 text-gray-600 hover:text-black"
        onClick={() => setOpen(true)}
      >
        <SearchIcon className="h-4 w-4" />
        <span>Search</span>
        <kbd className="ml-2 pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border border-gray-300 bg-gray-100 px-1.5 font-mono text-[10px] font-medium text-gray-600">
          <span className="text-xs">⌘</span>K
        </kbd>
      </Button>
      <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setOpen(true)}>
        <SearchIcon className="h-5 w-5" />
      </Button>
      <SearchDialog open={open} setOpen={setOpen} />
    </>
  )
}

function SearchDialog({ open, setOpen }: { open: boolean; setOpen: (open: boolean) => void }) {
  const [searchQuery, setSearchQuery] = useState("")
  const [results, setResults] = useState<SearchItem[]>([])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)
  const router = useRouter()

  // Focus input when dialog opens
  useEffect(() => {
    if (open && inputRef.current) {
      setTimeout(() => {
        inputRef.current?.focus()
      }, 50)
    }
  }, [open])

  // Handle search
  useEffect(() => {
    if (!searchQuery.trim()) {
      setResults([])
      return
    }

    const query = searchQuery.toLowerCase()
    const filtered = searchData.filter((item) => {
      // Check title, description, and keywords
      return (
        item.title.toLowerCase().includes(query) ||
        item.description.toLowerCase().includes(query) ||
        item.category.toLowerCase().includes(query) ||
        item.keywords.some((keyword) => keyword.toLowerCase().includes(query))
      )
    })

    setResults(filtered)
    setSelectedIndex(0)
  }, [searchQuery])

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault()
      setSelectedIndex((prev) => (prev < results.length - 1 ? prev + 1 : prev))
      // Scroll into view if needed
      const selectedElement = resultsRef.current?.children[selectedIndex + 1] as HTMLElement
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: "nearest" })
      }
    } else if (e.key === "ArrowUp") {
      e.preventDefault()
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : 0))
      // Scroll into view if needed
      const selectedElement = resultsRef.current?.children[selectedIndex - 1] as HTMLElement
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: "nearest" })
      }
    } else if (e.key === "Enter" && results.length > 0) {
      e.preventDefault()
      const selectedResult = results[selectedIndex]
      if (selectedResult) {
        router.push(selectedResult.path)
        setOpen(false)
      }
    }
  }

  // Group results by category
  const groupedResults = results.reduce(
    (acc, item) => {
      if (!acc[item.category]) {
        acc[item.category] = []
      }
      acc[item.category].push(item)
      return acc
    },
    {} as Record<string, SearchItem[]>,
  )

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-[550px] p-0">
        <div className="relative">
          <SearchIcon className="absolute left-4 top-3.5 h-5 w-5 text-gray-500" />
          <Input
            ref={inputRef}
            placeholder="Search for models, concepts, or topics..."
            className="pl-11 pr-10 py-6 border-0 border-b rounded-t-lg rounded-b-none focus-visible:ring-0 focus-visible:ring-offset-0"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-2 top-2.5 h-7 w-7 text-gray-500 hover:text-black"
              onClick={() => setSearchQuery("")}
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>

        <div className="max-h-[60vh] overflow-y-auto p-2" ref={resultsRef}>
          {results.length === 0 && searchQuery && (
            <div className="p-4 text-center text-gray-500">No results found for "{searchQuery}"</div>
          )}

          {Object.entries(groupedResults).map(([category, items]) => (
            <div key={category} className="mb-4">
              <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">{category}</div>
              <div className="space-y-1">
                {items.map((item, index) => {
                  const absoluteIndex = results.findIndex((r) => r.path === item.path)
                  return (
                    <Link
                      key={item.path}
                      href={item.path}
                      className={cn(
                        "flex items-start gap-3 px-3 py-2 rounded-md text-sm",
                        absoluteIndex === selectedIndex ? "bg-gray-100 text-black" : "hover:bg-gray-50 text-gray-700",
                      )}
                      onClick={() => setOpen(false)}
                    >
                      <div className="flex-shrink-0 mt-1 text-gray-500">{item.icon}</div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium">{item.title}</div>
                        <div className="text-xs text-gray-500 truncate">{item.description}</div>
                      </div>
                      <ArrowRight className="flex-shrink-0 h-4 w-4 mt-1 text-gray-400" />
                    </Link>
                  )
                })}
              </div>
            </div>
          ))}

          {searchQuery && results.length > 0 && (
            <div className="px-3 py-2 text-xs text-gray-500 border-t border-gray-200 mt-2">
              <div className="flex items-center justify-between">
                <span>
                  {results.length} result{results.length !== 1 ? "s" : ""}
                </span>
                <div className="flex gap-1">
                  <kbd className="inline-flex h-5 items-center gap-1 rounded border border-gray-300 bg-gray-100 px-1.5 font-mono text-[10px] font-medium text-gray-600">
                    ↑
                  </kbd>
                  <kbd className="inline-flex h-5 items-center gap-1 rounded border border-gray-300 bg-gray-100 px-1.5 font-mono text-[10px] font-medium text-gray-600">
                    ↓
                  </kbd>
                  <span className="text-gray-500">to navigate</span>
                  <kbd className="inline-flex h-5 items-center gap-1 rounded border border-gray-300 bg-gray-100 px-1.5 font-mono text-[10px] font-medium text-gray-600">
                    ↵
                  </kbd>
                  <span className="text-gray-500">to select</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
