import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { BookOpen, FileText, Video } from "lucide-react"

export default function ReferencesPage() {
  const books = [
    {
      title: "Deep Learning",
      authors: "Ian Goodfellow, Yoshua Bengio, and Aaron Courville",
      description:
        "A comprehensive introduction to deep learning, covering both theoretical foundations and practical applications.",
      link: "https://www.deeplearningbook.org/",
    },
    {
      title: "Pattern Recognition and Machine Learning",
      authors: "Christopher Bishop",
      description:
        "A classic textbook covering the mathematical foundations of machine learning with a Bayesian perspective.",
      link: "#",
    },
    {
      title: "The Elements of Statistical Learning",
      authors: "Trevor Hastie, Robert Tibshirani, and Jerome Friedman",
      description:
        "A comprehensive overview of statistical learning methods for data mining, inference, and prediction.",
      link: "https://web.stanford.edu/~hastie/ElemStatLearn/",
    },
    {
      title: "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow",
      authors: "Aurélien Géron",
      description: "A practical guide to implementing machine learning algorithms with popular Python libraries.",
      link: "#",
    },
  ]

  const papers = [
    {
      title: "ImageNet Classification with Deep Convolutional Neural Networks",
      authors: "Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton",
      year: 2012,
      description:
        "The seminal paper introducing AlexNet, which demonstrated the power of deep convolutional neural networks for image classification.",
      link: "#",
    },
    {
      title: "Attention Is All You Need",
      authors: "Ashish Vaswani et al.",
      year: 2017,
      description:
        "Introduced the Transformer architecture, which has revolutionized natural language processing and other sequence modeling tasks.",
      link: "#",
    },
    {
      title: "Deep Residual Learning for Image Recognition",
      authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun",
      year: 2015,
      description: "Introduced residual networks (ResNets), which enabled the training of much deeper neural networks.",
      link: "#",
    },
  ]

  const onlineCourses = [
    {
      title: "CS229: Machine Learning",
      provider: "Stanford University",
      instructor: "Andrew Ng",
      description:
        "A comprehensive introduction to machine learning covering supervised and unsupervised learning, deep learning, and reinforcement learning.",
      link: "https://cs229.stanford.edu/",
    },
    {
      title: "Deep Learning Specialization",
      provider: "Coursera",
      instructor: "Andrew Ng",
      description:
        "A series of courses covering the foundations of deep learning, CNNs, sequence models, and practical aspects of deep learning projects.",
      link: "#",
    },
    {
      title: "Practical Deep Learning for Coders",
      provider: "fast.ai",
      instructor: "Jeremy Howard and Rachel Thomas",
      description: "A top-down, practical approach to deep learning that gets students building models from day one.",
      link: "https://course.fast.ai/",
    },
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-6">References & Resources</h1>

        <p className="text-neutral-700 text-lg mb-8">
          A curated collection of books, research papers, online courses, and other resources to deepen your
          understanding of machine learning and deep learning.
        </p>

        <section className="mb-12">
          <div className="flex items-center mb-6">
            <BookOpen className="h-6 w-6 text-neutral-800 mr-2" />
            <h2 className="text-2xl font-bold text-neutral-900">Books</h2>
          </div>

          <div className="space-y-6">
            {books.map((book, index) => (
              <Card key={index} className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">{book.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="font-medium text-neutral-700 mb-2">by {book.authors}</p>
                  <p className="text-neutral-600 mb-2">{book.description}</p>
                  {book.link !== "#" && (
                    <a
                      href={book.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-neutral-900 font-medium hover:underline"
                    >
                      View Book →
                    </a>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section className="mb-12">
          <div className="flex items-center mb-6">
            <FileText className="h-6 w-6 text-neutral-800 mr-2" />
            <h2 className="text-2xl font-bold text-neutral-900">Research Papers</h2>
          </div>

          <div className="space-y-6">
            {papers.map((paper, index) => (
              <Card key={index} className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">
                    {paper.title} ({paper.year})
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="font-medium text-neutral-700 mb-2">by {paper.authors}</p>
                  <p className="text-neutral-600">{paper.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section>
          <div className="flex items-center mb-6">
            <Video className="h-6 w-6 text-neutral-800 mr-2" />
            <h2 className="text-2xl font-bold text-neutral-900">Online Courses</h2>
          </div>

          <div className="space-y-6">
            {onlineCourses.map((course, index) => (
              <Card key={index} className="border-neutral-300 bg-white">
                <CardHeader className="pb-2">
                  <CardTitle className="text-neutral-900">{course.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="font-medium text-neutral-700 mb-2">
                    {course.provider} • Instructor: {course.instructor}
                  </p>
                  <p className="text-neutral-600 mb-2">{course.description}</p>
                  {course.link !== "#" && (
                    <a
                      href={course.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-neutral-900 font-medium hover:underline"
                    >
                      View Course →
                    </a>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
