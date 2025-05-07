"use client"

import { useState } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Search, BookOpen, Tag, ArrowUpDown } from "lucide-react"

export default function GlossaryPage() {
  const [searchTerm, setSearchTerm] = useState("")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc")

  // Comprehensive glossary of machine learning terms
  const glossaryItems = [
    {
      term: "Activation Function",
      definition:
        "A mathematical function that determines the output of a neural network node. Common examples include ReLU, Sigmoid, and Tanh.",
      category: "Neural Networks",
    },
    {
      term: "Backpropagation",
      definition:
        "An algorithm for training neural networks that calculates gradients of the loss function with respect to the weights, propagating from output to input layers.",
      category: "Neural Networks",
    },
    {
      term: "Batch Normalization",
      definition:
        "A technique to normalize the inputs of each layer to improve training stability and speed by reducing internal covariate shift.",
      category: "Neural Networks",
    },
    {
      term: "Bias",
      definition:
        "1) A parameter in machine learning models that allows the model to fit the data better. 2) A systematic error that causes a model to favor certain outcomes.",
      category: "General",
    },
    {
      term: "Classification",
      definition:
        "A supervised learning task where the model predicts discrete class labels or categories for input data.",
      category: "Tasks",
    },
    {
      term: "Clustering",
      definition:
        "An unsupervised learning technique that groups similar data points together based on certain features.",
      category: "Tasks",
    },
    {
      term: "Convolutional Neural Network (CNN)",
      definition:
        "A type of neural network designed for processing grid-like data such as images, using convolutional layers to detect spatial patterns.",
      category: "Neural Networks",
    },
    {
      term: "Cross-Validation",
      definition:
        "A resampling procedure used to evaluate machine learning models where the dataset is split into multiple subsets for training and validation.",
      category: "Evaluation",
    },
    {
      term: "Decision Tree",
      definition:
        "A tree-like model that makes decisions based on feature values, splitting the data into branches at decision nodes.",
      category: "Models",
    },
    {
      term: "Deep Learning",
      definition:
        "A subset of machine learning using neural networks with many layers (deep neural networks) to model complex patterns in data.",
      category: "General",
    },
    {
      term: "Dropout",
      definition:
        "A regularization technique where randomly selected neurons are ignored during training to prevent overfitting.",
      category: "Neural Networks",
    },
    {
      term: "Embedding",
      definition:
        "A technique to represent discrete variables as continuous vectors in a lower-dimensional space, often used for text or categorical data.",
      category: "Feature Engineering",
    },
    {
      term: "Ensemble Learning",
      definition: "A technique that combines multiple machine learning models to improve performance and robustness.",
      category: "Models",
    },
    {
      term: "Epoch",
      definition:
        "One complete pass through the entire training dataset during the training of a machine learning model.",
      category: "Training",
    },
    {
      term: "Feature",
      definition:
        "An individual measurable property or characteristic of a phenomenon being observed, used as input to a machine learning model.",
      category: "Feature Engineering",
    },
    {
      term: "Feature Engineering",
      definition:
        "The process of selecting, transforming, or creating features from raw data to improve model performance.",
      category: "Feature Engineering",
    },
    {
      term: "Gradient Descent",
      definition:
        "An optimization algorithm that iteratively adjusts parameters to minimize a loss function by moving in the direction of steepest descent.",
      category: "Optimization",
    },
    {
      term: "Hyperparameter",
      definition:
        "A parameter whose value is set before the learning process begins, as opposed to parameters that are learned during training.",
      category: "Training",
    },
    {
      term: "K-Means Clustering",
      definition:
        "An unsupervised learning algorithm that partitions data into K clusters, where each data point belongs to the cluster with the nearest mean.",
      category: "Models",
    },
    {
      term: "Learning Rate",
      definition:
        "A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.",
      category: "Optimization",
    },
    {
      term: "Loss Function",
      definition:
        "A function that measures the difference between the model's predictions and the actual target values, used to guide the optimization process.",
      category: "Optimization",
    },
    {
      term: "LSTM (Long Short-Term Memory)",
      definition:
        "A type of recurrent neural network architecture designed to handle the vanishing gradient problem and better capture long-term dependencies in sequential data.",
      category: "Neural Networks",
    },
    {
      term: "Neural Network",
      definition:
        "A computational model inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers that process information.",
      category: "Neural Networks",
    },
    {
      term: "Normalization",
      definition:
        "The process of scaling features to a standard range, typically between 0 and 1 or -1 and 1, to improve model training and performance.",
      category: "Feature Engineering",
    },
    {
      term: "Overfitting",
      definition:
        "A modeling error where a model learns the training data too well, including noise and outliers, resulting in poor generalization to new data.",
      category: "Training",
    },
    {
      term: "Precision",
      definition:
        "A metric that measures the proportion of true positive predictions among all positive predictions made by a model.",
      category: "Evaluation",
    },
    {
      term: "Random Forest",
      definition:
        "An ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification or mean prediction for regression.",
      category: "Models",
    },
    {
      term: "Recall",
      definition:
        "A metric that measures the proportion of true positive predictions among all actual positive instances in the data.",
      category: "Evaluation",
    },
    {
      term: "Recurrent Neural Network (RNN)",
      definition:
        "A type of neural network designed for sequential data, with connections that form cycles to maintain memory of previous inputs.",
      category: "Neural Networks",
    },
    {
      term: "Regression",
      definition:
        "A supervised learning task where the model predicts continuous numerical values rather than discrete categories.",
      category: "Tasks",
    },
    {
      term: "Regularization",
      definition:
        "Techniques used to prevent overfitting by adding a penalty term to the loss function or modifying the model architecture.",
      category: "Training",
    },
    {
      term: "Reinforcement Learning",
      definition:
        "A type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.",
      category: "General",
    },
    {
      term: "Semi-Supervised Learning",
      definition:
        "A learning approach that combines a small amount of labeled data with a large amount of unlabeled data during training.",
      category: "General",
    },
    {
      term: "Supervised Learning",
      definition:
        "A type of machine learning where the model is trained on labeled data, learning to map inputs to known outputs.",
      category: "General",
    },
    {
      term: "Support Vector Machine (SVM)",
      definition:
        "A supervised learning model that finds the optimal hyperplane to separate different classes in the feature space.",
      category: "Models",
    },
    {
      term: "Transfer Learning",
      definition:
        "A technique where a model developed for one task is reused as the starting point for a model on a second task, often saving training time and improving performance.",
      category: "Training",
    },
    {
      term: "Underfitting",
      definition:
        "A modeling error where a model is too simple to capture the underlying pattern in the data, resulting in poor performance on both training and new data.",
      category: "Training",
    },
    {
      term: "Unsupervised Learning",
      definition:
        "A type of machine learning where the model is trained on unlabeled data, discovering patterns and relationships without explicit guidance.",
      category: "General",
    },
    {
      term: "Validation Set",
      definition:
        "A subset of the data used to tune hyperparameters and evaluate model performance during training, separate from the test set.",
      category: "Evaluation",
    },
    {
      term: "Variance",
      definition:
        "A measure of how much the predictions of a model change when trained on different subsets of the training data.",
      category: "Evaluation",
    },
    {
      term: "Weight",
      definition:
        "A parameter in a neural network or other machine learning model that determines the strength of connection between nodes or the importance of features.",
      category: "Neural Networks",
    },
    // New terms related to model types
    {
      term: "Batch Size",
      definition:
        "The number of training examples utilized in one iteration of model training. It affects both the optimization process and the time required to train the model.",
      category: "Training",
    },
    {
      term: "Confusion Matrix",
      definition:
        "A table used to describe the performance of a classification model, showing the counts of true positives, false positives, true negatives, and false negatives.",
      category: "Evaluation",
    },
    {
      term: "Convolution",
      definition:
        "A mathematical operation that applies a filter to an input to create a feature map that summarizes the presence of detected features in the input.",
      category: "Neural Networks",
    },
    {
      term: "Early Stopping",
      definition:
        "A form of regularization used to avoid overfitting by stopping training when performance on a validation set starts to degrade.",
      category: "Training",
    },
    {
      term: "Explainable AI (XAI)",
      definition:
        "Artificial intelligence systems whose actions can be easily understood by humans. It contrasts with the 'black box' concept in machine learning.",
      category: "General",
    },
    {
      term: "F1 Score",
      definition:
        "The harmonic mean of precision and recall, providing a single metric that balances both concerns. Particularly useful for imbalanced datasets.",
      category: "Evaluation",
    },
    {
      term: "Gini Impurity",
      definition:
        "A measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.",
      category: "Models",
    },
    {
      term: "Hyperparameter Tuning",
      definition:
        "The process of finding the optimal hyperparameters for a machine learning algorithm to maximize its performance on a specific task.",
      category: "Training",
    },
    {
      term: "Information Gain",
      definition:
        "A measure used in decision trees that quantifies how much 'information' a feature gives us about the class. It's based on the concept of entropy from information theory.",
      category: "Models",
    },
    {
      term: "L1 Regularization (Lasso)",
      definition:
        "A regularization technique that adds the absolute value of the magnitude of coefficients as a penalty term to the loss function, promoting sparsity in the model.",
      category: "Optimization",
    },
    {
      term: "L2 Regularization (Ridge)",
      definition:
        "A regularization technique that adds the squared magnitude of coefficients as a penalty term to the loss function, preventing any single feature from having too much influence.",
      category: "Optimization",
    },
    {
      term: "Mean Absolute Error (MAE)",
      definition:
        "A measure of errors between paired observations expressing the same phenomenon, calculated as the average of the absolute differences between prediction and actual observation.",
      category: "Evaluation",
    },
    {
      term: "Mean Squared Error (MSE)",
      definition:
        "A measure of the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.",
      category: "Evaluation",
    },
    {
      term: "One-Hot Encoding",
      definition:
        "A process by which categorical variables are converted into a form that could be provided to machine learning algorithms to improve predictions.",
      category: "Feature Engineering",
    },
    {
      term: "Pooling",
      definition:
        "A downsampling operation used in CNNs that reduces the dimensionality of feature maps, retaining the most important information while reducing computation.",
      category: "Neural Networks",
    },
    {
      term: "R-squared (Coefficient of Determination)",
      definition:
        "A statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.",
      category: "Evaluation",
    },
    {
      term: "Root Mean Squared Error (RMSE)",
      definition:
        "The square root of the mean of the squared differences between predicted values and observed values, providing an error measure in the same units as the target variable.",
      category: "Evaluation",
    },
    {
      term: "Self-Attention",
      definition:
        "A mechanism used in transformer models that allows the model to weigh the importance of different words in a sequence when making predictions, regardless of their position.",
      category: "Neural Networks",
    },
    {
      term: "Softmax Function",
      definition:
        "A function that converts a vector of real numbers into a probability distribution. It's often used as the activation function in the output layer of neural networks for multi-class classification.",
      category: "Neural Networks",
    },
    {
      term: "Stochastic Gradient Descent (SGD)",
      definition:
        "A variant of gradient descent that uses a single training example or a small batch to compute the gradient and update the parameters, making it more efficient for large datasets.",
      category: "Optimization",
    },
    {
      term: "Tokenization",
      definition:
        "The process of breaking down text into smaller units called tokens, which can be words, characters, or subwords, for processing in NLP tasks.",
      category: "Feature Engineering",
    },
    {
      term: "Transformer",
      definition:
        "A deep learning model architecture that relies entirely on self-attention mechanisms without using recurrent neural networks, primarily used for NLP tasks.",
      category: "Neural Networks",
    },
    {
      term: "Vanishing Gradient Problem",
      definition:
        "A difficulty found in training neural networks with gradient-based methods and backpropagation, where the gradient becomes extremely small, effectively preventing the weights from changing value.",
      category: "Neural Networks",
    },
  ]

  // Get unique categories
  const categories = Array.from(new Set(glossaryItems.map((item) => item.category)))

  // Filter items based on search term
  const filteredItems = glossaryItems.filter(
    (item) =>
      item.term.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.definition.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  // Sort items
  const sortedItems = [...filteredItems].sort((a, b) => {
    if (sortOrder === "asc") {
      return a.term.localeCompare(b.term)
    } else {
      return b.term.localeCompare(a.term)
    }
  })

  // Group items by first letter for alphabetical view
  const groupedByLetter = sortedItems.reduce(
    (acc, item) => {
      const firstLetter = item.term[0].toUpperCase()
      if (!acc[firstLetter]) {
        acc[firstLetter] = []
      }
      acc[firstLetter].push(item)
      return acc
    },
    {} as Record<string, typeof glossaryItems>,
  )

  // Get sorted letters for alphabetical navigation
  const letters = Object.keys(groupedByLetter).sort()

  return (
    <div className="container mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-6">Machine Learning Glossary</h1>
      <p className="text-lg mb-8">
        A comprehensive reference of machine learning terms, concepts, and techniques to help you understand the field.
      </p>

      <div className="flex flex-col md:flex-row gap-4 mb-8">
        <div className="relative flex-grow">
          <Search className="absolute left-3 top-3 h-4 w-4 text-neutral-500" />
          <Input
            placeholder="Search terms or definitions..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        <Button
          variant="outline"
          onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
          className="flex items-center gap-2"
        >
          <ArrowUpDown className="h-4 w-4" />
          Sort {sortOrder === "asc" ? "A-Z" : "Z-A"}
        </Button>
      </div>

      <Tabs defaultValue="all">
        <TabsList className="mb-6">
          <TabsTrigger value="all" className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            All Terms
          </TabsTrigger>
          <TabsTrigger value="categories" className="flex items-center gap-2">
            <Tag className="h-4 w-4" />
            By Category
          </TabsTrigger>
          <TabsTrigger value="alphabetical" className="flex items-center gap-2">
            <span className="font-bold">A</span>
            <span className="font-bold">Z</span>
            Alphabetical
          </TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-6">
          {sortedItems.length > 0 ? (
            sortedItems.map((item) => (
              <Card key={item.term} className="overflow-hidden">
                <CardHeader className="bg-neutral-50 pb-3">
                  <div className="flex justify-between items-start">
                    <CardTitle className="text-xl">{item.term}</CardTitle>
                    <span className="text-xs px-2 py-1 bg-neutral-200 rounded-full">{item.category}</span>
                  </div>
                </CardHeader>
                <CardContent className="pt-4">
                  <p>{item.definition}</p>
                </CardContent>
              </Card>
            ))
          ) : (
            <div className="text-center py-12">
              <p className="text-lg text-neutral-500">No terms found matching your search.</p>
            </div>
          )}
        </TabsContent>

        <TabsContent value="categories">
          {categories.map((category) => {
            const categoryItems = sortedItems.filter((item) => item.category === category)
            if (categoryItems.length === 0) return null

            return (
              <div key={category} className="mb-10">
                <h2 className="text-2xl font-bold mb-4">{category}</h2>
                <div className="space-y-4">
                  {categoryItems.map((item) => (
                    <Card key={item.term} className="overflow-hidden">
                      <CardHeader className="bg-neutral-50 pb-3">
                        <CardTitle className="text-xl">{item.term}</CardTitle>
                      </CardHeader>
                      <CardContent className="pt-4">
                        <p>{item.definition}</p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )
          })}
        </TabsContent>

        <TabsContent value="alphabetical">
          <div className="flex flex-wrap gap-2 mb-6">
            {letters.map((letter) => (
              <a
                key={letter}
                href={`#letter-${letter}`}
                className="w-8 h-8 flex items-center justify-center border rounded-md hover:bg-neutral-100"
              >
                {letter}
              </a>
            ))}
          </div>

          {letters.map((letter) => (
            <div key={letter} id={`letter-${letter}`} className="mb-10">
              <h2 className="text-2xl font-bold mb-4 flex items-center">
                <span className="w-10 h-10 flex items-center justify-center bg-neutral-100 rounded-full mr-3">
                  {letter}
                </span>
                Terms
              </h2>
              <div className="space-y-4">
                {groupedByLetter[letter].map((item) => (
                  <Card key={item.term} className="overflow-hidden">
                    <CardHeader className="bg-neutral-50 pb-3">
                      <div className="flex justify-between items-start">
                        <CardTitle className="text-xl">{item.term}</CardTitle>
                        <span className="text-xs px-2 py-1 bg-neutral-200 rounded-full">{item.category}</span>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-4">
                      <p>{item.definition}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  )
}
