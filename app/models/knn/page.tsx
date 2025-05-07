import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import NotebookCell from "@/components/notebook-cell"
import KNNVisualization from "@/components/knn-visualization"

export default function KNNPage() {
  return (
    <div className="container mx-auto px-4 py-12">
      <div className="mb-12">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">K-Nearest Neighbors (KNN)</h1>
        <p className="text-neutral-700 max-w-3xl">
          K-Nearest Neighbors is a simple, versatile, and non-parametric supervised learning algorithm used for both
          classification and regression tasks. It makes predictions based on the majority vote (for classification) or
          average value (for regression) of the k nearest data points.
        </p>
      </div>

      <Tabs defaultValue="overview" className="mb-12">
        <TabsList className="mb-8">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="visualization">Interactive Model</TabsTrigger>
          <TabsTrigger value="code">Code Notebook</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle>How KNN Works</CardTitle>
                <CardDescription>Understanding the algorithm</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <p>
                  KNN is based on the principle that similar data points exist in close proximity. The algorithm works
                  as follows:
                </p>
                <ol className="list-decimal list-inside space-y-2">
                  <li>Store all the training data points and their labels</li>
                  <li>For a new data point, calculate the distance to all training points</li>
                  <li>Select the K nearest neighbors (closest points)</li>
                  <li>
                    For classification: Take a majority vote of the K neighbors&apos; labels
                    <br />
                    For regression: Calculate the average of the K neighbors&apos; values
                  </li>
                  <li>Assign the resulting class or value to the new data point</li>
                </ol>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Key Characteristics</CardTitle>
                <CardDescription>Strengths and limitations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-4">
                  <h3 className="font-medium text-neutral-900 mb-2">Strengths</h3>
                  <ul className="list-disc list-inside space-y-1">
                    <li>Simple to understand and implement</li>
                    <li>No training phase (lazy learning)</li>
                    <li>Naturally handles multi-class problems</li>
                    <li>Can be effective for non-linear data</li>
                    <li>Makes no assumptions about the underlying data distribution</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-medium text-neutral-900 mb-2">Limitations</h3>
                  <ul className="list-disc list-inside space-y-1">
                    <li>Computationally expensive for large datasets</li>
                    <li>Sensitive to irrelevant features and the scale of the data</li>
                    <li>Requires feature scaling</li>
                    <li>The optimal value of K needs to be determined</li>
                    <li>Performance degrades in high-dimensional spaces (curse of dimensionality)</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Distance Metrics</CardTitle>
                <CardDescription>Different ways to measure distance between points</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="mb-4">
                  The choice of distance metric is crucial for KNN and depends on the type of data. Common distance
                  metrics include:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-neutral-50 p-4 rounded-lg">
                    <h3 className="font-medium text-neutral-900 mb-2">Euclidean Distance</h3>
                    <p className="text-sm text-neutral-700 mb-2">
                      The straight-line distance between two points in Euclidean space.
                    </p>
                    <div className="bg-white p-3 rounded border border-neutral-200">
                      <p className="font-mono text-sm">
                        d(p,q) = √(Σ(q<sub>i</sub> - p<sub>i</sub>)²)
                      </p>
                    </div>
                  </div>

                  <div className="bg-neutral-50 p-4 rounded-lg">
                    <h3 className="font-medium text-neutral-900 mb-2">Manhattan Distance</h3>
                    <p className="text-sm text-neutral-700 mb-2">
                      The sum of the absolute differences of their Cartesian coordinates.
                    </p>
                    <div className="bg-white p-3 rounded border border-neutral-200">
                      <p className="font-mono text-sm">
                        d(p,q) = Σ|p<sub>i</sub> - q<sub>i</sub>|
                      </p>
                    </div>
                  </div>

                  <div className="bg-neutral-50 p-4 rounded-lg">
                    <h3 className="font-medium text-neutral-900 mb-2">Minkowski Distance</h3>
                    <p className="text-sm text-neutral-700 mb-2">
                      A generalization of Euclidean and Manhattan distance.
                    </p>
                    <div className="bg-white p-3 rounded border border-neutral-200">
                      <p className="font-mono text-sm">
                        d(p,q) = (Σ|p<sub>i</sub> - q<sub>i</sub>|<sup>r</sup>)<sup>1/r</sup>
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="visualization">
          <Card>
            <CardHeader>
              <CardTitle>Interactive KNN Model</CardTitle>
              <CardDescription>Adjust parameters to see how they affect the KNN decision boundaries</CardDescription>
            </CardHeader>
            <CardContent>
              <KNNVisualization />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="code">
          <Card>
            <CardHeader>
              <CardTitle>KNN Implementation</CardTitle>
              <CardDescription>Explore a Python implementation of KNN using scikit-learn</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <NotebookCell
                cellType="markdown"
                content={`# K-Nearest Neighbors Implementation

In this notebook, we'll implement KNN for a classification task using scikit-learn and visualize the decision boundaries.`}
              />

              <NotebookCell
                cellType="code"
                content={`import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# Set the style for our plots
sns.set_style('whitegrid')
np.random.seed(42)`}
              />

              <NotebookCell
                cellType="markdown"
                content={`## Data Preparation

First, let's generate a synthetic dataset for classification.`}
              />

              <NotebookCell
                cellType="code"
                content={`# Generate a synthetic dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,  # 2 features for easy visualization
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', 
            edgecolors='k', s=50, alpha=0.8)
plt.title('Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.show()`}
              />

              <NotebookCell
                cellType="markdown"
                content={`## Training the KNN Model

Now, let's train a KNN classifier with k=5 and visualize the decision boundaries.`}
              />

              <NotebookCell
                cellType="code"
                content={`# Create and train the KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with k={k}: {accuracy:.4f}")

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model, title):
    h = 0.02  # Step size in the mesh
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict class for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                edgecolors='k', s=50, alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

# Plot the decision boundaries
plot_decision_boundaries(X_train, y_train, knn, f'KNN Decision Boundaries (k={k})')`}
              />

              <NotebookCell
                cellType="markdown"
                content={`## Finding the Optimal k Value

Let's try different values of k and see how they affect the model's performance.`}
              />

              <NotebookCell
                cellType="code"
                content={`# Test different k values
k_values = list(range(1, 31, 2))  # Odd numbers from 1 to 30
train_accuracies = []
test_accuracies = []

for k in k_values:
    # Create and train the model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Calculate training accuracy
    train_pred = knn.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_accuracies.append(train_acc)
    
    # Calculate test accuracy
    test_pred = knn.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_accuracies.append(test_acc)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(k_values, train_accuracies, 'o-', label='Training Accuracy')
plt.plot(k_values, test_accuracies, 's-', label='Test Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Performance vs. k Value')
plt.legend()
plt.grid(True)
plt.show()

# Find the optimal k value
optimal_k = k_values[np.argmax(test_accuracies)]
print(f"Optimal k value: {optimal_k}")
print(f"Best test accuracy: {max(test_accuracies):.4f}")

# Train the model with the optimal k value
optimal_knn = KNeighborsClassifier(n_neighbors=optimal_k)
optimal_knn.fit(X_train, y_train)

# Plot the decision boundaries with the optimal k
plot_decision_boundaries(X_train, y_train, optimal_knn, 
                        f'KNN Decision Boundaries (Optimal k={optimal_k})')`}
              />

              <NotebookCell
                cellType="markdown"
                content={`## Effect of Distance Metrics

Let's compare different distance metrics in KNN.`}
              />

              <NotebookCell
                cellType="code"
                content={`# Compare different distance metrics
metrics = ['euclidean', 'manhattan', 'minkowski']
plt.figure(figsize=(15, 5))

for i, metric in enumerate(metrics, 1):
    # Create and train the model
    knn = KNeighborsClassifier(n_neighbors=optimal_k, metric=metric)
    knn.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Plot the decision boundaries
    plt.subplot(1, 3, i)
    
    # Create a mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict class for each point in the mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', 
                edgecolors='k', s=30, alpha=0.8)
    
    plt.title(f'{metric.capitalize()} (Acc: {accuracy:.4f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()`}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
