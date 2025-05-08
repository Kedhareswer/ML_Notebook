import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight, Info, Code, BarChart } from "lucide-react"
import NotebookCell from "@/components/notebook-cell"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import LinearRegressionVisualization from "@/components/interactive-visualizations/linear-regression-viz"

export default function PolynomialRegressionPage() {
  return (
    <div className="container mx-auto px-4 py-12 max-w-5xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">Polynomial Regression</h1>
        <p className="text-neutral-700 text-lg">Extending linear models to capture non-linear relationships in data</p>
      </div>

      <Tabs defaultValue="overview" className="mb-12">
        <TabsList className="grid w-full grid-cols-3 mb-8">
          <TabsTrigger value="overview" className="flex items-center gap-2">
            <Info className="h-4 w-4" /> Overview
          </TabsTrigger>
          <TabsTrigger value="demo" className="flex items-center gap-2">
            <BarChart className="h-4 w-4" /> Interactive Demo
          </TabsTrigger>
          <TabsTrigger value="code" className="flex items-center gap-2">
            <Code className="h-4 w-4" /> Implementation
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-8">
          <section>
            <h2 className="text-2xl font-semibold text-neutral-800 mb-4">What is Polynomial Regression?</h2>
            <p className="text-neutral-700 mb-4">
              Polynomial regression is an extension of linear regression that models the relationship between the
              independent variable x and the dependent variable y as an nth degree polynomial. Unlike linear regression,
              which fits a straight line to the data, polynomial regression can capture more complex, non-linear
              patterns.
            </p>
            <p className="text-neutral-700">The general form of a polynomial regression model is:</p>
            <div className="py-4 px-6 bg-neutral-100 rounded-lg my-4 font-mono text-neutral-800">
              y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ + ε
            </div>
            <p className="text-neutral-700">
              where β₀, β₁, β₂, ..., βₙ are the regression coefficients and ε is the error term.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-neutral-800 mb-4">Key Concepts</h2>
            <div className="grid gap-6 md:grid-cols-2">
              <div className="bg-white p-6 rounded-lg border border-neutral-300">
                <h3 className="text-xl font-medium text-neutral-800 mb-2">Degree of Polynomial</h3>
                <p className="text-neutral-700">
                  The highest power of the independent variable in the polynomial equation. Higher degrees can fit more
                  complex patterns but risk overfitting.
                </p>
              </div>
              <div className="bg-white p-6 rounded-lg border border-neutral-300">
                <h3 className="text-xl font-medium text-neutral-800 mb-2">Basis Functions</h3>
                <p className="text-neutral-700">
                  Polynomial terms (x, x², x³, etc.) serve as basis functions that transform the original features into
                  a higher-dimensional space.
                </p>
              </div>
              <div className="bg-white p-6 rounded-lg border border-neutral-300">
                <h3 className="text-xl font-medium text-neutral-800 mb-2">Overfitting Risk</h3>
                <p className="text-neutral-700">
                  Higher-degree polynomials can lead to overfitting, where the model captures noise in the training data
                  rather than the underlying pattern.
                </p>
              </div>
              <div className="bg-white p-6 rounded-lg border border-neutral-300">
                <h3 className="text-xl font-medium text-neutral-800 mb-2">Feature Transformation</h3>
                <p className="text-neutral-700">
                  Polynomial regression can be implemented as a linear regression model after transforming the input
                  features to include polynomial terms.
                </p>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-neutral-800 mb-4">When to Use Polynomial Regression</h2>
            <ul className="list-disc pl-6 space-y-2 text-neutral-700">
              <li>When the relationship between variables follows a curvilinear pattern</li>
              <li>When linear models show systematic errors in residual plots</li>
              <li>When domain knowledge suggests non-linear relationships</li>
              <li>When modeling phenomena with diminishing returns or saturation effects</li>
              <li>As a simple approach to capture non-linearity before trying more complex models</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-neutral-800 mb-4">Comparison with Other Regression Models</h2>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="bg-neutral-100">
                    <th className="border border-neutral-300 px-4 py-2 text-left">Model</th>
                    <th className="border border-neutral-300 px-4 py-2 text-left">Strengths</th>
                    <th className="border border-neutral-300 px-4 py-2 text-left">Weaknesses</th>
                    <th className="border border-neutral-300 px-4 py-2 text-left">Best Use Cases</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="border border-neutral-300 px-4 py-2 font-medium">Linear Regression</td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>Simple and interpretable</li>
                        <li>Computationally efficient</li>
                        <li>Less prone to overfitting</li>
                      </ul>
                    </td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>Cannot capture non-linear relationships</li>
                        <li>Limited flexibility</li>
                      </ul>
                    </td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>Simple linear relationships</li>
                        <li>When interpretability is crucial</li>
                      </ul>
                    </td>
                  </tr>
                  <tr>
                    <td className="border border-neutral-300 px-4 py-2 font-medium">Polynomial Regression</td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>Can model non-linear relationships</li>
                        <li>Still relatively interpretable</li>
                        <li>Flexible degree selection</li>
                      </ul>
                    </td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>Prone to overfitting with high degrees</li>
                        <li>Sensitive to outliers</li>
                        <li>Extrapolation can be unreliable</li>
                      </ul>
                    </td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>Curvilinear relationships</li>
                        <li>When the pattern follows a polynomial form</li>
                      </ul>
                    </td>
                  </tr>
                  <tr>
                    <td className="border border-neutral-300 px-4 py-2 font-medium">Ridge/Lasso Regression</td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>Prevents overfitting</li>
                        <li>Handles multicollinearity</li>
                        <li>Feature selection (Lasso)</li>
                      </ul>
                    </td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>Requires tuning regularization parameter</li>
                        <li>Still limited to linear relationships</li>
                      </ul>
                    </td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>High-dimensional data</li>
                        <li>When features are correlated</li>
                      </ul>
                    </td>
                  </tr>
                  <tr>
                    <td className="border border-neutral-300 px-4 py-2 font-medium">Spline Regression</td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>Flexible for complex patterns</li>
                        <li>Smooth transitions at knot points</li>
                        <li>Better local fitting</li>
                      </ul>
                    </td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>More complex to implement</li>
                        <li>Requires knot selection</li>
                        <li>Less interpretable</li>
                      </ul>
                    </td>
                    <td className="border border-neutral-300 px-4 py-2">
                      <ul className="list-disc pl-4 text-sm">
                        <li>Complex non-linear patterns</li>
                        <li>When relationship changes across ranges</li>
                      </ul>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </TabsContent>

        <TabsContent value="demo">
          <section>
            <h2 className="text-2xl font-semibold text-neutral-800 mb-4">Interactive Polynomial Regression Demo</h2>
            <p className="text-neutral-700 mb-6">
              Explore how polynomial regression works by adjusting the degree of the polynomial and other parameters.
              Notice how higher-degree polynomials can fit more complex patterns but may lead to overfitting.
            </p>
            <LinearRegressionVisualization width={800} height={500} />
            <div className="mt-6 bg-neutral-50 p-4 rounded-lg border border-neutral-200">
              <h3 className="text-lg font-medium text-neutral-800 mb-2">How to use this demo:</h3>
              <ul className="list-disc pl-6 space-y-1 text-neutral-700">
                <li>Switch to the "Polynomial" tab to see polynomial regression in action</li>
                <li>Adjust the polynomial degree to see how it affects the fit</li>
                <li>Use the "Random Data" button to generate new datasets</li>
                <li>Notice how higher degrees can lead to overfitting, especially with noisy data</li>
                <li>Compare with simple linear regression to see the difference in flexibility</li>
              </ul>
            </div>
          </section>
        </TabsContent>

        <TabsContent value="code">
          <section>
            <h2 className="text-2xl font-semibold text-neutral-800 mb-4">Polynomial Regression in Practice</h2>
            <p className="text-neutral-700 mb-6">
              Below is a comprehensive implementation of polynomial regression using scikit-learn. The example
              demonstrates how to create polynomial features, fit a model, and evaluate its performance with different
              polynomial degrees.
            </p>
            <NotebookCell
              code={`import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Generate synthetic data with non-linear pattern
np.random.seed(42)
X = np.sort(np.random.uniform(0, 1, 100))[:, np.newaxis]
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and compare models with different polynomial degrees
degrees = [1, 2, 3, 5, 9]
plt.figure(figsize=(14, 10))

for i, degree in enumerate(degrees):
    ax = plt.subplot(2, 3, i + 1)
    
    # Create polynomial regression model
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear_reg", LinearRegression())
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    # Predict on a fine grid for plotting
    X_grid = np.linspace(0, 1, 1000)[:, np.newaxis]
    y_grid_pred = model.predict(X_grid)
    
    # Plot results
    plt.scatter(X_train, y_train, color='blue', s=20, alpha=0.5, label='Training data')
    plt.scatter(X_test, y_test, color='green', s=20, alpha=0.5, label='Testing data')
    plt.plot(X_grid, y_grid_pred, color='red', linewidth=2, label='Polynomial fit')
    plt.title(f"Degree {degree}\\nTrain MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}\\nR²: {r2:.4f}")
    plt.ylim((-1.5, 1.5))
    plt.xlabel("X")
    plt.ylabel("y")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()

# Analyze the relationship between polynomial degree and error
degrees_extended = range(1, 15)
train_errors = []
test_errors = []

for degree in degrees_extended:
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear_reg", LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# Plot the error curves
plt.figure(figsize=(10, 6))
plt.plot(degrees_extended, train_errors, 'o-', color='blue', label='Training MSE')
plt.plot(degrees_extended, test_errors, 'o-', color='red', label='Testing MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Error vs. Polynomial Degree')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Demonstrate regularized polynomial regression to prevent overfitting
from sklearn.linear_model import Ridge

plt.figure(figsize=(14, 5))

# High degree polynomial with standard linear regression (prone to overfitting)
plt.subplot(1, 2, 1)
model_no_reg = Pipeline([
    ("poly_features", PolynomialFeatures(degree=9, include_bias=False)),
    ("linear_reg", LinearRegression())
])
model_no_reg.fit(X_train, y_train)
y_pred_no_reg = model_no_reg.predict(X_grid)

plt.scatter(X_train, y_train, color='blue', s=20, alpha=0.5, label='Training data')
plt.scatter(X_test, y_test, color='green', s=20, alpha=0.5, label='Testing data')
plt.plot(X_grid, y_pred_no_reg, color='red', linewidth=2, label='Degree 9 (No Regularization)')
plt.title(f"Polynomial Degree 9 - No Regularization\\nTest MSE: {mean_squared_error(y_test, model_no_reg.predict(X_test)):.4f}")
plt.ylim((-1.5, 1.5))
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

# High degree polynomial with Ridge regression (reduces overfitting)
plt.subplot(1, 2, 2)
model_ridge = Pipeline([
    ("poly_features", PolynomialFeatures(degree=9, include_bias=False)),
    ("ridge_reg", Ridge(alpha=0.1))
])
model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_grid)

plt.scatter(X_train, y_train, color='blue', s=20, alpha=0.5, label='Training data')
plt.scatter(X_test, y_test, color='green', s=20, alpha=0.5, label='Testing data')
plt.plot(X_grid, y_pred_ridge, color='purple', linewidth=2, label='Degree 9 (Ridge Regularization)')
plt.title(f"Polynomial Degree 9 - With Ridge Regularization\\nTest MSE: {mean_squared_error(y_test, model_ridge.predict(X_test)):.4f}")
plt.ylim((-1.5, 1.5))
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()

# Real-world example: Boston Housing dataset
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load the Boston Housing dataset
boston = load_boston()
X_boston = boston.data
y_boston = boston.target

# Use just one feature for visualization (LSTAT: % lower status of the population)
X_boston_single = X_boston[:, 12].reshape(-1, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_boston_single, y_boston, test_size=0.3, random_state=42)

# Scale the feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different polynomial degrees
degrees = [1, 2, 3, 5]
plt.figure(figsize=(14, 10))

for i, degree in enumerate(degrees):
    ax = plt.subplot(2, 2, i + 1)
    
    # Create polynomial regression model
    model = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linear_reg", LinearRegression())
    ])
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    # Predict on a fine grid for plotting
    X_grid = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 1000).reshape(-1, 1)
    y_grid_pred = model.predict(X_grid)
    
    # Plot results
    plt.scatter(X_train_scaled, y_train, color='blue', s=20, alpha=0.5, label='Training data')
    plt.scatter(X_test_scaled, y_test, color='green', s=20, alpha=0.5, label='Testing data')
    plt.plot(X_grid, y_grid_pred, color='red', linewidth=2, label=f'Degree {degree}')
    plt.title(f"Boston Housing: Degree {degree}\\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}\\nR²: {r2:.4f}")
    plt.xlabel("LSTAT (scaled)")
    plt.ylabel("House Price ($1000s)")
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()`}
              language="python"
              showLineNumbers={true}
            />
            <div className="mt-6 bg-neutral-50 p-4 rounded-lg border border-neutral-200">
              <h3 className="text-lg font-medium text-neutral-800 mb-2">Key Implementation Points:</h3>
              <ul className="list-disc pl-6 space-y-1 text-neutral-700">
                <li>
                  Polynomial features are created using <code>PolynomialFeatures</code> from scikit-learn, which
                  transforms the original features into polynomial terms
                </li>
                <li>
                  A <code>Pipeline</code> is used to combine feature transformation and model fitting in a single step
                </li>
                <li>
                  The example demonstrates the trade-off between model complexity (polynomial degree) and overfitting
                </li>
                <li>
                  Ridge regression is shown as a way to reduce overfitting in high-degree polynomial models through
                  regularization
                </li>
                <li>
                  The Boston Housing dataset example shows how polynomial regression can be applied to real-world data
                </li>
              </ul>
            </div>
          </section>
        </TabsContent>
      </Tabs>

      <div className="flex justify-between items-center mt-12 border-t border-neutral-300 pt-6">
        <Button asChild variant="outline">
          <Link href="/models/linear-regression">
            <ChevronLeft className="mr-2 h-4 w-4" /> Linear Regression
          </Link>
        </Button>
        <Button asChild variant="default">
          <Link href="/models/regularized-regression">
            Ridge & Lasso Regression <ChevronRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </div>
  )
}
