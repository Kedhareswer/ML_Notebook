"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Download,
  Search,
  Code,
  Database,
  BarChart3,
  LineChart,
  Brain,
  BookOpen,
  FileCode,
  FileText,
  Filter,
  Copy,
  ExternalLink,
  Star,
  StarHalf,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"

export default function CheatSheetsPage() {
  const [searchTerm, setSearchTerm] = useState("")
  const [activeCategory, setActiveCategory] = useState("all")
  const [activeType, setActiveType] = useState("all")
  const [difficulty, setDifficulty] = useState("all")

  const cheatSheets = [
    {
      title: "Python Basics",
      description: "Essential Python syntax and functions for machine learning",
      category: "Programming",
      type: "Code",
      difficulty: "Beginner",
      icon: <Code className="h-6 w-6" />,
      stars: 5,
      content: [
        {
          title: "Data Structures",
          code: `# Lists
my_list = [1, 2, 3, 4, 5]
my_list.append(6)  # Add element
my_list.pop()      # Remove last element
my_list[0]         # Access element

# Dictionaries
my_dict = {'a': 1, 'b': 2}
my_dict['c'] = 3   # Add key-value pair
my_dict.get('a')   # Get value
my_dict.keys()     # Get all keys

# NumPy Arrays
import numpy as np
arr = np.array([1, 2, 3])
arr * 2            # Element-wise multiplication
np.mean(arr)       # Calculate mean
np.reshape(arr, (3, 1))  # Reshape array`,
        },
        {
          title: "Control Flow",
          code: `# Conditionals
if condition:
    # do something
elif another_condition:
    # do something else
else:
    # fallback

# Loops
for item in iterable:
    # process item

while condition:
    # do something
    # update condition

# List Comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]`,
        },
        {
          title: "Functions",
          code: `# Basic function
def my_function(arg1, arg2=default_value):
    """Docstring explaining function"""
    # function body
    return result

# Lambda functions
square = lambda x: x**2

# Map, Filter, Reduce
from functools import reduce
mapped = map(lambda x: x*2, [1, 2, 3])
filtered = filter(lambda x: x > 2, [1, 2, 3, 4])
reduced = reduce(lambda x, y: x + y, [1, 2, 3, 4])`,
        },
      ],
    },
    {
      title: "NumPy Essentials",
      description: "Key NumPy functions and operations for numerical computing",
      category: "Libraries",
      type: "Code",
      difficulty: "Intermediate",
      icon: <Database className="h-6 w-6" />,
      stars: 5,
      content: [
        {
          title: "Array Creation",
          code: `import numpy as np

# Creating arrays
a = np.array([1, 2, 3])           # From list
b = np.zeros((3, 3))              # Array of zeros
c = np.ones((2, 3))               # Array of ones
d = np.eye(3)                     # Identity matrix
e = np.random.random((2, 2))      # Random values
f = np.arange(10)                 # Range of values
g = np.linspace(0, 1, 5)          # Evenly spaced values`,
        },
        {
          title: "Array Operations",
          code: `# Basic operations
a + b                  # Element-wise addition
a - b                  # Element-wise subtraction
a * b                  # Element-wise multiplication
a / b                  # Element-wise division
a @ b                  # Matrix multiplication
a.dot(b)               # Matrix multiplication (alternative)

# Array manipulation
a.reshape(3, 1)        # Reshape array
a.T                    # Transpose
np.concatenate([a, b]) # Concatenate arrays
a.flatten()            # Flatten to 1D
np.split(a, 3)         # Split array`,
        },
        {
          title: "Statistical Functions",
          code: `# Statistics
np.mean(a)             # Mean
np.median(a)           # Median
np.std(a)              # Standard deviation
np.var(a)              # Variance
np.min(a)              # Minimum
np.max(a)              # Maximum
np.argmin(a)           # Index of minimum
np.argmax(a)           # Index of maximum
np.percentile(a, 75)   # 75th percentile`,
        },
      ],
    },
    {
      title: "Pandas for Data Analysis",
      description: "Common Pandas operations for data manipulation and analysis",
      category: "Libraries",
      type: "Code",
      difficulty: "Intermediate",
      icon: <Database className="h-6 w-6" />,
      stars: 5,
      content: [
        {
          title: "Data Structures",
          code: `import pandas as pd

# Series
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# Reading data
df_csv = pd.read_csv('file.csv')
df_excel = pd.read_excel('file.xlsx')
df_sql = pd.read_sql('query', connection)`,
        },
        {
          title: "Data Selection",
          code: `# Selecting data
df['A']                # Select column
df[['A', 'B']]         # Select multiple columns
df.loc[0]              # Select row by label
df.iloc[0]             # Select row by position
df.loc[0, 'A']         # Select cell by label
df.iloc[0, 0]          # Select cell by position
df[df['A'] > 2]        # Filter rows by condition`,
        },
        {
          title: "Data Manipulation",
          code: `# Manipulating data
df.dropna()            # Drop missing values
df.fillna(0)           # Fill missing values
df.sort_values('A')    # Sort by values
df.sort_index()        # Sort by index
df.groupby('A').mean() # Group by and aggregate
df.merge(df2, on='A')  # Merge DataFrames
df.join(df2)           # Join DataFrames
df.pivot_table(index='A', columns='B', values='C')  # Pivot table`,
        },
      ],
    },
    {
      title: "Matplotlib & Seaborn",
      description: "Visualization techniques for data exploration and presentation",
      category: "Visualization",
      type: "Code",
      difficulty: "Intermediate",
      icon: <BarChart3 className="h-6 w-6" />,
      stars: 4.5,
      content: [
        {
          title: "Basic Plotting",
          code: `import matplotlib.pyplot as plt
import seaborn as sns

# Basic plots
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.scatter(x, y)
plt.bar(x, y)
plt.hist(x, bins=10)
plt.boxplot(x)
plt.pie(x)

# Customization
plt.title('Title')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.legend(['Label 1', 'Label 2'])
plt.grid(True)
plt.savefig('plot.png')
plt.show()`,
        },
        {
          title: "Seaborn Plots",
          code: `# Set style
sns.set_theme(style="whitegrid")

# Statistical plots
sns.histplot(data=df, x='column')
sns.kdeplot(data=df, x='column')
sns.boxplot(data=df, x='category', y='value')
sns.violinplot(data=df, x='category', y='value')
sns.barplot(data=df, x='category', y='value')

# Relationship plots
sns.scatterplot(data=df, x='x', y='y', hue='category')
sns.lineplot(data=df, x='x', y='y')
sns.regplot(data=df, x='x', y='y')
sns.heatmap(correlation_matrix, annot=True)
sns.pairplot(df)`,
        },
        {
          title: "Subplots",
          code: `# Creating subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Accessing subplots
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
axes[1, 0].hist(x)
axes[1, 1].boxplot(x)

# Adjusting layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# FacetGrid in Seaborn
g = sns.FacetGrid(df, col='category', row='another_category')
g.map(plt.scatter, 'x', 'y')`,
        },
      ],
    },
    {
      title: "Scikit-Learn",
      description: "Machine learning algorithms and evaluation metrics",
      category: "Machine Learning",
      type: "Code",
      difficulty: "Intermediate",
      icon: <LineChart className="h-6 w-6" />,
      stars: 5,
      content: [
        {
          title: "Model Training",
          code: `from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_scaled, y_train)

svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train_scaled, y_train)`,
        },
        {
          title: "Model Evaluation",
          code: `from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Classification report
report = classification_report(y_test, y_pred)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)`,
        },
        {
          title: "Hyperparameter Tuning",
          code: `from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_`,
        },
      ],
    },
    {
      title: "TensorFlow & Keras",
      description: "Deep learning model building and training",
      category: "Deep Learning",
      type: "Code",
      difficulty: "Advanced",
      icon: <Brain className="h-6 w-6" />,
      stars: 4.5,
      content: [
        {
          title: "Model Building",
          code: `import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Sequential API
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# CNN model
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Functional API
inputs = tf.keras.Input(shape=(784,))
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)`,
        },
        {
          title: "Model Training",
          code: `# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)

# Predictions
predictions = model.predict(X_test)`,
        },
        {
          title: "Transfer Learning",
          code: `from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create new model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)`,
        },
      ],
    },
    {
      title: "Machine Learning Algorithms",
      description: "Overview of common ML algorithms and their applications",
      category: "Machine Learning",
      type: "PDF",
      difficulty: "Intermediate",
      icon: <FileText className="h-6 w-6" />,
      stars: 4.5,
      content: [
        {
          title: "Supervised Learning",
          code: `# This is a PDF resource with comprehensive information about supervised learning algorithms
# Including decision trees, random forests, SVMs, and neural networks
# With examples, use cases, and implementation guidelines`,
        },
        {
          title: "Unsupervised Learning",
          code: `# This is a PDF resource covering clustering algorithms
# Including K-means, hierarchical clustering, and DBSCAN
# With visualization techniques and evaluation metrics`,
        },
        {
          title: "Reinforcement Learning",
          code: `# This is a PDF resource explaining reinforcement learning concepts
# Including Q-learning, policy gradients, and deep reinforcement learning
# With practical examples and implementation strategies`,
        },
      ],
    },
    {
      title: "Statistics for Data Science",
      description: "Essential statistical concepts for machine learning",
      category: "Mathematics",
      type: "PDF",
      difficulty: "Intermediate",
      icon: <FileText className="h-6 w-6" />,
      stars: 4,
      content: [
        {
          title: "Descriptive Statistics",
          code: `# This is a PDF resource covering measures of central tendency and dispersion
# Including mean, median, mode, variance, standard deviation
# With practical examples and interpretations`,
        },
        {
          title: "Probability Distributions",
          code: `# This is a PDF resource explaining common probability distributions
# Including normal, binomial, Poisson, and exponential distributions
# With applications in machine learning`,
        },
        {
          title: "Hypothesis Testing",
          code: `# This is a PDF resource on statistical hypothesis testing
# Including t-tests, chi-square tests, and ANOVA
# With examples of when to use each test`,
        },
      ],
    },
    {
      title: "Linear Algebra for ML",
      description: "Key linear algebra concepts used in machine learning",
      category: "Mathematics",
      type: "PDF",
      difficulty: "Advanced",
      icon: <FileText className="h-6 w-6" />,
      stars: 4,
      content: [
        {
          title: "Vectors and Matrices",
          code: `# This is a PDF resource covering vector and matrix operations
# Including addition, multiplication, transposition, and inversion
# With applications in machine learning algorithms`,
        },
        {
          title: "Eigenvalues and Eigenvectors",
          code: `# This is a PDF resource explaining eigendecomposition
# Including calculation methods and geometric interpretation
# With applications in PCA and other dimensionality reduction techniques`,
        },
        {
          title: "Matrix Factorization",
          code: `# This is a PDF resource on matrix factorization techniques
# Including SVD, LU, and QR decomposition
# With applications in recommendation systems and image processing`,
        },
      ],
    },
    {
      title: "PyTorch Basics",
      description: "Getting started with PyTorch for deep learning",
      category: "Deep Learning",
      type: "Code",
      difficulty: "Intermediate",
      icon: <Brain className="h-6 w-6" />,
      stars: 4.5,
      content: [
        {
          title: "Tensors and Operations",
          code: `import torch

# Creating tensors
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 3)
z = torch.ones(2, 2)
w = torch.rand(3, 3)

# Basic operations
a = x + x
b = y * 2
c = torch.matmul(y, w)
d = torch.cat([y, z], dim=0)

# Moving to GPU
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    y_gpu = y.cuda()

# Converting to NumPy
x_np = x.numpy()
y_tensor = torch.from_numpy(numpy_array)`,
        },
        {
          title: "Neural Network Modules",
          code: `import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Instantiate the model
model = Net()`,
        },
        {
          title: "Training Loop",
          code: `# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}')
                running_loss = 0.0
                
    print('Finished Training')`,
        },
      ],
    },
    {
      title: "Data Preprocessing",
      description: "Techniques for preparing data for machine learning",
      category: "Data Science",
      type: "Code",
      difficulty: "Beginner",
      icon: <Database className="h-6 w-6" />,
      stars: 4.5,
      content: [
        {
          title: "Data Cleaning",
          code: `import pandas as pd
import numpy as np

# Handling missing values
df.isnull().sum()  # Count missing values
df.dropna()        # Drop rows with missing values
df.fillna(0)       # Fill missing values with 0
df.fillna(df.mean())  # Fill with mean

# Handling duplicates
df.duplicated().sum()  # Count duplicates
df.drop_duplicates()   # Remove duplicates

# Handling outliers
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
df_filtered = df[(df['column'] >= Q1 - 1.5 * IQR) & 
                 (df['column'] <= Q3 + 1.5 * IQR)]

# Data type conversion
df['column'] = df['column'].astype('int64')
df['date_column'] = pd.to_datetime(df['date_column'])`,
        },
        {
          title: "Feature Scaling",
          code: `from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization (z-score normalization)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
# Mean = 0, Std = 1

# Min-Max scaling
min_max_scaler = MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)
# Range [0, 1]

# Robust scaling (uses median and IQR)
robust_scaler = RobustScaler()
df_scaled = robust_scaler.fit_transform(df)
# Robust to outliers

# Log transformation
df['log_column'] = np.log1p(df['column'])  # log(1+x)`,
        },
        {
          title: "Feature Engineering",
          code: `# Creating new features
df['area'] = df['length'] * df['width']
df['density'] = df['weight'] / df['volume']

# Extracting date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                         labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded_category'] = le.fit_transform(df['category'])`,
        },
      ],
    },
    {
      title: "SQL for Data Science",
      description: "Essential SQL queries for data extraction and analysis",
      category: "Data Science",
      type: "Code",
      difficulty: "Beginner",
      icon: <Database className="h-6 w-6" />,
      stars: 4,
      content: [
        {
          title: "Basic Queries",
          code: `-- Select all columns
SELECT * FROM table_name;

-- Select specific columns
SELECT column1, column2 FROM table_name;

-- Filter rows
SELECT * FROM table_name WHERE condition;

-- Sort results
SELECT * FROM table_name ORDER BY column_name ASC/DESC;

-- Limit results
SELECT * FROM table_name LIMIT 10;

-- Distinct values
SELECT DISTINCT column_name FROM table_name;`,
        },
        {
          title: "Joins and Aggregations",
          code: `-- Inner join
SELECT a.column1, b.column2
FROM table_a a
INNER JOIN table_b b ON a.id = b.id;

-- Left join
SELECT a.column1, b.column2
FROM table_a a
LEFT JOIN table_b b ON a.id = b.id;

-- Group by with aggregation
SELECT category, COUNT(*) as count, AVG(value) as avg_value
FROM table_name
GROUP BY category;

-- Having clause
SELECT category, COUNT(*) as count
FROM table_name
GROUP BY category
HAVING COUNT(*) > 10;`,
        },
        {
          title: "Advanced Queries",
          code: `-- Subqueries
SELECT *
FROM table_name
WHERE column_name IN (
    SELECT column_name
    FROM another_table
    WHERE condition
);

-- Common Table Expressions (CTE)
WITH cte_name AS (
    SELECT column1, column2
    FROM table_name
    WHERE condition
)
SELECT * FROM cte_name;

-- Window functions
SELECT 
    column1,
    column2,
    AVG(column2) OVER (PARTITION BY column3) as avg_by_group,
    RANK() OVER (ORDER BY column2 DESC) as rank
FROM table_name;

-- Pivot
SELECT *
FROM (
    SELECT category, month, value
    FROM table_name
) AS source_table
PIVOT (
    SUM(value)
    FOR month IN ([Jan], [Feb], [Mar], [Apr])
) AS pivot_table;`,
        },
      ],
    },
    {
      title: "Neural Network Architectures",
      description: "Common neural network architectures and their applications",
      category: "Deep Learning",
      type: "PDF",
      difficulty: "Advanced",
      icon: <FileText className="h-6 w-6" />,
      stars: 5,
      content: [
        {
          title: "Convolutional Neural Networks",
          code: `# This is a PDF resource covering CNN architectures
# Including LeNet, AlexNet, VGG, ResNet, and Inception
# With applications in computer vision and image processing`,
        },
        {
          title: "Recurrent Neural Networks",
          code: `# This is a PDF resource explaining RNN architectures
# Including vanilla RNNs, LSTMs, and GRUs
# With applications in sequence modeling and NLP`,
        },
        {
          title: "Transformer Models",
          code: `# This is a PDF resource on transformer architectures
# Including self-attention mechanisms and multi-head attention
# With applications in NLP and beyond`,
        },
      ],
    },
    {
      title: "Git for Data Scientists",
      description: "Essential Git commands for version control in data science projects",
      category: "Tools",
      type: "Code",
      difficulty: "Beginner",
      icon: <Code className="h-6 w-6" />,
      stars: 4,
      content: [
        {
          title: "Basic Commands",
          code: `# Initialize a repository
git init

# Clone a repository
git clone https://github.com/username/repository.git

# Check status
git status

# Add files to staging
git add filename
git add .  # Add all files

# Commit changes
git commit -m "Commit message"

# Push to remote
git push origin branch_name

# Pull from remote
git pull origin branch_name`,
        },
        {
          title: "Branching and Merging",
          code: `# Create a new branch
git branch branch_name

# Switch to a branch
git checkout branch_name

# Create and switch to a new branch
git checkout -b branch_name

# List all branches
git branch

# Merge a branch into current branch
git merge branch_name

# Delete a branch
git branch -d branch_name`,
        },
        {
          title: "Advanced Operations",
          code: `# View commit history
git log
git log --oneline --graph

# Stash changes
git stash
git stash pop

# Revert a commit
git revert commit_hash

# Reset to a previous commit
git reset --hard commit_hash

# Cherry-pick a commit
git cherry-pick commit_hash

# Rebase
git rebase branch_name

# Configure user
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"`,
        },
      ],
    },
    {
      title: "Model Deployment",
      description: "Techniques for deploying machine learning models to production",
      category: "MLOps",
      type: "PDF",
      difficulty: "Advanced",
      icon: <FileText className="h-6 w-6" />,
      stars: 4.5,
      content: [
        {
          title: "Containerization",
          code: `# This is a PDF resource covering Docker and containerization
# Including Dockerfiles, Docker Compose, and Kubernetes
# With examples for packaging ML models`,
        },
        {
          title: "REST APIs",
          code: `# This is a PDF resource explaining REST API development
# Including Flask, FastAPI, and Django REST framework
# With examples for serving ML model predictions`,
        },
        {
          title: "Cloud Deployment",
          code: `# This is a PDF resource on cloud deployment options
# Including AWS SageMaker, Azure ML, and Google AI Platform
# With step-by-step deployment guides`,
        },
      ],
    },
  ]

  // Filter cheat sheets based on search term, category, type, and difficulty
  const filteredCheatSheets = cheatSheets.filter((sheet) => {
    const matchesSearch =
      sheet.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      sheet.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = activeCategory === "all" || sheet.category === activeCategory
    const matchesType = activeType === "all" || sheet.type === activeType
    const matchesDifficulty = difficulty === "all" || sheet.difficulty === difficulty
    return matchesSearch && matchesCategory && matchesType && matchesDifficulty
  })

  // Get unique categories, types, and difficulties
  const categories = ["all", ...new Set(cheatSheets.map((sheet) => sheet.category))]
  const types = ["all", ...new Set(cheatSheets.map((sheet) => sheet.type))]
  const difficulties = ["all", "Beginner", "Intermediate", "Advanced"]

  // Function to render star ratings
  const renderStars = (rating) => {
    const fullStars = Math.floor(rating)
    const hasHalfStar = rating % 1 !== 0

    return (
      <div className="flex items-center">
        {[...Array(fullStars)].map((_, i) => (
          <Star key={i} className="h-4 w-4 fill-yellow-400 text-yellow-400" />
        ))}
        {hasHalfStar && <StarHalf className="h-4 w-4 fill-yellow-400 text-yellow-400" />}
        {[...Array(5 - fullStars - (hasHalfStar ? 1 : 0))].map((_, i) => (
          <Star key={i + fullStars + (hasHalfStar ? 1 : 0)} className="h-4 w-4 text-neutral-300" />
        ))}
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="mb-12 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-neutral-900 mb-4">Machine Learning Cheat Sheets</h1>
        <p className="text-neutral-700 max-w-2xl mx-auto">
          Quick reference guides for machine learning concepts, algorithms, and implementation details
        </p>
      </div>

      <div className="flex flex-col space-y-4 mb-8">
        <div className="relative flex-grow">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-500 h-4 w-4" />
          <Input
            placeholder="Search cheat sheets..."
            className="pl-10"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-neutral-500" />
              <span className="text-sm font-medium text-neutral-700">Category</span>
            </div>
            <Tabs defaultValue="all" value={activeCategory} onValueChange={setActiveCategory} className="w-full">
              <TabsList className="w-full grid grid-cols-2 md:grid-cols-3 h-auto">
                {categories.map((category) => (
                  <TabsTrigger key={category} value={category} className="capitalize text-xs py-1.5">
                    {category}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <FileCode className="h-4 w-4 text-neutral-500" />
              <span className="text-sm font-medium text-neutral-700">Type</span>
            </div>
            <Tabs defaultValue="all" value={activeType} onValueChange={setActiveType} className="w-full">
              <TabsList className="w-full grid grid-cols-3 h-auto">
                {types.map((type) => (
                  <TabsTrigger key={type} value={type} className="text-xs py-1.5">
                    {type}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <BookOpen className="h-4 w-4 text-neutral-500" />
              <span className="text-sm font-medium text-neutral-700">Difficulty</span>
            </div>
            <Tabs defaultValue="all" value={difficulty} onValueChange={setDifficulty} className="w-full">
              <TabsList className="w-full grid grid-cols-4 h-auto">
                {difficulties.map((level) => (
                  <TabsTrigger key={level} value={level} className="text-xs py-1.5">
                    {level}
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredCheatSheets.map((sheet, index) => (
          <Card key={index} className="overflow-hidden hover:shadow-md transition-shadow duration-300">
            <CardHeader className="bg-neutral-50 border-b border-neutral-200 pb-4">
              <div className="flex items-start gap-3">
                <div className="bg-white p-2 rounded-md shadow-sm">{sheet.icon}</div>
                <div className="flex-1">
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-lg">{sheet.title}</CardTitle>
                    <Badge
                      variant="outline"
                      className={
                        sheet.difficulty === "Beginner"
                          ? "bg-green-50 text-green-700 border-green-200"
                          : sheet.difficulty === "Intermediate"
                            ? "bg-blue-50 text-blue-700 border-blue-200"
                            : "bg-purple-50 text-purple-700 border-purple-200"
                      }
                    >
                      {sheet.difficulty}
                    </Badge>
                  </div>
                  <CardDescription className="mt-1">{sheet.description}</CardDescription>
                  <div className="flex items-center gap-2 mt-2">
                    {renderStars(sheet.stars)}
                    <span className="text-xs text-neutral-500">{sheet.stars.toFixed(1)}</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <Tabs defaultValue={sheet.content[0].title} className="w-full">
                <TabsList className="w-full justify-start rounded-none border-b border-neutral-200 px-4 bg-white">
                  {sheet.content.map((section, i) => (
                    <TabsTrigger key={i} value={section.title} className="text-sm">
                      {section.title}
                    </TabsTrigger>
                  ))}
                </TabsList>
                {sheet.content.map((section, i) => (
                  <TabsContent key={i} value={section.title} className="p-0 m-0">
                    <div className="bg-neutral-900 text-neutral-50 p-4 overflow-x-auto">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs text-neutral-400">
                          {sheet.type === "Code" ? "Code Snippet" : "PDF Preview"}
                        </span>
                        <Button variant="ghost" size="sm" className="h-6 text-neutral-400 hover:text-white">
                          <Copy className="h-3.5 w-3.5 mr-1" /> Copy
                        </Button>
                      </div>
                      <pre className="text-sm">
                        <code>{section.code}</code>
                      </pre>
                    </div>
                  </TabsContent>
                ))}
              </Tabs>
            </CardContent>
            <CardFooter className="bg-neutral-50 border-t border-neutral-200 flex justify-between py-3">
              <Badge variant="secondary" className="bg-neutral-100 text-neutral-700 hover:bg-neutral-200">
                {sheet.category}
              </Badge>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="h-8">
                  <ExternalLink className="h-3.5 w-3.5 mr-1" /> View Full
                </Button>
                <Button variant="default" size="sm" className="h-8 bg-neutral-800 hover:bg-neutral-900">
                  <Download className="h-3.5 w-3.5 mr-1" /> Download
                </Button>
              </div>
            </CardFooter>
          </Card>
        ))}
      </div>

      {filteredCheatSheets.length === 0 && (
        <div className="text-center py-12 bg-neutral-50 rounded-lg border border-neutral-200">
          <div className="mb-4">
            <Search className="h-12 w-12 text-neutral-300 mx-auto" />
          </div>
          <p className="text-neutral-600 mb-4">No cheat sheets found matching your search criteria.</p>
          <Button
            variant="outline"
            onClick={() => {
              setSearchTerm("")
              setActiveCategory("all")
              setActiveType("all")
              setDifficulty("all")
            }}
          >
            Clear all filters
          </Button>
        </div>
      )}

      <div className="mt-12 p-6 border border-neutral-300 rounded-lg bg-neutral-50">
        <h2 className="text-xl font-bold text-neutral-900 mb-4">How to Use These Cheat Sheets</h2>
        <p className="text-neutral-700 mb-4">
          Our cheat sheets are designed to be quick references when implementing or studying machine learning models.
          They contain the most important functions, parameters, and implementation tips in a condensed format.
        </p>
        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <div className="bg-white p-4 rounded-md border border-neutral-200">
            <h3 className="text-lg font-medium text-neutral-800 mb-3 flex items-center">
              <BookOpen className="h-5 w-5 mr-2 text-neutral-700" /> For Learning
            </h3>
            <ul className="space-y-2">
              <li className="flex items-start">
                <div className="mr-2 text-neutral-800">•</div>
                <span className="text-neutral-700">Use as quick reference during courses or tutorials</span>
              </li>
              <li className="flex items-start">
                <div className="mr-2 text-neutral-800">•</div>
                <span className="text-neutral-700">Review key concepts before exams or interviews</span>
              </li>
              <li className="flex items-start">
                <div className="mr-2 text-neutral-800">•</div>
                <span className="text-neutral-700">Compare different algorithms and their properties</span>
              </li>
              <li className="flex items-start">
                <div className="mr-2 text-neutral-800">•</div>
                <span className="text-neutral-700">Understand the mathematical foundations of ML techniques</span>
              </li>
            </ul>
          </div>
          <div className="bg-white p-4 rounded-md border border-neutral-200">
            <h3 className="text-lg font-medium text-neutral-800 mb-3 flex items-center">
              <Code className="h-5 w-5 mr-2 text-neutral-700" /> For Implementation
            </h3>
            <ul className="space-y-2">
              <li className="flex items-start">
                <div className="mr-2 text-neutral-800">•</div>
                <span className="text-neutral-700">Copy code snippets directly into your projects</span>
              </li>
              <li className="flex items-start">
                <div className="mr-2 text-neutral-800">•</div>
                <span className="text-neutral-700">Use as templates for common ML workflows</span>
              </li>
              <li className="flex items-start">
                <div className="mr-2 text-neutral-800">•</div>
                <span className="text-neutral-700">Reference API parameters and function signatures</span>
              </li>
              <li className="flex items-start">
                <div className="mr-2 text-neutral-800">•</div>
                <span className="text-neutral-700">Troubleshoot common issues with implementation examples</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
