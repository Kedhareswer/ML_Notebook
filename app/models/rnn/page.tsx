"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, BookOpen, Code, Network } from "lucide-react"
import Link from "next/link"
import NotebookCell from "@/components/notebook-cell"
import RNNVisualization from "@/components/interactive-visualizations/rnn-viz"

export default function RNNPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Model: "sequential"
          <br />
          _________________________________________________________________
          <br />
          Layer (type) Output Shape Param # <br />
          =================================================================
          <br />
          embedding (Embedding) (None, 100, 32) 160000 <br />
          _________________________________________________________________
          <br />
          lstm (LSTM) (None, 100, 64) 24832 <br />
          _________________________________________________________________
          <br />
          lstm_1 (LSTM) (None, 32) 12416 <br />
          _________________________________________________________________
          <br />
          dense (Dense) (None, 1) 33 <br />
          =================================================================
          <br />
          Total params: 197,281
          <br />
          Trainable params: 197,281
          <br />
          Non-trainable params: 0<br />
          _________________________________________________________________
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Epoch 1/5
          <br />
          625/625 [==============================] - 42s 67ms/step - loss: 0.6932 - accuracy: 0.5024 - val_loss: 0.6931
          - val_accuracy: 0.5040
          <br />
          Epoch 2/5
          <br />
          625/625 [==============================] - 41s 66ms/step - loss: 0.6931 - accuracy: 0.5040 - val_loss: 0.6931
          - val_accuracy: 0.5040
          <br />
          Epoch 3/5
          <br />
          625/625 [==============================] - 41s 66ms/step - loss: 0.6931 - accuracy: 0.5040 - val_loss: 0.6931
          - val_accuracy: 0.5040
          <br />
          Epoch 4/5
          <br />
          625/625 [==============================] - 41s 66ms/step - loss: 0.6931 - accuracy: 0.5040 - val_loss: 0.6931
          - val_accuracy: 0.5040
          <br />
          Epoch 5/5
          <br />
          625/625 [==============================] - 41s 66ms/step - loss: 0.6931 - accuracy: 0.5040 - val_loss: 0.6931
          - val_accuracy: 0.5040
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Test accuracy: 0.8764
          <br />
          <br />
          Classification Report:
          <br />
          <br />
          precision recall f1-score support
          <br />
          <br />
          negative 0.88 0.87 0.88 5000
          <br />
          positive 0.87 0.88 0.88 5000
          <br />
          <br />
          accuracy 0.88 10000
          <br />
          macro avg 0.88 0.88 0.88 10000
          <br />
          weighted avg 0.88 0.88 0.88 10000
        </div>
      )
    } else if (cellId === "cell4") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Prediction: Negative (0.12)
          <br />
          Actual: Negative
          <br />
          <br />
          Prediction: Positive (0.89)
          <br />
          Actual: Positive
          <br />
          <br />
          Prediction: Positive (0.76)
          <br />
          Actual: Positive
          <br />
          <br />
          Prediction: Negative (0.23)
          <br />
          Actual: Negative
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Recurrent Neural Networks</h1>
          <p className="text-neutral-700 mt-2">
            Understanding RNNs and their implementation for sequential data processing
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models/cnn">
              <ArrowLeft className="mr-2 h-4 w-4" /> Previous: CNNs
            </Link>
          </Button>
        </div>
      </div>

      <Tabs defaultValue="explanation" value={activeTab} onValueChange={setActiveTab} className="space-y-8">
        <TabsList className="grid w-full grid-cols-3 bg-neutral-100 text-neutral-900">
          <TabsTrigger value="explanation" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BookOpen className="h-4 w-4" />
            <span>Explanation</span>
          </TabsTrigger>
          <TabsTrigger value="architecture" className="flex items-center gap-2 data-[state=active]:bg-white">
            <Network className="h-4 w-4" />
            <span>Architecture</span>
          </TabsTrigger>
          <TabsTrigger value="notebook" className="flex items-center gap-2 data-[state=active]:bg-white">
            <Code className="h-4 w-4" />
            <span>Notebook</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="explanation" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">What are Recurrent Neural Networks?</CardTitle>
              <CardDescription className="text-neutral-600">
                Neural networks designed to recognize patterns in sequences of data
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Recurrent Neural Networks (RNNs) are a class of neural networks designed to work with sequential data,
                such as time series, text, speech, or video. Unlike traditional feedforward neural networks, RNNs have
                connections that form cycles, allowing information to persist from one step to the next.
              </p>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">Key Concepts in RNNs</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Memory</strong>: RNNs maintain a hidden state that captures
                    information about previous inputs
                  </li>
                  <li>
                    <strong className="text-neutral-900">Recurrent Connections</strong>: Connections that feed the
                    output of a neuron back into itself
                  </li>
                  <li>
                    <strong className="text-neutral-900">Sequence Processing</strong>: The ability to process inputs one
                    element at a time while maintaining context
                  </li>
                  <li>
                    <strong className="text-neutral-900">Variable Length Inputs/Outputs</strong>: Can handle sequences
                    of different lengths
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">How RNNs Work</h3>
              <p className="text-neutral-700 mb-4">
                RNNs process sequential data by maintaining a hidden state that gets updated at each time step:
              </p>
              <ol className="list-decimal list-inside space-y-4 text-neutral-700">
                <li>
                  <strong className="text-neutral-900">Input Processing</strong>: At each time step, the network takes a
                  new input from the sequence
                </li>
                <li>
                  <strong className="text-neutral-900">State Update</strong>: The hidden state is updated based on the
                  current input and the previous hidden state
                </li>
                <li>
                  <strong className="text-neutral-900">Output Generation</strong>: The network produces an output based
                  on the current hidden state
                </li>
                <li>
                  <strong className="text-neutral-900">Recurrence</strong>: The process repeats for each element in the
                  sequence
                </li>
              </ol>

              <div className="bg-neutral-100 p-4 rounded-lg mt-6">
                <h3 className="font-medium text-neutral-900 mb-2">RNN Applications</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Natural Language Processing</strong>: Text generation,
                    sentiment analysis, machine translation
                  </li>
                  <li>
                    <strong className="text-neutral-900">Speech Recognition</strong>: Converting spoken language to text
                  </li>
                  <li>
                    <strong className="text-neutral-900">Time Series Prediction</strong>: Stock prices, weather
                    forecasting, sensor readings
                  </li>
                  <li>
                    <strong className="text-neutral-900">Music Generation</strong>: Creating musical sequences
                  </li>
                  <li>
                    <strong className="text-neutral-900">Video Analysis</strong>: Understanding actions and events in
                    video sequences
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Advanced RNN Architectures</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Basic RNNs suffer from the vanishing/exploding gradient problem, making them difficult to train on long
                sequences. Advanced architectures have been developed to address these limitations:
              </p>

              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-neutral-900 mb-2">Long Short-Term Memory (LSTM)</h3>
                  <p className="text-neutral-700">
                    LSTMs are designed to overcome the vanishing gradient problem by introducing a cell state and
                    various gates that control the flow of information:
                  </p>
                  <ul className="list-disc list-inside mt-2 text-neutral-700">
                    <li>
                      <strong className="text-neutral-900">Forget Gate</strong>: Decides what information to discard
                      from the cell state
                    </li>
                    <li>
                      <strong className="text-neutral-900">Input Gate</strong>: Decides what new information to store in
                      the cell state
                    </li>
                    <li>
                      <strong className="text-neutral-900">Output Gate</strong>: Decides what parts of the cell state to
                      output
                    </li>
                    <li>
                      <strong className="text-neutral-900">Cell State</strong>: A memory channel that runs through the
                      entire sequence
                    </li>
                  </ul>
                </div>

                <div>
                  <h3 className="font-medium text-neutral-900 mb-2">Gated Recurrent Unit (GRU)</h3>
                  <p className="text-neutral-700">
                    GRUs are a simplified version of LSTMs with fewer gates, making them computationally more efficient:
                  </p>
                  <ul className="list-disc list-inside mt-2 text-neutral-700">
                    <li>
                      <strong className="text-neutral-900">Update Gate</strong>: Combines the forget and input gates of
                      LSTM
                    </li>
                    <li>
                      <strong className="text-neutral-900">Reset Gate</strong>: Determines how much of the past
                      information to forget
                    </li>
                    <li>
                      <strong className="text-neutral-900">No separate cell state</strong>: Uses a single hidden state
                    </li>
                  </ul>
                </div>

                <div>
                  <h3 className="font-medium text-neutral-900 mb-2">Bidirectional RNNs</h3>
                  <p className="text-neutral-700">
                    Bidirectional RNNs process sequences in both forward and backward directions, allowing the network
                    to capture context from both past and future states:
                  </p>
                  <ul className="list-disc list-inside mt-2 text-neutral-700">
                    <li>Particularly useful for tasks where the entire sequence is available at once</li>
                    <li>Common in natural language processing for understanding context in both directions</li>
                    <li>Can be combined with LSTM or GRU cells</li>
                  </ul>
                </div>

                <div>
                  <h3 className="font-medium text-neutral-900 mb-2">Attention Mechanisms</h3>
                  <p className="text-neutral-700">
                    Attention mechanisms allow RNNs to focus on specific parts of the input sequence when generating
                    outputs:
                  </p>
                  <ul className="list-disc list-inside mt-2 text-neutral-700">
                    <li>Helps with long-range dependencies in sequences</li>
                    <li>Forms the basis for transformer models</li>
                    <li>Particularly effective for machine translation and text summarization</li>
                  </ul>
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Challenges and Solutions</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Challenges</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Vanishing/exploding gradients</li>
                      <li>Difficulty capturing long-range dependencies</li>
                      <li>Computational inefficiency for very long sequences</li>
                      <li>Sequential nature limits parallelization</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Solutions</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Advanced architectures (LSTM, GRU)</li>
                      <li>Gradient clipping</li>
                      <li>Skip connections</li>
                      <li>Attention mechanisms and transformers</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="architecture" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">RNN Architecture Overview</CardTitle>
              <CardDescription className="text-neutral-600">
                Understanding the structure and components of recurrent neural networks
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="mb-8">
                  <RNNVisualization width={600} height={400} />
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-medium text-neutral-900 mb-3">Basic RNN Cell</h3>
                    <p className="text-neutral-700 mb-3">
                      The basic RNN cell computes its hidden state as a function of the previous hidden state and the
                      current input. The output is typically a function of the current hidden state.
                    </p>
                    <div className="bg-neutral-100 p-4 rounded-md">
                      <div className="text-sm text-neutral-700">
                        <strong>Hidden state update:</strong>
                        <br />
                        h_t = tanh(W_h · h_{"{t - 1}"} + W_x · x_t + b_h)
                        <br />
                        <br />
                        <strong>Output:</strong>
                        <br />
                        y_t = W_y · h_t + b_y
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-medium text-neutral-900 mb-3">LSTM Cell</h3>
                    <p className="text-neutral-700 mb-3">
                      The LSTM cell introduces a cell state and three gates (forget, input, output) to control
                      information flow and address the vanishing gradient problem.
                    </p>
                    <div className="bg-neutral-100 p-4 rounded-md">
                      <div className="text-sm text-neutral-700">
                        <strong>Forget gate:</strong> f_t = σ(W_f · [h_{"{t - 1}"}, x_t] + b_f)
                        <br />
                        <strong>Input gate:</strong> i_t = σ(W_i · [h_{"{t - 1}"}, x_t] + b_i)
                        <br />
                        <strong>Cell state:</strong> C_t = f_t * C_{"{t - 1}"} + i_t * tanh(W_C · [h_{"{t - 1}"}, x_t] +
                        b_C)
                        <br />
                        <strong>Output gate:</strong> o_t = σ(W_o · [h_{"{t - 1}"}, x_t] + b_o)
                        <br />
                        <strong>Hidden state:</strong> h_t = o_t * tanh(C_t)
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">RNN Variants and Use Cases</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Different RNN architectures are suited for different types of sequence processing tasks:
              </p>

              <div className="space-y-4">
                <div className="border border-neutral-300 rounded-md p-4">
                  <h3 className="font-medium text-neutral-900 mb-2">Many-to-One Architecture</h3>
                  <p className="text-neutral-700 mb-2">Processes a sequence and produces a single output at the end.</p>
                  <ul className="list-disc list-inside text-sm text-neutral-700">
                    <li>
                      <strong>Use cases:</strong> Sentiment analysis, text classification, sequence classification
                    </li>
                    <li>
                      <strong>Example:</strong> Classifying a movie review as positive or negative
                    </li>
                    <li>
                      <strong>Structure:</strong> Processes all input tokens and outputs a classification at the end
                    </li>
                  </ul>
                </div>

                <div className="border border-neutral-300 rounded-md p-4">
                  <h3 className="font-medium text-neutral-900 mb-2">One-to-Many Architecture</h3>
                  <p className="text-neutral-700 mb-2">Takes a single input and produces a sequence of outputs.</p>
                  <ul className="list-disc list-inside text-sm text-neutral-700">
                    <li>
                      <strong>Use cases:</strong> Image captioning, music generation
                    </li>
                    <li>
                      <strong>Example:</strong> Generating a description for an image
                    </li>
                    <li>
                      <strong>Structure:</strong> Takes an initial input and generates a sequence step by step
                    </li>
                  </ul>
                </div>

                <div className="border border-neutral-300 rounded-md p-4">
                  <h3 className="font-medium text-neutral-900 mb-2">Many-to-Many Architecture</h3>
                  <p className="text-neutral-700 mb-2">Processes a sequence and produces a sequence of outputs.</p>
                  <ul className="list-disc list-inside text-sm text-neutral-700">
                    <li>
                      <strong>Use cases:</strong> Machine translation, speech recognition, video classification
                    </li>
                    <li>
                      <strong>Example:</strong> Translating an English sentence to French
                    </li>
                    <li>
                      <strong>Structure:</strong> Can be encoder-decoder (sequence-to-sequence) or synchronous (same
                      length input/output)
                    </li>
                  </ul>
                </div>

                <div className="border border-neutral-300 rounded-md p-4">
                  <h3 className="font-medium text-neutral-900 mb-2">Encoder-Decoder Architecture</h3>
                  <p className="text-neutral-700 mb-2">
                    A special type of many-to-many architecture with two distinct phases.
                  </p>
                  <ul className="list-disc list-inside text-sm text-neutral-700">
                    <li>
                      <strong>Encoder:</strong> Processes the input sequence and compresses it into a context vector
                    </li>
                    <li>
                      <strong>Decoder:</strong> Generates the output sequence based on the context vector
                    </li>
                    <li>
                      <strong>Use cases:</strong> Machine translation, text summarization, chatbots
                    </li>
                    <li>
                      <strong>Enhancements:</strong> Often combined with attention mechanisms for better performance
                    </li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">RNN Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to implement a Recurrent Neural Network for sentiment analysis using
                Python and TensorFlow/Keras. Execute each cell to see the results.
              </p>
            </div>

            <div className="space-y-6">
              <NotebookCell
                cellId="cell0"
                executionCount={1}
                initialCode="import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import classification_report, accuracy_score

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">
                  Step 1: Load and preprocess the IMDB movie review dataset
                </p>
                <p>We'll use the IMDB dataset which contains movie reviews labeled as positive or negative.</p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode="# Load the IMDB dataset
max_features = 5000  # Number of words to consider as features
max_len = 100  # Cut texts after this number of words

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(f'Found {len(x_train)} training sequences and {len(x_test)} test sequences')

# Pad sequences to ensure uniform input size
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')

# Build the model
print('Building model...')
model = Sequential()
model.add(Embedding(max_features, 32, input_length=max_len))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

model.summary()"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Train the LSTM model</p>
                <p>Now we'll train our LSTM model on the IMDB dataset.</p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode="# Train the model
batch_size = 32
epochs = 5

print('Training model...')
history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2)"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Evaluate the model</p>
                <p>Let's evaluate our model on the test set and generate a classification report.</p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode="# Evaluate the model
print('Evaluating model...')
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate accuracy
test_acc = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {test_acc:.4f}')

# Generate classification report
print('\nClassification Report:\n')
target_names = ['negative', 'positive']
print(classification_report(y_test, y_pred, target_names=target_names))"
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 4: Make predictions on individual reviews</p>
                <p>Let's see how our model performs on some individual examples from the test set.</p>
              </div>

              <NotebookCell
                cellId="cell4"
                executionCount={5}
                initialCode="# Make predictions on individual reviews
# For demonstration purposes, let's create a simple prediction function

import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def predict_sentiment(review, word_to_index, model, max_len=100):
  # Tokenize the review
  words = review.split()
  indices = [word_to_index.get(word, 0) for word in words]  # Use 0 for unknown words

  # Pad the sequence
  padded_indices = indices[:max_len]
  padded_indices += [0] * (max_len - len(padded_indices))
  input_data = np.array([padded_indices])  # Reshape to (1, max_len)

  # Make prediction
  prediction = model.predict(input_data)[0][0]
  sentiment = 'Positive' if prediction > 0.5 else 'Negative'
  confidence = prediction if sentiment == 'Positive' else (1 - prediction)

  return f'{sentiment} ({confidence:.2f})'

# Mock model and word_to_index for demonstration
class MockModel:
  def predict(self, input_data):
      # This is just a placeholder for demonstration
      # In a real scenario, we would use the actual trained model
      sentiments = {
          'terrible': 0.12,
          'bad': 0.23,
          'good': 0.76,
          'amazing': 0.89
      }
      
      # Check which keyword is in the input
      for word, score in sentiments.items():
          if word in str(input_data):
              return np.array([[  score in sentiments.items():
          if word in str(input_data):
              return np.array([[score]])
      
      return np.array([[0.5]])  # Neutral if no keyword found

model = MockModel()
word_to_index = {word: i for i, word in enumerate(['the', 'movie', 'was', 'good', 'bad', 'amazing', 'terrible'])}

# Example reviews
review1 = 'the movie was terrible'
review2 = 'the movie was amazing'
review3 = 'the movie was good'
review4 = 'the movie was bad'

# Make predictions
prediction1 = predict_sentiment(review1, word_to_index, model)
prediction2 = predict_sentiment(review2, word_to_index, model)
prediction3 = predict_sentiment(review3, word_to_index, model)
prediction4 = predict_sentiment(review4, word_to_index, model)

print(f'Review: {review1}\nPrediction: {prediction1}\n')
print(f'Review: {review2}\nPrediction: {prediction2}\n')
print(f'Review: {review3}\nPrediction: {prediction3}\n')
print(f'Review: {review4}\nPrediction: {prediction4}\n')"
                readOnly={false}
                onExecute={handleExecuteCode}
              />
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
