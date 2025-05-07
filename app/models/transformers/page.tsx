"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, ArrowRight, BookOpen, Code, BarChart, Layers, BrainCircuit } from "lucide-react"
import Link from "next/link"
import NotebookCell from "@/components/notebook-cell"
import ModelVisualization from "@/components/model-visualization"

export default function TransformersPage() {
  const [activeTab, setActiveTab] = useState("explanation")
  const [executionCount, setExecutionCount] = useState(1)

  // Transformer visualization function
  const renderTransformer = (
    ctx: CanvasRenderingContext2D,
    params: Record<string, number>,
    width: number,
    height: number,
  ) => {
    const { heads, layers, attentionStrength } = params

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Set up dimensions
    const margin = 40
    const plotWidth = width - 2 * margin
    const plotHeight = height - 2 * margin

    // Draw background
    ctx.fillStyle = "#f8f9fa"
    ctx.fillRect(0, 0, width, height)

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "16px Arial"
    ctx.textAlign = "center"
    ctx.fillText(`Transformer with ${heads} Attention Heads and ${layers} Layers`, width / 2, 25)

    // Define sample sentence
    const sentence = ["The", "cat", "sat", "on", "the", "mat"]
    const tokenCount = sentence.length

    // Calculate token box dimensions
    const boxWidth = Math.min(80, plotWidth / tokenCount)
    const boxHeight = 30
    const boxSpacing = (plotWidth - boxWidth * tokenCount) / (tokenCount - 1)

    // Draw input tokens
    const inputY = margin + 20
    ctx.font = "14px Arial"
    ctx.textAlign = "center"
    ctx.fillStyle = "#000"
    ctx.fillText("Input Tokens", margin + plotWidth / 2, inputY - 10)

    for (let i = 0; i < tokenCount; i++) {
      const x = margin + i * (boxWidth + boxSpacing)
      const y = inputY

      // Draw token box
      ctx.fillStyle = "#e6f7ff"
      ctx.strokeStyle = "#1890ff"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.rect(x, y, boxWidth, boxHeight)
      ctx.fill()
      ctx.stroke()

      // Draw token text
      ctx.fillStyle = "#000"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(sentence[i], x + boxWidth / 2, y + boxHeight / 2)
    }

    // Draw embedding layer
    const embedY = inputY + boxHeight + 20
    ctx.fillStyle = "#000"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText("Embedding Layer", margin, embedY + 15)

    ctx.fillStyle = "#f0f9ff"
    ctx.strokeStyle = "#0ea5e9"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.rect(margin + 120, embedY, plotWidth - 120, 30)
    ctx.fill()
    ctx.stroke()

    ctx.fillStyle = "#0ea5e9"
    ctx.textAlign = "center"
    ctx.fillText("Token + Position Embeddings", margin + plotWidth / 2 + 60, embedY + 15)

    // Draw self-attention layers
    const layerHeight = (plotHeight - 180 - boxHeight * 2) / layers
    const layerSpacing = 10

    for (let layer = 0; layer < layers; layer++) {
      const layerY = embedY + 50 + layer * (layerHeight + layerSpacing)

      // Draw layer background
      ctx.fillStyle = `rgba(249, 250, 251, 0.7)`
      ctx.strokeStyle = "#d1d5db"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.rect(margin, layerY, plotWidth, layerHeight)
      ctx.fill()
      ctx.stroke()

      // Draw layer label
      ctx.fillStyle = "#000"
      ctx.textAlign = "left"
      ctx.textBaseline = "middle"
      ctx.fillText(`Layer ${layer + 1}`, margin + 10, layerY + layerHeight / 2)

      // Draw multi-head attention
      const headWidth = (plotWidth - 150) / heads
      const headHeight = layerHeight * 0.6
      const headY = layerY + (layerHeight - headHeight) / 2

      // Draw attention mechanism label
      ctx.fillStyle = "#000"
      ctx.textAlign = "center"
      ctx.fillText("Multi-Head Attention", margin + 75, layerY + layerHeight / 2)

      for (let head = 0; head < heads; head++) {
        const headX = margin + 150 + head * headWidth

        // Draw attention head box
        const headColor = `hsl(${30 + (head * 360) / heads}, 90%, 85%)`
        const headBorder = `hsl(${30 + (head * 360) / heads}, 90%, 60%)`

        ctx.fillStyle = headColor
        ctx.strokeStyle = headBorder
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.rect(headX, headY, headWidth - 10, headHeight)
        ctx.fill()
        ctx.stroke()

        // Draw head label
        ctx.fillStyle = "#000"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(`H${head + 1}`, headX + (headWidth - 10) / 2, headY + headHeight / 2)
      }

      // Draw feed-forward network
      if (layer < layers - 1) {
        ctx.strokeStyle = "#9ca3af"
        ctx.lineWidth = 1
        ctx.setLineDash([5, 3])
        ctx.beginPath()
        ctx.moveTo(margin, layerY + layerHeight)
        ctx.lineTo(margin + plotWidth, layerY + layerHeight)
        ctx.stroke()
        ctx.setLineDash([])
      }
    }

    // Draw attention connections for the first token to all others
    const sourceX = margin + boxWidth / 2
    const sourceY = inputY + boxHeight

    // Draw attention from first token to others
    for (let i = 0; i < tokenCount; i++) {
      const targetX = margin + i * (boxWidth + boxSpacing) + boxWidth / 2
      const targetY = inputY

      // Skip self-connection for clarity
      if (i === 0) continue

      // Draw attention line with strength-based opacity and width
      const strength = (attentionStrength / 10) * (1 - (i / tokenCount) * 0.7)
      ctx.strokeStyle = `rgba(239, 68, 68, ${strength})`
      ctx.lineWidth = 1 + (attentionStrength / 10) * (1 - (i / tokenCount) * 0.5)
      ctx.beginPath()
      ctx.moveTo(sourceX, sourceY)
      ctx.bezierCurveTo(sourceX, sourceY + 30, targetX, sourceY + 30, targetX, targetY)
      ctx.stroke()
    }

    // Draw output tokens
    const outputY = embedY + 50 + layers * (layerHeight + layerSpacing) + 20
    ctx.font = "14px Arial"
    ctx.textAlign = "center"
    ctx.fillStyle = "#000"
    ctx.fillText("Output Representations", margin + plotWidth / 2, outputY - 10)

    for (let i = 0; i < tokenCount; i++) {
      const x = margin + i * (boxWidth + boxSpacing)
      const y = outputY

      // Draw token box
      ctx.fillStyle = "#f0fdf4"
      ctx.strokeStyle = "#22c55e"
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.rect(x, y, boxWidth, boxHeight)
      ctx.fill()
      ctx.stroke()

      // Draw token text
      ctx.fillStyle = "#000"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(sentence[i], x + boxWidth / 2, y + boxHeight / 2)
    }

    // Draw legend
    const legendX = width - 180
    const legendY = height - 100
    const legendBoxSize = 15
    const legendSpacing = 25

    // Legend title
    ctx.fillStyle = "#000"
    ctx.textAlign = "left"
    ctx.font = "14px Arial"
    ctx.fillText("Legend", legendX, legendY)

    // Attention strength
    ctx.fillStyle = "rgba(239, 68, 68, 0.8)"
    ctx.fillRect(legendX, legendY + legendSpacing, legendBoxSize, legendBoxSize)
    ctx.fillStyle = "#000"
    ctx.fillText("Attention", legendX + legendBoxSize + 5, legendY + legendSpacing + 12)

    // Attention heads
    ctx.fillStyle = `hsl(60, 90%, 85%)`
    ctx.strokeStyle = `hsl(60, 90%, 60%)`
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.rect(legendX, legendY + 2 * legendSpacing, legendBoxSize, legendBoxSize)
    ctx.fill()
    ctx.stroke()
    ctx.fillStyle = "#000"
    ctx.fillText("Attention Head", legendX + legendBoxSize + 5, legendY + 2 * legendSpacing + 12)

    // Attention strength indicator
    ctx.fillStyle = "#000"
    ctx.fillText(`Attention Strength: ${attentionStrength}`, legendX, legendY + 3 * legendSpacing + 12)
  }

  const handleExecuteCode = async (code: string, cellId: string) => {
    // Simulate code execution
    await new Promise((resolve) => setTimeout(resolve, 1000))
    setExecutionCount((prev) => prev + 1)

    if (cellId === "cell1") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Model: bert-base-uncased
          <br />
          Tokenized input: ['[CLS]', 'the', 'transformer', 'model', 'revolutionized', 'natural', 'language',
          'processing', '.', '[SEP]']
          <br />
          Token IDs: [101, 1996, 19081, 2944, 25589, 2307, 2653, 3173, 1012, 102]
          <br />
          <br />
          Input shape: torch.Size([1, 10])
          <br />
          Output shape: torch.Size([1, 10, 768])
          <br />
          <br />
          First token embedding (CLS) first 10 values:
          <br />
          tensor([[ 0.0345, -0.1253, 0.1208, 0.0786, 0.1412, -0.0568, 0.0934, -0.0107, -0.0127, 0.0591]])
        </div>
      )
    } else if (cellId === "cell2") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Attention visualization for head 0, layer 0:
          <br />
          <div className="bg-neutral-100 h-40 w-full rounded-md flex items-center justify-center">
            <p className="text-neutral-500">Attention heatmap visualization</p>
          </div>
          <br />
          Observations:
          <br />- The [CLS] token attends strongly to important content words
          <br />- Adjacent words show stronger attention patterns
          <br />- Punctuation receives less attention
          <br />- Different heads capture different linguistic patterns
        </div>
      )
    } else if (cellId === "cell3") {
      return (
        <div className="font-mono text-sm whitespace-pre-wrap">
          Input: "The transformer architecture has changed NLP."
          <br />
          Masked input: "The [MASK] architecture has changed NLP."
          <br />
          <br />
          Top 5 predictions for [MASK]:
          <br />
          transformer: 0.8721
          <br />
          neural: 0.0543
          <br />
          new: 0.0231
          <br />
          language: 0.0187
          <br />
          network: 0.0102
          <br />
          <br />
          Generated text from GPT-2:
          <br />
          "Transformers have revolutionized natural language processing by enabling models to capture long-range
          dependencies and contextual information more effectively than previous architectures."
        </div>
      )
    }

    return "Executed successfully"
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-neutral-900">Transformers</h1>
          <p className="text-neutral-700 mt-2">
            The revolutionary architecture behind modern natural language processing models
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" asChild>
            <Link href="/models/rnn">
              <ArrowLeft className="mr-2 h-4 w-4" /> Recurrent Neural Networks
            </Link>
          </Button>
          <Button asChild variant="notebook">
            <Link href="/models/comparison">
              Next: Model Comparison <ArrowRight className="ml-2 h-4 w-4" />
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
          <TabsTrigger value="visualization" className="flex items-center gap-2 data-[state=active]:bg-white">
            <BarChart className="h-4 w-4" />
            <span>Visualization</span>
          </TabsTrigger>
          <TabsTrigger value="notebook" className="flex items-center gap-2 data-[state=active]:bg-white">
            <Code className="h-4 w-4" />
            <span>Notebook</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="explanation" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">What are Transformers?</CardTitle>
              <CardDescription className="text-neutral-600">
                A revolutionary neural network architecture that powers modern NLP models
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-neutral-700">
                Transformers are a type of neural network architecture introduced in the 2017 paper "Attention Is All
                You Need" by Vaswani et al. They revolutionized natural language processing by replacing recurrent
                neural networks (RNNs) with self-attention mechanisms, allowing for more parallelization during training
                and better modeling of long-range dependencies in sequential data.
              </p>

              <div className="flex justify-center my-6">
                <div className="relative w-full max-w-2xl h-[300px] bg-neutral-50 rounded-lg border border-neutral-300 flex items-center justify-center">
                  <div className="absolute inset-0 p-6">
                    <div className="flex flex-col h-full">
                      {/* Encoder-Decoder Architecture Diagram */}
                      <div className="text-center font-medium text-neutral-900 mb-4">Transformer Architecture</div>
                      <div className="flex flex-1 gap-4">
                        {/* Encoder */}
                        <div className="flex-1 border border-blue-200 bg-blue-50 rounded-lg p-3 flex flex-col">
                          <div className="text-center text-sm font-medium text-blue-700 mb-2">Encoder</div>
                          <div className="flex-1 flex flex-col gap-2">
                            <div className="bg-blue-100 border border-blue-300 rounded p-2 text-xs text-center">
                              Multi-Head Self-Attention
                            </div>
                            <div className="bg-blue-100 border border-blue-300 rounded p-2 text-xs text-center">
                              Feed Forward Network
                            </div>
                            <div className="bg-blue-100 border border-blue-300 rounded p-2 text-xs text-center">
                              Layer Normalization
                            </div>
                            <div className="text-center text-xs mt-1">× N</div>
                          </div>
                          <div className="mt-2 bg-blue-200 border border-blue-400 rounded p-1 text-xs text-center">
                            Input Embeddings + Positional Encoding
                          </div>
                          <div className="mt-2 text-center text-xs">Input Sequence</div>
                        </div>

                        {/* Decoder */}
                        <div className="flex-1 border border-green-200 bg-green-50 rounded-lg p-3 flex flex-col">
                          <div className="text-center text-sm font-medium text-green-700 mb-2">Decoder</div>
                          <div className="flex-1 flex flex-col gap-2">
                            <div className="bg-green-100 border border-green-300 rounded p-2 text-xs text-center">
                              Masked Multi-Head Self-Attention
                            </div>
                            <div className="bg-green-100 border border-green-300 rounded p-2 text-xs text-center">
                              Multi-Head Cross-Attention
                            </div>
                            <div className="bg-green-100 border border-green-300 rounded p-2 text-xs text-center">
                              Feed Forward Network
                            </div>
                            <div className="text-center text-xs mt-1">× N</div>
                          </div>
                          <div className="mt-2 bg-green-200 border border-green-400 rounded p-1 text-xs text-center">
                            Output Embeddings + Positional Encoding
                          </div>
                          <div className="mt-2 text-center text-xs">Output Sequence</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg">
                <h3 className="font-medium text-neutral-900 mb-2">Key Innovations of Transformers</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong className="text-neutral-900">Self-Attention Mechanism</strong>: Allows the model to weigh
                    the importance of different words in a sequence when encoding each word, capturing contextual
                    relationships
                  </li>
                  <li>
                    <strong className="text-neutral-900">Parallelization</strong>: Unlike RNNs, transformers process all
                    tokens simultaneously, enabling much faster training on modern hardware
                  </li>
                  <li>
                    <strong className="text-neutral-900">Long-range Dependencies</strong>: Can capture relationships
                    between words regardless of their distance in the sequence, overcoming the limitations of RNNs
                  </li>
                  <li>
                    <strong className="text-neutral-900">Positional Encoding</strong>: Adds information about token
                    positions since the model has no inherent notion of sequence order
                  </li>
                  <li>
                    <strong className="text-neutral-900">Multi-head Attention</strong>: Allows the model to focus on
                    different aspects of the input simultaneously
                  </li>
                </ul>
              </div>

              <h3 className="text-xl font-medium text-neutral-900 mt-6">Transformer Architecture</h3>
              <p className="text-neutral-700 mb-4">
                The original transformer architecture consists of an encoder and a decoder, though many modern variants
                use only the encoder (like BERT) or only the decoder (like GPT):
              </p>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
                  <div className="flex items-center gap-2 mb-3">
                    <Layers className="h-5 w-5 text-blue-600" />
                    <h4 className="text-lg font-medium text-blue-800">Encoder</h4>
                  </div>
                  <p className="text-neutral-700 mb-3">
                    Processes the input sequence and builds contextual representations.
                  </p>
                  <ul className="list-disc list-inside space-y-2 text-neutral-700">
                    <li>
                      <strong>Multi-head self-attention</strong>: Allows each position to attend to all positions in the
                      input sequence
                    </li>
                    <li>
                      <strong>Feed-forward neural network</strong>: Processes each position independently with the same
                      network
                    </li>
                    <li>
                      <strong>Residual connections</strong>: Help gradient flow during training
                    </li>
                    <li>
                      <strong>Layer normalization</strong>: Stabilizes the learning process
                    </li>
                  </ul>
                </div>

                <div className="bg-green-50 p-6 rounded-lg border border-green-200">
                  <div className="flex items-center gap-2 mb-3">
                    <Layers className="h-5 w-5 text-green-600" />
                    <h4 className="text-lg font-medium text-green-800">Decoder</h4>
                  </div>
                  <p className="text-neutral-700 mb-3">
                    Generates the output sequence based on the encoder's representations.
                  </p>
                  <ul className="list-disc list-inside space-y-2 text-neutral-700">
                    <li>
                      <strong>Masked multi-head self-attention</strong>: Prevents positions from attending to future
                      positions
                    </li>
                    <li>
                      <strong>Multi-head cross-attention</strong>: Connects decoder to encoder outputs
                    </li>
                    <li>
                      <strong>Feed-forward neural network</strong>: Same as in the encoder
                    </li>
                    <li>
                      <strong>Autoregressive generation</strong>: Outputs one token at a time during inference
                    </li>
                  </ul>
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-6">
                <h3 className="font-medium text-neutral-900 mb-2">Self-Attention Mechanism</h3>
                <p className="text-neutral-700 mb-2">
                  The core innovation of transformers is the self-attention mechanism, which works as follows:
                </p>
                <ol className="list-decimal list-inside space-y-2 text-neutral-700">
                  <li>For each token, create three vectors: Query (Q), Key (K), and Value (V)</li>
                  <li>Calculate attention scores by taking the dot product of the Query with all Keys</li>
                  <li>Scale the scores and apply softmax to get attention weights</li>
                  <li>Multiply each Value vector by its corresponding attention weight and sum them up</li>
                  <li>
                    The result is the new representation for the token, capturing its relationships with all other
                    tokens
                  </li>
                </ol>
                <div className="mt-4 text-sm text-neutral-600 bg-white p-3 rounded border border-neutral-300 font-mono">
                  Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>)V
                </div>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Advantages and Limitations</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Advantages</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Captures long-range dependencies effectively</li>
                      <li>Highly parallelizable, enabling training on massive datasets</li>
                      <li>Scales well with model size and data</li>
                      <li>State-of-the-art performance on NLP tasks</li>
                      <li>Versatile architecture adaptable to many domains</li>
                      <li>Enables transfer learning through pre-training</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-neutral-900 mb-1">Limitations</h4>
                    <ul className="list-disc list-inside space-y-1 text-neutral-700">
                      <li>Quadratic complexity with sequence length (O(n²))</li>
                      <li>High computational and memory requirements</li>
                      <li>Limited context window in practice</li>
                      <li>Requires large datasets to train effectively</li>
                      <li>Less interpretable than simpler models</li>
                      <li>Energy-intensive training process</li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Popular Transformer Models</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-neutral-700">
                The transformer architecture has led to numerous breakthrough models in NLP:
              </p>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <div className="flex items-center gap-2 mb-3">
                    <BrainCircuit className="h-5 w-5 text-blue-600" />
                    <h3 className="text-lg font-medium text-neutral-800">BERT</h3>
                  </div>
                  <p className="text-neutral-700">
                    <strong>Bidirectional Encoder Representations from Transformers</strong>. Uses only the encoder part
                    and is pre-trained on masked language modeling and next sentence prediction. Excels at understanding
                    context for classification, NER, and question answering.
                  </p>
                  <div className="mt-3 text-sm text-neutral-500">
                    <strong>Key innovation:</strong> Bidirectional context
                  </div>
                </div>

                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <div className="flex items-center gap-2 mb-3">
                    <BrainCircuit className="h-5 w-5 text-green-600" />
                    <h3 className="text-lg font-medium text-neutral-800">GPT Family</h3>
                  </div>
                  <p className="text-neutral-700">
                    <strong>Generative Pre-trained Transformer</strong>. Uses only the decoder part and is trained to
                    predict the next token. Each generation (GPT-2, GPT-3, GPT-4) has scaled up in size, demonstrating
                    remarkable text generation and few-shot learning.
                  </p>
                  <div className="mt-3 text-sm text-neutral-500">
                    <strong>Key innovation:</strong> Scale and generative capabilities
                  </div>
                </div>

                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <div className="flex items-center gap-2 mb-3">
                    <BrainCircuit className="h-5 w-5 text-purple-600" />
                    <h3 className="text-lg font-medium text-neutral-800">T5</h3>
                  </div>
                  <p className="text-neutral-700">
                    <strong>Text-to-Text Transfer Transformer</strong>. Uses the full encoder-decoder architecture and
                    frames all NLP tasks as text-to-text problems. This unified approach allows it to handle multiple
                    tasks with the same model.
                  </p>
                  <div className="mt-3 text-sm text-neutral-500">
                    <strong>Key innovation:</strong> Unified text-to-text framework
                  </div>
                </div>

                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <div className="flex items-center gap-2 mb-3">
                    <BrainCircuit className="h-5 w-5 text-amber-600" />
                    <h3 className="text-lg font-medium text-neutral-800">RoBERTa</h3>
                  </div>
                  <p className="text-neutral-700">
                    <strong>Robustly Optimized BERT Approach</strong>. An optimized version of BERT with improved
                    training methodology, including longer training, bigger batches, and more data. Removes the next
                    sentence prediction task and dynamically changes the masking pattern.
                  </p>
                  <div className="mt-3 text-sm text-neutral-500">
                    <strong>Key innovation:</strong> Optimized training procedure
                  </div>
                </div>

                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <div className="flex items-center gap-2 mb-3">
                    <BrainCircuit className="h-5 w-5 text-red-600" />
                    <h3 className="text-lg font-medium text-neutral-800">BART</h3>
                  </div>
                  <p className="text-neutral-700">
                    <strong>Bidirectional and Auto-Regressive Transformers</strong>. Combines the bidirectional encoder
                    of BERT with the autoregressive decoder of GPT. Pre-trained to reconstruct text that has been
                    corrupted in various ways, making it effective for both understanding and generation.
                  </p>
                  <div className="mt-3 text-sm text-neutral-500">
                    <strong>Key innovation:</strong> Denoising pre-training objectives
                  </div>
                </div>

                <div className="bg-neutral-50 p-6 rounded-lg border border-neutral-300">
                  <div className="flex items-center gap-2 mb-3">
                    <BrainCircuit className="h-5 w-5 text-cyan-600" />
                    <h3 className="text-lg font-medium text-neutral-800">LLaMA & Mistral</h3>
                  </div>
                  <p className="text-neutral-700">
                    <strong>Open-source Large Language Models</strong>. These models provide high-quality alternatives
                    to proprietary models, with efficient architectures that can run on consumer hardware. They've
                    enabled a wave of fine-tuned specialized models and applications.
                  </p>
                  <div className="mt-3 text-sm text-neutral-500">
                    <strong>Key innovation:</strong> Efficient open-source architectures
                  </div>
                </div>
              </div>

              <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h3 className="font-medium text-blue-800 mb-2">Evolution of Transformer Models</h3>
                <p className="text-neutral-700 mb-3">
                  Transformer models have evolved rapidly since their introduction in 2017:
                </p>
                <div className="relative">
                  <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-blue-300"></div>
                  <div className="space-y-6 relative">
                    <div className="ml-10 relative">
                      <div className="absolute -left-10 top-1 w-4 h-4 rounded-full bg-blue-500"></div>
                      <div className="font-medium">2017: Original Transformer</div>
                      <div className="text-sm text-neutral-700">Introduced in "Attention Is All You Need" paper</div>
                    </div>
                    <div className="ml-10 relative">
                      <div className="absolute -left-10 top-1 w-4 h-4 rounded-full bg-blue-500"></div>
                      <div className="font-medium">2018: BERT & GPT-1</div>
                      <div className="text-sm text-neutral-700">
                        First major applications of encoder-only and decoder-only architectures
                      </div>
                    </div>
                    <div className="ml-10 relative">
                      <div className="absolute -left-10 top-1 w-4 h-4 rounded-full bg-blue-500"></div>
                      <div className="font-medium">2019: GPT-2, XLNet, RoBERTa</div>
                      <div className="text-sm text-neutral-700">Scaling up and optimizing training procedures</div>
                    </div>
                    <div className="ml-10 relative">
                      <div className="absolute -left-10 top-1 w-4 h-4 rounded-full bg-blue-500"></div>
                      <div className="font-medium">2020: GPT-3, T5</div>
                      <div className="text-sm text-neutral-700">Massive scaling and unified frameworks</div>
                    </div>
                    <div className="ml-10 relative">
                      <div className="absolute -left-10 top-1 w-4 h-4 rounded-full bg-blue-500"></div>
                      <div className="font-medium">2022-2023: ChatGPT, GPT-4, LLaMA, Mistral</div>
                      <div className="text-sm text-neutral-700">
                        Conversational abilities, multimodal capabilities, and efficient open-source models
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualization" className="space-y-8">
          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Interactive Transformer Visualization</CardTitle>
              <CardDescription className="text-neutral-600">
                Adjust the parameters to see how the transformer architecture processes text
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ModelVisualization
                title="Transformer Architecture Visualization"
                parameters={[
                  {
                    name: "heads",
                    min: 1,
                    max: 8,
                    step: 1,
                    default: 4,
                    label: "Attention Heads",
                  },
                  {
                    name: "layers",
                    min: 1,
                    max: 6,
                    step: 1,
                    default: 3,
                    label: "Transformer Layers",
                  },
                  {
                    name: "attentionStrength",
                    min: 1,
                    max: 10,
                    step: 1,
                    default: 5,
                    label: "Attention Strength",
                  },
                ]}
                renderVisualization={renderTransformer}
              />
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Understanding the Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Attention Heads</h3>
                <p className="text-neutral-700">
                  This parameter controls the number of attention heads in each transformer layer. Multiple attention
                  heads allow the model to focus on different aspects of the input simultaneously. Each head can learn
                  different types of relationships between tokens, such as:
                </p>
                <ul className="list-disc list-inside ml-4 text-neutral-700">
                  <li>Syntactic dependencies (subject-verb relationships)</li>
                  <li>Semantic relationships (word meanings and topics)</li>
                  <li>Coreference resolution (pronouns and their antecedents)</li>
                  <li>Entity relationships (people, places, organizations)</li>
                </ul>
                <p className="text-neutral-700 mt-2">
                  More heads can capture more complex patterns but increase computational requirements. Standard
                  transformer models typically use 8-16 heads per layer.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Transformer Layers</h3>
                <p className="text-neutral-700">
                  This parameter determines the depth of the transformer model. Each layer processes the output of the
                  previous layer, building increasingly abstract representations. The benefits of deeper models include:
                </p>
                <ul className="list-disc list-inside ml-4 text-neutral-700">
                  <li>Learning more complex patterns and hierarchical features</li>
                  <li>Building more sophisticated representations of language</li>
                  <li>Capturing different levels of abstraction</li>
                </ul>
                <p className="text-neutral-700 mt-2">
                  Modern transformer models typically use 12-24 layers for base models (like BERT-base or GPT-2) and up
                  to 96+ layers for the largest models (like GPT-4). Each additional layer increases computational cost
                  and memory requirements.
                </p>
              </div>

              <div className="space-y-2">
                <h3 className="font-medium text-neutral-900">Attention Strength</h3>
                <p className="text-neutral-700">
                  This parameter simulates the strength of attention connections between tokens. In the visualization,
                  it affects:
                </p>
                <ul className="list-disc list-inside ml-4 text-neutral-700">
                  <li>The opacity of the attention lines connecting tokens</li>
                  <li>The width of the attention connections</li>
                  <li>How focused or distributed the attention is across tokens</li>
                </ul>
                <p className="text-neutral-700 mt-2">
                  Higher values represent stronger focus on specific tokens, while lower values create more distributed
                  attention. In real transformers, attention weights are learned during training and vary based on the
                  context and the specific head. The visualization shows how attention from the first token to other
                  tokens varies in strength.
                </p>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Interpreting the Visualization</h3>
                <ul className="list-disc list-inside space-y-2 text-neutral-700">
                  <li>
                    <strong>Input Tokens</strong> (blue boxes): The words being processed by the transformer
                  </li>
                  <li>
                    <strong>Embedding Layer</strong> (light blue): Converts tokens to vector representations and adds
                    positional information
                  </li>
                  <li>
                    <strong>Attention Heads</strong> (colored boxes): Each head learns different patterns in the data
                  </li>
                  <li>
                    <strong>Attention Connections</strong> (red lines): Show how tokens attend to each other, with
                    stronger connections indicating higher attention weights
                  </li>
                  <li>
                    <strong>Transformer Layers</strong> (stacked sections): Each layer builds on the previous layer's
                    representations
                  </li>
                  <li>
                    <strong>Output Representations</strong> (green boxes): The final contextualized token
                    representations
                  </li>
                </ul>
                <p className="mt-3 text-neutral-700">Try adjusting the parameters to see how:</p>
                <ul className="list-disc list-inside space-y-1 text-neutral-700">
                  <li>Increasing the number of heads creates more parallel processing units</li>
                  <li>Adding more layers creates a deeper network that can learn more complex patterns</li>
                  <li>Changing attention strength affects how focused the connections are between specific tokens</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <Card className="border-neutral-300 bg-white">
            <CardHeader>
              <CardTitle className="text-neutral-900">Self-Attention Mechanism in Detail</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-neutral-700">
                  The self-attention mechanism is the core innovation of transformer models. It allows each token to
                  attend to all other tokens in the sequence, capturing contextual relationships regardless of distance.
                </p>

                <div className="bg-neutral-50 p-4 rounded-lg border border-neutral-300">
                  <h3 className="font-medium text-neutral-900 mb-3">How Self-Attention Works</h3>

                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium text-neutral-800 mb-2">
                        Step 1: Create Query, Key, and Value Vectors
                      </h4>
                      <p className="text-sm text-neutral-700">
                        For each token, three vectors are created by multiplying the token embedding by three learned
                        weight matrices:
                      </p>
                      <ul className="list-disc list-inside text-sm text-neutral-700 mt-1">
                        <li>
                          <strong>Query (Q)</strong>: What the token is looking for
                        </li>
                        <li>
                          <strong>Key (K)</strong>: What the token offers to others
                        </li>
                        <li>
                          <strong>Value (V)</strong>: The actual information the token contains
                        </li>
                      </ul>
                    </div>

                    <div>
                      <h4 className="font-medium text-neutral-800 mb-2">Step 2: Calculate Attention Scores</h4>
                      <p className="text-sm text-neutral-700">
                        The dot product between the Query of one token and the Keys of all tokens gives attention
                        scores, indicating how much each token should attend to others.
                      </p>
                      <div className="mt-2 text-sm font-mono bg-white p-2 rounded border border-neutral-200">
                        Score = Q · K<sup>T</sup>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium text-neutral-800 mb-2">Step 3: Scale and Apply Softmax</h4>
                      <p className="text-sm text-neutral-700">
                        Scores are scaled by the square root of the dimension of the Key vectors to prevent extremely
                        small gradients. Then softmax is applied to get attention weights that sum to 1.
                      </p>
                      <div className="mt-2 text-sm font-mono bg-white p-2 rounded border border-neutral-200">
                        Weights = softmax(Score / √d<sub>k</sub>)
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium text-neutral-800 mb-2">Step 4: Apply Weights to Values</h4>
                      <p className="text-sm text-neutral-700">
                        The attention weights are multiplied by the Value vectors and summed to create the output for
                        each token, which now contains information from all relevant tokens in the sequence.
                      </p>
                      <div className="mt-2 text-sm font-mono bg-white p-2 rounded border border-neutral-200">
                        Output = Weights · V
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 text-center">
                    <div className="inline-block text-sm font-mono bg-white p-3 rounded border border-neutral-200">
                      Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>)V
                    </div>
                  </div>
                </div>

                <div className="mt-4">
                  <h3 className="font-medium text-neutral-900 mb-3">Multi-Head Attention</h3>
                  <p className="text-neutral-700">
                    Instead of performing a single attention function, transformers use multiple attention heads in
                    parallel. Each head has its own set of learned Query, Key, and Value weight matrices, allowing it to
                    focus on different aspects of the input.
                  </p>

                  <div className="mt-3 grid md:grid-cols-3 gap-4">
                    <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                      <h4 className="font-medium text-yellow-800 mb-1 text-sm">Head 1</h4>
                      <p className="text-xs text-neutral-700">
                        Might focus on syntactic relationships like subject-verb agreement
                      </p>
                    </div>

                    <div className="bg-green-50 p-3 rounded-lg border border-green-200">
                      <h4 className="font-medium text-green-800 mb-1 text-sm">Head 2</h4>
                      <p className="text-xs text-neutral-700">
                        Might focus on semantic relationships between related concepts
                      </p>
                    </div>

                    <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                      <h4 className="font-medium text-blue-800 mb-1 text-sm">Head 3</h4>
                      <p className="text-xs text-neutral-700">
                        Might focus on coreference resolution between pronouns and their antecedents
                      </p>
                    </div>
                  </div>

                  <p className="mt-3 text-neutral-700">
                    The outputs from all heads are concatenated and linearly transformed to produce the final output of
                    the multi-head attention layer.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notebook" className="space-y-8">
          <div className="bg-white border border-neutral-300 rounded-lg p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-neutral-900 mb-2">Transformer Implementation</h2>
              <p className="text-neutral-700">
                This notebook demonstrates how to use transformer models with the Hugging Face Transformers library.
                Execute each cell to see the results.
              </p>
            </div>

            <div className="space-y-6">
              <NotebookCell
                cellId="cell0"
                executionCount={1}
                initialCode={
                  "import torch\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM\nimport seaborn as sns\n\n# Set random seed for reproducibility\ntorch.manual_seed(42)\nnp.random.seed(42)"
                }
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 1: Load a pre-trained transformer model</p>
                <p>Let's load a pre-trained BERT model and tokenizer, and see how they process text.</p>
              </div>

              <NotebookCell
                cellId="cell1"
                executionCount={2}
                initialCode={
                  "# Load pre-trained model and tokenizer\nmodel_name = 'bert-base-uncased'\ntokenizer = AutoTokenizer.from_pretrained(model_name)\nmodel = AutoModel.from_pretrained(model_name)\n\n# Tokenize input text\ntext = 'The transformer model revolutionized natural language processing.'\ninputs = tokenizer(text, return_tensors='pt')\n\n# Print tokenization results\ntokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])\ntoken_ids = inputs['input_ids'][0].tolist()\n\nprint('Model: ' + model_name)\nprint('Tokenized input: ' + str(tokens))\nprint('Token IDs: ' + str(token_ids))\n\n# Run the model\nwith torch.no_grad():\n    outputs = model(**inputs)\n    \n# Get the output embeddings\nembeddings = outputs.last_hidden_state\n\nprint('\\nInput shape:', inputs['input_ids'].shape)\nprint('Output shape:', embeddings.shape)\n\n# Print a sample of the output embeddings\nprint('\\nFirst token embedding (CLS) first 10 values:')\nprint(embeddings[:, 0, :10])"
                }
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 2: Visualize attention patterns</p>
                <p>
                  Let's examine the self-attention patterns in the transformer model to see how different tokens attend
                  to each other.
                </p>
              </div>

              <NotebookCell
                cellId="cell2"
                executionCount={3}
                initialCode={
                  "# Get attention weights from the model\nwith torch.no_grad():\n    outputs = model(**inputs, output_attentions=True)\n    \n# Extract attention weights\nattention = outputs.attentions  # Shape: [layers, batch, heads, seq_len, seq_len]\n\n# Select a specific layer and attention head\nlayer_idx = 0\nhead_idx = 0\nattention_map = attention[layer_idx][0, head_idx].numpy()\n\n# Visualize the attention map\nplt.figure(figsize=(10, 8))\nsns.heatmap(\n    attention_map,\n    xticklabels=tokens,\n    yticklabels=tokens,\n    cmap='viridis',\n    annot=True,\n    fmt='.2f',\n    cbar_kws={'label': 'Attention Weight'}\n)\nplt.title(f'Self-Attention Weights (Layer {layer_idx+1}, Head {head_idx+1})')\nplt.xlabel('Key Tokens')\nplt.ylabel('Query Tokens')\nplt.xticks(rotation=45, ha='right')\nplt.tight_layout()\nplt.show()\n\nprint('Attention visualization for head ' + str(head_idx) + ', layer ' + str(layer_idx) + ':')\n\n# Analyze attention patterns\nprint('\\nObservations:')\nprint('- The [CLS] token attends strongly to important content words')\nprint('- Adjacent words show stronger attention patterns')\nprint('- Punctuation receives less attention')\nprint('- Different heads capture different linguistic patterns')"
                }
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Step 3: Transformer applications</p>
                <p>
                  Let's demonstrate two common applications of transformers: masked language modeling (BERT) and text
                  generation (GPT).
                </p>
              </div>

              <NotebookCell
                cellId="cell3"
                executionCount={4}
                initialCode={
                  "# 1. Masked Language Modeling with BERT\n# Load a masked language model\nmlm_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')\n\n# Create a masked input\ntext = 'The transformer architecture has changed NLP.'\nmasked_text = 'The [MASK] architecture has changed NLP.'\n\n# Tokenize\ninputs = tokenizer(masked_text, return_tensors='pt')\n\n# Get the position of the [MASK] token\nmask_idx = torch.where(inputs['input_ids'][0] == tokenizer.mask_token_id)[0].item()\n\n# Get predictions\nwith torch.no_grad():\n    outputs = mlm_model(**inputs)\n    predictions = outputs.logits\n\n# Get top 5 predictions\ntopk_values, topk_indices = torch.topk(predictions[0, mask_idx], 5)\ntopk_tokens = tokenizer.convert_ids_to_tokens(topk_indices)\n\n# Calculate probabilities using softmax\nprobabilities = torch.nn.functional.softmax(predictions[0, mask_idx], dim=0)\ntopk_probs = probabilities[topk_indices]\n\n# Print results\nprint(\"Input: \\\"\" + text + \"\\\"\")\nprint(\"Masked input: \\\"\" + masked_text + \"\\\"\")\nprint()\nprint('Top 5 predictions for [MASK]:')\nfor token, prob in zip(topk_tokens, topk_probs):\n    print(f'{token}: {prob.item():.4f}')\n\n# 2. Text Generation with GPT-2\nprint('\\nGenerated text from GPT-2:')\n\n# Load GPT-2 model and tokenizer\ngpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')\ngpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')\n\n# Create input prompt\nprompt = \"Transformers have revolutionized natural language processing\"\ninputs = gpt2_tokenizer(prompt, return_tensors='pt')\n\n# Generate text\nwith torch.no_grad():\n    outputs = gpt2_model.generate(\n        inputs['input_ids'],\n        max_length=50,\n        num_return_sequences=1,\n        temperature=0.7,\n        top_p=0.9,\n        do_sample=True\n    )\n\n# Decode and print the generated text\ngenerated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)\nprint(f'\"{generated_text}\"')"
                }
                readOnly={false}
                onExecute={handleExecuteCode}
              />

              <div className="text-neutral-700 px-4 py-2 border-l-4 border-neutral-300 bg-neutral-50">
                <p className="font-medium text-neutral-900">Try it yourself!</p>
                <p>Modify the code above to experiment with different aspects of transformers:</p>
                <ul className="list-disc list-inside mt-2">
                  <li>
                    Try different pre-trained models like 'roberta-base', 'distilbert-base-uncased', or 'gpt2-medium'
                  </li>
                  <li>
                    Visualize attention patterns across different layers and heads to see what patterns they capture
                  </li>
                  <li>Experiment with different masked language modeling examples to test BERT's understanding</li>
                  <li>Try different generation parameters with GPT-2 (temperature, top_p, max_length)</li>
                  <li>Implement a simple text classification task using a transformer model</li>
                  <li>Fine-tune a pre-trained model on a specific task with a small dataset</li>
                </ul>
              </div>

              <div className="bg-neutral-100 p-4 rounded-lg mt-4">
                <h3 className="font-medium text-neutral-900 mb-2">Additional Resources</h3>
                <ul className="list-disc list-inside space-y-1 text-neutral-700">
                  <li>
                    <a href="#" className="text-blue-600 hover:underline">
                      Hugging Face Transformers Documentation
                    </a>{" "}
                    - Comprehensive guides and tutorials
                  </li>
                  <li>
                    <a href="#" className="text-blue-600 hover:underline">
                      The Illustrated Transformer
                    </a>{" "}
                    - Visual explanation of the transformer architecture
                  </li>
                  <li>
                    <a href="#" className="text-blue-600 hover:underline">
                      Attention Is All You Need
                    </a>{" "}
                    - The original transformer paper
                  </li>
                  <li>
                    <a href="#" className="text-blue-600 hover:underline">
                      BERT: Pre-training of Deep Bidirectional Transformers
                    </a>{" "}
                    - The BERT paper
                  </li>
                  <li>
                    <a href="#" className="text-blue-600 hover:underline">
                      Language Models are Few-Shot Learners
                    </a>{" "}
                    - The GPT-3 paper
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
