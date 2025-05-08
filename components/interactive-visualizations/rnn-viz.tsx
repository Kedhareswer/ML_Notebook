"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Shuffle, Download, Play, Pause } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface RNNVisualizationProps {
  width?: number
  height?: number
}

export default function RNNVisualization({ width = 600, height = 400 }: RNNVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [activeTab, setActiveTab] = useState("basic")
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationStep, setAnimationStep] = useState(0)

  // Parameters for basic RNN
  const [sequenceLength, setSequenceLength] = useState(5)
  const [hiddenSize, setHiddenSize] = useState(3)
  const [timeStep, setTimeStep] = useState(0)

  // Parameters for LSTM/GRU
  const [cellType, setCellType] = useState("lstm")
  const [gateVisibility, setGateVisibility] = useState(0.7)
  const [memoryStrength, setMemoryStrength] = useState(0.8)

  // Parameters for sequence tasks
  const [taskType, setTaskType] = useState("classification")
  const [inputFeatures, setInputFeatures] = useState(3)
  const [outputClasses, setOutputClasses] = useState(2)

  // Generate random data
  const generateRandomData = () => {
    if (activeTab === "basic") {
      setSequenceLength(Math.floor(Math.random() * 5) + 3)
      setHiddenSize(Math.floor(Math.random() * 4) + 2)
      setTimeStep(Math.floor(Math.random() * sequenceLength))
    } else if (activeTab === "lstm_gru") {
      setCellType(Math.random() > 0.5 ? "lstm" : "gru")
      setGateVisibility(Math.random() * 0.6 + 0.4)
      setMemoryStrength(Math.random() * 0.6 + 0.4)
    } else if (activeTab === "sequence_tasks") {
      const tasks = ["classification", "generation", "translation"]
      setTaskType(tasks[Math.floor(Math.random() * tasks.length)])
      setInputFeatures(Math.floor(Math.random() * 4) + 2)
      setOutputClasses(Math.floor(Math.random() * 3) + 2)
    }
  }

  // Toggle animation
  const toggleAnimation = () => {
    setIsAnimating(!isAnimating)
  }

  // Animation loop
  useEffect(() => {
    if (isAnimating) {
      const animate = () => {
        setAnimationStep((prev) => (prev + 1) % 60) // 60 frames for a complete cycle
        setTimeStep((prev) => (prev + 1) % sequenceLength)
        animationRef.current = requestAnimationFrame(animate)
      }
      animationRef.current = requestAnimationFrame(animate)
    } else if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isAnimating, sequenceLength])

  // Render basic RNN
  const renderBasicRNN = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Define dimensions
    const cellSize = 40
    const cellSpacing = 80
    const verticalSpacing = 70
    const startX = (width - (sequenceLength - 1) * cellSpacing) / 2
    const startY = 80

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "18px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    ctx.fillText("Recurrent Neural Network (Unfolded)", width / 2, 20)

    // Draw time steps
    for (let t = 0; t < sequenceLength; t++) {
      const x = startX + t * cellSpacing

      // Draw input node
      ctx.beginPath()
      ctx.arc(x, startY, cellSize / 2, 0, Math.PI * 2)
      ctx.fillStyle = "#f0f0f0"
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(`x${t}`, x, startY)

      // Draw hidden state node
      ctx.beginPath()
      ctx.arc(x, startY + verticalSpacing, cellSize / 2, 0, Math.PI * 2)

      // Highlight current time step
      if (t === timeStep) {
        ctx.fillStyle = "#d0d0d0"
      } else {
        ctx.fillStyle = "#f0f0f0"
      }

      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = t === timeStep ? 2 : 1
      ctx.stroke()

      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(`h${t}`, x, startY + verticalSpacing)

      // Draw output node
      ctx.beginPath()
      ctx.arc(x, startY + 2 * verticalSpacing, cellSize / 2, 0, Math.PI * 2)
      ctx.fillStyle = "#f0f0f0"
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(`y${t}`, x, startY + 2 * verticalSpacing)

      // Draw vertical connections
      // Input to hidden
      ctx.beginPath()
      ctx.moveTo(x, startY + cellSize / 2)
      ctx.lineTo(x, startY + verticalSpacing - cellSize / 2)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(x, startY + verticalSpacing - cellSize / 2)
      ctx.lineTo(x - 3, startY + verticalSpacing - cellSize / 2 - 6)
      ctx.lineTo(x + 3, startY + verticalSpacing - cellSize / 2 - 6)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()

      // Hidden to output
      ctx.beginPath()
      ctx.moveTo(x, startY + verticalSpacing + cellSize / 2)
      ctx.lineTo(x, startY + 2 * verticalSpacing - cellSize / 2)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(x, startY + 2 * verticalSpacing - cellSize / 2)
      ctx.lineTo(x - 3, startY + 2 * verticalSpacing - cellSize / 2 - 6)
      ctx.lineTo(x + 3, startY + 2 * verticalSpacing - cellSize / 2 - 6)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()

      // Draw recurrent connections (except for the last time step)
      if (t < sequenceLength - 1) {
        ctx.beginPath()
        ctx.moveTo(x + cellSize / 2, startY + verticalSpacing)
        ctx.lineTo(x + cellSpacing - cellSize / 2, startY + verticalSpacing)

        // Use dashed line for future connections
        if (t >= timeStep) {
          ctx.setLineDash([3, 3])
        } else {
          ctx.setLineDash([])
        }

        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1
        ctx.stroke()
        ctx.setLineDash([])

        // Draw arrowhead
        ctx.beginPath()
        ctx.moveTo(x + cellSpacing - cellSize / 2, startY + verticalSpacing)
        ctx.lineTo(x + cellSpacing - cellSize / 2 - 6, startY + verticalSpacing - 3)
        ctx.lineTo(x + cellSpacing - cellSize / 2 - 6, startY + verticalSpacing + 3)
        ctx.closePath()
        ctx.fillStyle = "#000"
        ctx.fill()
      }
    }

    // Draw hidden state details
    const detailsStartY = startY + 2 * verticalSpacing + 60

    ctx.fillStyle = "#000"
    ctx.font = "16px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    ctx.fillText(`Hidden State at t=${timeStep}`, width / 2, detailsStartY)

    // Draw hidden state neurons
    const neuronSpacing = 50
    const neuronRadius = 15
    const neuronStartX = (width - (hiddenSize - 1) * neuronSpacing) / 2
    const neuronY = detailsStartY + 50

    for (let i = 0; i < hiddenSize; i++) {
      const x = neuronStartX + i * neuronSpacing

      // Draw neuron
      ctx.beginPath()
      ctx.arc(x, neuronY, neuronRadius, 0, Math.PI * 2)
      ctx.fillStyle = "#f0f0f0"
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw activation value (random for visualization)
      const value = Math.random().toFixed(1)
      ctx.fillStyle = "#000"
      ctx.font = "10px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(value, x, neuronY)
    }

    // Draw labels
    ctx.fillStyle = "#000"
    ctx.font = "12px sans-serif"
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    ctx.fillText("Inputs:", startX - 20, startY)
    ctx.fillText("Hidden States:", startX - 20, startY + verticalSpacing)
    ctx.fillText("Outputs:", startX - 20, startY + 2 * verticalSpacing)

    // Draw time step labels
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    for (let t = 0; t < sequenceLength; t++) {
      const x = startX + t * cellSpacing
      ctx.fillText(`t=${t}`, x, startY + 2 * verticalSpacing + 30)
    }
  }

  // Render LSTM/GRU
  const renderLSTMGRU = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Define dimensions
    const cellWidth = 180
    const cellHeight = 120
    const startX = (width - cellWidth) / 2
    const startY = 60

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "18px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    ctx.fillText(
      cellType === "lstm" ? "Long Short-Term Memory (LSTM) Cell" : "Gated Recurrent Unit (GRU) Cell",
      width / 2,
      20,
    )

    // Draw cell outline
    ctx.fillStyle = "#f0f0f0"
    ctx.fillRect(startX, startY, cellWidth, cellHeight)
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.strokeRect(startX, startY, cellWidth, cellHeight)

    if (cellType === "lstm") {
      // Draw LSTM cell internals

      // Cell state line (horizontal line through the middle) with better positioning
      ctx.beginPath()
      ctx.moveTo(startX - 30, startY + cellHeight / 2)
      ctx.lineTo(startX + cellWidth + 30, startY + cellHeight / 2)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.setLineDash([]) // Reset line dash to ensure solid lines for other elements

      // Draw cell state label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "left"
      ctx.textBaseline = "bottom"
      ctx.fillText("Cell State (C_t)", startX + cellWidth + 10, startY + cellHeight / 2 - 5)

      // Draw gates
      const gateRadius = 15
      const gateSpacing = cellWidth / 4

      // Forget gate
      const forgetX = startX + gateSpacing
      const forgetY = startY + cellHeight / 2

      ctx.beginPath()
      ctx.arc(forgetX, forgetY, gateRadius, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(150, 150, 150, ${gateVisibility})`
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("f", forgetX, forgetY)

      // Input gate
      const inputX = startX + 2 * gateSpacing
      const inputY = startY + cellHeight / 2

      ctx.beginPath()
      ctx.arc(inputX, inputY, gateRadius, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(100, 100, 100, ${gateVisibility})`
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("i", inputX, inputY)

      // Output gate
      const outputX = startX + 3 * gateSpacing
      const outputY = startY + cellHeight / 2

      ctx.beginPath()
      ctx.arc(outputX, outputY, gateRadius, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(50, 50, 50, ${gateVisibility})`
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("o", outputX, outputY)

      // Draw gate labels
      ctx.fillStyle = "#000"
      ctx.font = "10px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText("Forget Gate", forgetX, forgetY + gateRadius + 5)
      ctx.fillText("Input Gate", inputX, inputY + gateRadius + 5)
      ctx.fillText("Output Gate", outputX, outputY + gateRadius + 5)

      // Draw input and output arrows
      // Input arrow
      ctx.beginPath()
      ctx.moveTo(startX - 30, startY + cellHeight / 4)
      ctx.lineTo(startX, startY + cellHeight / 4)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(startX, startY + cellHeight / 4)
      ctx.lineTo(startX - 6, startY + cellHeight / 4 - 3)
      ctx.lineTo(startX - 6, startY + cellHeight / 4 + 3)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()

      // Input label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "right"
      ctx.textBaseline = "middle"
      ctx.fillText("x_t", startX - 40, startY + cellHeight / 4)

      // Previous hidden state arrow
      ctx.beginPath()
      ctx.moveTo(startX - 30, startY + (3 * cellHeight) / 4)
      ctx.lineTo(startX, startY + (3 * cellHeight) / 4)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(startX, startY + (3 * cellHeight) / 4)
      ctx.lineTo(startX - 6, startY + (3 * cellHeight) / 4 - 3)
      ctx.lineTo(startX - 6, startY + (3 * cellHeight) / 4 + 3)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()

      // Previous hidden state label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "right"
      ctx.textBaseline = "middle"
      ctx.fillText("h_{t-1}", startX - 40, startY + (3 * cellHeight) / 4)

      // Output arrow (hidden state)
      ctx.beginPath()
      ctx.moveTo(startX + cellWidth, startY + (3 * cellHeight) / 4)
      ctx.lineTo(startX + cellWidth + 30, startY + (3 * cellHeight) / 4)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(startX + cellWidth + 30, startY + (3 * cellHeight) / 4)
      ctx.lineTo(startX + cellWidth + 24, startY + (3 * cellHeight) / 4 - 3)
      ctx.lineTo(startX + cellWidth + 24, startY + (3 * cellHeight) / 4 + 3)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()

      // Output label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "left"
      ctx.textBaseline = "middle"
      ctx.fillText("h_t", startX + cellWidth + 40, startY + (3 * cellHeight) / 4)

      // Draw memory strength indicator
      const memoryBarWidth = 100
      const memoryBarHeight = 20
      const memoryBarX = (width - memoryBarWidth) / 2
      const memoryBarY = startY + cellHeight + 40

      ctx.fillStyle = "#000"
      ctx.font = "14px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "bottom"
      ctx.fillText("Memory Strength", width / 2, memoryBarY - 5)

      // Draw memory bar background
      ctx.fillStyle = "#e0e0e0"
      ctx.fillRect(memoryBarX, memoryBarY, memoryBarWidth, memoryBarHeight)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.strokeRect(memoryBarX, memoryBarY, memoryBarWidth, memoryBarHeight)

      // Draw memory bar fill
      ctx.fillStyle = "#000"
      ctx.fillRect(memoryBarX, memoryBarY, memoryBarWidth * memoryStrength, memoryBarHeight)

      // Draw LSTM equations
      const equationsY = memoryBarY + memoryBarHeight + 30

      ctx.fillStyle = "#000"
      ctx.font = "14px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText("LSTM Equations:", width / 2, equationsY)

      ctx.font = "12px sans-serif"
      ctx.textAlign = "left"
      ctx.textBaseline = "top"
      const equations = [
        "f_t = σ(W_f·[h_{t-1}, x_t] + b_f)",
        "i_t = σ(W_i·[h_{t-1}, x_t] + b_i)",
        "C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)",
        "C_t = f_t * C_{t-1} + i_t * C̃_t",
        "o_t = σ(W_o·[h_{t-1}, x_t] + b_o)",
        "h_t = o_t * tanh(C_t)",
      ]

      for (let i = 0; i < equations.length; i++) {
        ctx.fillText(equations[i], (width - 200) / 2, equationsY + 25 + i * 20)
      }
    } else {
      // Draw GRU cell internals

      // Draw gates
      const gateRadius = 15
      const gateSpacing = cellWidth / 3

      // Update gate
      const updateX = startX + gateSpacing
      const updateY = startY + cellHeight / 3

      ctx.beginPath()
      ctx.arc(updateX, updateY, gateRadius, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(255, 0, 0, ${gateVisibility})`
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("z", updateX, updateY)

      // Reset gate
      const resetX = startX + 2 * gateSpacing
      const resetY = startY + cellHeight / 3

      ctx.beginPath()
      ctx.arc(resetX, resetY, gateRadius, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(0, 0, 255, ${gateVisibility})`
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("r", resetX, resetY)

      // Draw gate labels
      ctx.fillStyle = "#000"
      ctx.font = "10px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText("Update Gate", updateX, updateY + gateRadius + 5)
      ctx.fillText("Reset Gate", resetX, resetY + gateRadius + 5)

      // Draw hidden state line
      ctx.beginPath()
      ctx.moveTo(startX - 30, startY + (2 * cellHeight) / 3)
      ctx.lineTo(startX + cellWidth + 30, startY + (2 * cellHeight) / 3)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 2
      ctx.stroke()

      // Draw hidden state label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "bottom"
      ctx.fillText("Hidden State (h_t)", width / 2, startY + (2 * cellHeight) / 3 - 5)

      // Draw input and output arrows
      // Input arrow
      ctx.beginPath()
      ctx.moveTo(startX - 30, startY + cellHeight / 4)
      ctx.lineTo(startX, startY + cellHeight / 4)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(startX, startY + cellHeight / 4)
      ctx.lineTo(startX - 6, startY + cellHeight / 4 - 3)
      ctx.lineTo(startX - 6, startY + cellHeight / 4 + 3)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()

      // Input label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "right"
      ctx.textBaseline = "middle"
      ctx.fillText("x_t", startX - 40, startY + cellHeight / 4)

      // Previous hidden state arrow
      ctx.beginPath()
      ctx.moveTo(startX - 30, startY + (2 * cellHeight) / 3)
      ctx.lineTo(startX, startY + (2 * cellHeight) / 3)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Previous hidden state label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "right"
      ctx.textBaseline = "middle"
      ctx.fillText("h_{t-1}", startX - 40, startY + (2 * cellHeight) / 3)

      // Output arrow (hidden state)
      ctx.beginPath()
      ctx.moveTo(startX + cellWidth, startY + (2 * cellHeight) / 3)
      ctx.lineTo(startX + cellWidth + 30, startY + (2 * cellHeight) / 3)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(startX + cellWidth + 30, startY + (2 * cellHeight) / 3)
      ctx.lineTo(startX + cellWidth + 24, startY + (2 * cellHeight) / 3 - 3)
      ctx.lineTo(startX + cellWidth + 24, startY + (2 * cellHeight) / 3 + 3)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()

      // Output label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "left"
      ctx.textBaseline = "middle"
      ctx.fillText("h_t", startX + cellWidth + 40, startY + (2 * cellHeight) / 3)

      // Draw memory strength indicator
      const memoryBarWidth = 100
      const memoryBarHeight = 20
      const memoryBarX = (width - memoryBarWidth) / 2
      const memoryBarY = startY + cellHeight + 40

      ctx.fillStyle = "#000"
      ctx.font = "14px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "bottom"
      ctx.fillText("Memory Strength", width / 2, memoryBarY - 5)

      // Draw memory bar background
      ctx.fillStyle = "#e0e0e0"
      ctx.fillRect(memoryBarX, memoryBarY, memoryBarWidth, memoryBarHeight)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.strokeRect(memoryBarX, memoryBarY, memoryBarWidth, memoryBarHeight)

      // Draw memory bar fill
      ctx.fillStyle = "#000"
      ctx.fillRect(memoryBarX, memoryBarY, memoryBarWidth * memoryStrength, memoryBarHeight)

      // Draw GRU equations
      const equationsY = memoryBarY + memoryBarHeight + 30

      ctx.fillStyle = "#000"
      ctx.font = "14px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText("GRU Equations:", width / 2, equationsY)

      ctx.font = "12px sans-serif"
      ctx.textAlign = "left"
      ctx.textBaseline = "top"
      const equations = [
        "z_t = σ(W_z·[h_{t-1}, x_t] + b_z)",
        "r_t = σ(W_r·[h_{t-1}, x_t] + b_r)",
        "h̃_t = tanh(W·[r_t * h_{t-1}, x_t] + b)",
        "h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t",
      ]

      for (let i = 0; i < equations.length; i++) {
        ctx.fillText(equations[i], (width - 200) / 2, equationsY + 25 + i * 20)
      }
    }
  }

  // Render sequence tasks
  const renderSequenceTasks = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Define dimensions
    const sequenceLength = 5
    const cellSize = 40
    const cellSpacing = 70
    const verticalSpacing = 60
    const startX = (width - (sequenceLength - 1) * cellSpacing) / 2
    const startY = 60

    // Draw title
    ctx.fillStyle = "#000"
    ctx.font = "18px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "top"

    let title
    if (taskType === "classification") {
      title = "Sequence Classification"
    } else if (taskType === "generation") {
      title = "Sequence Generation"
    } else {
      title = "Sequence-to-Sequence (Translation)"
    }
    ctx.fillText(title, width / 2, 20)

    // Draw input sequence
    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText("Input Sequence:", startX - 120, startY)

    for (let t = 0; t < sequenceLength; t++) {
      const x = startX + t * cellSpacing

      // Draw input node
      ctx.beginPath()
      ctx.arc(x, startY, cellSize / 2, 0, Math.PI * 2)
      ctx.fillStyle = "#f0f0f0"
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw feature vector representation
      const vectorHeight = inputFeatures * 5
      const vectorWidth = 10
      const vectorX = x - vectorWidth / 2
      const vectorY = startY - vectorHeight / 2

      ctx.fillStyle = "#ddd"
      ctx.fillRect(vectorX, vectorY, vectorWidth, vectorHeight)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.strokeRect(vectorX, vectorY, vectorWidth, vectorHeight)

      // Draw feature lines
      for (let i = 1; i < inputFeatures; i++) {
        const lineY = vectorY + i * (vectorHeight / inputFeatures)
        ctx.beginPath()
        ctx.moveTo(vectorX, lineY)
        ctx.lineTo(vectorX + vectorWidth, lineY)
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 0.5
        ctx.stroke()
      }
    }

    // Draw RNN processing
    const rnnY = startY + verticalSpacing

    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText("RNN Processing:", startX - 120, rnnY)

    // Draw RNN block
    const rnnWidth = (sequenceLength - 1) * cellSpacing + cellSize
    const rnnHeight = 40
    const rnnX = startX - cellSize / 2

    ctx.fillStyle = "#f0f0f0"
    ctx.fillRect(rnnX, rnnY - rnnHeight / 2, rnnWidth, rnnHeight)
    ctx.strokeStyle = "#000"
    ctx.lineWidth = 2
    ctx.strokeRect(rnnX, rnnY - rnnHeight / 2, rnnWidth, rnnHeight)

    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "center"
    ctx.textBaseline = "middle"
    ctx.fillText("Recurrent Neural Network", rnnX + rnnWidth / 2, rnnY)

    // Draw arrows from input to RNN
    for (let t = 0; t < sequenceLength; t++) {
      const x = startX + t * cellSpacing

      ctx.beginPath()
      ctx.moveTo(x, startY + cellSize / 2)
      ctx.lineTo(x, rnnY - rnnHeight / 2)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(x, rnnY - rnnHeight / 2)
      ctx.lineTo(x - 3, rnnY - rnnHeight / 2 - 6)
      ctx.lineTo(x + 3, rnnY - rnnHeight / 2 - 6)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()
    }

    // Draw output based on task type
    const outputY = rnnY + verticalSpacing

    ctx.fillStyle = "#000"
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText("Output:", startX - 120, outputY)

    if (taskType === "classification") {
      // Draw single output node for classification
      const outputX = startX + ((sequenceLength - 1) * cellSpacing) / 2

      // Draw arrow from RNN to output
      ctx.beginPath()
      ctx.moveTo(outputX, rnnY + rnnHeight / 2)
      ctx.lineTo(outputX, outputY - cellSize / 2)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(outputX, outputY - cellSize / 2)
      ctx.lineTo(outputX - 3, outputY - cellSize / 2 - 6)
      ctx.lineTo(outputX + 3, outputY - cellSize / 2 - 6)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()

      // Draw output node
      ctx.beginPath()
      ctx.arc(outputX, outputY, cellSize / 2, 0, Math.PI * 2)
      ctx.fillStyle = "#f0f0f0"
      ctx.fill()
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw class probabilities
      const barWidth = 10
      const barSpacing = 5
      const totalBarWidth = outputClasses * barWidth + (outputClasses - 1) * barSpacing
      const barsStartX = outputX - totalBarWidth / 2
      const maxBarHeight = 30

      for (let i = 0; i < outputClasses; i++) {
        const barHeight = Math.random() * maxBarHeight + 10
        const barX = barsStartX + i * (barWidth + barSpacing)
        const barY = outputY - barHeight / 2

        ctx.fillStyle = "#000"
        ctx.fillRect(barX, barY, barWidth, barHeight)

        // Add class label
        ctx.fillStyle = "#000"
        ctx.font = "10px sans-serif"
        ctx.textAlign = "center"
        ctx.textBaseline = "top"
        ctx.fillText(`C${i}`, barX + barWidth / 2, outputY + cellSize / 2 + 5)
      }

      // Add classification label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText("Class Probabilities", outputX, outputY + cellSize / 2 + 20)
    } else if (taskType === "generation") {
      // Draw output sequence for generation
      for (let t = 0; t < sequenceLength; t++) {
        const x = startX + t * cellSpacing

        // Draw arrow from RNN to output
        ctx.beginPath()
        ctx.moveTo(x, rnnY + rnnHeight / 2)
        ctx.lineTo(x, outputY - cellSize / 2)
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1
        ctx.stroke()

        // Draw arrowhead
        ctx.beginPath()
        ctx.moveTo(x, outputY - cellSize / 2)
        ctx.lineTo(x - 3, outputY - cellSize / 2 - 6)
        ctx.lineTo(x + 3, outputY - cellSize / 2 - 6)
        ctx.closePath()
        ctx.fillStyle = "#000"
        ctx.fill()

        // Draw output node
        ctx.beginPath()
        ctx.arc(x, outputY, cellSize / 2, 0, Math.PI * 2)
        ctx.fillStyle = "#f0f0f0"
        ctx.fill()
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1
        ctx.stroke()

        // Add text inside node
        ctx.fillStyle = "#000"
        ctx.font = "12px sans-serif"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(`y${t}`, x, outputY)
      }

      // Add feedback arrows for autoregressive generation
      for (let t = 0; t < sequenceLength - 1; t++) {
        const x1 = startX + t * cellSpacing
        const x2 = startX + (t + 1) * cellSpacing
        const arrowY = outputY + cellSize / 2 + 20

        ctx.beginPath()
        ctx.moveTo(x1, outputY + cellSize / 2)
        ctx.lineTo(x1, arrowY)
        ctx.lineTo(x2, arrowY)
        ctx.lineTo(x2, outputY + cellSize / 2)
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1
        ctx.stroke()

        // Draw arrowhead
        ctx.beginPath()
        ctx.moveTo(x2, outputY + cellSize / 2)
        ctx.lineTo(x2 - 3, outputY + cellSize / 2 + 6)
        ctx.lineTo(x2 + 3, outputY + cellSize / 2 + 6)
        ctx.closePath()
        ctx.fillStyle = "#000"
        ctx.fill()
      }

      // Add generation label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText("Autoregressive Generation", width / 2, outputY + cellSize / 2 + 30)
    } else {
      // Draw encoder-decoder for translation
      const encoderWidth = ((sequenceLength - 1) * cellSpacing) / 2
      const decoderWidth = encoderWidth
      const encoderX = startX
      const decoderX = startX + (sequenceLength - 1) * cellSpacing - encoderWidth

      // Draw encoder label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText("Encoder", encoderX + encoderWidth / 2, rnnY + rnnHeight / 2 + 5)

      // Draw decoder label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText("Decoder", decoderX + decoderWidth / 2, rnnY + rnnHeight / 2 + 5)

      // Draw context vector
      const contextX = startX + ((sequenceLength - 1) * cellSpacing) / 2
      const contextY = rnnY + verticalSpacing / 2
      const contextWidth = 40
      const contextHeight = 20

      ctx.fillStyle = "#ddd"
      ctx.fillRect(contextX - contextWidth / 2, contextY - contextHeight / 2, contextWidth, contextHeight)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.strokeRect(contextX - contextWidth / 2, contextY - contextHeight / 2, contextWidth, contextHeight)

      // Add context vector label
      ctx.fillStyle = "#000"
      ctx.font = "10px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("Context", contextX, contextY)

      // Draw arrow from encoder to context
      ctx.beginPath()
      ctx.moveTo(encoderX + encoderWidth / 2, rnnY + rnnHeight / 2)
      ctx.lineTo(contextX, contextY - contextHeight / 2)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(contextX, contextY - contextHeight / 2)
      ctx.lineTo(contextX - 3, contextY - contextHeight / 2 - 6)
      ctx.lineTo(contextX + 3, contextY - contextHeight / 2 - 6)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()

      // Draw arrow from context to decoder
      ctx.beginPath()
      ctx.moveTo(contextX, contextY + contextHeight / 2)
      ctx.lineTo(decoderX + decoderWidth / 2, rnnY + rnnHeight / 2)
      ctx.strokeStyle = "#000"
      ctx.lineWidth = 1
      ctx.stroke()

      // Draw arrowhead
      ctx.beginPath()
      ctx.moveTo(decoderX + decoderWidth / 2, rnnY + rnnHeight / 2)
      ctx.lineTo(decoderX + decoderWidth / 2 - 6, rnnY + rnnHeight / 2 - 3)
      ctx.lineTo(decoderX + decoderWidth / 2 - 6, rnnY + rnnHeight / 2 + 3)
      ctx.closePath()
      ctx.fillStyle = "#000"
      ctx.fill()

      // Draw output sequence
      for (let t = 0; t < sequenceLength; t++) {
        const x = startX + t * cellSpacing

        // Draw output node
        ctx.beginPath()
        ctx.arc(x, outputY, cellSize / 2, 0, Math.PI * 2)
        ctx.fillStyle = "#f0f0f0"
        ctx.fill()
        ctx.strokeStyle = "#000"
        ctx.lineWidth = 1
        ctx.stroke()

        // Add text inside node
        ctx.fillStyle = "#000"
        ctx.font = "12px sans-serif"
        ctx.textAlign = "center"
        ctx.textBaseline = "middle"
        ctx.fillText(`y${t}`, x, outputY)
      }

      // Draw arrows from decoder to outputs
      for (let t = 0; t < sequenceLength; t++) {
        const x = startX + t * cellSpacing

        // Only draw arrows for outputs on the decoder side
        if (t >= Math.floor(sequenceLength / 2)) {
          ctx.beginPath()
          ctx.moveTo(decoderX + decoderWidth / 2, rnnY + rnnHeight / 2)
          ctx.lineTo(x, outputY - cellSize / 2)
          ctx.strokeStyle = "#000"
          ctx.lineWidth = 1
          ctx.stroke()

          // Draw arrowhead
          ctx.beginPath()
          ctx.moveTo(x, outputY - cellSize / 2)
          ctx.lineTo(x - 3, outputY - cellSize / 2 - 6)
          ctx.lineTo(x + 3, outputY - cellSize / 2 - 6)
          ctx.closePath()
          ctx.fillStyle = "#000"
          ctx.fill()
        }
      }

      // Add translation label
      ctx.fillStyle = "#000"
      ctx.font = "12px sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "top"
      ctx.fillText("Sequence-to-Sequence Translation", width / 2, outputY + cellSize / 2 + 20)
    }
  }

  // Update visualization when parameters change
  useEffect(() => {
    if (activeTab === "basic") {
      renderBasicRNN()
    } else if (activeTab === "lstm_gru") {
      renderLSTMGRU()
    } else if (activeTab === "sequence_tasks") {
      renderSequenceTasks()
    }
  }, [
    activeTab,
    sequenceLength,
    hiddenSize,
    timeStep,
    cellType,
    gateVisibility,
    memoryStrength,
    taskType,
    inputFeatures,
    outputClasses,
    animationStep,
  ])

  return (
    <Card className="w-full border-neutral-300 bg-white">
      <CardContent className="pt-6">
        <div className="flex justify-between items-center mb-4">
          <div className="text-lg font-medium text-neutral-900">RNN Visualization</div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={generateRandomData} className="flex items-center gap-1">
              <Shuffle className="h-4 w-4" /> Random Data
            </Button>
            {activeTab === "basic" && (
              <Button variant="outline" size="sm" onClick={toggleAnimation} className="flex items-center gap-1">
                {isAnimating ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {isAnimating ? "Pause" : "Animate"}
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-1"
              onClick={() => {
                const canvas = canvasRef.current
                if (canvas) {
                  const link = document.createElement("a")
                  link.download = `rnn-${activeTab}.png`
                  link.href = canvas.toDataURL("image/png")
                  link.click()
                }
              }}
            >
              <Download className="h-4 w-4" /> Save Image
            </Button>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-4">
          <TabsList className="grid w-full grid-cols-3 bg-neutral-100 text-neutral-900">
            <TabsTrigger value="basic" className="data-[state=active]:bg-white">
              Basic RNN
            </TabsTrigger>
            <TabsTrigger value="lstm_gru" className="data-[state=active]:bg-white">
              LSTM/GRU
            </TabsTrigger>
            <TabsTrigger value="sequence_tasks" className="data-[state=active]:bg-white">
              Sequence Tasks
            </TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1 order-2 lg:order-1">
            <canvas
              ref={canvasRef}
              width={width}
              height={height}
              className="w-full h-auto bg-white border border-neutral-300 rounded-md"
            />
          </div>

          <div className="w-full lg:w-64 space-y-6 order-1 lg:order-2">
            {activeTab === "basic" && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="sequenceLength" className="text-neutral-900">
                      Sequence Length
                    </Label>
                    <span className="text-sm text-neutral-600">{sequenceLength}</span>
                  </div>
                  <Slider
                    id="sequenceLength"
                    min={2}
                    max={7}
                    step={1}
                    value={[sequenceLength]}
                    onValueChange={(value) => setSequenceLength(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="hiddenSize" className="text-neutral-900">
                      Hidden Size
                    </Label>
                    <span className="text-sm text-neutral-600">{hiddenSize}</span>
                  </div>
                  <Slider
                    id="hiddenSize"
                    min={1}
                    max={6}
                    step={1}
                    value={[hiddenSize]}
                    onValueChange={(value) => setHiddenSize(value[0])}
                    className="notebook-slider"
                  />
                </div>

                {!isAnimating && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="timeStep" className="text-neutral-900">
                        Time Step
                      </Label>
                      <span className="text-sm text-neutral-600">{timeStep}</span>
                    </div>
                    <Slider
                      id="timeStep"
                      min={0}
                      max={sequenceLength - 1}
                      step={1}
                      value={[timeStep]}
                      onValueChange={(value) => setTimeStep(value[0])}
                      className="notebook-slider"
                    />
                  </div>
                )}
              </>
            )}

            {activeTab === "lstm_gru" && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="cellType" className="text-neutral-900">
                    Cell Type
                  </Label>
                  <Select value={cellType} onValueChange={setCellType}>
                    <SelectTrigger id="cellType">
                      <SelectValue placeholder="Select cell type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="lstm">LSTM</SelectItem>
                      <SelectItem value="gru">GRU</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="gateVisibility" className="text-neutral-900">
                      Gate Visibility
                    </Label>
                    <span className="text-sm text-neutral-600">{gateVisibility.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="gateVisibility"
                    min={0.2}
                    max={1}
                    step={0.05}
                    value={[gateVisibility]}
                    onValueChange={(value) => setGateVisibility(value[0])}
                    className="notebook-slider"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="memoryStrength" className="text-neutral-900">
                      Memory Strength
                    </Label>
                    <span className="text-sm text-neutral-600">{memoryStrength.toFixed(2)}</span>
                  </div>
                  <Slider
                    id="memoryStrength"
                    min={0.1}
                    max={1}
                    step={0.05}
                    value={[memoryStrength]}
                    onValueChange={(value) => setMemoryStrength(value[0])}
                    className="notebook-slider"
                  />
                </div>
              </>
            )}

            {activeTab === "sequence_tasks" && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="taskType" className="text-neutral-900">
                    Task Type
                  </Label>
                  <Select value={taskType} onValueChange={setTaskType}>
                    <SelectTrigger id="taskType">
                      <SelectValue placeholder="Select task type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="classification">Classification</SelectItem>
                      <SelectItem value="generation">Generation</SelectItem>
                      <SelectItem value="translation">Translation</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label htmlFor="inputFeatures" className="text-neutral-900">
                      Input Features
                    </Label>
                    <span className="text-sm text-neutral-600">{inputFeatures}</span>
                  </div>
                  <Slider
                    id="inputFeatures"
                    min={1}
                    max={5}
                    step={1}
                    value={[inputFeatures]}
                    onValueChange={(value) => setInputFeatures(value[0])}
                    className="notebook-slider"
                  />
                </div>

                {taskType === "classification" && (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label htmlFor="outputClasses" className="text-neutral-900">
                        Output Classes
                      </Label>
                      <span className="text-sm text-neutral-600">{outputClasses}</span>
                    </div>
                    <Slider
                      id="outputClasses"
                      min={2}
                      max={5}
                      step={1}
                      value={[outputClasses]}
                      onValueChange={(value) => setOutputClasses(value[0])}
                      className="notebook-slider"
                    />
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
