"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism"
import { Button } from "@/components/ui/button"
import { PlayIcon, PauseIcon, RefreshCwIcon } from "lucide-react"

interface NotebookCellProps {
  cellId: string
  executionCount: number
  initialCode: string
  language?: string
  readOnly?: boolean
  autoExecute?: boolean
  onExecute?: (code: string, cellId: string) => Promise<React.ReactNode>
}

const NotebookCell: React.FC<NotebookCellProps> = ({
  cellId,
  executionCount,
  initialCode,
  language = "python",
  readOnly = false,
  autoExecute = false,
  onExecute,
}) => {
  const [code, setCode] = useState(initialCode)
  const [output, setOutput] = useState<string>("")
  const [isExecuting, setIsExecuting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isEditing, setIsEditing] = useState(false)

  // Mock execution environment
  const executeCode = async (codeToExecute: string) => {
    setIsExecuting(true)
    setOutput("")
    setError(null)

    try {
      // Simulate network delay for realistic execution feel
      await new Promise((resolve) => setTimeout(resolve, 800))

      // Simple Python code execution simulation with more realistic outputs
      if (language === "python") {
        if (codeToExecute.includes("import numpy") || codeToExecute.includes("import np")) {
          if (codeToExecute.includes("np.array") || codeToExecute.includes("numpy.array")) {
            setOutput("array([1, 2, 3, 4, 5])")
          } else if (codeToExecute.includes("random")) {
            setOutput("array([0.23, 0.54, 0.12, 0.87, 0.42])")
          } else {
            setOutput("NumPy imported successfully")
          }
        } else if (codeToExecute.includes("import pandas") || codeToExecute.includes("import pd")) {
          setOutput("Pandas imported successfully")
          if (codeToExecute.includes("read_csv") || codeToExecute.includes("DataFrame")) {
            setOutput(`   feature1  feature2  target
0       1.2       2.3     0
1       2.4       3.1     1
2       0.5       1.0     0
3       3.2       4.1     1
4       2.8       3.7     1`)
          }
        } else if (codeToExecute.includes("import matplotlib") || codeToExecute.includes("plt.")) {
          setOutput("[Matplotlib figure displayed]")
        } else if (codeToExecute.includes("print(")) {
          const match = codeToExecute.match(/print$$(.*)$$/)
          if (match && match[1]) {
            setOutput(match[1].replace(/['"]/g, ""))
          } else {
            setOutput("")
          }
        } else if (codeToExecute.includes("for") && codeToExecute.includes("range")) {
          let output = ""
          const rangeMatch = codeToExecute.match(/range$$(\d+)$$/)
          if (rangeMatch && rangeMatch[1]) {
            const limit = Math.min(Number.parseInt(rangeMatch[1]), 10) // Limit to 10 iterations for display
            for (let i = 0; i < limit; i++) {
              output += `Iteration ${i}\n`
            }
            setOutput(output)
          }
        } else if (codeToExecute.includes("sklearn")) {
          if (codeToExecute.includes("fit(")) {
            setOutput(`Model fitted successfully.
Training accuracy: 0.92
Validation accuracy: 0.87`)
          } else if (codeToExecute.includes("predict(")) {
            setOutput("array([0, 1, 1, 0, 1])")
          } else {
            setOutput("Scikit-learn model initialized")
          }
        } else if (
          codeToExecute.includes("torch") ||
          codeToExecute.includes("tensorflow") ||
          codeToExecute.includes("tf.")
        ) {
          if (codeToExecute.includes("model.fit") || codeToExecute.includes("model.train")) {
            setOutput(`Epoch 1/10
32/32 [==============================] - 1s 2ms/step - loss: 0.6932 - accuracy: 0.5342
Epoch 2/10
32/32 [==============================] - 1s 2ms/step - loss: 0.5423 - accuracy: 0.7231
Epoch 3/10
32/32 [==============================] - 1s 2ms/step - loss: 0.3245 - accuracy: 0.8543`)
          } else {
            setOutput("Deep learning framework initialized successfully")
          }
        } else {
          // Generic execution result
          setOutput("Code executed successfully")
        }
      } else if (language === "javascript") {
        // JavaScript execution simulation
        if (codeToExecute.includes("console.log")) {
          const match = codeToExecute.match(/console\.log$$(.*)$$/)
          if (match && match[1]) {
            setOutput(match[1].replace(/['"]/g, ""))
          }
        } else if (codeToExecute.includes("Math.")) {
          setOutput("3.14159265359")
        } else {
          setOutput("JavaScript code executed")
        }
      }
    } catch (err) {
      setError(`Error: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsExecuting(false)
    }
  }

  useEffect(() => {
    if (autoExecute) {
      executeCode(code)
    }
  }, [])

  const handleCodeChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCode(e.target.value)
  }

  const handleExecute = () => {
    executeCode(code)
  }

  const handleReset = () => {
    setCode(initialCode)
    setOutput("")
    setError(null)
  }

  return (
    <div className="mb-8 border rounded-lg overflow-hidden bg-gray-50 dark:bg-gray-900">
      <div className="p-4 bg-gray-100 dark:bg-gray-800 border-b flex justify-between items-center">
        <div className="font-mono text-sm text-gray-500 dark:text-gray-400">
          {language === "python" ? "Python" : language}
        </div>
        <div className="flex space-x-2">
          {!readOnly && (
            <Button variant="outline" size="sm" onClick={() => setIsEditing(!isEditing)} className="text-xs">
              {isEditing ? "View" : "Edit"}
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={handleReset} className="text-xs">
            <RefreshCwIcon className="h-3 w-3 mr-1" />
            Reset
          </Button>
          <Button variant="default" size="sm" onClick={handleExecute} disabled={isExecuting} className="text-xs">
            {isExecuting ? (
              <>
                <PauseIcon className="h-3 w-3 mr-1 animate-pulse" />
                Running...
              </>
            ) : (
              <>
                <PlayIcon className="h-3 w-3 mr-1" />
                Run
              </>
            )}
          </Button>
        </div>
      </div>

      <div className="p-4">
        {isEditing || !readOnly ? (
          <textarea
            value={code}
            onChange={handleCodeChange}
            className="w-full h-40 font-mono text-sm p-2 border rounded bg-white dark:bg-gray-950 dark:text-gray-200"
            disabled={readOnly}
          />
        ) : (
          <SyntaxHighlighter
            language={language}
            style={vscDarkPlus}
            customStyle={{ margin: 0, borderRadius: "4px" }}
            wrapLines={true}
          >
            {code}
          </SyntaxHighlighter>
        )}
      </div>

      {(output || error) && (
        <div className="p-4 border-t bg-black text-white font-mono text-sm overflow-x-auto">
          <div className="text-xs text-gray-400 mb-2">Output:</div>
          {error ? (
            <pre className="text-red-400 whitespace-pre-wrap">{error}</pre>
          ) : (
            <pre className="whitespace-pre-wrap">{output}</pre>
          )}
        </div>
      )}
    </div>
  )
}

export default NotebookCell
