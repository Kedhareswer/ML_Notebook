"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Play, Copy, Check } from "lucide-react"

interface CodeEditorProps {
  initialCode: string
  language?: string
  onRun?: (code: string) => void
  readOnly?: boolean
}

export default function CodeEditor({ initialCode, language = "python", onRun, readOnly = false }: CodeEditorProps) {
  const [code, setCode] = useState(initialCode)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    // Only update if initialCode actually changes and is different from current code
    if (initialCode !== code) {
      setCode(initialCode)
    }
  }, [initialCode, code])

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleRun = () => {
    if (onRun) {
      onRun(code)
    }
  }

  return (
    <div className="rounded-lg border bg-muted/50">
      <div className="flex items-center justify-between px-4 py-2 border-b bg-muted">
        <div className="text-sm font-medium">{language}</div>
        <div className="flex gap-2">
          <Button variant="ghost" size="icon" onClick={handleCopy} className="h-8 w-8" aria-label="Copy code">
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          </Button>
          {onRun && (
            <Button variant="ghost" size="icon" onClick={handleRun} className="h-8 w-8" aria-label="Run code">
              <Play className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
      <div className="p-4 overflow-auto">
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className={`w-full font-mono text-sm bg-transparent outline-none resize-none ${
            readOnly ? "cursor-default" : ""
          }`}
          style={{ minHeight: "150px" }}
          readOnly={readOnly}
          spellCheck="false"
        />
      </div>
    </div>
  )
}
