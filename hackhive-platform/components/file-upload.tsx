"use client"

import type React from "react"

import { useState, useCallback, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, FileText, AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

export function FileUpload() {
  const [dragActive, setDragActive] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [error, setError] = useState<string>("")
  const inputRef = useRef<HTMLInputElement | null>(null)

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    setError("")

    const files = e.dataTransfer.files
    if (files && files[0]) {
      validateAndSetFile(files[0])
    }
  }, [])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    setError("")
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0])
    }
  }

  const validateAndSetFile = (file: File) => {
    const maxSize = 10 * 1024 * 1024 // 10MB

    // Some browsers / OSes report CSV as application/vnd.ms-excel or empty string
    const mime = file.type
    const nameLower = file.name.toLowerCase()
    const ext = nameLower.split('.').pop() || ''

    const isCSV = mime === 'text/csv' || mime === 'application/vnd.ms-excel' || ext === 'csv'
    const isJSON = mime === 'application/json' || ext === 'json'
    const isTXT = mime === 'text/plain' || ext === 'txt'

    if (!(isCSV || isJSON || isTXT)) {
      setError(`Unsupported file type: ${mime || '(none)'} — please upload CSV, JSON, or TXT`)
      return
    }

    if (file.size > maxSize) {
      setError("File size must be less than 10MB")
      return
    }

    setFile(file)
  }

  const handleUpload = async () => {
    if (file) {
      console.log("Processing file:", file.name)
      
      // Redirect to analysis page with file data
      // For now, we'll use localStorage to pass file data
      const reader = new FileReader()
      reader.onload = (e) => {
        const content = e.target?.result as string
        localStorage.setItem('analysisFile', JSON.stringify({
          name: file.name,
          content: content,
          type: file.type,
          size: file.size
        }))
        
        // Navigate to analysis page
        window.location.href = '/dashboard/analysis'
      }
      reader.readAsText(file)
    }
  }

  return (
    <Card className="w-full max-w-2xl mx-auto relative z-[55]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5" />
          Upload Data File
        </CardTitle>
        <CardDescription>
          Upload structured feedback data in CSV, JSON, or TXT format for sentiment analysis
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors z-[60] ${
            dragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.json,.txt"
            onChange={handleChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-[70]"
            aria-label="Select data file for analysis"
          />

          <div className="space-y-4">
            <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
              <Upload className="h-8 w-8 text-primary" />
            </div>

            <div>
              <p className="text-lg font-medium text-foreground">Drop your file here, or click to browse</p>
              <p className="text-sm text-muted-foreground mt-2">Supports CSV, JSON, and TXT files up to 10MB</p>
            </div>
          </div>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {file && (
          <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
            <div className="flex items-center gap-3">
              <FileText className="h-5 w-5 text-primary" />
              <div>
                <p className="font-medium text-foreground">{file.name}</p>
                <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>
            </div>
            <Button onClick={handleUpload} className="bg-primary hover:bg-primary/90">
              Upload & Analyze
            </Button>
          </div>
        )}

        {!file && (
          <div className="flex justify-center">
            <Button
              variant="secondary"
              onClick={(e) => {
                e.preventDefault()
                console.log('Browse button clicked - triggering file input')
                if (inputRef.current) {
                  inputRef.current.click()
                } else {
                  console.error('inputRef is null')
                }
              }}
              className="mt-2"
            >
              Browse File
            </Button>
          </div>
        )}

        <div className="text-sm text-muted-foreground">
          <p className="font-medium mb-2">Supported formats:</p>
          <ul className="space-y-1 ml-4">
            <li>• CSV files with comment/feedback columns</li>
            <li>• JSON files with structured feedback data</li>
            <li>• TXT files with line-separated comments</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}
