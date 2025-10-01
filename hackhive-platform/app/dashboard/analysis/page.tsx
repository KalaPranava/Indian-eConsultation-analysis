"use client"

import { useState, useEffect } from "react"
import { AnalysisDashboard } from "@/components/analysis-dashboard"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowLeft, FileText } from "lucide-react"
import Link from "next/link"

interface FileData {
  name: string
  content: string
  type: string
  size: number
}

export default function AnalysisPage() {
  const [fileData, setFileData] = useState<FileData | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Get file data from localStorage
    const storedFile = localStorage.getItem('analysisFile')
    if (storedFile) {
      try {
        const parsed = JSON.parse(storedFile)
        setFileData(parsed)
      } catch (error) {
        console.error('Error parsing stored file:', error)
      }
    }
    setIsLoading(false)
  }, [])

  if (isLoading) {
    return (
      <main className="min-h-screen bg-background">
        <div className="flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
            <p className="text-muted-foreground">Loading analysis...</p>
          </div>
        </div>
      </main>
    )
  }

  if (!fileData) {
    return (
      <main className="min-h-screen bg-background">
        <div className="pb-16">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <Card>
              <CardHeader>
                <CardTitle className="text-2xl text-destructive">No File Found</CardTitle>
                <CardDescription>
                  No file data found for analysis. Please upload a file first.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Link href="/dashboard">
                  <Button>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back to Dashboard
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="pb-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <Link href="/dashboard">
                  <Button variant="outline">
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Back
                  </Button>
                </Link>
                <div>
                  <h1 className="text-2xl md:text-3xl font-bold text-foreground">Analysis Results</h1>
                  <div className="flex items-center mt-2 text-muted-foreground">
                    <FileText className="mr-2 h-4 w-4" />
                    <span>{fileData.name}</span>
                    <span className="mx-2">•</span>
                    <span>{(fileData.size / 1024).toFixed(1)} KB</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Analysis Dashboard */}
          <AnalysisDashboard fileData={fileData} />
        </div>
      </div>
    </main>
  )
}