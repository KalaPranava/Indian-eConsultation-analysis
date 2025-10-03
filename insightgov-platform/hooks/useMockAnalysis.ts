import { useState } from 'react'

interface FileData {
  name: string
  content: string
  type: string
  size: number
}

interface MockAnalysisResult {
  results: any[]
  statistics: any
  overallSummary: any
  models_used: any
}

export function useMockAnalysis() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentBatch, setCurrentBatch] = useState(0)
  const [totalBatches, setTotalBatches] = useState(0)

  const simulateAnalysis = async (fileData: FileData): Promise<MockAnalysisResult> => {
    setIsAnalyzing(true)
    setProgress(0)
    setCurrentBatch(0)
    setTotalBatches(4) // Simulate 4 batches

    // Simulate progressive loading with realistic delays
    const stages = [
      { progress: 25, batch: 1, delay: 1500, message: "Loading sentiment models..." },
      { progress: 50, batch: 2, delay: 2000, message: "Processing emotions..." },
      { progress: 75, batch: 3, delay: 1800, message: "Generating summaries..." },
      { progress: 100, batch: 4, delay: 1200, message: "Finalizing analysis..." }
    ]

    for (const stage of stages) {
      await new Promise(resolve => setTimeout(resolve, stage.delay))
      setProgress(stage.progress)
      setCurrentBatch(stage.batch)
    }

    // Load the sample analysis results
    try {
      const response = await fetch('/sample-analysis-results.json')
      const mockData = await response.json()
      
      // Add some realistic variations based on file data
      const enhancedData = {
        ...mockData,
        results: mockData.results.map((result: any, index: number) => ({
          ...result,
          id: index + 1,
          timestamp: new Date(Date.now() - (mockData.results.length - index) * 120000).toISOString()
        })),
        statistics: {
          ...mockData.statistics,
          // Simulate dynamic statistics based on "file size"
          total: Math.max(6, Math.floor(fileData.size / 1000))
        }
      }

      setIsAnalyzing(false)
      return enhancedData
    } catch (error) {
      console.error('Error loading mock data:', error)
      setIsAnalyzing(false)
      
      // Fallback data if JSON loading fails
      return {
        results: [],
        statistics: {
          total: 0,
          sentimentCounts: { positive: 0, negative: 0, neutral: 0 },
          emotionCounts: { joy: 0, anger: 0, neutral: 0 },
          languageCounts: { English: 0 },
          avgConfidence: 0
        },
        overallSummary: null,
        models_used: {
          sentiment: "Mock Analysis Mode",
          emotion: "Demo Version",
          summarization: "Sample Data",
          language_detection: "Static Results"
        }
      }
    }
  }

  const simulateDemoAnalysis = async (): Promise<MockAnalysisResult> => {
    // For demo, we'll use a fake file object
    const demoFile: FileData = {
      name: 'policy_feedback.csv',
      content: 'demo content',
      type: 'text/csv',
      size: 15000 // Simulate realistic file size
    }

    return simulateAnalysis(demoFile)
  }

  return {
    isAnalyzing,
    progress,
    currentBatch,
    totalBatches,
    simulateAnalysis,
    simulateDemoAnalysis
  }
}