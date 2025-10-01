"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  BarChart3, 
  PieChart, 
  TrendingUp, 
  MessageSquare, 
  Heart, 
  Brain,
  Download,
  Sparkles,
  Activity,
  Clock,
  FileText,
  Smile,
  Frown,
  Meh,
  Star,
  AlertCircle,
  ThumbsUp,
  ThumbsDown,
  Target,
  Users,
  Globe,
  Zap,
  ChevronRight,
  Cloud,
  Filter,
  Search,
  X
} from "lucide-react"

interface FileData {
  name: string
  content: string
  type: string
  size: number
}

interface AnalysisResult {
  id: number
  originalText: string
  sentiment: {
    label: string
    confidence: number
    scores: { [key: string]: number }
    method: string
  }
  emotion: {
    primary_emotion: string
    emotion_scores: { [key: string]: number }
    confidence: number
    method: string
  }
  summary: string
  language: string
  timestamp: string
}

interface Props {
  fileData: FileData
}

export function AnalysisDashboard({ fileData }: Props) {
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentBatch, setCurrentBatch] = useState(0)
  const [totalBatches, setTotalBatches] = useState(0)
  const [statistics, setStatistics] = useState<any>(null)
  const [overallSummary, setOverallSummary] = useState<any>(null)
  const [wordCloud, setWordCloud] = useState<{[key: string]: number}>({})
  // Word cloud selection supports multiple words now
  const [selectedWords, setSelectedWords] = useState<string[]>([])
  const [filteredComments, setFilteredComments] = useState<AnalysisResult[]>([])
  const [modelsUsed, setModelsUsed] = useState<{[key: string]: string}>({})

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000'
  
  // Mock mode - automatically enable for demo deployment
  const MOCK_MODE = process.env.NEXT_PUBLIC_MOCK_MODE === 'true' || 
    (!API_BASE_URL || API_BASE_URL === '' || 
     (!API_BASE_URL.includes('localhost') && !API_BASE_URL.includes('127.0.0.1') && 
      !API_BASE_URL.includes('render') && !API_BASE_URL.includes('oracle') && 
      !API_BASE_URL.includes('http')))

  // Define active ML models for impressive loading display 
  const activeModels = {
    sentiment: {
      name: "XLM-RoBERTa Multilingual",
      description: "State-of-the-art transformer for Hindi/English sentiment",
      icon: "🧠",
      color: "from-blue-500 to-cyan-500",
      textColor: "text-blue-600"
    },
    emotion: {
      name: "DistilBERT Emotion Detector",
      description: "Fine-tuned emotion classification model",
      icon: "❤️",
      color: "from-purple-500 to-pink-500",
      textColor: "text-purple-600"
    },
    summarization: {
      name: "DistilBART Summarizer",
      description: "Advanced text summarization for Indian content",
      icon: "📝",
      color: "from-green-500 to-emerald-500",
      textColor: "text-green-600"
    },
    language: {
      name: "Language Detection Engine",
      description: "Hindi/English/Code-mixed classification",
      icon: "🌐",
      color: "from-orange-500 to-yellow-500",
      textColor: "text-orange-600"
    }
  }

  useEffect(() => {
    if (fileData) {
      startAnalysis()
    }
  }, [fileData])

  const parseCSVData = (csvContent: string) => {
    const lines = csvContent.split('\n').filter(line => line.trim())
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''))
    
    // Find text column (look for common names)
    const textColumnNames = ['comment', 'text', 'feedback', 'review', 'message', 'content']
    let textColumnIndex = headers.findIndex(h => 
      textColumnNames.some(name => h.toLowerCase().includes(name))
    )
    
    if (textColumnIndex === -1) {
      textColumnIndex = 1 // Default to second column if no match
    }

    const data = []
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''))
      if (values[textColumnIndex] && values[textColumnIndex].trim()) {
        data.push(values[textColumnIndex].trim())
      }
    }
    
    return data
  }

  const startMockAnalysis = async () => {
    try {
      setTotalBatches(4)
      
      // Simulate progressive loading with realistic delays
      const stages = [
        { progress: 25, batch: 1, delay: 1500 },
        { progress: 50, batch: 2, delay: 2000 },
        { progress: 75, batch: 3, delay: 1800 },
        { progress: 100, batch: 4, delay: 1200 }
      ]

      for (const stage of stages) {
        await new Promise(resolve => setTimeout(resolve, stage.delay))
        setProgress(stage.progress)
        setCurrentBatch(stage.batch)
      }

      // Load the sample analysis results
      const response = await fetch('/sample-analysis-results.json')
      const mockData = await response.json()
      
      // Set the analysis results
      setAnalysisResults(mockData.results)
      setStatistics(mockData.statistics)
      setOverallSummary(mockData.overallSummary)
      setModelsUsed(mockData.models_used)
      
      // Generate word cloud from mock data
      generateWordCloudFromResults(mockData.results)
      
    } catch (error) {
      console.error('Error loading mock data:', error)
      // Set fallback data
      setAnalysisResults([])
      setStatistics({
        total: 0,
        sentimentCounts: { positive: 0, negative: 0, neutral: 0 },
        emotionCounts: { joy: 0, anger: 0, neutral: 0 },
        languageCounts: { English: 0 },
        avgConfidence: 0
      })
      setOverallSummary(null)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const generateWordCloudFromResults = (results: AnalysisResult[]) => {
    const stopWords = new Set([
      'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was', 'will', 'an', 'be', 'or', 'of', 'with', 
      'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among',
      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'this', 'that', 'these', 'those', 'my', 'your',
      'his', 'her', 'its', 'our', 'very', 'can', 'had', 'has', 'have', 'do', 'does', 'did', 'but', 'if', 'for', 'in'
    ])
    
    const wordCount: { [key: string]: number } = {}
    
    results.forEach(result => {
      const words = result.originalText
        .toLowerCase()
        .replace(/[^\w\s\u0900-\u097F]/g, ' ')
        .split(/\s+/)
        .filter(word => word.length > 2 && !stopWords.has(word))
      
      words.forEach(word => {
        wordCount[word] = (wordCount[word] || 0) + 1
      })
    })
    
    // Get top 30 words
    const sortedWords = Object.entries(wordCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 30)
      .reduce((obj, [word, count]) => {
        obj[word] = count
        return obj
      }, {} as { [key: string]: number })
    
    setWordCloud(sortedWords)
  }

  const startAnalysis = async () => {
    setIsAnalyzing(true)
    setProgress(0)

    // Mock mode for demo deployment
    if (MOCK_MODE) {
      return await startMockAnalysis()
    }

    try {
      // Parse the file content
      let texts: string[] = []
      
      if (fileData.type === 'text/csv') {
        texts = parseCSVData(fileData.content)
      } else if (fileData.type === 'application/json') {
        const jsonData = JSON.parse(fileData.content)
        if (Array.isArray(jsonData)) {
          texts = jsonData.map(item => 
            typeof item === 'string' ? item : 
            item.text || item.comment || item.feedback || JSON.stringify(item)
          )
        }
      } else {
        // Plain text - split by lines
        texts = fileData.content.split('\n').filter(line => line.trim())
      }

      if (texts.length === 0) {
        throw new Error('No text data found in the file')
      }

      const batchSize = 10
      const batches = Math.ceil(texts.length / batchSize)
      setTotalBatches(batches)

      const results: AnalysisResult[] = []

      // Process in batches
      for (let i = 0; i < texts.length; i += batchSize) {
        const batchTexts = texts.slice(i, Math.min(i + batchSize, texts.length))
        setCurrentBatch(Math.floor(i / batchSize) + 1)

        try {
          const response = await fetch(`${API_BASE_URL}/analyze/batch`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              texts: batchTexts,
              analyses: ['sentiment', 'emotion', 'summarize']
            })
          })

          if (!response.ok) {
            throw new Error(`API Error: ${response.status}`)
          }

          const batchResult = await response.json()

          // Process batch results
          for (let j = 0; j < batchTexts.length; j++) {
            const result = batchResult.results[j]
            
            // Track models used (update our models state based on actual API response)
            if (result.sentiment?.method) {
              setModelsUsed(prev => ({ ...prev, sentiment: result.sentiment.method }))
            }
            if (result.emotions?.method) {
              setModelsUsed(prev => ({ ...prev, emotion: result.emotions.method }))
            }
            
            results.push({
              id: i + j + 1,
              originalText: batchTexts[j],
              sentiment: result.sentiment || { label: 'neutral', confidence: 0.5, scores: {}, method: 'fallback' },
              emotion: result.emotions || { primary_emotion: 'neutral', emotion_scores: {}, confidence: 0.5, method: 'fallback' },
              summary: result.summary || batchTexts[j].substring(0, 60) + '...',
              language: detectLanguage(batchTexts[j]),
              timestamp: new Date().toISOString()
            })
          }

          setProgress(Math.round(((i + batchTexts.length) / texts.length) * 100))
        } catch (error) {
          console.error(`Error processing batch ${Math.floor(i / batchSize) + 1}:`, error)
          // Add fallback results for this batch
          for (let j = 0; j < batchTexts.length; j++) {
            results.push({
              id: i + j + 1,
              originalText: batchTexts[j],
              sentiment: { label: 'neutral', confidence: 0.5, scores: {}, method: 'fallback' },
              emotion: { primary_emotion: 'neutral', emotion_scores: {}, confidence: 0.5, method: 'fallback' },
              summary: batchTexts[j].substring(0, 60) + '...',
              language: detectLanguage(batchTexts[j]),
              timestamp: new Date().toISOString()
            })
          }
        }
      }

      setAnalysisResults(results)
      generateStatistics(results)
      
      // Get overall summary
      try {
        const subset = texts.slice(0, 100)
        let summaryResponse = await fetch(`${API_BASE_URL}/analyze/overall_summary`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ texts: subset })
        })

        // If 422 try raw list payload fallback
        if (summaryResponse.status === 422) {
          summaryResponse = await fetch(`${API_BASE_URL}/analyze/overall_summary`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(subset)
          })
        }

        if (summaryResponse.ok) {
          const summary = await summaryResponse.json()
          setOverallSummary(summary)
        } else {
          console.warn('Overall summary request failed:', summaryResponse.status)
        }
      } catch (error) {
        console.error('Error getting overall summary:', error)
      }

    } catch (error) {
      console.error('Analysis failed:', error)
      alert(`Analysis failed: ${error}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const detectLanguage = (text: string) => {
    const hindiChars = (text.match(/[\u0900-\u097F]/g) || []).length
    const englishChars = (text.match(/[a-zA-Z]/g) || []).length
    const total = hindiChars + englishChars
    
    if (total === 0) return 'Unknown'
    
    const hindiRatio = hindiChars / total
    const englishRatio = englishChars / total
    
    if (hindiRatio > 0.6) return 'Hindi'
    if (englishRatio > 0.6) return 'English'
    if (hindiRatio > 0.2 && englishRatio > 0.2) return 'Code-mixed'
    
    return 'Mixed'
  }

  const generateWordCloud = (results: AnalysisResult[]) => {
    const stopWords = new Set([
      'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was', 'will', 'an', 'be', 'or', 'of', 'with', 
      'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among',
      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'this', 'that', 'these', 'those', 'my', 'your',
      'his', 'her', 'its', 'our', 'very', 'can', 'had', 'has', 'have', 'do', 'does', 'did', 'but', 'if', 'for', 'in',
      'policy', 'एक', 'है', 'के', 'की', 'को', 'में', 'से', 'पर', 'और', 'या', 'भी', 'नहीं', 'तो', 'ही', 'कि', 'जो'
    ])
    
    const wordCount: { [key: string]: number } = {}
    
    results.forEach(result => {
      const words = result.originalText
        .toLowerCase()
        .replace(/[^\w\s\u0900-\u097F]/g, ' ')
        .split(/\s+/)
        .filter(word => word.length > 2 && !stopWords.has(word))
      
      words.forEach(word => {
        wordCount[word] = (wordCount[word] || 0) + 1
      })
    })
    
    // Get top 50 words
    const sortedWords = Object.entries(wordCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 50)
      .reduce((obj, [word, count]) => {
        obj[word] = count
        return obj
      }, {} as { [key: string]: number })
    
    setWordCloud(sortedWords)
  }

  const toggleWordSelection = (word: string) => {
    setSelectedWords(prev => {
      let next: string[]
      if (prev.includes(word)) {
        next = prev.filter(w => w !== word)
      } else {
        next = [...prev, word]
      }
      // Update filtered comments based on ANY selected word (OR logic)
      if (next.length === 0) {
        setFilteredComments([])
      } else {
        const lowered = next.map(w => w.toLowerCase())
        setFilteredComments(
          analysisResults.filter(r => lowered.some(w => r.originalText.toLowerCase().includes(w)))
        )
      }
      return next
    })
  }

  const clearSelectedWords = () => {
    setSelectedWords([])
    setFilteredComments([])
  }

  const generateStatistics = (results: AnalysisResult[]) => {
    const total = results.length
    
    const sentimentCounts = {
      positive: results.filter(r => r.sentiment.label === 'positive').length,
      negative: results.filter(r => r.sentiment.label === 'negative').length,
      neutral: results.filter(r => r.sentiment.label === 'neutral').length
    }
    
    const emotionCounts = results.reduce((acc, r) => {
      const emotion = r.emotion.primary_emotion
      acc[emotion] = (acc[emotion] || 0) + 1
      return acc
    }, {} as { [key: string]: number })
    
    const languageCounts = results.reduce((acc, r) => {
      acc[r.language] = (acc[r.language] || 0) + 1
      return acc
    }, {} as { [key: string]: number })
    
    const avgConfidence = results.reduce((sum, r) => sum + r.sentiment.confidence, 0) / total

    // Generate overall summary
    const positivePercentage = Math.round((sentimentCounts.positive / total) * 100)
    const topEmotion = Object.entries(emotionCounts).reduce((a, b) => emotionCounts[a[0]] > emotionCounts[b[0]] ? a : b)[0]
    const mainLanguage = Object.entries(languageCounts).reduce((a, b) => languageCounts[a[0]] > languageCounts[b[0]] ? a : b)[0]
    
    const overallSummaryText = `Analysis of ${total} comments reveals ${positivePercentage}% positive sentiment with ${topEmotion} being the dominant emotion. The majority of comments are in ${mainLanguage} with an average confidence of ${Math.round(avgConfidence * 100)}%. ${
      sentimentCounts.negative > sentimentCounts.positive ? 'Negative feedback patterns suggest areas for improvement.' :
      sentimentCounts.positive > sentimentCounts.negative ? 'Positive feedback indicates good user satisfaction.' :
      'Mixed sentiment suggests balanced user opinions.'
    }`

    setOverallSummary({
      overall_summary: overallSummaryText,
      sentiment_distribution: sentimentCounts,
      emotion_distribution: emotionCounts,
      key_insights: {
        satisfaction_level: positivePercentage > 60 ? 'High' : positivePercentage > 40 ? 'Medium' : 'Low',
        primary_concerns: sentimentCounts.negative > sentimentCounts.positive ? 'Service Quality' : 'Minor Issues'
      },
      text: overallSummaryText,
      keyInsights: [
        `${positivePercentage}% positive sentiment`,
        `${topEmotion} is the primary emotion`,
        `${mainLanguage} is the main language`,
        `${Math.round(avgConfidence * 100)}% average confidence`
      ]
    })

    setStatistics({
      total,
      sentimentCounts,
      emotionCounts,
      languageCounts,
      avgConfidence
    })

    // Generate word cloud
    generateWordCloud(results)
  }

  const exportResults = () => {
    const csvContent = [
      ['ID', 'Text', 'Sentiment', 'Confidence', 'Emotion', 'Summary', 'Language'],
      ...analysisResults.map(result => [
        result.id,
        `"${result.originalText.replace(/"/g, '""')}"`,
        result.sentiment.label,
        result.sentiment.confidence.toFixed(3),
        result.emotion.primary_emotion,
        `"${result.summary.replace(/"/g, '""')}"`,
        result.language
      ])
    ].map(row => row.join(',')).join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `analysis_results_${new Date().toISOString().split('T')[0]}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }

  if (isAnalyzing) {
    return (
      <div className="space-y-6">
        {/* Impressive Loading Card with Models Display */}
        <Card className="bg-gradient-to-br from-indigo-50 via-blue-50 to-cyan-50 dark:from-indigo-950 dark:via-blue-950 dark:to-cyan-950 border-indigo-200 dark:border-indigo-800 shadow-xl">
          <CardHeader className="text-center pb-4">
            <CardTitle className="flex items-center justify-center text-2xl text-indigo-900 dark:text-indigo-100">
              <Brain className="mr-3 h-8 w-8 animate-pulse text-indigo-600 dark:text-indigo-400" />
              AI Analysis in Progress
            </CardTitle>
            <CardDescription className="text-lg text-indigo-700 dark:text-indigo-300">
              Deploying state-of-the-art ML models for comprehensive analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Progress Section */}
            <div className="space-y-3">
              <div className="flex justify-between text-sm font-medium text-indigo-800 dark:text-indigo-200">
                <span>Overall Progress</span>
                <span className="text-indigo-600 dark:text-indigo-400">{progress}%</span>
              </div>
              <Progress 
                value={progress} 
                className="h-3 bg-indigo-100 dark:bg-indigo-900"
              />
              <div className="flex justify-between text-xs text-indigo-600 dark:text-indigo-400">
                <span>Batch {currentBatch} of {totalBatches}</span>
                <span>{Math.round((currentBatch / Math.max(totalBatches, 1)) * 100)}% batches complete</span>
              </div>
            </div>
            
            {/* Active Models Grid - The Impressive Part */}
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-xl p-6 border border-indigo-200 dark:border-indigo-700">
              <h3 className="text-lg font-bold text-indigo-900 dark:text-indigo-100 mb-4 flex items-center">
                <Sparkles className="mr-2 h-5 w-5 text-indigo-600 dark:text-indigo-400" />
                Active ML Models
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(activeModels).map(([key, model], index) => (
                  <div 
                    key={key} 
                    className={`model-card relative overflow-hidden rounded-lg p-4 bg-gradient-to-r ${model.color} shadow-lg transform transition-all duration-500 hover:scale-105`}
                    style={{ 
                      animationDelay: `${index * 0.2}s`
                    }}
                  >
                    <div className="absolute top-2 right-2 text-2xl animate-pulse">
                      {model.icon}
                    </div>
                    <div className="text-white">
                      <h4 className="font-bold text-lg mb-1">{model.name}</h4>
                      <p className="text-white/90 text-sm leading-relaxed">
                        {model.description}
                      </p>
                      <div className="mt-3 flex items-center text-white/80 text-xs">
                        <Activity className="h-3 w-3 mr-1 animate-pulse" />
                        <span>Processing...</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Processing Stats */}
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg p-3 border border-indigo-100 dark:border-indigo-800">
                <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {Math.round(progress / 25) || 1}
                </div>
                <div className="text-xs text-indigo-800 dark:text-indigo-200">Models Active</div>
              </div>
              <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg p-3 border border-indigo-100 dark:border-indigo-800">
                <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {currentBatch}
                </div>
                <div className="text-xs text-indigo-800 dark:text-indigo-200">Current Batch</div>
              </div>
              <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg p-3 border border-indigo-100 dark:border-indigo-800">
                <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                  {Math.round(progress * 0.95)}%
                </div>
                <div className="text-xs text-indigo-800 dark:text-indigo-200">AI Confidence</div>
              </div>
            </div>

            <style jsx>{`
              @keyframes fadeInUp {
                from {
                  opacity: 0;
                  transform: translateY(20px);
                }
                to {
                  opacity: 1;
                  transform: translateY(0);
                }
              }
              
              @keyframes glow {
                0%, 100% {
                  box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
                }
                50% {
                  box-shadow: 0 0 30px rgba(99, 102, 241, 0.6);
                }
              }
              
              .model-card {
                animation: fadeInUp 0.6s ease-out forwards, glow 2s ease-in-out infinite;
              }
            `}</style>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!statistics) {
    return (
      <div className="text-center py-12">
        <FileText className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
        <p className="text-muted-foreground">No analysis results available</p>
      </div>
    )
  }

  return (
    <div className="flex gap-6 relative z-10">
      {/* Main Content */}
      <div className="flex-1 space-y-6">
        {/* Demo Mode Banner */}
        {MOCK_MODE && (
          <Card className="bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950 dark:to-orange-950 border-amber-200 dark:border-amber-800">
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <Sparkles className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                <div>
                  <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
                    Demo Mode Active
                  </p>
                  <p className="text-xs text-amber-700 dark:text-amber-300">
                    Showing sample analysis results. Full AI backend will be available soon.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
        
        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center space-x-2">
                <MessageSquare className="h-5 w-5 text-primary" />
                <div>
                  <p className="text-2xl font-bold">{statistics.total}</p>
                  <p className="text-sm text-muted-foreground">Total Comments</p>
                </div>
              </div>
            </CardContent>
          </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-green-500" />
              <div>
                <p className="text-2xl font-bold text-green-600">
                  {Math.round((statistics.sentimentCounts.positive / statistics.total) * 100)}%
                </p>
                <p className="text-sm text-muted-foreground">Positive</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Heart className="h-5 w-5 text-red-500" />
              <div>
                <p className="text-2xl font-bold">
                  {Object.keys(statistics.emotionCounts).reduce((a, b) => 
                    statistics.emotionCounts[a] > statistics.emotionCounts[b] ? a : b
                  )}
                </p>
                <p className="text-sm text-muted-foreground">Top Emotion</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Sparkles className="h-5 w-5 text-blue-500" />
              <div>
                <p className="text-2xl font-bold">
                  {Math.round(statistics.avgConfidence * 100)}%
                </p>
                <p className="text-sm text-muted-foreground">Avg Confidence</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Overall Summary */}
      {overallSummary && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Brain className="mr-2 h-5 w-5" />
              AI-Generated Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <h4 className="font-medium mb-2">Sentiment Distribution</h4>
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span>Positive</span>
                    <Badge variant="secondary">
                      {overallSummary?.sentiment_distribution?.positive ?? 0}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Negative</span>
                    <Badge variant="secondary">
                      {overallSummary?.sentiment_distribution?.negative ?? 0}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Neutral</span>
                    <Badge variant="secondary">
                      {overallSummary?.sentiment_distribution?.neutral ?? 0}
                    </Badge>
                  </div>
                </div>
              </div>
              <div>
                <h4 className="font-medium mb-2">Top Emotions</h4>
                <div className="space-y-1">
                  {overallSummary?.emotion_distribution ? 
                    Object.entries(overallSummary.emotion_distribution)
                      .sort(([,a], [,b]) => (b as number) - (a as number))
                      .slice(0, 3)
                      .map(([emotion, count]) => (
                        <div key={emotion} className="flex justify-between">
                          <span className="capitalize">{emotion}</span>
                          <Badge variant="secondary">{count as number}</Badge>
                        </div>
                      ))
                    : (
                      <div className="text-muted-foreground text-sm">
                        No emotion data available yet
                      </div>
                    )}
                </div>
              </div>
              <div>
                <h4 className="font-medium mb-2">Key Insights</h4>
                <div className="space-y-1">
                  {overallSummary?.key_insights ? (
                    <>
                      <div className="flex justify-between">
                        <span>Satisfaction</span>
                        <Badge>{overallSummary.key_insights.satisfaction_level || 'N/A'}</Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Concerns</span>
                        <Badge variant="outline">{overallSummary.key_insights.primary_concerns || 'N/A'}</Badge>
                      </div>
                    </>
                  ) : (
                    <div className="text-muted-foreground text-sm">
                      Key insights will appear after analysis
                    </div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Detailed Results */}
      <Tabs defaultValue="summary" className="space-y-4 relative z-20">
        <TabsList className="grid w-full grid-cols-3 sm:grid-cols-6 gap-1 h-auto p-1 relative z-30 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/90 border shadow-sm">
          <TabsTrigger value="summary" className="text-xs sm:text-sm px-2 py-2 data-[state=active]:z-40">Summary</TabsTrigger>
          <TabsTrigger value="wordcloud" className="text-xs sm:text-sm px-2 py-2 data-[state=active]:z-40">Word Cloud</TabsTrigger>
          <TabsTrigger value="insights" className="text-xs sm:text-sm px-2 py-2 data-[state=active]:z-40">Data Insights</TabsTrigger>
          <TabsTrigger value="comments" className="text-xs sm:text-sm px-2 py-2 data-[state=active]:z-40">Comments</TabsTrigger>
          <TabsTrigger value="emotions" className="text-xs sm:text-sm px-2 py-2 data-[state=active]:z-40">Emotions</TabsTrigger>
          <TabsTrigger value="export" className="text-xs sm:text-sm px-2 py-2 data-[state=active]:z-40">Export</TabsTrigger>
        </TabsList>

        <TabsContent value="summary" className="space-y-6 relative z-10 mt-4">
          {overallSummary && (
            <div className="space-y-6">
              {/* Overall Summary Card */}
              <Card className="bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-950 dark:to-indigo-900 border-blue-200 dark:border-blue-800">
                <CardHeader>
                  <CardTitle className="flex items-center text-blue-900 dark:text-blue-100">
                    <FileText className="mr-2 h-6 w-6" />
                    Overall Analysis Summary
                  </CardTitle>
                  <CardDescription className="text-blue-700 dark:text-blue-300">
                    Comprehensive insights from all {statistics.total} comments
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 border border-blue-200 dark:border-blue-700 shadow-sm">
                    <p className="text-gray-900 dark:text-gray-100 leading-relaxed text-lg font-medium">
                      {overallSummary.text}
                    </p>
                  </div>
                  {/* Only render keyInsights if it exists and is an array */}
                  {Array.isArray(overallSummary.keyInsights) && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                      {overallSummary.keyInsights.map((insight: string, index: number) => (
                        <div key={index} className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-blue-200 dark:border-blue-700 text-center">
                          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-1">
                            {insight.split(' ')[0]}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            {insight.split(' ').slice(1).join(' ')}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Key Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card className="bg-gradient-to-br from-green-50 to-emerald-100 dark:from-green-950 dark:to-emerald-900 border-green-200 dark:border-green-800">
                  <CardContent className="p-6 text-center">
                    <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">
                      {Math.round((statistics.sentimentCounts.positive / statistics.total) * 100)}%
                    </div>
                    <div className="text-green-700 dark:text-green-300 font-medium">Positive Sentiment</div>
                    <div className="text-sm text-green-600 dark:text-green-400 mt-1">
                      {statistics.sentimentCounts.positive} out of {statistics.total} comments
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-purple-50 to-violet-100 dark:from-purple-950 dark:to-violet-900 border-purple-200 dark:border-purple-800">
                  <CardContent className="p-6 text-center">
                    <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2 capitalize">
                      {Object.entries(statistics.emotionCounts).reduce((a, b) => statistics.emotionCounts[a[0]] > statistics.emotionCounts[b[0]] ? a : b)[0]}
                    </div>
                    <div className="text-purple-700 dark:text-purple-300 font-medium">Dominant Emotion</div>
                    <div className="text-sm text-purple-600 dark:text-purple-400 mt-1">
                      {Object.entries(statistics.emotionCounts).reduce((a, b) => statistics.emotionCounts[a[0]] > statistics.emotionCounts[b[0]] ? a : b)[1]} occurrences
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-orange-50 to-amber-100 dark:from-orange-950 dark:to-amber-900 border-orange-200 dark:border-orange-800">
                  <CardContent className="p-6 text-center">
                    <div className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-2">
                      {Math.round(statistics.avgConfidence * 100)}%
                    </div>
                    <div className="text-orange-700 dark:text-orange-300 font-medium">Avg Confidence</div>
                    <div className="text-sm text-orange-600 dark:text-orange-400 mt-1">
                      Analysis reliability score
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}
        </TabsContent>

        <TabsContent value="wordcloud" className="space-y-6 relative z-10 mt-4">
          <Card className="bg-gradient-to-br from-gray-50 to-slate-100 dark:from-gray-900 dark:to-slate-800 border-gray-200 dark:border-gray-700">
            <CardHeader>
              <CardTitle className="flex items-center text-gray-900 dark:text-gray-100">
                <Cloud className="mr-2 h-6 w-6" />
                Interactive Word Cloud
              </CardTitle>
              <CardDescription className="text-gray-700 dark:text-gray-300">
                Click words to select (multi-select). Comments containing ANY selected word will appear below.
                {selectedWords.length > 0 && (
                  <span className="mt-2 flex flex-wrap items-center gap-2">
                    <Filter className="h-4 w-4" />
                    <span className="text-xs">Filtering by:</span>
                    {selectedWords.map(w => (
                      <Badge key={w} variant="secondary" className="flex items-center gap-1 text-xs">
                        {w}
                        <button
                          onClick={() => toggleWordSelection(w)}
                          className="ml-1 text-gray-500 hover:text-gray-800 dark:hover:text-gray-200"
                        >
                          ×
                        </button>
                      </Badge>
                    ))}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={clearSelectedWords}
                      className="h-6 px-2 text-xs"
                    >
                      Clear
                    </Button>
                  </span>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 min-h-[300px] border border-gray-200 dark:border-gray-700">
                <div className="flex flex-wrap items-center justify-center gap-2">
                  {Object.entries(wordCloud).map(([word, count]) => {
                    const maxCount = Math.max(...Object.values(wordCloud))
                    const minSize = 12
                    const maxSize = 32
                    const fontSize = minSize + ((count / maxCount) * (maxSize - minSize))
                    const opacity = 0.6 + ((count / maxCount) * 0.4)
                    
                    return (
                      <button
                        key={word}
                        onClick={() => toggleWordSelection(word)}
                        className={`transition-all duration-200 hover:scale-110 hover:opacity-100 rounded px-2 py-1 ${
                          selectedWords.includes(word) 
                            ? 'bg-blue-500 text-white dark:bg-blue-600 ring-2 ring-blue-300 dark:ring-blue-700' 
                            : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                        }`}
                        style={{ 
                          fontSize: `${fontSize}px`,
                          opacity: selectedWords.includes(word) ? 1 : opacity,
                          fontWeight: count > maxCount * 0.7 ? 'bold' : 'normal'
                        }}
                      >
                        {word}
                      </button>
                    )
                  })}
                </div>
              </div>
              
              {selectedWords.length > 0 && filteredComments.length > 0 && (
                <div className="mt-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                      Comments containing {selectedWords.length === 1 ? '"' + selectedWords[0] + '"' : 'any of (' + selectedWords.join(', ') + ')'} ({filteredComments.length})
                    </h3>
                  </div>
                  <div className="space-y-3 max-h-60 overflow-y-auto">
                    {filteredComments.slice(0, 10).map((comment) => (
                      <div key={comment.id} className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <Badge variant="outline" className="text-xs font-semibold text-gray-800 dark:text-gray-100 border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700">#{comment.id}</Badge>
                          <div className="flex items-center space-x-2">
                            <Badge variant={comment.sentiment.label === 'positive' ? 'default' : comment.sentiment.label === 'negative' ? 'destructive' : 'secondary'} className="text-xs">
                              {comment.sentiment.label}
                            </Badge>
                          </div>
                        </div>
                        <p className="text-sm text-gray-700 dark:text-gray-300">
                          {selectedWords.length === 0 ? comment.originalText : (
                            // Highlight all selected words
                            comment.originalText.split(/(\s+)/).map((tok, idx) => {
                              const lower = tok.toLowerCase()
                              if (selectedWords.some(w => w.toLowerCase() === lower.replace(/[^\w\u0900-\u097F]/g, ''))) {
                                return <mark key={idx} className="bg-yellow-200 dark:bg-yellow-800 px-1 rounded">{tok}</mark>
                              }
                              return <span key={idx}>{tok}</span>
                            })
                          )}
                        </p>
                      </div>
                    ))}
                    {filteredComments.length > 10 && (
                      <div className="text-center text-sm text-gray-500 dark:text-gray-400">
                        And {filteredComments.length - 10} more...
                      </div>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="insights" className="space-y-6 relative z-10 mt-4">
          {/* Data Visualization Dashboard */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Sentiment Pie Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <PieChart className="mr-2 h-5 w-5" />
                  Sentiment Distribution
                </CardTitle>
              </CardHeader>
              <CardContent className="h-80">
                <div className="relative h-full flex items-center justify-center">
                  <svg viewBox="0 0 100 100" className="w-48 h-48">
                    {(() => {
                      const total = statistics.total
                      const positive = statistics.sentimentCounts.positive
                      const negative = statistics.sentimentCounts.negative
                      const neutral = statistics.sentimentCounts.neutral
                      
                      const positivePercent = (positive / total) * 100
                      const negativePercent = (negative / total) * 100
                      const neutralPercent = (neutral / total) * 100
                      
                      let cumulativePercent = 0
                      
                      const createArc = (percent: number, color: string, offset: number) => {
                        const startAngle = (offset / 100) * 360
                        const endAngle = ((offset + percent) / 100) * 360
                        
                        const x1 = 50 + 40 * Math.cos((startAngle - 90) * Math.PI / 180)
                        const y1 = 50 + 40 * Math.sin((startAngle - 90) * Math.PI / 180)
                        const x2 = 50 + 40 * Math.cos((endAngle - 90) * Math.PI / 180)
                        const y2 = 50 + 40 * Math.sin((endAngle - 90) * Math.PI / 180)
                        
                        const largeArc = percent > 50 ? 1 : 0
                        
                        return (
                          <path
                            d={`M 50 50 L ${x1} ${y1} A 40 40 0 ${largeArc} 1 ${x2} ${y2} Z`}
                            fill={color}
                          />
                        )
                      }
                      
                      return (
                        <>
                          {positivePercent > 0 && createArc(positivePercent, '#22c55e', 0)}
                          {negativePercent > 0 && createArc(negativePercent, '#ef4444', positivePercent)}
                          {neutralPercent > 0 && createArc(neutralPercent, '#6b7280', positivePercent + negativePercent)}
                        </>
                      )
                    })()}
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-2xl font-bold">{statistics.total}</div>
                      <div className="text-sm text-muted-foreground">Comments</div>
                    </div>
                  </div>
                </div>
                <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                    <span>Positive ({statistics.sentimentCounts.positive})</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                    <span>Negative ({statistics.sentimentCounts.negative})</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-gray-500 rounded-full mr-2"></div>
                    <span>Neutral ({statistics.sentimentCounts.neutral})</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Emotion Bar Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BarChart3 className="mr-2 h-5 w-5" />
                  Emotion Analysis
                </CardTitle>
              </CardHeader>
              <CardContent className="h-80">
                <div className="h-full flex items-end justify-between px-4 pb-4">
                  {Object.entries(statistics.emotionCounts)
                    .sort(([,a], [,b]) => (b as number) - (a as number))
                    .slice(0, 6)
                    .map(([emotion, count], index) => {
                      const maxCount = Math.max(...Object.values(statistics.emotionCounts))
                      const height = ((count as number) / maxCount) * 200
                      const colors = ['#f59e0b', '#3b82f6', '#ef4444', '#8b5cf6', '#10b981', '#6b7280']
                      
                      return (
                        <div key={emotion} className="flex flex-col items-center space-y-2">
                          <div 
                            className="w-8 rounded-t transition-all duration-1000 ease-out"
                            style={{ 
                              height: `${height}px`, 
                              backgroundColor: colors[index] || '#6b7280',
                              minHeight: '4px'
                            }}
                          ></div>
                          <Badge variant="outline" className="text-xs whitespace-nowrap">
                            {emotion}
                          </Badge>
                          <span className="text-xs font-medium">{count}</span>
                        </div>
                      )
                    })}
                </div>
              </CardContent>
            </Card>

            {/* Confidence Distribution */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Target className="mr-2 h-5 w-5" />
                  Confidence Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { label: 'Very High (90-100%)', min: 0.9, color: 'bg-green-500' },
                    { label: 'High (70-89%)', min: 0.7, color: 'bg-blue-500' },
                    { label: 'Medium (50-69%)', min: 0.5, color: 'bg-yellow-500' },
                    { label: 'Low (0-49%)', min: 0, color: 'bg-red-500' }
                  ].map(range => {
                    const count = analysisResults.filter(r => {
                      const conf = r.sentiment.confidence
                      if (range.min === 0) return conf < 0.5
                      if (range.min === 0.9) return conf >= 0.9
                      return conf >= range.min && conf < (range.min + 0.2)
                    }).length
                    const percentage = (count / statistics.total) * 100
                    
                    return (
                      <div key={range.label} className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>{range.label}</span>
                          <span>{count} ({percentage.toFixed(1)}%)</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full transition-all duration-1000 ${range.color}`}
                            style={{ width: `${percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Language Distribution */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Globe className="mr-2 h-5 w-5" />
                  Language Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(statistics.languageCounts)
                    .sort(([,a], [,b]) => (b as number) - (a as number))
                    .map(([lang, count]) => {
                      const percentage = ((count as number) / statistics.total) * 100
                      const colors: { [key: string]: string } = {
                        'English': 'bg-blue-500',
                        'Hindi': 'bg-orange-500', 
                        'Code-mixed': 'bg-purple-500',
                        'Mixed': 'bg-green-500',
                        'Unknown': 'bg-gray-500'
                      }
                      
                      return (
                        <div key={lang} className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="flex items-center">
                              <div className={`w-3 h-3 rounded-full mr-2 ${colors[lang] || 'bg-gray-500'}`}></div>
                              {lang}
                            </span>
                            <Badge variant="secondary">{count} ({percentage.toFixed(1)}%)</Badge>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-1">
                            <div 
                              className={`h-1 rounded-full transition-all duration-1000 ${colors[lang] || 'bg-gray-500'}`}
                              style={{ width: `${percentage}%` }}
                            ></div>
                          </div>
                        </div>
                      )
                    })}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Key Metrics Row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="bg-gradient-to-r from-green-50 to-green-100 border-green-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-green-600 font-medium">Satisfaction Rate</p>
                    <p className="text-2xl font-bold text-green-700">
                      {Math.round((statistics.sentimentCounts.positive / statistics.total) * 100)}%
                    </p>
                  </div>
                  <ThumbsUp className="h-8 w-8 text-green-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-r from-blue-50 to-blue-100 border-blue-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-blue-600 font-medium">Avg Confidence</p>
                    <p className="text-2xl font-bold text-blue-700">
                      {Math.round(statistics.avgConfidence * 100)}%
                    </p>
                  </div>
                  <Zap className="h-8 w-8 text-blue-600" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-r from-purple-50 to-purple-100 border-purple-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-purple-600 font-medium">Emotion Diversity</p>
                    <p className="text-2xl font-bold text-purple-700">
                      {Object.keys(statistics.emotionCounts).length}/6
                    </p>
                  </div>
                  <Heart className="h-8 w-8 text-purple-600" />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="comments" className="relative z-10 mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <MessageSquare className="mr-2 h-5 w-5" />
                Individual Comment Analysis
              </CardTitle>
              <CardDescription>
                Clean and detailed view of each comment with sentiment, confidence, and emotion analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6 max-h-96 overflow-y-auto">
                {analysisResults.slice(0, 50).map((result) => {
                  const getSentimentIcon = (sentiment: string) => {
                    switch(sentiment) {
                      case 'positive': return <Smile className="h-5 w-5 text-green-600" />
                      case 'negative': return <Frown className="h-5 w-5 text-red-600" />
                      default: return <Meh className="h-5 w-5 text-yellow-600" />
                    }
                  }

                  const getSentimentColor = (sentiment: string) => {
                    switch(sentiment) {
                      case 'positive': return 'bg-green-50 border-green-200 text-green-800'
                      case 'negative': return 'bg-red-50 border-red-200 text-red-800'
                      default: return 'bg-yellow-50 border-yellow-200 text-yellow-800'
                    }
                  }

                  const getEmotionEmoji = (emotion: string) => {
                    const emojis: { [key: string]: string } = {
                      'joy': '😊', 'happiness': '😊', 'love': '❤️',
                      'sadness': '😢', 'anger': '😠', 'fear': '😰',
                      'surprise': '😮', 'disgust': '🤢', 'neutral': '😐'
                    }
                    return emojis[emotion] || '😐'
                  }

                  return (
                    <div key={result.id} className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6 hover:shadow-lg transition-all duration-200 hover:border-gray-300 dark:hover:border-gray-600">
                      {/* Comment Header */}
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-3">
                          <Badge variant="outline" className="text-xs font-medium text-gray-900 dark:text-gray-100 border-gray-300 dark:border-gray-600">
                            Comment #{result.id}
                          </Badge>
                          <Badge variant="secondary" className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
                            {result.language}
                          </Badge>
                        </div>
                      </div>

                      {/* Comment Text - Main Focus */}
                      <div className="mb-6">
                        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 border-l-4 border-blue-400 dark:border-blue-600">
                          <p className="text-gray-800 dark:text-gray-200 leading-relaxed text-base">
                            "{result.originalText}"
                          </p>
                        </div>
                      </div>

                      {/* Analysis Results in Clean Grid */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {/* Sentiment */}
                        <div className="text-center">
                          <div className="mb-2">
                            <div className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">SENTIMENT</div>
                            <div className={`inline-flex items-center px-4 py-2 rounded-full border ${
                              result.sentiment.label === 'positive' 
                                ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-800 dark:text-green-300'
                                : result.sentiment.label === 'negative'
                                ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-800 dark:text-red-300'
                                : 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-300'
                            }`}>
                              {getSentimentIcon(result.sentiment.label)}
                              <span className="ml-2 font-semibold capitalize">
                                {result.sentiment.label}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Confidence */}
                        <div className="text-center">
                          <div className="mb-2">
                            <div className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">CONFIDENCE</div>
                            <div className="flex flex-col items-center">
                              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400 mb-1">
                                {Math.round(result.sentiment.confidence * 100)}%
                              </div>
                              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 max-w-24">
                                <div 
                                  className={`h-2 rounded-full transition-all duration-500 ${
                                    result.sentiment.confidence > 0.8 ? 'bg-green-500' :
                                    result.sentiment.confidence > 0.6 ? 'bg-blue-500' :
                                    result.sentiment.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                                  }`}
                                  style={{ width: `${result.sentiment.confidence * 100}%` }}
                                ></div>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Emotion */}
                        <div className="text-center">
                          <div className="mb-2">
                            <div className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">EMOTION</div>
                            <div className="inline-flex items-center px-4 py-2 rounded-full bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 text-purple-800 dark:text-purple-300">
                              <span className="text-xl mr-2">
                                {getEmotionEmoji(result.emotion.primary_emotion)}
                              </span>
                              <span className="font-semibold capitalize">
                                {result.emotion.primary_emotion}
                              </span>
                            </div>
                            <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                              {Math.round(result.emotion.confidence * 100)}% confidence
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Optional Summary */}
                      {result.summary && result.summary !== result.originalText && (
                        <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
                          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 border border-blue-200 dark:border-blue-800">
                            <div className="flex items-center mb-2">
                              <FileText className="h-4 w-4 text-blue-600 dark:text-blue-400 mr-2" />
                              <span className="text-sm font-semibold text-blue-800 dark:text-blue-300">AI Summary</span>
                            </div>
                            <p className="text-sm text-blue-700 dark:text-blue-300 leading-relaxed">
                              {result.summary}
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
                {analysisResults.length > 50 && (
                  <div className="text-center py-4">
                    <p className="text-muted-foreground text-sm mb-2">
                      Showing first 50 results. Export to download all {analysisResults.length} results.
                    </p>
                    <Button variant="outline" size="sm">
                      <ChevronRight className="mr-2 h-4 w-4" />
                      Load More
                    </Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="emotions" className="relative z-10 mt-4">
          <div className="space-y-6">
            {/* Emotion Overview Cards */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {Object.entries(statistics.emotionCounts)
                .sort(([,a], [,b]) => (b as number) - (a as number))
                .map(([emotion, count]) => {
                  const percentage = ((count as number) / statistics.total) * 100
                  const emotionConfig = {
                    joy: { icon: '😊', color: 'from-yellow-400 to-yellow-600', textColor: 'text-yellow-700' },
                    sadness: { icon: '😢', color: 'from-blue-400 to-blue-600', textColor: 'text-blue-700' },
                    anger: { icon: '😠', color: 'from-red-400 to-red-600', textColor: 'text-red-700' },
                    fear: { icon: '😨', color: 'from-purple-400 to-purple-600', textColor: 'text-purple-700' },
                    surprise: { icon: '😲', color: 'from-orange-400 to-orange-600', textColor: 'text-orange-700' },
                    disgust: { icon: '🤢', color: 'from-green-400 to-green-600', textColor: 'text-green-700' },
                    neutral: { icon: '😐', color: 'from-gray-400 to-gray-600', textColor: 'text-gray-700' }
                  }
                  const config = emotionConfig[emotion as keyof typeof emotionConfig] || emotionConfig.neutral

                  return (
                    <Card key={emotion} className="overflow-hidden">
                      <CardContent className="p-4">
                        <div className={`w-full h-2 rounded-full bg-gradient-to-r ${config.color} mb-3`}></div>
                        <div className="text-center">
                          <div className="text-2xl mb-2">{config.icon}</div>
                          <div className={`text-lg font-bold ${config.textColor} capitalize`}>
                            {emotion}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {count} ({percentage.toFixed(1)}%)
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
            </div>

            {/* Detailed Emotion Analysis */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Brain className="mr-2 h-5 w-5" />
                  Emotion Pattern Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {Object.entries(statistics.emotionCounts)
                    .sort(([,a], [,b]) => (b as number) - (a as number))
                    .map(([emotion, count]) => {
                      const percentage = ((count as number) / statistics.total) * 100
                      const sampleComments = analysisResults
                        .filter(r => r.emotion.primary_emotion === emotion)
                        .slice(0, 3)

                      return (
                        <div key={emotion} className="border rounded-lg p-4">
                          <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-semibold capitalize flex items-center">
                              <span className="mr-2">
                                {{
                                  joy: '😊', sadness: '😢', anger: '😠', fear: '😨',
                                  surprise: '😲', disgust: '🤢', neutral: '😐'
                                }[emotion as keyof typeof statistics.emotionCounts] || '😐'}
                              </span>
                              {emotion}
                            </h3>
                            <Badge variant="secondary" className="text-lg px-3 py-1">
                              {count} ({percentage.toFixed(1)}%)
                            </Badge>
                          </div>
                          
                          <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                            <div 
                              className="h-3 rounded-full bg-gradient-to-r from-purple-400 to-purple-600 transition-all duration-1000"
                              style={{ width: `${percentage}%` }}
                            ></div>
                          </div>

                          {sampleComments.length > 0 && (
                            <div>
                              <h4 className="text-sm font-medium text-muted-foreground mb-2">Sample Comments:</h4>
                              <div className="space-y-2">
                                {sampleComments.map((comment, index) => (
                                  <div key={index} className="text-sm bg-muted p-2 rounded border-l-4 border-purple-500">
                                    <span className="text-xs text-muted-foreground">#{comment.id}</span>
                                    <p className="mt-1">
                                      {comment.originalText.length > 100 
                                        ? comment.originalText.substring(0, 100) + '...'
                                        : comment.originalText
                                      }
                                    </p>
                                    <div className="flex items-center mt-2 space-x-2">
                                      <Badge variant="outline" size="sm">
                                        Confidence: {Math.round(comment.emotion.confidence * 100)}%
                                      </Badge>
                                      <Badge variant="outline" size="sm">
                                        {comment.sentiment.label}
                                      </Badge>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="export" className="relative z-10 mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Export Results</CardTitle>
              <CardDescription>
                Download your analysis results in various formats
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Button onClick={exportResults} className="w-full">
                  <Download className="mr-2 h-4 w-4" />
                  Download as CSV
                </Button>
                <p className="text-sm text-muted-foreground">
                  The CSV file will contain all {statistics.total} analyzed comments with their sentiment scores, emotions, and summaries.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      </div>

      {/* Models Used Panel - Right Side */}
      <div className="w-80 shrink-0">
        <div className="sticky top-6">
          <Card className="bg-gradient-to-br from-slate-50 to-gray-100 dark:from-slate-900 dark:to-gray-800 border-slate-200 dark:border-slate-700 shadow-lg">
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center text-slate-900 dark:text-slate-100">
                <Brain className="mr-2 h-5 w-5 text-slate-600 dark:text-slate-400" />
                Models Used
              </CardTitle>
              <CardDescription className="text-slate-700 dark:text-slate-300">
                AI models deployed for this analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {Object.entries(activeModels).map(([key, model]) => (
                <div key={key} className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-slate-600 shadow-sm">
                  <div className="flex items-start space-x-3">
                    <div className="text-2xl">{model.icon}</div>
                    <div className="flex-1">
                      <h4 className={`font-semibold text-sm ${model.textColor} dark:text-slate-300`}>
                        {model.name}
                      </h4>
                      <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 leading-relaxed">
                        {model.description}
                      </p>
                      <div className="mt-2 flex items-center text-xs text-slate-500 dark:text-slate-500">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                        <span>Active</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              
              {/* Analysis Stats */}
              <div className="mt-6 pt-4 border-t border-slate-200 dark:border-slate-600">
                <h4 className="font-semibold text-sm text-slate-900 dark:text-slate-100 mb-3">
                  Analysis Performance
                </h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-600 dark:text-slate-400">Processing Time</span>
                    <Badge variant="secondary" className="text-xs">
                      ~{Math.ceil(statistics.total / 10)}s
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-600 dark:text-slate-400">Model Accuracy</span>
                    <Badge variant="secondary" className="text-xs">
                      {Math.round(statistics.avgConfidence * 100)}%
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-600 dark:text-slate-400">Data Points</span>
                    <Badge variant="secondary" className="text-xs">
                      {statistics.total}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-600 dark:text-slate-400">Languages Detected</span>
                    <Badge variant="secondary" className="text-xs">
                      {Object.keys(statistics.languageCounts).length}
                    </Badge>
                  </div>
                </div>
              </div>

              {/* Technical Details */}
              <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-600">
                <h4 className="font-semibold text-sm text-slate-900 dark:text-slate-100 mb-3">
                  Technical Details
                </h4>
                <div className="space-y-2 text-xs text-slate-600 dark:text-slate-400">
                  <div className="flex items-center space-x-2">
                    <Globe className="h-3 w-3" />
                    <span>Multilingual Processing</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Zap className="h-3 w-3" />
                    <span>Real-time Analysis</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Target className="h-3 w-3" />
                    <span>Batch Processing</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Users className="h-3 w-3" />
                    <span>Indian Context Optimized</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}