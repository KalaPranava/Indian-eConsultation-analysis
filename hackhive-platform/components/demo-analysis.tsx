"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Play, FileText, Download, Users, Calendar, Sparkles, ArrowRight } from "lucide-react"

export function DemoAnalysis() {
  const [isLoading, setIsLoading] = useState(false)

  const handleDemoAnalysis = async () => {
    setIsLoading(true)
    
    try {
      // Fetch the demo CSV file from the public directory
      const response = await fetch('/policy_feedback.csv')
      const csvText = await response.text()
      
      // Store the demo file data in localStorage
      const demoFileData = {
        name: 'policy_feedback.csv',
        content: csvText,
        type: 'text/csv',
        size: csvText.length
      }
      
      localStorage.setItem('analysisFile', JSON.stringify(demoFileData))
      
      // Navigate to analysis page
      window.location.href = '/dashboard/analysis'
    } catch (error) {
      console.error('Failed to load demo file:', error)
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Divider */}
      <div className="flex items-center justify-center">
        <div className="flex-1 border-t border-border"></div>
        <div className="px-4 text-sm text-muted-foreground font-medium">OR</div>
        <div className="flex-1 border-t border-border"></div>
      </div>

      {/* Demo Analysis Card */}
      <Card className="bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-950 dark:to-indigo-900 border-blue-200 dark:border-blue-800 relative z-[55]">
        <CardHeader className="text-center">
          <div className="mx-auto w-16 h-16 bg-blue-500/10 rounded-full flex items-center justify-center mb-4">
            <Sparkles className="h-8 w-8 text-blue-600 dark:text-blue-400" />
          </div>
          <CardTitle className="text-blue-900 dark:text-blue-100 text-2xl">
            Try Demo Analysis
          </CardTitle>
          <CardDescription className="text-blue-700 dark:text-blue-300 text-lg">
            Evaluate our system instantly with real policy feedback data
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {/* Sample Data Info */}
          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 border border-blue-200 dark:border-blue-700">
            <div className="flex items-start space-x-4">
              <FileText className="h-6 w-6 text-blue-600 dark:text-blue-400 mt-1 flex-shrink-0" />
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
                  Sample Dataset: Policy Feedback Analysis
                </h3>
                <p className="text-gray-700 dark:text-gray-300 text-sm mb-4">
                  Real citizen feedback on government policy initiatives from various stakeholders including citizens, experts, academia, and journalists.
                </p>
                
                {/* Dataset Stats */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
                  <div className="flex items-center space-x-2">
                    <Users className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">300+ Comments</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Calendar className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">Recent Data</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <FileText className="h-4 w-4 text-blue-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">CSV Format</span>
                  </div>
                </div>

                {/* Stakeholder Types */}
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary" className="text-xs">Citizens</Badge>
                  <Badge variant="secondary" className="text-xs">Academia</Badge>
                  <Badge variant="secondary" className="text-xs">Experts</Badge>
                  <Badge variant="secondary" className="text-xs">Journalists</Badge>
                  <Badge variant="secondary" className="text-xs">Government</Badge>
                </div>
              </div>
            </div>
          </div>

          {/* What You'll See */}
          <div className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg p-4 border border-blue-200 dark:border-blue-700">
            <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3 flex items-center">
              <ArrowRight className="h-4 w-4 mr-2 text-blue-500" />
              What you'll analyze:
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm text-gray-700 dark:text-gray-300">
              <div>• Sentiment distribution across stakeholders</div>
              <div>• Emotional analysis of feedback</div>
              <div>• Word cloud of key terms</div>
              <div>• Comprehensive summary insights</div>
            </div>
          </div>

          {/* Action Button */}
          <div className="text-center pt-4">
            <Button
              onClick={handleDemoAnalysis}
              disabled={isLoading}
              size="lg"
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-6 text-lg shadow-lg hover:shadow-xl transition-all duration-200 relative z-[60]"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                  Loading Demo...
                </>
              ) : (
                <>
                  <Play className="mr-3 h-5 w-5" />
                  Analyze Using This File
                </>
              )}
            </Button>
            
            <p className="text-sm text-blue-600 dark:text-blue-400 mt-3">
              No upload required • Instant analysis • Full feature showcase
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}