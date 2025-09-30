import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BarChart3, Cloud, FileText, TrendingUp, Brain, Sparkles, Upload, Eye } from "lucide-react"
import Link from "next/link"

export function DashboardPreview() {
  return (
    <section className="py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Interactive Dashboard</h2>
          <p className="text-xl text-muted-foreground mb-8">
            Upload comments, visualize trends, and explore AI-powered insights.
          </p>
          <Link href="/dashboard">
            <Button size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground text-lg px-8 py-6">
              Start Analysis Now
              <Upload className="ml-2 h-5 w-5" />
            </Button>
          </Link>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mt-16">
          <Card className="bg-card border-border hover:border-primary/50 transition-colors">
            <CardHeader className="text-center">
              <Upload className="h-12 w-12 text-primary mx-auto mb-4" />
              <CardTitle className="text-card-foreground">Upload & Process</CardTitle>
            </CardHeader>
            <CardContent className="text-center space-y-3">
              <CardDescription>Upload CSV, JSON, or TXT files with citizen feedback</CardDescription>
              <div className="flex flex-wrap justify-center gap-1">
                <Badge variant="secondary">CSV</Badge>
                <Badge variant="secondary">JSON</Badge>
                <Badge variant="secondary">TXT</Badge>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card border-border hover:border-primary/50 transition-colors">
            <CardHeader className="text-center">
              <Brain className="h-12 w-12 text-primary mx-auto mb-4" />
              <CardTitle className="text-card-foreground">AI Analysis</CardTitle>
            </CardHeader>
            <CardContent className="text-center space-y-3">
              <CardDescription>Advanced ML models analyze sentiment and emotions</CardDescription>
              <div className="flex flex-wrap justify-center gap-1">
                <Badge variant="outline">DistilBERT</Badge>
                <Badge variant="outline">VADER</Badge>
                <Badge variant="outline">TF-IDF</Badge>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card border-border hover:border-primary/50 transition-colors">
            <CardHeader className="text-center">
              <Sparkles className="h-12 w-12 text-primary mx-auto mb-4" />
              <CardTitle className="text-card-foreground">Rich Insights</CardTitle>
            </CardHeader>
            <CardContent className="text-center space-y-3">
              <CardDescription>Interactive charts, summaries, and actionable reports</CardDescription>
              <div className="flex flex-wrap justify-center gap-1">
                <Badge variant="secondary">Charts</Badge>
                <Badge variant="secondary">Export</Badge>
                <Badge variant="secondary">Summary</Badge>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Feature Highlight */}
        <div className="mt-16 text-center">
          <Card className="max-w-4xl mx-auto bg-gradient-to-r from-primary/5 to-secondary/5 border-primary/20">
            <CardContent className="p-8">
              <div className="flex items-center justify-center mb-4">
                <TrendingUp className="h-8 w-8 text-primary mr-3" />
                <h3 className="text-2xl font-bold">Real-time Processing</h3>
              </div>
              <p className="text-lg text-muted-foreground mb-6">
                Process thousands of comments in seconds with our optimized ML pipeline. 
                Get instant insights with confidence scores and detailed breakdowns.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="flex items-center justify-center space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span>99.2% Accuracy</span>
                </div>
                <div className="flex items-center justify-center space-x-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span>Multi-language Support</span>
                </div>
                <div className="flex items-center justify-center space-x-2">
                  <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                  <span>Batch Processing</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
