"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Twitter, Youtube, Brain, BarChart3, TrendingUp, MessageSquare, Users, Target, Eye } from "lucide-react"
import { SiReddit as Reddit } from "react-icons/si"

// NOTE: Imported from duplicate folder 'hackhive-platform (1)' and adapted (Reddit icon swapped with react-icons)
export function OperationalAdvantages() {
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null)

  useEffect(() => {
    // Helper to open feature and smoothly scroll to section
    const openFeature = (feature: 'policy' | 'comparative' | 'social') => {
      setSelectedFeature(feature)
      // Scroll after a tiny delay to allow layout to settle
      setTimeout(() => {
        document
          .getElementById('operational-advantages')
          ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }, 60)
    }

    // 1) Listen for custom preview events (in-page clicks)
    const handlePreview = (e: any) => {
      const feature = e?.detail?.feature as 'policy' | 'comparative' | 'social' | undefined
      if (feature) openFeature(feature)
    }

    // 2) Handle URL hash for cross-route navigation and refresh
    const handleHash = () => {
      const hash = window.location.hash
      if (hash === '#preview-policy') openFeature('policy')
      else if (hash === '#preview-comparative') openFeature('comparative')
      else if (hash === '#preview-social') openFeature('social')
    }

    // Mount: attach listeners and check current hash
    window.addEventListener('openPreview', handlePreview)
    window.addEventListener('hashchange', handleHash)
    handleHash()

    return () => {
      window.removeEventListener('openPreview', handlePreview)
      window.removeEventListener('hashchange', handleHash)
    }
  }, [])

  const features = {
    social: {
      title: "Social Media Analysis",
      description: "360° sentiment across major platforms. Real‑time trends, topics, and influencer signals.",
      preview: {
        stats: [
          { label: "Platforms Monitored", value: "3+", icon: Users },
          { label: "Real-time Updates", value: "24/7", icon: TrendingUp },
          { label: "Sentiment Accuracy", value: "94%", icon: Target },
        ],
        features: [
          "Track sentiment across Twitter, Reddit, and YouTube simultaneously",
          "Identify trending topics and viral discussions in real-time",
          "Detect influencer opinions and their impact on public sentiment",
          "Cross-platform mood tracking with unified dashboard",
          "Automated alert system for sentiment shifts",
        ],
        useCase: "Monitor public reaction to a new policy announcement across all major social platforms within minutes, identifying key concerns and positive feedback patterns."
      }
    },
    policy: {
      title: "Policy Amendment Suggestor",
      description: "Detects gaps and recommends targeted edits. Improve clarity and fairness to lift feedback.",
      preview: {
        stats: [
          { label: "Gap Detection", value: "AI-Powered", icon: Brain },
          { label: "Suggestions", value: "Actionable", icon: MessageSquare },
          { label: "Improvement", value: "87%", icon: TrendingUp },
        ],
        features: [
          "Automatic detection of ambiguous or unclear policy language",
          "Identify sections receiving the most negative feedback",
          "Generate specific, actionable amendment recommendations",
          "Priority ranking based on sentiment impact score",
          "Side-by-side comparison of original vs suggested text",
        ],
        useCase: "Automatically analyze a 50-page policy document, identify the 5 most problematic sections based on public feedback, and receive specific amendment suggestions with improved wording to address citizen concerns effectively."
      }
    },
    comparative: {
      title: "Comparative Analysis",
      description: "Compare new vs previous policies with clear visuals. See shifts in public opinion at a glance.",
      preview: {
        stats: [
          { label: "Visual Comparison", value: "Charts", icon: BarChart3 },
          { label: "Time Period", value: "Before/After", icon: TrendingUp },
          { label: "Analysis Depth", value: "Multi-level", icon: Target },
        ],
        features: [
          "Side-by-side sentiment comparison of current vs previous policy versions",
          "Track sentiment evolution over time with interactive timeline charts",
          "Compare feedback patterns across different regions and demographics",
          "Visualize topic shifts and emerging themes between policy iterations",
          "Generate comprehensive comparison reports with actionable insights",
        ],
        useCase: "Compare public sentiment for the 2024 vs 2025 education policy across 15 states, showing a 23% increase in positive feedback and 18% decrease in negative comments after incorporating suggested amendments from previous e-consultation rounds."
      }
    }
  }

  return (
    <section id="operational-advantages" aria-labelledby="operational-advantages-heading" className="py-16 sm:py-24 scroll-mt-28">
      <div className="mx-auto max-w-6xl px-4">
        <header className="mb-10 sm:mb-12 text-center">
          <h2
            id="operational-advantages-heading"
            className="text-balance text-3xl font-semibold tracking-tight text-foreground sm:text-4xl"
          >
            Operational Advantages
          </h2>
          <p className="mt-3 text-pretty text-base text-muted-foreground sm:text-lg">
            Unlock practical benefits that streamline analysis, inform policy improvements, and clarify public sentiment
            with clear, actionable insights.
          </p>
        </header>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <Card 
            className="bg-card text-card-foreground border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:shadow-xl hover:shadow-primary/10 hover:-translate-y-2 hover:scale-105 relative group"
            onClick={() => setSelectedFeature('social')}
          >
            <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/30">
                <Eye className="w-3 h-3 mr-1" />
                Preview
              </Badge>
            </div>
            <CardHeader className="text-center">
              <div className="mb-4 flex items-center justify-center gap-4">
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center transition-all duration-300 hover:bg-primary/20 hover:scale-110">
                  <Twitter aria-hidden="true" className="h-8 w-8 text-primary" />
                </div>
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center transition-all duration-300 hover:bg-primary/20 hover:scale-110">
                  <Reddit aria-hidden="true" className="h-8 w-8 text-primary" />
                </div>
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center transition-all duration-300 hover:bg-primary/20 hover:scale-110">
                  <Youtube aria-hidden="true" className="h-8 w-8 text-primary" />
                </div>
                <span className="sr-only">Twitter, Reddit, and YouTube</span>
              </div>
              <CardTitle className="text-xl">Social Media Analysis</CardTitle>
              <CardDescription>
                360° sentiment across major platforms. Real‑time trends, topics, and influencer signals.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc pl-5 text-sm text-muted-foreground">
                <li>Cross‑platform mood tracking</li>
                <li>Trend and influencer insights</li>
              </ul>
            </CardContent>
          </Card>

          <Card 
            className="bg-card text-card-foreground border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:shadow-xl hover:shadow-primary/10 hover:-translate-y-2 hover:scale-105 relative group"
            onClick={() => setSelectedFeature('policy')}
          >
            <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/30">
                <Eye className="w-3 h-3 mr-1" />
                Preview
              </Badge>
            </div>
            <CardHeader className="text-center">
              <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4 transition-all duration-300 hover:bg-primary/20 hover:scale-110">
                <Brain aria-hidden="true" className="h-8 w-8 text-primary" />
                <span className="sr-only">AI Brain</span>
              </div>
              <CardTitle className="text-xl">Policy Amendment Suggestor</CardTitle>
              <CardDescription>
                Detects gaps and recommends targeted edits. Improve clarity and fairness to lift feedback.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc pl-5 text-sm text-muted-foreground">
                <li>Prioritized drawback detection</li>
                <li>Actionable amendment suggestions</li>
              </ul>
            </CardContent>
          </Card>

          <Card 
            className="bg-card text-card-foreground border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:shadow-xl hover:shadow-primary/10 hover:-translate-y-2 hover:scale-105 relative group"
            onClick={() => setSelectedFeature('comparative')}
          >
            <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
              <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/30">
                <Eye className="w-3 h-3 mr-1" />
                Preview
              </Badge>
            </div>
            <CardHeader className="text-center">
              <div className="mb-4 flex items-center justify-center gap-4">
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center transition-all duration-300 hover:bg-primary/20 hover:scale-110">
                  <BarChart3 aria-hidden="true" className="h-8 w-8 text-primary" />
                </div>
                <span aria-hidden="true" className="text-sm font-semibold text-primary">
                  VS
                </span>
                <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center transition-all duration-300 hover:bg-primary/20 hover:scale-110">
                  <BarChart3 aria-hidden="true" className="h-8 w-8 text-primary" />
                </div>
              </div>
              <CardTitle className="text-xl">Comparative Analysis</CardTitle>
              <CardDescription>
                Compare new vs previous policies with clear visuals. See shifts in public opinion at a glance.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc pl-5 text-sm text-muted-foreground">
                <li>Before/after sentiment snapshots</li>
                <li>Topic & region comparisons</li>
              </ul>
            </CardContent>
          </Card>
        </div>

        {/* Feature Preview Modals */}
        {selectedFeature && (
          <Dialog open={!!selectedFeature} onOpenChange={() => setSelectedFeature(null)}>
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
              <DialogHeader className="space-y-3">
                <DialogTitle className="text-xl sm:text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  {features[selectedFeature as keyof typeof features].title}
                </DialogTitle>
                <DialogDescription className="text-sm sm:text-base text-left leading-relaxed">
                  {features[selectedFeature as keyof typeof features].description}
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-6 mt-4">
                {/* Stats Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
                  {features[selectedFeature as keyof typeof features].preview.stats.map((stat, idx) => {
                    const Icon = stat.icon
                    return (
                      <div key={idx} className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950 dark:to-pink-950 rounded-lg p-4 border border-purple-200 dark:border-purple-800 min-h-[110px]">
                        <div className="flex flex-col items-center justify-center text-center h-full gap-2">
                          <div className="flex-shrink-0">
                            <Icon className="w-7 h-7 text-purple-600 dark:text-purple-400 mx-auto" />
                          </div>
                          <div className="space-y-1">
                            <div className="text-base sm:text-lg font-bold text-purple-600 dark:text-purple-400 break-words">{stat.value}</div>
                            <div className="text-xs text-muted-foreground leading-tight break-words">{stat.label}</div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Features List */}
                <div>
                  <h4 className="font-semibold text-lg mb-4 flex items-center">
                    <Target className="w-5 h-5 mr-2 text-purple-600" />
                    Key Features
                  </h4>
                  <ul className="space-y-3">
                    {features[selectedFeature as keyof typeof features].preview.features.map((feature, idx) => (
                      <li key={idx} className="flex items-start gap-3">
                        <span className="inline-block w-2 h-2 rounded-full bg-gradient-to-r from-purple-600 to-pink-600 mt-2 flex-shrink-0"></span>
                        <span className="text-sm text-foreground leading-relaxed">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Use Case */}
                <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg p-[1px]">
                  <div className="bg-background rounded-lg p-5">
                    <h4 className="font-semibold text-lg mb-3 flex items-center">
                      <MessageSquare className="w-5 h-5 mr-2 text-purple-600" />
                      Real-World Use Case
                    </h4>
                    <p className="text-sm text-foreground leading-relaxed">
                      {features[selectedFeature as keyof typeof features].preview.useCase}
                    </p>
                  </div>
                </div>

                {/* CTA */}
                <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950/50 dark:to-pink-950/50 rounded-lg p-5 text-center border border-purple-200 dark:border-purple-800">
                  <p className="text-sm sm:text-base font-medium text-purple-700 dark:text-purple-300 leading-relaxed">
                    ✨ Try this feature in the demo dashboard above!
                  </p>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        )}
      </div>
    </section>
  )
}