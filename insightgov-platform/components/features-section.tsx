"use client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart3, Cloud, FileText, ArrowDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useCallback } from "react"

export function FeaturesSection() {
  const scrollToGoals = useCallback(() => {
    document.getElementById("goals")?.scrollIntoView({ behavior: "smooth" })
  }, [])
  const features = [
    {
      icon: BarChart3,
      title: "Sentiment Analysis",
      description:
        "Advanced AI algorithms analyze citizen feedback to determine emotional tone and public sentiment trends, providing clear insights into community opinions.",
    },
    {
      icon: Cloud,
      title: "Word Cloud",
      description:
        "Visual representation of the most frequently mentioned topics and concerns, making it easy to identify key themes in public discourse.",
    },
    {
      icon: FileText,
      title: "Summary Generation",
      description:
        "Automatically generate concise summaries of large volumes of feedback, highlighting key points and actionable recommendations for policymakers.",
    },
  ]

  return (
  <section id="features" className="pt-16 pb-20 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-14">
          <Button
            onClick={scrollToGoals}
            size="lg"
            className="mb-6 inline-flex items-center gap-2 px-8 py-6 text-base md:text-lg bg-primary hover:bg-primary/90 text-primary-foreground shadow-sm hover:shadow transition-all group"
          >
            Know us before getting started
            <ArrowDown className="h-5 w-5 transition-transform duration-300 group-hover:translate-y-1" />
          </Button>
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Core Features</h2>
          <p className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            Powerful AI tools designed to transform citizen feedback into actionable insights
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <Card key={index} className="bg-card border-border hover:border-primary/50 transition-colors">
              <CardHeader className="text-center">
                <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4">
                  <feature.icon className="h-8 w-8 text-primary" />
                </div>
                <CardTitle className="text-xl text-card-foreground">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-center text-muted-foreground leading-relaxed">
                  {feature.description}
                </CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  )
}
