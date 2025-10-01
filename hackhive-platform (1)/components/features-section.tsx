import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart3, Cloud, FileText } from "lucide-react"

export function FeaturesSection() {
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
    <section id="features" className="py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Core Features</h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Powerful AI tools designed to transform citizen feedback into actionable insights
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <Card
              key={index}
              className="bg-card border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:shadow-xl hover:shadow-primary/10 hover:-translate-y-2 hover:scale-105"
            >
              <CardHeader className="text-center">
                <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4 transition-all duration-300 hover:bg-primary/20 hover:scale-110">
                  <feature.icon className="h-8 w-8 text-primary transition-all duration-300" />
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
