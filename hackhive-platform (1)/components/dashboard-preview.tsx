import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, TrendingUp, Eye } from "lucide-react"
import Link from "next/link"

export function DashboardPreview() {
  return (
    <section id="dashboard" className="py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Interactive Dashboard</h2>
          <p className="text-xl text-muted-foreground mb-8">
            Upload comments, visualize trends, and explore AI-powered insights.
          </p>
          <Link href="/dashboard">
            <Button
              size="lg"
              className="bg-primary hover:bg-primary/90 text-primary-foreground cursor-pointer hover:scale-105 hover:-translate-y-1 transition-all duration-300 shadow-lg hover:shadow-xl"
            >
              Get Started
            </Button>
          </Link>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mt-16">
          <Card className="bg-card border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:shadow-lg hover:-translate-y-1 hover:scale-105">
            <CardHeader className="text-center">
              <Upload className="h-12 w-12 text-primary mx-auto mb-4" />
              <CardTitle className="text-card-foreground">Upload Data</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="text-center">Upload structured feedback data for analysis</CardDescription>
            </CardContent>
          </Card>

          <Card className="bg-card border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:shadow-lg hover:-translate-y-1 hover:scale-105">
            <CardHeader className="text-center">
              <TrendingUp className="h-12 w-12 text-primary mx-auto mb-4" />
              <CardTitle className="text-card-foreground">Analyze Trends</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="text-center">
                Discover patterns and sentiment trends in real-time
              </CardDescription>
            </CardContent>
          </Card>

          <Card className="bg-card border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:shadow-lg hover:-translate-y-1 hover:scale-105">
            <CardHeader className="text-center">
              <Eye className="h-12 w-12 text-primary mx-auto mb-4" />
              <CardTitle className="text-card-foreground">Visualize Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="text-center">Generate actionable reports and visualizations</CardDescription>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
