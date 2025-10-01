import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Twitter, RedoDot as Reddit, Youtube, Brain, BarChart3 } from "lucide-react"

export function OperationalAdvantages() {
  return (
    <section id="operational-advantages" aria-labelledby="operational-advantages-heading" className="py-16 sm:py-24">
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
          <Card className="bg-card text-card-foreground border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:shadow-xl hover:shadow-primary/10 hover:-translate-y-2 hover:scale-105">
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

          <Card className="bg-card text-card-foreground border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:shadow-xl hover:shadow-primary/10 hover:-translate-y-2 hover:scale-105">
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

          <Card className="bg-card text-card-foreground border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:shadow-xl hover:shadow-primary/10 hover:-translate-y-2 hover:scale-105">
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
      </div>
    </section>
  )
}
