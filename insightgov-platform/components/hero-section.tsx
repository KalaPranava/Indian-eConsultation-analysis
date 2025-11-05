"use client"

import { Button } from "@/components/ui/button"
import { ArrowDown, ArrowRight } from "lucide-react"
import { SplitText } from "./split-text"
import Link from "next/link"

export function HeroSection() {
  const scrollToFeatures = () => {
    document.getElementById("features")?.scrollIntoView({ behavior: "smooth" })
  }

  return (
    <section className="min-h-[70vh] sm:min-h-[82vh] flex items-center justify-center relative overflow-hidden gradient-bg pt-20 sm:pt-24 pb-12 sm:pb-16">
      <div className="absolute inset-0 grid-pattern opacity-20 pointer-events-none"></div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
        <div className="space-y-4 sm:space-y-6 md:space-y-8">
          <h1 className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl font-bold text-balance leading-tight">
            <SplitText text="InsightGov:" className="text-primary" />{" "}
            <SplitText text="Turning Voices into Action" className="text-foreground" delay={400} />
          </h1>

          <div className="text-base sm:text-lg md:text-xl lg:text-2xl text-muted-foreground max-w-4xl mx-auto text-pretty leading-relaxed px-2">
            <SplitText text="When people speak, data listens." className="text-muted-foreground" delay={800} />
            <br />
            <SplitText text="When data speaks, policies improve." className="text-muted-foreground" delay={1200} />
          </div>

          <div className="pt-4 sm:pt-6 opacity-0 animate-in fade-in slide-in-from-bottom-4 duration-1000 delay-1600 flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center px-4">
            <Button
              size="lg"
              onClick={scrollToFeatures}
              variant="outline"
              className="text-sm sm:text-base md:text-lg px-4 sm:px-6 md:px-8 py-4 sm:py-5 md:py-6 w-full sm:w-auto"
            >
              <span className="hidden sm:inline">Know us before getting started</span>
              <span className="sm:hidden">Learn More</span>
              <ArrowDown className="ml-2 h-4 w-4 sm:h-5 sm:w-5" />
            </Button>
            <Button
              asChild
              size="lg"
              className="text-sm sm:text-base md:text-lg px-4 sm:px-6 md:px-8 py-4 sm:py-5 md:py-6 bg-primary hover:bg-primary/90 text-primary-foreground w-full sm:w-auto"
            >
              <Link href="/dashboard" aria-label="Start Analysis Now" prefetch>
                <span className="inline-flex items-center">Start Analysis<ArrowRight className="ml-2 h-4 w-4 sm:h-5 sm:w-5" /></span>
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}
