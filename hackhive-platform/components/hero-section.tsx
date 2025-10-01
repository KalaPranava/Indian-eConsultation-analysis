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
    <section className="min-h-[82vh] flex items-center justify-center relative overflow-hidden gradient-bg">
      <div className="absolute inset-0 grid-pattern opacity-20 pointer-events-none"></div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
        <div className="space-y-8">
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold text-balance">
            <SplitText text="HackHive:" className="text-primary" />{" "}
            <SplitText text="Turning Voices into Action" className="text-foreground" delay={400} />
          </h1>

          <div className="text-xl md:text-2xl text-muted-foreground max-w-4xl mx-auto text-pretty leading-relaxed">
            <SplitText text="When people speak, data listens." className="text-muted-foreground" delay={800} />
            <br />
            <SplitText text="When data speaks, policies improve." className="text-muted-foreground" delay={1200} />
          </div>

          <div className="pt-6 opacity-0 animate-in fade-in slide-in-from-bottom-4 duration-1000 delay-1600 flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              size="lg"
              onClick={scrollToFeatures}
              variant="outline"
              className="text-lg px-8 py-6"
            >
              Know us before getting started
              <ArrowDown className="ml-2 h-5 w-5" />
            </Button>
            <Button
              asChild
              size="lg"
              className="text-lg px-8 py-6 bg-primary hover:bg-primary/90 text-primary-foreground"
            >
              <Link href="/dashboard" aria-label="Start Analysis Now" prefetch>
                <span className="inline-flex items-center">Start Analysis Now<ArrowRight className="ml-2 h-5 w-5" /></span>
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}
