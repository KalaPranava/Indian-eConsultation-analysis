import { Navigation } from "@/components/navigation"
import { HeroSection } from "@/components/hero-section"
import { FeaturesSection } from "@/components/features-section"
import { GoalsSection } from "@/components/goals-section"
import { DashboardPreview } from "@/components/dashboard-preview"

export default function HomePage() {
  return (
    <main className="min-h-screen">
      <Navigation />
      <HeroSection />
      <FeaturesSection />
      <GoalsSection />
      <DashboardPreview />
    </main>
  )
}
