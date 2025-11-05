import { HeroSection } from "@/components/hero-section"
import { FeaturesSection } from "@/components/features-section"
import { GoalsSection } from "@/components/goals-section"
import { DashboardPreview } from "@/components/dashboard-preview"
import { OperationalAdvantagesWrapper } from "@/components/operational-advantages-wrapper"

export default function HomePage() {
  return (
    <main className="min-h-screen">
      <HeroSection />
      <FeaturesSection />
      <GoalsSection />
  <DashboardPreview />
  <OperationalAdvantagesWrapper />
    </main>
  )
}
