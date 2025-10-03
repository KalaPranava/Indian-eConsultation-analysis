import { HeroSection } from "@/components/hero-section"
import { FeaturesSection } from "@/components/features-section"
import { GoalsSection } from "@/components/goals-section"
import { DashboardPreview } from "@/components/dashboard-preview"
import { OperationalAdvantages } from "@/components/operational-advantages"

export default function HomePage() {
  return (
    <main className="min-h-screen">
      <HeroSection />
      <FeaturesSection />
      <GoalsSection />
  <DashboardPreview />
  <OperationalAdvantages />
    </main>
  )
}
