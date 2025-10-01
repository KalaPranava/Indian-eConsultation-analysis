import { HeroSection } from "@/components/hero-section"
import { FeaturesSection } from "@/components/features-section"
import { GoalsSection } from "@/components/goals-section"
import { DashboardPreview } from "@/components/dashboard-preview"
import { ContactSection } from "@/components/contact-section"
import { OperationalAdvantages } from "@/components/operational-advantages"

export default function HomePage() {
  return (
    <main className="min-h-screen">
      {/* Tubelight navbar renders globally in layout */}
      <HeroSection />
      <FeaturesSection />
      <OperationalAdvantages />
      <GoalsSection />
      <DashboardPreview />
      <ContactSection />
    </main>
  )
}
