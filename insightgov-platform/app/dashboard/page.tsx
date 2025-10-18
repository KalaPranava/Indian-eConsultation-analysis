import { FileUpload } from "@/components/file-upload"
import { DemoAnalysis } from "@/components/demo-analysis"

export default function DashboardPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="pb-8 sm:pb-12 md:pb-16">
        <div className="max-w-7xl mx-auto px-3 sm:px-4 md:px-6 lg:px-8">
          <div className="text-center mb-6 sm:mb-8 md:mb-12">
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-foreground mb-2 sm:mb-4">Upload & Analyze</h1>
            <p className="text-base sm:text-lg md:text-xl text-muted-foreground px-2">
              Upload your data to begin sentiment analysis and generate insights
            </p>
          </div>

          {/* Side-by-side layout for desktop, stacked for mobile */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 md:gap-8 items-start">
            {/* File Upload Section */}
            <div className="w-full">
              <FileUpload />
            </div>

            {/* Demo Analysis Section */}
            <div className="w-full">
              <DemoAnalysis />
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
