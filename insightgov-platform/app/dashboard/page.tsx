import { FileUpload } from "@/components/file-upload"
import { DemoAnalysis } from "@/components/demo-analysis"

export default function DashboardPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="pb-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Upload & Analyze</h1>
            <p className="text-xl text-muted-foreground">
              Upload your data to begin sentiment analysis and generate insights
            </p>
          </div>

          <FileUpload />

          {/* Demo Analysis Section */}
          <div id="demo" className="mt-16">
            <DemoAnalysis />
          </div>
        </div>
      </div>
    </main>
  )
}
