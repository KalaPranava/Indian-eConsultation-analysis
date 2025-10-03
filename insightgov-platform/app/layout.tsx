import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import { Suspense } from "react"
import { TubelightNavBar } from "@/components/tubelight-navbar"
import Link from "next/link"
import "./globals.css"

export const metadata: Metadata = {
  title: "InsightGov - Turning Voices into Action",
  description: "AI-powered sentiment analysis platform for citizen feedback and policy insights",
  generator: "v0.app",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable} antialiased`}>
        {/* Brand logo (restored) */}
  <div className="fixed top-[43.5px] left-8 z-50">
          <Link href="/" className="text-2xl font-bold tracking-tight text-primary drop-shadow-sm hover:opacity-90 transition-opacity">
            InsightGov
          </Link>
        </div>
        <TubelightNavBar />
  <div className="pt-[130px] pb-24">{/* reduced top padding to tighten space before sections */}
          <Suspense fallback={null}>{children}</Suspense>
        </div>
        <Analytics />
      </body>
    </html>
  )
}
