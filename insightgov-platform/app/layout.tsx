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
  description: "AI-powered sentiment analysis platform for citizen feedback and policy insights - Demo Mode",
  generator: "v0.app",
  icons: {
    icon: [
      { url: '/favicon.png', type: 'image/png' },
      { url: '/favicon.ico' },
    ],
    apple: [
      { url: '/favicon.png', sizes: '180x180', type: 'image/png' },
    ],
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable} antialiased`}>
        {/* Brand logo (restored) - responsive sizing */}
  <div className="fixed top-[38px] sm:top-[43.5px] left-3 sm:left-6 md:left-8 z-50">
          <Link href="/" className="text-lg sm:text-xl md:text-2xl font-bold tracking-tight text-primary drop-shadow-sm hover:opacity-90 transition-opacity">
            InsightGov
          </Link>
        </div>
        <TubelightNavBar />
  <div className="pt-[100px] sm:pt-[120px] md:pt-[130px] pb-12 sm:pb-16 md:pb-24">{/* responsive padding for all screen sizes */}
          <Suspense fallback={null}>{children}</Suspense>
        </div>
        <Analytics />
      </body>
    </html>
  )
}
