import type React from "react"
import type { Metadata } from "next"
import { Suspense } from "react"
import "./globals.css"
import { NavBar } from "@/components/ui/tubelight-navbar"
import { Home, LayoutDashboard, Mail } from "lucide-react"

export const metadata: Metadata = {
  title: "HackHive - Turning Voices into Action",
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
      <body className="font-sans antialiased">
        {/* Tubelight navbar overlay (fixed bottom on mobile, top on desktop) */}
        <NavBar
          items={[
            { name: "Home", url: "/", icon: Home },
            { name: "Dashboard", url: "/dashboard", icon: LayoutDashboard },
            { name: "Contact", url: "#contact", icon: Mail },
          ]}
        />
        {/* App content */}
        <Suspense fallback={null}>{children}</Suspense>
      </body>
    </html>
  )
}
