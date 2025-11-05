"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import Link from "next/link"
import { usePathname } from "next/navigation"
import type { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { Home, LayoutDashboard, Layers3, Brain, BarChart3 } from "lucide-react"

interface NavItem {
  name: string
  url: string
  icon: LucideIcon
}

interface NavBarProps {
  items?: NavItem[]
  className?: string
}

// Glass / tubelight navbar adapted from duplicate project variant
export function TubelightNavBar({ items, className }: NavBarProps) {
  const defaultItems: NavItem[] = items || [
    { name: "Home", url: "/", icon: Home },
    { name: "Dashboard", url: "/dashboard", icon: LayoutDashboard },
    // Use absolute path so navigating from /dashboard goes back to home and scrolls
    { name: "Advantages", url: "/#operational-advantages", icon: Layers3 },
    { name: "Policy AI", url: "/#preview-policy", icon: Brain },
    { name: "Compare", url: "/#preview-comparative", icon: BarChart3 },
  ]

  const [activeTab, setActiveTab] = useState<string>(defaultItems[0]?.name ?? "")
  const [isMobile, setIsMobile] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768)
    handleResize()
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  // Update active tab based on current route
  useEffect(() => {
    if (pathname === "/") {
      setActiveTab("Home")
    } else if (pathname.startsWith("/dashboard")) {
      setActiveTab("Dashboard")
    } else if (pathname.includes("operational-advantages")) {
      setActiveTab("Advantages")
    }
  }, [pathname])

  return (
    <div
      className={cn(
        // shift down slightly on desktop to align with brand logo
        "fixed bottom-0 sm:top-[43.5px] left-1/2 -translate-x-1/2 z-50 mb-6 sm:pt-0",
        // Constrain size to prevent full-page coverage
        "w-auto h-auto max-w-fit max-h-fit pointer-events-none",
        className,
      )}
    >
      <div className="flex items-center gap-3 bg-background/40 dark:bg-background/30 border border-border/60 backdrop-blur-lg py-1 px-1 rounded-full shadow-lg supports-[backdrop-filter]:bg-background/30 pointer-events-auto w-fit h-fit">
        {defaultItems.map((item) => {
          const Icon = item.icon
          const isActive = activeTab === item.name

            return (
            <Link
              key={item.name}
              href={item.url}
              onClick={(e) => {
                setActiveTab(item.name)
                // Dispatch custom event for Policy AI and Compare
                if (item.name === "Policy AI") {
                  console.log('Policy AI clicked in tubelight navbar')
                  window.dispatchEvent(new CustomEvent('openPreview', { detail: { feature: 'policy' } }))
                } else if (item.name === "Compare") {
                  console.log('Compare clicked in tubelight navbar')
                  window.dispatchEvent(new CustomEvent('openPreview', { detail: { feature: 'comparative' } }))
                }
              }}
              className={cn(
                "relative cursor-pointer text-xs sm:text-sm font-medium px-5 py-2 rounded-full transition-colors",
                "text-foreground/80 hover:text-primary",
                isActive && "bg-muted/70 text-primary",
              )}
            >
              <span className="hidden md:inline">{item.name}</span>
              <span className="md:hidden">
                <Icon size={18} strokeWidth={2.5} />
              </span>
              {isActive && (
                <motion.div
                  layoutId="lamp"
                  className="absolute inset-0 w-full bg-primary/5 rounded-full -z-10"
                  initial={false}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                >
                  <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-8 h-1 bg-primary rounded-t-full">
                    <div className="absolute w-12 h-6 bg-primary/25 rounded-full blur-md -top-2 -left-2" />
                    <div className="absolute w-8 h-6 bg-primary/25 rounded-full blur-md -top-1" />
                    <div className="absolute w-4 h-4 bg-primary/25 rounded-full blur-sm top-0 left-2" />
                  </div>
                </motion.div>
              )}
            </Link>
          )
        })}
      </div>
    </div>
  )
}
