"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Menu, X } from "lucide-react"

export function Navigation() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="fixed top-0 w-full z-[100] bg-background/95 backdrop-blur-md border-b border-border shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <Link href="/" className="text-2xl font-bold text-primary">
              InsightGov
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-8">
              <Link href="/" className="text-foreground hover:text-primary transition-colors">
                Home
              </Link>
              <Link href="/dashboard" className="text-foreground hover:text-primary transition-colors">
                Dashboard
              </Link>
              <Link href="#operational-advantages" className="text-foreground hover:text-primary transition-colors">
                Advantages
              </Link>
              <button
                className="text-foreground hover:text-primary transition-colors cursor-pointer bg-transparent border-none"
                onClick={() => {
                  console.log('Policy AI button clicked!')
                  const event = new CustomEvent('openPreview', { detail: { feature: 'policy' } })
                  console.log('Dispatching event:', event)
                  window.dispatchEvent(event)
                  console.log('Event dispatched')
                }}
              >
                Policy AI
              </button>
              <button
                className="text-foreground hover:text-primary transition-colors cursor-pointer bg-transparent border-none"
                onClick={() => {
                  console.log('Compare button clicked!')
                  const event = new CustomEvent('openPreview', { detail: { feature: 'comparative' } })
                  console.log('Dispatching event:', event)
                  window.dispatchEvent(event)
                  console.log('Event dispatched')
                }}
              >
                Compare
              </button>
              <Link href="#contact" className="text-foreground hover:text-primary transition-colors">
                Contact
              </Link>
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <Button variant="ghost" size="sm" onClick={() => setIsOpen(!isOpen)}>
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      {isOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-background border-b border-border shadow-lg">
            <Link
              href="/"
              className="block px-3 py-2 text-foreground hover:text-primary transition-colors"
              onClick={() => setIsOpen(false)}
            >
              Home
            </Link>
            <Link
              href="/dashboard"
              className="block px-3 py-2 text-foreground hover:text-primary transition-colors"
              onClick={() => setIsOpen(false)}
            >
              Dashboard
            </Link>
            <Link
              href="#operational-advantages"
              className="block px-3 py-2 text-foreground hover:text-primary transition-colors"
              onClick={() => setIsOpen(false)}
            >
              Advantages
            </Link>
            <button
              className="block px-3 py-2 text-foreground hover:text-primary transition-colors cursor-pointer text-left w-full bg-transparent border-none"
              onClick={() => {
                console.log('Mobile Policy AI button clicked!')
                const event = new CustomEvent('openPreview', { detail: { feature: 'policy' } })
                window.dispatchEvent(event)
                setIsOpen(false)
              }}
            >
              Policy AI
            </button>
            <button
              className="block px-3 py-2 text-foreground hover:text-primary transition-colors cursor-pointer text-left w-full bg-transparent border-none"
              onClick={() => {
                console.log('Mobile Compare button clicked!')
                const event = new CustomEvent('openPreview', { detail: { feature: 'comparative' } })
                window.dispatchEvent(event)
                setIsOpen(false)
              }}
            >
              Compare
            </button>
            <Link
              href="#contact"
              className="block px-3 py-2 text-foreground hover:text-primary transition-colors"
              onClick={() => setIsOpen(false)}
            >
              Contact
            </Link>
          </div>
        </div>
      )}
    </nav>
  )
}
