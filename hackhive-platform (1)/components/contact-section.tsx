"use client"

import { Button } from "@/components/ui/button"
import { Mail } from "lucide-react"

export function ContactSection() {
  return (
    <section id="contact" className="py-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">Get in Touch</h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Ready to transform citizen feedback into actionable insights? Contact our team to learn more about HackHive.
          </p>
        </div>

        <div className="text-center">
          <div className="inline-flex items-center space-x-4 bg-gray-900/50 border border-gray-800 rounded-lg p-8">
            <div className="w-12 h-12 bg-purple-600/20 rounded-lg flex items-center justify-center">
              <Mail className="w-6 h-6 text-purple-400" />
            </div>
            <div className="flex flex-col items-center">
              <h3 className="text-lg font-semibold text-white mb-2 text-center">Contact Email</h3>
              <p className="text-2xl text-purple-400 font-mono text-center">contact@hackhive.ai</p>
              <Button
                className="mt-4 bg-purple-600 hover:bg-purple-700 text-white px-8 py-3 rounded-lg font-semibold transition-all duration-300 cursor-pointer hover:scale-105 hover:shadow-lg"
                onClick={() => (window.location.href = "mailto:contact@hackhive.ai")}
              >
                Send us an Email
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
