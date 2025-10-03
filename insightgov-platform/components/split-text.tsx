"use client"

import { useEffect, useState } from "react"

interface SplitTextProps {
  text: string
  className?: string
  delay?: number
}

export function SplitText({ text, className = "", delay = 0 }: SplitTextProps) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true)
    }, delay)

    return () => clearTimeout(timer)
  }, [delay])

  return (
    <span className={className}>
      {text.split("").map((char, index) => (
        <span
          key={index}
          className={`inline-block transition-all duration-700 ease-out ${
            isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
          }`}
          style={{
            transitionDelay: `${index * 50}ms`,
          }}
        >
          {char === " " ? "\u00A0" : char}
        </span>
      ))}
    </span>
  )
}
