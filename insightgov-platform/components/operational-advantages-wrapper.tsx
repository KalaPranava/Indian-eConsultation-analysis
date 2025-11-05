"use client"

import { Suspense } from "react"
import { OperationalAdvantages } from "./operational-advantages"

export function OperationalAdvantagesWrapper() {
  return (
    <Suspense fallback={<div className="py-16 sm:py-24" />}>
      <OperationalAdvantages />
    </Suspense>
  )
}
