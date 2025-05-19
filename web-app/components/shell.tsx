"use client"

import type React from "react"
import { AppLayout } from "@/components/layout/app-layout"

interface ShellProps {
  children: React.ReactNode
}

/**
 * Main application shell component
 * Uses the new AppLayout component for improved structure
 */
export function Shell({ children }: ShellProps) {
  return <AppLayout>{children}</AppLayout>
}
