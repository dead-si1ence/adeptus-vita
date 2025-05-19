"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { AppHeader } from "@/components/layout/app-header"
import { AppSidebar } from "@/components/layout/app-sidebar"
import { AppFooter } from "@/components/layout/app-footer"
import { AppBreadcrumb } from "@/components/layout/app-breadcrumb"

interface AppLayoutProps {
  children: React.ReactNode
}

/**
 * Main application layout component
 * Provides the overall structure with sidebar, header, content area, and footer
 */
export function AppLayout({ children }: AppLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isMobile, setIsMobile] = useState(false)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const pathname = usePathname()

  // Close mobile sidebar when route changes
  useEffect(() => {
    setMobileSidebarOpen(false)
  }, [pathname])

  // Handle responsive behavior
  useEffect(() => {
    const checkIsMobile = () => {
      setIsMobile(window.innerWidth < 1024)
      if (window.innerWidth < 1024) {
        setSidebarOpen(false)
      } else {
        setSidebarOpen(true)
      }
    }

    // Initial check
    checkIsMobile()

    // Add event listener for window resize
    window.addEventListener("resize", checkIsMobile)

    // Cleanup
    return () => window.removeEventListener("resize", checkIsMobile)
  }, [])

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar - fixed position on desktop, overlay on mobile */}
      <AppSidebar
        isOpen={isMobile ? mobileSidebarOpen : sidebarOpen}
        onToggle={() => (isMobile ? setMobileSidebarOpen(!mobileSidebarOpen) : setSidebarOpen(!sidebarOpen))}
        isMobile={isMobile}
      />

      {/* Main content area - FIX: Use max-width to prevent content from overflowing */}
      <div
        className={cn(
          "flex flex-col flex-1 w-full transition-all duration-300 ease-in-out",
          // Remove the conditional padding that causes content to shift
          "lg:ml-20", // Fixed margin for collapsed sidebar
          sidebarOpen && !isMobile && "lg:ml-72", // Additional margin when sidebar is open
        )}
        style={{
          width: "100%",
          maxWidth: "100vw", // Ensure content doesn't overflow viewport
          overflow: "hidden", // Prevent horizontal scrolling
        }}
      >
        {/* Header */}
        <AppHeader
          sidebarOpen={sidebarOpen}
          onSidebarToggle={() => (isMobile ? setMobileSidebarOpen(!mobileSidebarOpen) : setSidebarOpen(!sidebarOpen))}
          isMobile={isMobile}
        />

        {/* Main content */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden pt-16 pb-6 px-4 md:px-6 lg:px-8 bg-grid-pattern">
          <div className="mx-auto w-full max-w-6xl pt-4">
            <AppBreadcrumb />
            {children}
          </div>
        </main>

        {/* Footer */}
        <AppFooter sidebarOpen={sidebarOpen} isMobile={isMobile} />
      </div>

      {/* Mobile sidebar overlay */}
      {isMobile && mobileSidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-background/80 backdrop-blur-sm"
          onClick={() => setMobileSidebarOpen(false)}
        />
      )}
    </div>
  )
}
