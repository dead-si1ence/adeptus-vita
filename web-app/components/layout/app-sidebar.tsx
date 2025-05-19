"use client"

import { useEffect, useRef } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Brain, FileText, Home, ChevronLeft, ChevronRight, Info, Settings } from "lucide-react"
import { useTheme } from "next-themes"

interface AppSidebarProps {
  isOpen: boolean
  onToggle: () => void
  isMobile: boolean
}

/**
 * Application sidebar component
 * Provides navigation and branding
 */
export function AppSidebar({ isOpen, onToggle, isMobile }: AppSidebarProps) {
  const pathname = usePathname()
  const { theme } = useTheme()
  const sidebarRef = useRef<HTMLDivElement>(null)

  // Navigation items
  const mainNavItems = [
    {
      title: "Home",
      href: "/",
      icon: Home,
      active: pathname === "/",
    },
    {
      title: "Diagnostic",
      href: "/model",
      icon: Brain,
      active: pathname === "/model",
    },
    {
      title: "Blog",
      href: "/blog",
      icon: FileText,
      active: pathname === "/blog" || pathname.startsWith("/blog/"),
    },
  ]

  // Secondary navigation items
  const secondaryNavItems = [
    {
      title: "About",
      href: "/about",
      icon: Info,
      active: pathname === "/about",
    },
    {
      title: "Settings",
      href: "/settings",
      icon: Settings,
      active: pathname === "/settings",
    },
  ]

  // Handle click outside to close mobile sidebar
  useEffect(() => {
    if (!isMobile) return

    const handleClickOutside = (event: MouseEvent) => {
      if (sidebarRef.current && !sidebarRef.current.contains(event.target as Node)) {
        // Only close if it's open
        if (isOpen) {
          onToggle()
        }
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [isMobile, isOpen, onToggle])

  return (
    <>
      <aside
        ref={sidebarRef}
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex flex-col border-r border-border/40 bg-card transition-all duration-300 ease-in-out",
          isOpen ? "w-72" : "w-20",
          isMobile && (isOpen ? "translate-x-0" : "-translate-x-full"),
        )}
      >
        {/* Sidebar header with logo */}
        <div className="flex h-16 items-center justify-between border-b border-border/40 px-4">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary/10 text-primary">
              <Brain className="h-5 w-5" suppressHydrationWarning />
            </div>
            {isOpen && <span className="font-bold text-xl tracking-tight">ADEPTUS·VITA</span>}
          </Link>

          {/* Toggle button - only show on desktop */}
          {!isMobile && (
            <Button suppressContentEditableWarning variant="ghost" size="icon" onClick={onToggle} className="h-8 w-8 rounded-full hover:bg-accent">
              {isOpen ? <ChevronLeft suppressHydrationWarning className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            </Button>
          )}
        </div>

        {/* Scrollable sidebar content */}
        <ScrollArea className="flex-1 py-4">
          <div className="px-3 space-y-6">
            {/* Main navigation */}
            <div className="space-y-1">
              <h3 className={cn("text-xs font-medium text-muted-foreground px-2 mb-2", !isOpen && "sr-only")}>
                Navigation
              </h3>
              <nav className="space-y-1">
                {mainNavItems.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                      item.active
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground hover:bg-accent/50 hover:text-accent-foreground",
                      !isOpen && "justify-center px-0",
                    )}
                    title={!isOpen ? item.title : undefined}
                  >
                    <item.icon suppressHydrationWarning className={cn("h-5 w-5 flex-shrink-0", item.active && "text-primary")} />
                    {isOpen && <span>{item.title}</span>}
                  </Link>
                ))}
              </nav>
            </div>

            {/* Separator */}
            <Separator className="mx-1" />

            {/* Secondary navigation */}
            <div className="space-y-1">
              <h3 className={cn("text-xs font-medium text-muted-foreground px-2 mb-2", !isOpen && "sr-only")}>
                Support
              </h3>
              <nav className="space-y-1">
                {secondaryNavItems.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                      pathname === item.href
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground hover:bg-accent/50 hover:text-accent-foreground",
                      !isOpen && "justify-center px-0",
                    )}
                    title={!isOpen ? item.title : undefined}
                  >
                    <item.icon className="h-5 w-5 flex-shrink-0" />
                    {isOpen && <span>{item.title}</span>}
                  </Link>
                ))}
              </nav>
            </div>
          </div>
        </ScrollArea>

        {/* Sidebar footer */}
        <div className="border-t border-border/40 p-4">
          <div className={cn("flex items-center", isOpen ? "justify-between" : "justify-center")}>
            {isOpen && (
              <div className="flex items-center gap-2">
                <div className={cn("h-2 w-2 rounded-full", theme === "dark" ? "bg-green-500" : "bg-primary")} />
                <span className="text-xs text-muted-foreground">v1.0.0</span>
              </div>
            )}
            <div className={cn("text-xs text-muted-foreground", !isOpen && "hidden")}>© 2025 Adeptus Vita</div>
          </div>
        </div>
      </aside>
    </>
  )
}
