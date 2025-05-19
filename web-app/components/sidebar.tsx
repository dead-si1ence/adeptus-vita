"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { useSidebar } from "@/components/ui/sidebar"
import { Brain, FileText, Home } from "lucide-react"

/**
 * Sidebar navigation component
 * Displays a collapsible sidebar with navigation links and icons
 */
export function Sidebar() {
  const pathname = usePathname()
  const { state } = useSidebar()
  const isExpanded = state === "expanded"

  // Navigation route definitions with icons and active state
  const routes = [
    {
      href: "/",
      label: "Home",
      icon: Home,
      active: pathname === "/",
    },
    {
      href: "/model",
      label: "Diagnostic",
      icon: Brain,
      active: pathname === "/model",
    },
    {
      href: "/blog",
      label: "Blog",
      icon: FileText,
      active: pathname === "/blog" || pathname.startsWith("/blog/"),
    },
  ]

  return (
    <aside
      className={cn(
        "fixed top-0 left-0 z-20 h-screen border-r border-border/40 bg-card transition-all duration-300 ease-in-out",
        isExpanded ? "w-64" : "w-16",
        "hidden md:block", // Hide on mobile
      )}
    >
      {/* Logo/brand in the sidebar header */}
      <div className="flex h-16 items-center justify-center border-b border-border/40 px-4">
        <Link href="/" className="flex items-center justify-center">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/10 text-primary">
            <Brain className="h-4 w-4" />
          </div>
          {isExpanded && <span className="ml-2 font-bold text-lg">ADEPTUS·VITA</span>}
        </Link>
      </div>

      {/* Navigation links */}
      <nav className="flex flex-col pt-6 space-y-2 px-2">
        {routes.map((route) => (
          <Link
            key={route.href}
            href={route.href}
            className={cn(
              "flex items-center rounded-md p-2 transition-colors",
              route.active
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:bg-accent/50 hover:text-accent-foreground",
              isExpanded ? "justify-start" : "justify-center",
            )}
            title={!isExpanded ? route.label : undefined}
          >
            <route.icon className="h-5 w-5 flex-shrink-0" />
            {isExpanded && <span className="ml-2">{route.label}</span>}
          </Link>
        ))}
      </nav>
    </aside>
  )
}
