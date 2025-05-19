"use client"

import type React from "react"

import { useState, useEffect } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { ModeToggle } from "@/components/mode-toggle"
import { SidebarTrigger, useSidebar } from "@/components/ui/sidebar"
import { Brain, FileText, Home, Menu } from "lucide-react"

/**
 * Navigation link type definition
 */
type NavLink = {
  href: string
  label: string
  icon: React.ElementType
}

/**
 * Header component with responsive navigation
 * Displays brand/logo, navigation links, and theme toggle
 * Includes mobile menu drawer and desktop navigation
 */
export function Header() {
  const [open, setOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const pathname = usePathname()
  const { state } = useSidebar()
  const isExpanded = state === "expanded"

  // Handle scroll effect for header
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10)
    }
    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  // Navigation links configuration
  const navLinks: NavLink[] = [
    { href: "/", label: "Home", icon: Home },
    { href: "/model", label: "Diagnostic", icon: Brain },
    { href: "/blog", label: "Blog", icon: FileText },
  ]

  return (
    <header
      className={cn(
        "fixed top-0 w-full border-b z-10 transition-all duration-200",
        scrolled
          ? "border-border/40 bg-background/95 backdrop-blur-md shadow-sm"
          : "border-transparent bg-background/50 backdrop-blur-sm",
        isExpanded ? "md:pl-64" : "md:pl-16",
      )}
    >
      <div className="flex h-16 items-center justify-between px-4 max-w-7xl mx-auto">
        {/* Left section: Sidebar trigger on desktop, mobile menu on mobile */}
        <div className="flex items-center">
          {/* Mobile Menu Trigger */}
          <div className="md:hidden">
            <Sheet open={open} onOpenChange={setOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="h-9 w-9">
                  <Menu className="h-5 w-5" />
                  <span className="sr-only">Toggle menu</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-[240px] sm:w-[280px] pr-0">
                <div className="flex flex-col gap-6">
                  <Link href="/" className="flex items-center gap-2 mt-4" onClick={() => setOpen(false)}>
                    <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/10 text-primary">
                      <Brain className="h-4 w-4" />
                    </div>
                    <span className="font-bold text-lg">ADEPTUS·VITA</span>
                  </Link>

                  <nav className="flex flex-col gap-1">
                    {navLinks.map((link) => (
                      <Link
                        key={link.href}
                        href={link.href}
                        className={cn(
                          "flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                          pathname === link.href || (link.href !== "/" && pathname.startsWith(link.href))
                            ? "bg-accent text-accent-foreground"
                            : "hover:bg-accent/50 text-muted-foreground hover:text-foreground",
                        )}
                        onClick={() => setOpen(false)}
                      >
                        <link.icon className="h-4 w-4" />
                        {link.label}
                      </Link>
                    ))}
                  </nav>

                  <div className="mt-auto border-t border-border/40 pt-4">
                    <ModeToggle />
                  </div>
                </div>
              </SheetContent>
            </Sheet>
          </div>

          {/* Sidebar Trigger (visible on desktop) */}
          <div className="hidden md:flex">
            <SidebarTrigger className="mr-2" />
          </div>

          {/* Logo/Brand section (visible on mobile) */}
          <div className="flex items-center md:hidden">
            <Link href="/" className="flex items-center">
              <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary/10 text-primary mr-2">
                <Brain className="h-4 w-4" />
              </div>
              <span className="font-bold text-lg tracking-tight">ADEPTUS·VITA</span>
            </Link>
          </div>
        </div>

        {/* Right section: Navigation links on desktop, theme toggle on all */}
        <div className="flex items-center gap-1">
          {/* Desktop Navigation Links */}
          <div className="hidden md:flex items-center gap-1 mr-2">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "px-3 py-2 rounded-md text-sm font-medium transition-colors",
                  pathname === link.href || (link.href !== "/" && pathname.startsWith(link.href))
                    ? "bg-accent/50 text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/30",
                )}
              >
                {link.label}
              </Link>
            ))}
          </div>
          <ModeToggle />
        </div>
      </div>
    </header>
  )
}
