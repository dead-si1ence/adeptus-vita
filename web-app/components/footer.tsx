"use client"

import React from "react"
import Link from "next/link"
import { Separator } from "@/components/ui/separator"
import { Brain } from "lucide-react"
import { useSidebar } from "@/components/ui/sidebar"
import { cn } from "@/lib/utils"

/**
 * Navigation link type definition
 */
type FooterLink = {
  href: string
  label: string
}

/**
 * Application footer component
 * Displays copyright information and navigation links
 */
export function Footer() {
  const { state } = useSidebar()
  const isExpanded = state === "expanded"

  // Footer navigation links
  const footerLinks: FooterLink[] = [
    { href: "/", label: "Home" },
    { href: "/model", label: "Diagnostic" },
    { href: "/blog", label: "Blog" },
  ]

  // Get current year for copyright
  const currentYear = new Date().getFullYear()

  return (
    <footer
      className={cn(
        "border-t border-border/40 bg-card/50 w-full mt-auto transition-all duration-300",
        isExpanded ? "md:pl-64" : "md:pl-16",
      )}
    >
      <div className="container flex flex-col items-center justify-between gap-4 py-6 md:h-16 md:flex-row md:py-0">
        {/* Logo and copyright notice */}
        <div className="flex flex-col items-center gap-2 md:flex-row md:gap-4">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-primary/10 text-primary">
              <Brain className="h-3 w-3" />
            </div>
            <span className="text-sm font-medium">ADEPTUS·VITA</span>
          </div>
          <p className="text-center text-xs text-muted-foreground md:text-left">
            © {currentYear} Adeptus Vita. All rights reserved.
          </p>
        </div>

        {/* Footer navigation */}
        <nav className="flex items-center space-x-4">
          {footerLinks.map((link, index) => (
            <React.Fragment key={link.href}>
              {index > 0 && <Separator orientation="vertical" className="h-4" />}
              <Link
                href={link.href}
                className="text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
              >
                {link.label}
              </Link>
            </React.Fragment>
          ))}
        </nav>
      </div>
    </footer>
  )
}
