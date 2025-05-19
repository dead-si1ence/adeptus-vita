"use client";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { Brain, Github, Instagram, Twitter } from "lucide-react";
import { Button } from "@/components/ui/button";

interface AppFooterProps {
  sidebarOpen: boolean;
  isMobile: boolean;
}

/**
 * Application footer component
 * Displays copyright information, links, and social media
 */
export function AppFooter({ sidebarOpen, isMobile }: AppFooterProps) {
  // Footer navigation links
  const footerLinks = [
    { href: "/", label: "Home" },
    { href: "/model", label: "Diagnostic" },
    { href: "/blog", label: "Blog" },
    { href: "/about", label: "About" },
    { href: "/settings", label: "Settings" },
  ];

  // Get current year for copyright
  const currentYear = new Date().getFullYear();

  return (
    <footer
      className={cn(
        "border-t border-border/40 bg-card/50 transition-all duration-300 ease-in-out",
        sidebarOpen && !isMobile ? "lg:pl-72" : "lg:pl-20"
      )}
    >
      <div className="mx-auto max-w-6xl px-4 md:px-6 py-4">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Brand and copyright */}
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-primary/10 text-primary">
              <Brain className="h-3 w-3" />
            </div>
            <span className="text-sm font-medium">ADEPTUS·VITA</span>
            <span className="text-xs text-muted-foreground">
              © {currentYear} All rights reserved
            </span>
          </div>

          {/* Links */}
          <nav className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2">
            {footerLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors"
              >
                {link.label}
              </Link>
            ))}
          </nav>

          {/* Social links */}
          <div className="flex items-center gap-2">
            <a
              href="https://instagram.com/m0ntas3r"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-full"
              >
                <Instagram className="h-4 w-4" />
                <span className="sr-only">Instagram</span>
              </Button>
            </a>
            <a 
              href="https://github.com/dead-si1ence"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-full"
              >
                <Github className="h-4 w-4" />
                <span className="sr-only">GitHub</span>
              </Button>
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
