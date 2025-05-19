import type React from "react"
import type { Metadata } from "next"
import { JetBrains_Mono } from "next/font/google"
import "./globals.css"

import { ThemeProvider } from "@/components/ThemeProvider"
import { cn } from "@/lib/utils"

// Load JetBrains Mono font with proper configuration
const jetBrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-jetbrains-mono",
})

// Application metadata for SEO
export const metadata: Metadata = {
  title: "Adeptus Vita",
  description: "AI-powered diagnosis of Alzheimer's and dementia using MRI scans",
  keywords: ["alzheimer's", "dementia", "diagnosis", "MRI", "neural network", "healthcare", "AI"],
    generator: 'v0.dev'
}

/**
 * Root layout component for the entire application
 * Sets up font, theme provider, and base HTML structure
 */
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={jetBrainsMono.variable}>
      <head>
        {/* Script to prevent flash of wrong theme on page load */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
            (function() {
              try {
                const theme = localStorage.getItem('theme') || 'dark';
                document.documentElement.classList.add(theme === 'dark' ? 'dark' : 'light');
                document.documentElement.style.colorScheme = theme;
              } catch (e) {
                console.error('Failed to initialize theme', e);
              }
            })();
          `,
          }}
        />
      </head>
      <body className={cn("min-h-screen font-sans antialiased bg-background grid-lines overflow-x-hidden")}>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
