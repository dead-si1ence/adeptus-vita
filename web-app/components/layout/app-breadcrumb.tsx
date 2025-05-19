"use client"

import { usePathname } from "next/navigation"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"

/**
 * Breadcrumb component for navigation
 */
export function AppBreadcrumb() {
  const pathname = usePathname()

  // Skip breadcrumbs on home page
  if (pathname === "/") {
    return null
  }

  // Generate breadcrumb items from pathname
  const pathSegments = pathname.split("/").filter(Boolean)

  // Map path segments to readable names
  const segmentNames: Record<string, string> = {
    model: "Diagnostic",
    blog: "Blog",
    search: "Search",
    settings: "Settings",
    about: "About",
    notifications: "Notifications",
  }

  return (
    <Breadcrumb className="mb-4">
      <BreadcrumbList>
        <BreadcrumbItem>
          {/* Fix: Don't use asChild with Link, use direct href instead */}
          <BreadcrumbLink href="/">Home</BreadcrumbLink>
        </BreadcrumbItem>
        <BreadcrumbSeparator />

        {pathSegments.map((segment, index) => {
          const segmentPath = `/${pathSegments.slice(0, index + 1).join("/")}`
          const isLastSegment = index === pathSegments.length - 1
          const segmentName = segmentNames[segment] || segment

          return (
            <BreadcrumbItem key={segment}>
              {isLastSegment ? (
                <BreadcrumbPage>{segmentName}</BreadcrumbPage>
              ) : (
                <>
                  {/* Fix: Don't use asChild with Link, use direct href instead */}
                  <BreadcrumbLink href={segmentPath}>{segmentName}</BreadcrumbLink>
                  <BreadcrumbSeparator />
                </>
              )}
            </BreadcrumbItem>
          )
        })}
      </BreadcrumbList>
    </Breadcrumb>
  )
}
