"use client"

import type React from "react"

import { useEffect, useState } from "react"
import { useSearchParams } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import { Brain, FileText, SearchIcon, ArrowLeft, Clock, Calendar } from "lucide-react"
import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// Mock search results
const mockBlogResults = [
  {
    id: "1",
    title: "Recent Advancements in MRI Technology",
    excerpt:
      "Explore the latest developments in MRI technology that are revolutionizing how we detect and diagnose Alzheimer's and dementia.",
    category: "Technology",
    date: "2023-11-15",
    url: "/blog/advancements-in-mri-technology",
  },
  {
    id: "2",
    title: "The Benefits of Early Detection in Alzheimer's Disease",
    excerpt:
      "Understanding why early detection matters and how it can significantly improve patient outcomes and quality of life.",
    category: "Research",
    date: "2023-10-27",
    url: "/blog/early-detection-benefits",
  },
  {
    id: "3",
    title: "AI in Healthcare: Transforming Diagnostic Procedures",
    excerpt:
      "How artificial intelligence is transforming diagnostic procedures across the healthcare industry, with a focus on neurology.",
    category: "AI",
    date: "2023-10-12",
    url: "/blog/ai-in-healthcare",
  },
]

const mockDiagnosticResults = [
  {
    id: "1",
    title: "MRI Analysis Report",
    date: "2023-11-10",
    result: "Negative",
    confidence: 94,
    url: "/model",
  },
  {
    id: "2",
    title: "Brain Scan Evaluation",
    date: "2023-10-28",
    result: "Positive - Early Stage",
    confidence: 87,
    url: "/model",
  },
  {
    id: "3",
    title: "Cognitive Assessment Results",
    date: "2023-09-15",
    result: "Negative",
    confidence: 91,
    url: "/model",
  },
]

/**
 * Search page component
 * Provides search functionality across the application
 */
export default function SearchPage() {
  const searchParams = useSearchParams()
  const query = searchParams.get("q") || ""
  const [searchQuery, setSearchQuery] = useState(query)
  const [activeTab, setActiveTab] = useState("all")
  const [isLoading, setIsLoading] = useState(true)
  const [blogResults, setBlogResults] = useState<typeof mockBlogResults>([])
  const [diagnosticResults, setDiagnosticResults] = useState<typeof mockDiagnosticResults>([])
  const [sortOrder, setSortOrder] = useState("relevance")

  // Simulate search results loading
  useEffect(() => {
    if (query) {
      setIsLoading(true)
      // Simulate API call delay
      const timer = setTimeout(() => {
        setBlogResults(mockBlogResults)
        setDiagnosticResults(mockDiagnosticResults)
        setIsLoading(false)
      }, 1500)

      return () => clearTimeout(timer)
    } else {
      setBlogResults([])
      setDiagnosticResults([])
      setIsLoading(false)
    }
  }, [query])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    // Update URL with search query
    const url = new URL(window.location.href)
    url.searchParams.set("q", searchQuery)
    window.history.pushState({}, "", url.toString())

    // Trigger search
    setIsLoading(true)
    setTimeout(() => {
      setBlogResults(mockBlogResults)
      setDiagnosticResults(mockDiagnosticResults)
      setIsLoading(false)
    }, 1500)
  }

  // Sort results based on selected order
  const sortResults = (results: any[]) => {
    if (sortOrder === "newest") {
      return [...results].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    } else if (sortOrder === "oldest") {
      return [...results].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    }
    return results // Default: relevance (as returned from API)
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Search Results</h1>
          <p className="text-muted-foreground">
            {query ? `Showing results for "${query}"` : "Enter a search term to find content"}
          </p>
        </div>
        <Button variant="outline" asChild className="gap-2">
          <Link href="/">
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>
        </Button>
      </div>

      {/* Search form */}
      <form onSubmit={handleSearch} className="flex w-full max-w-lg gap-2">
        <Input
          type="search"
          placeholder="Search..."
          className="flex-1"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        <Button type="submit">
          <SearchIcon className="mr-2 h-4 w-4" />
          Search
        </Button>
      </form>

      {/* Filter and sort options */}
      {(blogResults.length > 0 || diagnosticResults.length > 0) && (
        <div className="flex flex-col sm:flex-row gap-2 justify-between">
          <div className="flex items-center gap-2">
            <Select defaultValue="relevance" onValueChange={setSortOrder}>
              <SelectTrigger className="w-[180px]">
                <Clock className="h-4 w-4 mr-2" />
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="relevance">Sort by Relevance</SelectItem>
                <SelectItem value="newest">Sort by Newest</SelectItem>
                <SelectItem value="oldest">Sort by Oldest</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <Badge variant="outline" className="px-2 py-1">
              {blogResults.length + diagnosticResults.length} Results
            </Badge>
            <Badge variant="outline" className="px-2 py-1">
              {query}
            </Badge>
          </div>
        </div>
      )}

      {/* Results tabs */}
      <Tabs defaultValue="all" value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList>
          <TabsTrigger value="all">All Results</TabsTrigger>
          <TabsTrigger value="blog">Blog Posts</TabsTrigger>
          <TabsTrigger value="diagnostic">Diagnostic</TabsTrigger>
        </TabsList>

        {/* All results */}
        <TabsContent value="all" className="space-y-6">
          {isLoading ? (
            <div className="space-y-4">
              <SearchResultSkeleton />
              <SearchResultSkeleton />
              <SearchResultSkeleton />
              <SearchResultSkeleton />
            </div>
          ) : query ? (
            <>
              {blogResults.length > 0 && (
                <div className="space-y-4">
                  <h2 className="text-xl font-semibold">Blog Posts</h2>
                  {sortResults(blogResults).map((result) => (
                    <BlogResultCard key={result.id} result={result} />
                  ))}
                </div>
              )}

              {diagnosticResults.length > 0 && (
                <div className="space-y-4">
                  <h2 className="text-xl font-semibold">Diagnostic Results</h2>
                  {sortResults(diagnosticResults).map((result) => (
                    <DiagnosticResultCard key={result.id} result={result} />
                  ))}
                </div>
              )}

              {blogResults.length === 0 && diagnosticResults.length === 0 && (
                <div className="text-center py-12">
                  <SearchIcon className="mx-auto h-12 w-12 text-muted-foreground" />
                  <h2 className="mt-4 text-lg font-medium">No results found</h2>
                  <p className="mt-2 text-sm text-muted-foreground">
                    We couldn't find any matches for "{query}". Try different keywords.
                  </p>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <SearchIcon className="mx-auto h-12 w-12 text-muted-foreground" />
              <h2 className="mt-4 text-lg font-medium">Enter a search term</h2>
              <p className="mt-2 text-sm text-muted-foreground">Type in the search box above to find content</p>
            </div>
          )}
        </TabsContent>

        {/* Blog results */}
        <TabsContent value="blog" className="space-y-4">
          {isLoading ? (
            <div className="space-y-4">
              <SearchResultSkeleton />
              <SearchResultSkeleton />
            </div>
          ) : blogResults.length > 0 ? (
            sortResults(blogResults).map((result) => <BlogResultCard key={result.id} result={result} />)
          ) : (
            <div className="text-center py-12">
              <FileText className="mx-auto h-12 w-12 text-muted-foreground" />
              <h2 className="mt-4 text-lg font-medium">No blog posts found</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                We couldn't find any blog posts matching your search.
              </p>
            </div>
          )}
        </TabsContent>

        {/* Diagnostic results */}
        <TabsContent value="diagnostic" className="space-y-4">
          {isLoading ? (
            <div className="space-y-4">
              <SearchResultSkeleton />
              <SearchResultSkeleton />
            </div>
          ) : diagnosticResults.length > 0 ? (
            sortResults(diagnosticResults).map((result) => <DiagnosticResultCard key={result.id} result={result} />)
          ) : (
            <div className="text-center py-12">
              <Brain className="mx-auto h-12 w-12 text-muted-foreground" />
              <h2 className="mt-4 text-lg font-medium">No diagnostic results found</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                We couldn't find any diagnostic results matching your search.
              </p>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

// Blog result card component
function BlogResultCard({ result }: { result: (typeof mockBlogResults)[0] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">{result.category}</span>
          </div>
          <div className="flex items-center gap-2">
            <Calendar className="h-3 w-3 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">{result.date}</span>
          </div>
        </div>
        <CardTitle className="text-lg">
          <Link href={result.url} className="hover:underline">
            {result.title}
          </Link>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <CardDescription className="line-clamp-2">{result.excerpt}</CardDescription>
        <div className="mt-4">
          <Button variant="link" asChild className="px-0">
            <Link href={result.url}>Read more</Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// Diagnostic result card component
function DiagnosticResultCard({ result }: { result: (typeof mockDiagnosticResults)[0] }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Diagnostic</span>
          </div>
          <div className="flex items-center gap-2">
            <Calendar className="h-3 w-3 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">{result.date}</span>
          </div>
        </div>
        <CardTitle className="text-lg">
          <Link href={result.url} className="hover:underline">
            {result.title}
          </Link>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className={`text-sm font-medium ${
                result.result.includes("Negative") ? "text-green-500" : "text-red-500"
              }`}
            >
              {result.result}
            </span>
          </div>
          <span className="text-sm text-muted-foreground">Confidence: {result.confidence}%</span>
        </div>
        <div className="mt-4">
          <Button variant="link" asChild className="px-0">
            <Link href={result.url}>View details</Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

// Skeleton loader for search results
function SearchResultSkeleton() {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-20" />
        </div>
        <Skeleton className="h-6 w-3/4 mt-2" />
      </CardHeader>
      <CardContent>
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6 mt-2" />
        <Skeleton className="h-8 w-24 mt-4" />
      </CardContent>
    </Card>
  )
}
