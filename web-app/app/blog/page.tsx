import Link from "next/link"
import { Shell } from "@/components/shell"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { CalendarIcon, Clock } from "lucide-react"

interface BlogPost {
  slug: string
  title: string
  description: string
  date: string
  readingTime: string
  category: string
}

const blogPosts: BlogPost[] = [
  {
    slug: "advancements-in-mri-technology",
    title: "Recent Advancements in MRI Technology for Neurodegenerative Disease Detection",
    description:
      "Explore the latest developments in MRI technology that are revolutionizing how we detect and diagnose Alzheimer's and dementia.",
    date: "2023-11-15",
    readingTime: "6 min read",
    category: "Technology",
  },
  {
    slug: "early-detection-benefits",
    title: "The Benefits of Early Detection in Alzheimer's Disease",
    description:
      "Understanding why early detection matters and how it can significantly improve patient outcomes and quality of life.",
    date: "2023-10-27",
    readingTime: "8 min read",
    category: "Research",
  },
  {
    slug: "ai-in-healthcare",
    title: "The Role of AI in Modern Healthcare Diagnostics",
    description:
      "How artificial intelligence is transforming diagnostic procedures across the healthcare industry, with a focus on neurology.",
    date: "2023-10-12",
    readingTime: "5 min read",
    category: "AI",
  },
  {
    slug: "patient-stories",
    title: "Patient Stories: How Early Diagnosis Changed Lives",
    description:
      "Real-world examples of how early diagnosis of neurodegenerative diseases has impacted patient treatment plans and outcomes.",
    date: "2023-09-30",
    readingTime: "10 min read",
    category: "Stories",
  },
  {
    slug: "future-of-diagnosis",
    title: "The Future of Neurological Disease Diagnosis",
    description:
      "Predictions and insights into how diagnostic techniques for Alzheimer's and dementia will evolve over the next decade.",
    date: "2023-09-18",
    readingTime: "7 min read",
    category: "Future",
  },
]

function formatDate(dateString: string) {
  const options: Intl.DateTimeFormatOptions = { year: "numeric", month: "long", day: "numeric" }
  return new Date(dateString).toLocaleDateString("en-US", options)
}

export default function BlogPage() {
  return (
    <Shell>
      <div className="py-10 space-y-10 animate-fade-in">
        <div className="space-y-4">
          <h1 className="text-3xl font-bold tracking-tight">Blog</h1>
          <p className="text-lg text-muted-foreground">
            Stay updated with the latest research, advancements, and news in neurodegenerative disease diagnosis.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {blogPosts.map((post) => (
            <Link href={`/blog/${post.slug}`} key={post.slug} className="group">
              <Card className="h-full transition-all border-border/40 hover:border-primary/50 hover:shadow-md">
                <CardHeader className="space-y-1">
                  <div className="flex items-center text-sm text-muted-foreground">
                    <span className="px-2 py-1 text-xs rounded-md bg-primary/10 text-primary">{post.category}</span>
                    <span className="flex items-center ml-auto">
                      <CalendarIcon className="w-3 h-3 mr-1" />
                      {formatDate(post.date)}
                    </span>
                  </div>
                  <CardTitle className="group-hover:text-primary transition-colors">{post.title}</CardTitle>
                  <CardDescription className="line-clamp-2">{post.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-24 bg-muted rounded-md grid-lines flex items-center justify-center mb-2">
                    <span className="text-sm text-muted-foreground">Featured Image</span>
                  </div>
                </CardContent>
                <CardFooter>
                  <div className="flex items-center text-sm text-muted-foreground">
                    <Clock className="w-3 h-3 mr-1" />
                    {post.readingTime}
                  </div>
                </CardFooter>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </Shell>
  )
}
