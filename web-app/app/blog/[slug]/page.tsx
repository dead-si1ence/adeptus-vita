"use client"

import { useParams } from "next/navigation"
import Link from "next/link"
import { Shell } from "@/components/shell"
import { Button } from "@/components/ui/button"
import { CalendarIcon, Clock, ArrowLeft } from "lucide-react"

// Mock blog posts content
const blogPosts = {
  "advancements-in-mri-technology": {
    title: "Recent Advancements in MRI Technology for Neurodegenerative Disease Detection",
    date: "2025-5-14",
    readingTime: "6 min read",
    category: "Technology",
    author: "Montaser Amoor",
    content: `
      <h2>The Evolution of MRI Technology</h2>
      <p>Magnetic Resonance Imaging (MRI) has come a long way since its inception in the 1970s. The technology has evolved from simple structural imaging to sophisticated methods that can capture not only the anatomy but also the function and metabolism of the brain.</p>
      
      <p>Recent advancements in MRI technology have significantly improved our ability to detect neurodegenerative diseases like Alzheimer's and dementia at earlier stages. These improvements include higher field strengths, more sensitive detection coils, and advanced image reconstruction algorithms.</p>
      
      <h2>High-Field and Ultra-High-Field MRI</h2>
      <p>Traditional clinical MRI scanners operate at field strengths of 1.5 to 3 Tesla. However, recent research has utilized high-field (7 Tesla) and ultra-high-field (>7 Tesla) MRI systems. These higher field strengths provide improved signal-to-noise ratio, allowing for higher spatial resolution and better visualization of small brain structures that are often affected early in neurodegenerative diseases.</p>
      
      <p>The hippocampus, for example, which plays a critical role in memory formation and is one of the first regions affected by Alzheimer's disease, can be visualized with unprecedented detail using these advanced systems. This enables detection of subtle changes that would be invisible on conventional MRI scans.</p>
      
      <h2>Diffusion Tensor Imaging (DTI)</h2>
      <p>Diffusion Tensor Imaging is an MRI technique that measures the diffusion of water molecules in biological tissues. In the brain, this can help visualize white matter tracts and assess their integrity. Since many neurodegenerative diseases affect white matter connectivity, DTI has become an invaluable tool in early detection.</p>
      
      <p>Recent advances in DTI acquisition and analysis methods have improved its sensitivity to subtle changes in white matter structure. These improvements allow researchers and clinicians to identify alterations in brain connectivity that may precede clinical symptoms by years or even decades.</p>
      
      <h2>Functional MRI (fMRI)</h2>
      <p>Functional MRI measures brain activity by detecting changes in blood flow. When a brain area is in use, blood flow to that region increases. By tracking these changes, fMRI creates maps of brain activity during various tasks or at rest.</p>
      
      <p>Recent advancements in fMRI technology include higher temporal resolution, reduced artifacts, and improved analytical methods. These improvements have enabled researchers to identify alterations in brain activity patterns that are characteristic of neurodegenerative diseases, often before structural changes become apparent.</p>
      
      <h2>Artificial Intelligence and MRI</h2>
      <p>Perhaps the most significant recent advancement in MRI technology for neurodegenerative disease detection is the integration of artificial intelligence (AI) with imaging data. Machine learning algorithms, particularly deep learning, can analyze MRI scans and identify patterns that may be invisible to the human eye.</p>
      
      <p>These AI systems are trained on thousands of MRI scans from individuals with and without neurodegenerative diseases. Once trained, they can recognize subtle patterns associated with early-stage disease, potentially enabling diagnosis years before symptoms appear.</p>
      
      <p>Our own system at Adeptus Vita utilizes these cutting-edge technologies to provide accurate, early detection of Alzheimer's and other forms of dementia using MRI scans.</p>
      
      <h2>Conclusion</h2>
      <p>The field of MRI technology continues to advance rapidly, offering new opportunities for early detection and diagnosis of neurodegenerative diseases. By combining cutting-edge hardware with sophisticated software and artificial intelligence, we are entering an era where detection of these conditions at their earliest, most treatable stages is becoming a reality.</p>
      
      <p>As these technologies continue to evolve and become more widely available, we can hope for better outcomes for patients with Alzheimer's disease and other forms of dementia through earlier intervention and treatment.</p>
    `,
  },
  "early-detection-benefits": {
    title: "The Benefits of Early Detection in Alzheimer's Disease",
    date: "2023-10-27",
    readingTime: "8 min read",
    category: "Research",
    author: "Dr. Michael Chen",
    content: `
      <h2>Why Early Detection Matters</h2>
      <p>Early detection of Alzheimer's disease and other forms of dementia is crucial for several reasons. While there is currently no cure for Alzheimer's, early diagnosis allows for interventions that can significantly improve quality of life and potentially slow disease progression.</p>
      
      <p>In this article, we'll explore the numerous benefits of early detection and how advanced diagnostic tools like Adeptus Vita's AI-powered MRI analysis are making early diagnosis more accessible.</p>
      
      <h2>Access to Treatments and Clinical Trials</h2>
      <p>One of the most significant benefits of early detection is access to treatments that may help manage symptoms and slow disease progression. Medications such as cholinesterase inhibitors and memantine can be more effective when started early in the disease course.</p>
      
      <p>Additionally, early diagnosis opens the door to participation in clinical trials testing new treatments. These trials often seek participants who are in the earliest stages of the disease, where interventions may have the greatest impact.</p>
      
      <h2>Lifestyle Modifications</h2>
      <p>Research has shown that certain lifestyle modifications can help slow cognitive decline in people with early-stage Alzheimer's. These include regular physical exercise, cognitive stimulation, a healthy diet, good sleep hygiene, and stress management.</p>
      
      <p>When diagnosis occurs early, individuals have more time to implement these changes and potentially preserve cognitive function for longer periods.</p>
      
      <h2>Planning for the Future</h2>
      <p>An early diagnosis allows individuals and their families to plan for the future while the person with dementia can still actively participate in decision-making. This includes financial planning, legal arrangements, advance directives, and discussions about care preferences.</p>
      
      <p>Having these plans in place reduces stress on both the individual and their caregivers as the disease progresses.</p>
      
      <h2>Support and Resources</h2>
      <p>Early diagnosis connects individuals and families with support services and resources sooner. This includes support groups, educational programs, respite care, and other community services designed to help those affected by Alzheimer's and their caregivers.</p>
      
      <p>These resources can provide valuable information, emotional support, and practical assistance throughout the disease journey.</p>
      
      <h2>Reduced Healthcare Costs</h2>
      <p>Studies have shown that early diagnosis and intervention can actually reduce overall healthcare costs associated with Alzheimer's disease. This is due to better management of symptoms, fewer hospitalizations, delayed need for long-term care, and improved caregiver well-being.</p>
      
      <p>From a public health perspective, early detection and intervention strategies have the potential to significantly reduce the economic burden of Alzheimer's disease.</p>
      
      <h2>Improved Quality of Life</h2>
      <p>Perhaps most importantly, early detection can lead to improved quality of life for both individuals with Alzheimer's and their caregivers. By addressing symptoms early, implementing appropriate interventions, and planning for the future, many of the challenges associated with dementia can be better managed.</p>
      
      <p>This can result in more good days, preserved independence for longer periods, and a greater sense of control over the disease process.</p>
      
      <h2>Conclusion</h2>
      <p>The benefits of early detection in Alzheimer's disease are clear and multifaceted. As diagnostic technologies continue to advance, including AI-powered tools like Adeptus Vita's platform, we can hope to identify these conditions at increasingly earlier stages.</p>
      
      <p>Early detection isn't just about knowing sooner—it's about having more time: more time for treatment, for planning, for living well despite the diagnosis. And that makes all the difference.</p>
    `,
  },
}

function formatDate(dateString: string) {
  const options: Intl.DateTimeFormatOptions = { year: "numeric", month: "long", day: "numeric" }
  return new Date(dateString).toLocaleDateString("en-US", options)
}

export default function BlogPostPage() {
  const params = useParams()
  const slug = params.slug as string

  // Get post data or show not found
  const post = blogPosts[slug as keyof typeof blogPosts]

  if (!post) {
    return (
      <Shell>
        <div className="py-10 space-y-8 animate-fade-in">
          <div className="space-y-4">
            <h1 className="text-3xl font-bold tracking-tight">Blog Post Not Found</h1>
            <p className="text-lg text-muted-foreground">
              The blog post you're looking for doesn't exist or has been removed.
            </p>
            <Link href="/blog">
              <Button variant="outline">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Blog
              </Button>
            </Link>
          </div>
        </div>
      </Shell>
    )
  }

  return (
    <Shell>
      <article className="py-10 space-y-8 animate-fade-in">
        {/* Back link */}
        <Link href="/blog">
          <Button variant="ghost" className="mb-8 pl-0">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Blog
          </Button>
        </Link>

        {/* Post header */}
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            <span className="px-2 py-1 text-xs rounded-md bg-primary/10 text-primary">{post.category}</span>
            <span className="flex items-center">
              <CalendarIcon className="w-3 h-3 mr-1" />
              {formatDate(post.date)}
            </span>
            <span className="flex items-center">
              <Clock className="w-3 h-3 mr-1" />
              {post.readingTime}
            </span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight md:text-4xl">{post.title}</h1>
          <p className="text-lg text-muted-foreground">By {post.author}</p>
        </div>

        {/* Featured image placeholder */}
        <div className="w-full h-64 bg-muted rounded-lg grid-lines flex items-center justify-center">
          <span className="text-muted-foreground">Featured Image</span>
        </div>

        {/* Post content */}
        <div
          className="prose prose-zinc dark:prose-invert max-w-none"
          dangerouslySetInnerHTML={{ __html: post.content }}
        />

        {/* Post footer */}
        <div className="border-t border-border pt-6 mt-10">
          <h3 className="text-lg font-semibold mb-4">Continue Reading</h3>
          <div className="flex flex-col sm:flex-row gap-4">
            <Link href="/blog" className="w-full">
              <Button variant="outline" className="w-full">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Return to Blog
              </Button>
            </Link>
            <Link href="/model" className="w-full">
              <Button className="w-full">Try the Diagnostic Model</Button>
            </Link>
          </div>
        </div>
      </article>
    </Shell>
  )
}
