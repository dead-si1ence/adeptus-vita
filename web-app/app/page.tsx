import type React from "react";
import Link from "next/link";
import { Shell } from "@/components/shell";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Brain, FileText, AlertCircle } from "lucide-react";

/**
 * Card information type for feature cards
 */
type FeatureCard = {
  icon: React.ElementType;
  title: string;
  description: string;
  content: string;
  link: {
    href: string;
    label: string;
  };
  delay?: string;
};

/**
 * Hero section component
 */
function HeroSection() {
  return (
    <section className="py-12 md:py-16 animate-fade-in">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center space-y-4 text-center">
          <div className="space-y-2">
            <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
              <span className="text-primary">ADEPTUS</span>·
              <span className="text-primary">VITA</span>
            </h1>
            <p className="max-w-[700px] text-lg text-muted-foreground md:text-xl">
              AI-powered diagnosis of Alzheimer's and dementia using MRI scans
            </p>
          </div>
          <div className="space-x-4">
            <Link href="/model">
              <Button className="bg-primary text-primary-foreground">
                Start Diagnosis
              </Button>
            </Link>
            <Link href="/blog">
              <Button variant="outline">Learn More</Button>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

/**
 * Feature card component
 */
function FeatureCard({
  icon: Icon,
  title,
  description,
  content,
  link,
  delay,
}: FeatureCard) {
  return (
    <Card
      className={`bg-card animate-slide-up border-border/40 ${delay || ""}`}
    >
      <CardHeader>
        <Icon className="h-8 w-8 mb-2 text-primary" />
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground">{content}</p>
      </CardContent>
      <CardFooter>
        <Link href={link.href}>
          <Button variant="ghost" className="text-primary w-full">
            {link.label}
          </Button>
        </Link>
      </CardFooter>
    </Card>
  );
}

/**
 * Features section component
 */
function FeaturesSection() {
  const features: FeatureCard[] = [
    {
      icon: Brain,
      title: "Artificial Neural Network",
      description:
        "State-of-the-art artificial neural networks for accurate diagnosis",
      content:
        "Our model has been trained on thousands of MRI scans to detect early signs of neurodegenerative diseases with high accuracy.",
      link: { href: "/model", label: "Try it now" },
    },
    {
      icon: FileText,
      title: "Research & Resources",
      description: "Stay updated with the latest research and developments",
      content:
        "Our blog features the latest research findings, technological advancements, and educational resources in the field of neurodegenerative disease diagnosis.",
      link: { href: "/blog", label: "Read the blog" },
      delay: "delay-100",
    },
    {
      icon: AlertCircle,
      title: "Early Detection",
      description: "Identify signs before symptoms appear",
      content:
        "Early detection of neurodegenerative diseases like Alzheimer's can lead to better management and improved quality of life for patients.",
      link: { href: "/blog", label: "Learn more" },
      delay: "delay-200",
    },
  ];

  return (
    <section className="py-8 md:py-12">
      <div className="container px-4 md:px-6">
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => (
            <FeatureCard key={index} {...feature} />
          ))}
        </div>
      </div>
    </section>
  );
}

/**
 * About section component
 */
function AboutSection() {
  return (
    <section className="py-8 md:py-12 bg-card/50 border-y">
      <div className="container px-4 md:px-6">
        <div className="grid gap-10 md:grid-cols-2 items-center">
          <div className="space-y-4">
            <h2 className="text-2xl font-bold tracking-tighter md:text-3xl">
              About Adeptus Vita
            </h2>
            <p className="text-muted-foreground">
              Adeptus Vita was created to harness the power of AI in the fight
              against Alzheimer's and other forms of dementia. Our mission is to
              provide accessible, accurate diagnostic tools that can help
              identify neurodegenerative diseases at their earliest stages.
            </p>
            <p className="text-muted-foreground">
              Using advanced machine learning techniques and neural networks,
              our system analyzes MRI scans to detect subtle patterns and
              changes that may indicate the onset of these conditions, often
              before clinical symptoms appear.
            </p>
          </div>
          <div className="bg-muted p-6 rounded-lg grid-lines">
            <div className="space-y-2 animate-pulse">
              <div className="h-2 bg-primary/20 rounded w-3/4"></div>
              <div className="h-2 bg-primary/20 rounded w-1/2"></div>
              <div className="h-2 bg-primary/20 rounded w-5/6"></div>
              <div className="h-10 mt-4 bg-primary/10 rounded w-full"></div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/**
 * Home page component
 * Main landing page for the application
 */
export default function HomePage() {
  return (
    <Shell>
      <HeroSection />
      <FeaturesSection />
      <AboutSection />
    </Shell>
  );
}
