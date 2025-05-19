import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Brain, FileText, Users, Award, Building, Microscope } from "lucide-react"
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="space-y-6 py-10 px-20">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            About Adeptus Vita
          </h1>
          <p className="text-lg text-muted-foreground">
            Pioneering AI-powered diagnosis of Alzheimer's and dementia using
            MRI scans.
          </p>
        </div>
        <Button variant="outline" asChild className="gap-2">
          <Link href="/">
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>
        </Button>
      </div>

      {/* Hero section */}
      <div className="relative w-full h-64 md:h-80 rounded-lg overflow-hidden bg-gradient-to-r from-primary/20 to-primary/5 grid-lines flex items-center justify-center">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center space-y-4 p-6 backdrop-blur-sm bg-background/30 rounded-lg max-w-xl">
            <Brain className="h-12 w-12 mx-auto text-primary" />
            <h2 className="text-2xl font-bold">
              Early Detection, Better Outcomes
            </h2>
            <p className="text-muted-foreground">
              Our mission is to harness the power of artificial intelligence to
              detect neurodegenerative diseases at their earliest stages, when
              intervention is most effective.
            </p>
          </div>
        </div>
      </div>

      {/* Tabs section */}
      <Tabs defaultValue="mission" className="w-full">
        <TabsList className="grid w-full grid-cols-2 md:grid-cols-4">
          <TabsTrigger value="mission">
            <Award className="mr-2 h-4 w-4" />
            <span className="hidden sm:inline">Mission</span>
            <span className="sm:hidden">Mission</span>
          </TabsTrigger>
          <TabsTrigger value="team">
            <Users className="mr-2 h-4 w-4" />
            <span className="hidden sm:inline">Team</span>
            <span className="sm:hidden">Team</span>
          </TabsTrigger>
          <TabsTrigger value="technology">
            <Brain className="mr-2 h-4 w-4" />
            <span className="hidden sm:inline">Technology</span>
            <span className="sm:hidden">Tech</span>
          </TabsTrigger>
          <TabsTrigger value="research">
            <Microscope className="mr-2 h-4 w-4" />
            <span className="hidden sm:inline">Research</span>
            <span className="sm:hidden">Research</span>
          </TabsTrigger>
        </TabsList>

        {/* Mission Tab */}
        <TabsContent value="mission" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Our Mission</CardTitle>
              <CardDescription>
                Transforming neurological healthcare through early detection and
                intervention
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                Adeptus Vita was founded with a singular purpose: to
                revolutionize the detection and diagnosis of neurodegenerative
                diseases like Alzheimer's and dementia. By leveraging
                cutting-edge artificial intelligence and machine learning
                technologies, we aim to identify these conditions at their
                earliest stages, when treatment interventions are most
                effective.
              </p>
              <p>
                Our vision is a world where neurodegenerative diseases no longer
                rob individuals of their memories, independence, and dignity.
                Through early detection, we believe we can significantly improve
                patient outcomes, reduce healthcare costs, and ultimately
                contribute to the global effort to combat these devastating
                conditions.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center">
                      <Award className="mr-2 h-5 w-5 text-primary" />
                      Excellence
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      We are committed to the highest standards of scientific
                      rigor and clinical excellence in everything we do.
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center">
                      <Users className="mr-2 h-5 w-5 text-primary" />
                      Accessibility
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      We believe advanced diagnostic tools should be accessible
                      to all, regardless of geographic or economic barriers.
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center">
                      <FileText className="mr-2 h-5 w-5 text-primary" />
                      Transparency
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      We maintain complete transparency in our methods, results,
                      and limitations to build trust with patients and
                      providers.
                    </p>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Team Tab */}
        <TabsContent value="team" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Our Team</CardTitle>
              <CardDescription>
                A multidisciplinary group of experts dedicated to our mission
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Team Member 1 */}
                <div className="flex flex-col items-center text-center space-y-2">
                  <div className="w-24 h-24 rounded-full bg-muted flex items-center justify-center">
                    <Users className="h-12 w-12 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="font-medium">Montaser N. Amoor</h3>
                    <p className="text-sm text-muted-foreground">
                      Data Science Student
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Aspiring data scientist with a passion for AI and
                    healthcare technology.
                  </p>
                </div>

                {/* Team Member 2 */}
                <div className="flex flex-col items-center text-center space-y-2">
                  <div className="w-24 h-24 rounded-full bg-muted flex items-center justify-center">
                    <Users className="h-12 w-12 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="font-medium">Reta H. Basbous</h3>
                    <p className="text-sm text-muted-foreground">
                      Data Science Student
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Data science enthusiast with a focus on machine learning
                    applications in healthcare.
                  </p>
                </div>

                {/* Team Member 3 */}
                <div className="flex flex-col items-center text-center space-y-2">
                  <div className="w-24 h-24 rounded-full bg-muted flex items-center justify-center">
                    <Users className="h-12 w-12 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="font-medium">Raghad W. Sayen</h3>
                    <p className="text-sm text-muted-foreground">
                      Data Science Student
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Aspiring data scientist with a passion for AI and
                    healthcare technology.
                  </p>
                </div>

                {/* Team Member 4 */}
                <div className="flex flex-col items-center text-center space-y-2">
                  <div className="w-24 h-24 rounded-full bg-muted flex items-center justify-center">
                    <Users className="h-12 w-12 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="font-medium">Laila B. Abu-Khalaf</h3>
                    <p className="text-sm text-muted-foreground">
                      Data Science Student
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Aspiring data scientist with a passion for AI and
                    healthcare technology.
                  </p>
                </div>

                {/* Team Member 5 */}
                <div className="flex flex-col items-center text-center space-y-2">
                  <div className="w-24 h-24 rounded-full bg-muted flex items-center justify-center">
                    <Users className="h-12 w-12 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="font-medium">Tsneem H. Sbaih</h3>
                    <p className="text-sm text-muted-foreground">
                      Data Science Student
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Aspiring data scientist with a passion for AI and
                    healthcare technology.
                  </p>
                </div>

                {/* Team Member 6 */}
                <div className="flex flex-col items-center text-center space-y-2">
                  <div className="w-24 h-24 rounded-full bg-muted flex items-center justify-center">
                    <Users className="h-12 w-12 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="font-medium">Asim G. Al-Akhras</h3>
                    <p className="text-sm text-muted-foreground">
                      Data Science Student
                    </p>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Machine learning enthusiast with a focus on medical data
                    analysis and predictive modeling.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Technology Tab */}
        <TabsContent value="technology" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Our Technology</CardTitle>
              <CardDescription>
                Advanced AI systems for accurate and early diagnosis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                At the core of Adeptus Vita's capabilities is our proprietary
                neural network architecture, specifically designed to analyze
                MRI scans for subtle patterns and changes that may indicate the
                onset of neurodegenerative diseases. Our technology has been
                trained on thousands of MRI scans, allowing it to identify
                markers that might be invisible to the human eye.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">
                      Neural Network Architecture
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Our artificial neural network (ANN) is specifically
                      optimized for medical imaging analysis, with particular
                      attention to the structural changes associated with
                      neurodegenerative diseases.
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">ML Framework</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      We utilize TensorFlow and Keras for model training and
                      evaluation, ensuring a robust and scalable solution for
                      our diagnostic system.
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Website</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Our web application is built using Next.js, providing a
                      user-friendly interface using React UI, Tailwind CSS, and Shadcn components.
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">
                      Programming Languages
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      Our tech stack uses Python for the machine learning and TypeScript for the web application, ensuring a seamless integration between the backend and frontend.
                    </p>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Research Tab */}
        <TabsContent value="research" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Our Research</CardTitle>
              <CardDescription>
                Advancing the science of neurodegenerative disease detection
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                Research is at the heart of everything we do at Adeptus Vita.
                Our team of scientists and clinicians are constantly working to
                improve our diagnostic capabilities, validate our methods
                through rigorous studies, and contribute to the broader
                scientific understanding of neurodegenerative diseases.
              </p>

              <h3 className="text-lg font-medium mt-6">Key Research Areas</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                <div className="flex gap-2">
                  <Microscope className="h-5 w-5 text-primary flex-shrink-0" />
                  <div>
                    <h4 className="font-medium">Biomarker Identification</h4>
                    <p className="text-sm text-muted-foreground">
                      Identifying novel biomarkers for earlier and more accurate
                      diagnosis.
                    </p>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Brain className="h-5 w-5 text-primary flex-shrink-0" />
                  <div>
                    <h4 className="font-medium">Algorithm Optimization</h4>
                    <p className="text-sm text-muted-foreground">
                      Refining our neural networks for improved sensitivity and
                      specificity.
                    </p>
                  </div>
                </div>

                <div className="flex gap-2">
                  <FileText className="h-5 w-5 text-primary flex-shrink-0" />
                  <div>
                    <h4 className="font-medium">Clinical Validation</h4>
                    <p className="text-sm text-muted-foreground">
                      Conducting rigorous studies to validate our diagnostic
                      approach.
                    </p>
                  </div>
                </div>

                <div className="flex gap-2">
                  <Users className="h-5 w-5 text-primary flex-shrink-0" />
                  <div>
                    <h4 className="font-medium">Population Studies</h4>
                    <p className="text-sm text-muted-foreground">
                      Analyzing disease patterns across diverse demographic
                      groups.
                    </p>
                  </div>
                </div>
              </div>

              <h3 className="text-lg font-medium mt-6">Publications</h3>
              <ul className="space-y-2 mt-2">
                <li className="text-sm">
                  <span className="font-medium">Johnson S, Chen M, et al.</span>{" "}
                  (2023). "Early Detection of Alzheimer's Disease Using
                  Convolutional Neural Networks and MRI Analysis."{" "}
                  <em>Journal of Neurological Imaging</em>, 45(3), 234-248.
                </li>
                <li className="text-sm">
                  <span className="font-medium">
                    Rodriguez E, Wilson J, et al.
                  </span>{" "}
                  (2022). "Identification of Novel Biomarkers for Dementia Risk
                  Assessment." <em>Neurodegenerative Disease Research</em>,
                  18(2), 112-125.
                </li>
                <li className="text-sm">
                  <span className="font-medium">Kim S, Thompson A, et al.</span>{" "}
                  (2022). "Machine Learning Approaches to Predict Cognitive
                  Decline: A Comparative Analysis." <em>AI in Medicine</em>,
                  12(4), 345-359.
                </li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Partners section */}
      <Card>
        <CardHeader>
          <CardTitle>Our Partners</CardTitle>
          <CardDescription>
            Collaborating with leading institutions to advance our mission
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="flex flex-col items-center text-center">
              <div className="h-16 w-16 bg-muted rounded-md flex items-center justify-center mb-2">
                <Building className="h-8 w-8 text-muted-foreground" />
              </div>
              <h4 className="text-sm font-medium">University Medical Center</h4>
            </div>
            <div className="flex flex-col items-center text-center">
              <div className="h-16 w-16 bg-muted rounded-md flex items-center justify-center mb-2">
                <Building className="h-8 w-8 text-muted-foreground" />
              </div>
              <h4 className="text-sm font-medium">
                National Research Institute
              </h4>
            </div>
            <div className="flex flex-col items-center text-center">
              <div className="h-16 w-16 bg-muted rounded-md flex items-center justify-center mb-2">
                <Building className="h-8 w-8 text-muted-foreground" />
              </div>
              <h4 className="text-sm font-medium">Global Health Foundation</h4>
            </div>
            <div className="flex flex-col items-center text-center">
              <div className="h-16 w-16 bg-muted rounded-md flex items-center justify-center mb-2">
                <Building className="h-8 w-8 text-muted-foreground" />
              </div>
              <h4 className="text-sm font-medium">Tech Innovation Labs</h4>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
