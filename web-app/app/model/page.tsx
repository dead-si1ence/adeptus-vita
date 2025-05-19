"use client"

import type React from "react"

import { useState } from "react"
import Image from "next/image"
import { Shell } from "@/components/shell"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Brain, Upload, History, X, FileUp, AlertCircle, Check } from "lucide-react"

import NonDemented from "@/public/test/non-demented.jpg"
import MildDemented from "@/public/test/mild-demented.jpg"
import ModerateDemented from "@/public/test/moderate-demented.jpg"

interface Prediction {
  id: string
  date: string
  result: string
  // confidence: number
  imageUrl: string
}

export default function ModelPage() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<{ diagnosis: string; confidence: number } | null>(null)
  const [history, setHistory] = useState<Prediction[]>([
    {
      id: "1",
      date: "2025-5-14",
      result: "Negative",
      // confidence: 94,
      imageUrl: NonDemented.src,
    },
    {
      id: "2",
      date: "2025-5-14",
      result: "Positive - Early Stage",
      // confidence: 87,
      imageUrl: MildDemented.src,
    },
    {
      id: "3",
      date: "2025-5-14",
      result: "Positive - Moderate",
      // confidence: 92,
      imageUrl: ModerateDemented.src,
    },
  ]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      const objectUrl = URL.createObjectURL(selectedFile)
      setPreview(objectUrl)
      setResult(null)
    }
  }

  const resetUpload = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
  }

  const analyzeImage = async (e: React.FormEvent) => {
    if (!file) return;

    console.log("Analyzing image:", file.name);
    setIsLoading(true);

    try {
      // Create form data to send the file
      const formData = new FormData();
      formData.append("file", file);

      console.log("Sending request to /api with file:", file.name);

      // Send the file to the API endpoint
      const response = await fetch("/api", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to process image");
      }

      const formattedResult = {
        diagnosis: data.prediction.diagnosis,
        confidence: data.prediction.confidence
      };
      
      console.log("Received prediction:", formattedResult);
      setResult(formattedResult);

    // Simulate AI processing with a timeout
    // setTimeout(() => {
    //   const mockResults = [
    //     { diagnosis: "Negative", confidence: 94 },
    //     { diagnosis: "Positive - Early Stage", confidence: 87 },
    //     { diagnosis: "Positive - Moderate", confidence: 92 },
    //   ];

    //   const randomResult =
    //     mockResults[Math.floor(Math.random() * mockResults.length)];
    //   setResult(randomResult);

      // Add to history
      if (preview) {
        const newPrediction: Prediction = {
          id: Date.now().toString(),
          date: new Date().toISOString().split("T")[0],
          result: formattedResult.diagnosis,
          // confidence: formattedResult.confidence,
          imageUrl: preview,
        };

        setHistory((prev) => [newPrediction, ...prev]);
      }

      setIsLoading(false);
    } catch (error) {
      console.error("Error analyzing image:", error);
      setIsLoading(false);
      // Show a friendly error message
      setResult({
        diagnosis: "Error processing image",
        confidence: 0
      });
    }
  };

  return (
    <Shell>
      <div className="py-10 animate-fade-in">
        <h1 className="text-3xl font-bold tracking-tight mb-6">Diagnostic Model</h1>
        <p className="text-lg text-muted-foreground mb-8">
          Upload an MRI scan to get an AI-powered diagnosis for signs of Alzheimer's or dementia.
        </p>

        <Tabs defaultValue="upload" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="upload">
              <Upload className="mr-2 h-4 w-4" />
              Upload Scan
            </TabsTrigger>
            <TabsTrigger value="history">
              <History className="mr-2 h-4 w-4" />
              Prediction History
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="mt-0">
            <div className="grid md:grid-cols-2 gap-6">
              <Card className="md:col-span-1 border-border/40">
                <CardHeader>
                  <CardTitle>Upload MRI Scan</CardTitle>
                  <CardDescription>Upload a brain MRI scan in JPG, PNG or DICOM format</CardDescription>
                </CardHeader>
                <CardContent>
                  {!preview ? (
                    <div
                      className="border-2 border-dashed border-border rounded-lg p-12 text-center cursor-pointer hover:bg-muted/50 transition-colors"
                      onClick={() => document.getElementById("file-upload")?.click()}
                    >
                      <FileUp className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground mb-2">Click to upload or drag and drop</p>
                      <p className="text-xs text-muted-foreground">JPG, PNG or DICOM (max 10MB)</p>
                      <input
                        id="file-upload"
                        type="file"
                        className="hidden"
                        accept="image/jpeg,image/png,application/dicom"
                        onChange={handleFileChange}
                      />
                    </div>
                  ) : (
                    <div className="relative">
                      <Button
                        size="icon"
                        variant="secondary"
                        className="absolute top-2 right-2 z-10"
                        onClick={resetUpload}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                      <div className="rounded-lg overflow-hidden border border-border">
                        <Image
                          src={preview}
                          alt="MRI Preview"
                          width={400}
                          height={400}
                          className="w-full object-cover"
                        />
                      </div>
                    </div>
                  )}
                </CardContent>
                <CardFooter>
                  <Button onClick={analyzeImage} disabled={!file || isLoading} className="w-full">
                    {isLoading ? (
                      <>
                        <span className="mr-2">Processing</span>
                        <span className="animate-pulse">...</span>
                      </>
                    ) : (
                      <>
                        <Brain className="mr-2 h-4 w-4" />
                        Analyze Scan
                      </>
                    )}
                  </Button>
                </CardFooter>
              </Card>

              <Card className="md:col-span-1 border-border/40">
                <CardHeader>
                  <CardTitle>Diagnosis Results</CardTitle>
                  <CardDescription>AI-powered analysis of your MRI scan</CardDescription>
                </CardHeader>
                <CardContent className="min-h-[300px] flex items-center justify-center">
                  {!file && (
                    <div className="text-center text-muted-foreground">
                      <AlertCircle className="h-12 w-12 mx-auto mb-4" />
                      <p>Upload an MRI scan to see the diagnosis</p>
                    </div>
                  )}

                  {file && !result && !isLoading && (
                    <div className="text-center text-muted-foreground">
                      <Brain className="h-12 w-12 mx-auto mb-4" />
                      <p>Click "Analyze Scan" to process the image</p>
                    </div>
                  )}

                  {isLoading && (
                    <div className="text-center">
                      <div className="mb-4 relative h-20 w-20 mx-auto">
                        <div className="absolute inset-0 border-t-2 border-primary rounded-full animate-spin"></div>
                        <Brain className="h-12 w-12 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-primary" />
                      </div>
                      <p className="text-muted-foreground">Analyzing MRI scan...</p>
                      <p className="text-xs text-muted-foreground mt-2">This may take a moment</p>
                    </div>
                  )}

                  {result && (
                    <div className="w-full">
                      <div
                        className={`p-6 rounded-lg mb-6 ${
                          result.diagnosis.toLowerCase().includes("non")
                            ? "bg-green-500/10 border border-green-500/20"
                            : "bg-red-500/10 border border-red-500/20"
                        }`}
                      >
                        <div className="flex items-center mb-4">
                          {result.diagnosis.toLowerCase().includes("non") ? (
                            <Check className="h-8 w-8 text-green-500 mr-2" />
                          ) : (
                            <AlertCircle className="h-8 w-8 text-red-500 mr-2" />
                          )}
                          <h3 className="text-xl font-semibold">{result.diagnosis}</h3>
                        </div>
                        <div className="mb-4">
                          <div className="text-sm text-muted-foreground mb-1">Confidence: {result.confidence}%</div>
                          <div className="w-full bg-muted rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${
                                result.diagnosis.toLowerCase().includes("non") ? "bg-green-500" : "bg-red-500"
                              }`}
                              style={{ width: `${result.confidence}%` }}
                            ></div>
                          </div>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {result.diagnosis.toLowerCase().includes("non")
                            ? "No significant indicators of neurodegenerative disease detected."
                            : "Indicators of potential neurodegenerative disease detected. Please consult with a healthcare professional for further evaluation."}
                        </p>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Disclaimer: This is an automated analysis and should not replace professional medical advice.
                        Always consult with a healthcare provider for proper diagnosis and treatment.
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="history">
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {history.map((item) => (
                <Card key={item.id} className="border-border/40">
                  <CardHeader className="pb-2">
                    <CardTitle
                      className={`text-lg ${item.result.toLowerCase().includes("non") ? "text-green-500" : "text-red-500"}`}
                    >
                      {item.result}
                    </CardTitle>
                    {/* <CardDescription>
                      {item.date} • {item.confidence}% confidence
                    </CardDescription> */}
                  </CardHeader>
                  <CardContent>
                    <div className="rounded-md overflow-hidden">
                      <Image
                        src={item.imageUrl}
                        alt={`MRI Scan ${item.id}`}
                        width={300}
                        height={300}
                        className="w-full object-cover border border-border"
                      />
                    </div>
                  </CardContent>
                </Card>
              ))}

              {history.length === 0 && (
                <div className="col-span-full text-center py-12">
                  <History className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-muted-foreground">No prediction history available</p>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </Shell>
  )
}
