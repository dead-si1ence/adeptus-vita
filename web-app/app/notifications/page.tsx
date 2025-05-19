"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Skeleton } from "@/components/ui/skeleton"
import {
  BellOff,
  Brain,
  Calendar,
  Check,
  FileText,
  Info,
  Settings,
  Trash2,
  ArrowLeft,
  Bell,
  Filter,
  RefreshCw,
  CheckCircle,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import Link from "next/link"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// Mock notification data
const mockNotifications = [
  {
    id: "1",
    title: "Diagnostic Result Ready",
    description: "Your MRI scan analysis has been completed. View the results now.",
    date: "2 hours ago",
    read: false,
    type: "diagnostic",
    action: "/model",
    priority: "high",
  },
  {
    id: "2",
    title: "New Blog Post",
    description: "Recent Advancements in MRI Technology for Neurodegenerative Disease Detection",
    date: "1 day ago",
    read: true,
    type: "blog",
    action: "/blog/advancements-in-mri-technology",
    priority: "medium",
  },
  {
    id: "3",
    title: "Reminder: Follow-up Scan",
    description: "Your scheduled follow-up scan is due in 2 weeks. Please confirm your appointment.",
    date: "2 days ago",
    read: false,
    type: "reminder",
    action: "#",
    priority: "high",
  },
  {
    id: "4",
    title: "System Update",
    description: "We've improved our diagnostic algorithm for better accuracy. Learn more about the changes.",
    date: "1 week ago",
    read: true,
    type: "system",
    action: "#",
    priority: "low",
  },
  {
    id: "5",
    title: "New Research Publication",
    description: "A new research paper on early detection methods has been published in our blog.",
    date: "1 week ago",
    read: false,
    type: "blog",
    action: "/blog",
    priority: "medium",
  },
]

/**
 * Notifications page component
 * Displays user notifications and notification settings
 */
export default function NotificationsPage() {
  const [notifications, setNotifications] = useState(mockNotifications)
  const [isLoading, setIsLoading] = useState(false)
  const [filter, setFilter] = useState("all")

  // Count unread notifications
  const unreadCount = notifications.filter((n) => !n.read).length

  // Mark all as read
  const markAllAsRead = () => {
    setIsLoading(true)
    setTimeout(() => {
      setNotifications(notifications.map((n) => ({ ...n, read: true })))
      setIsLoading(false)
    }, 1000)
  }

  // Mark a single notification as read
  const markAsRead = (id: string) => {
    setNotifications(notifications.map((n) => (n.id === id ? { ...n, read: true } : n)))
  }

  // Delete a notification
  const deleteNotification = (id: string) => {
    setNotifications(notifications.filter((n) => n.id !== id))
  }

  // Clear all notifications
  const clearAll = () => {
    setIsLoading(true)
    setTimeout(() => {
      setNotifications([])
      setIsLoading(false)
    }, 1000)
  }

  // Refresh notifications (simulate)
  const refreshNotifications = () => {
    setIsLoading(true)
    setTimeout(() => {
      setIsLoading(false)
    }, 1500)
  }

  return (
    <div className="space-y-6 py-10 px-20">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Notifications</h1>
          <p className="text-muted-foreground mt-2">
            Stay updated with important information and updates.
          </p>
        </div>
        <Button variant="outline" asChild className="gap-2">
          <Link href="/">
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>
        </Button>
      </div>

      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={refreshNotifications}
            disabled={isLoading}
            className="gap-2"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </Button>

          <Select defaultValue="all" onValueChange={setFilter}>
            <SelectTrigger className="w-[180px]">
              <Filter className="h-4 w-4 mr-2" />
              <SelectValue placeholder="Filter by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Notifications</SelectItem>
              <SelectItem value="unread">Unread Only</SelectItem>
              <SelectItem value="high">High Priority</SelectItem>
              <SelectItem value="diagnostic">Diagnostic</SelectItem>
              <SelectItem value="system">System</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          {unreadCount > 0 && (
            <Button
              variant="outline"
              onClick={markAllAsRead}
              disabled={isLoading}
              className="gap-2 w-full sm:w-auto"
            >
              <CheckCircle className="h-4 w-4" />
              Mark all as read
            </Button>
          )}
          {notifications.length > 0 && (
            <Button
              variant="outline"
              onClick={clearAll}
              disabled={isLoading}
              className="gap-2 w-full sm:w-auto"
            >
              <Trash2 className="h-4 w-4" />
              Clear all
            </Button>
          )}
        </div>
      </div>

      <Tabs defaultValue="all" className="w-full">
        <TabsList>
          <TabsTrigger value="all">
            All
            {unreadCount > 0 && (
              <Badge variant="secondary" className="ml-2">
                {unreadCount}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="unread">Unread</TabsTrigger>
          <TabsTrigger value="diagnostic">Diagnostic</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-4">
          {isLoading ? (
            <div className="space-y-4">
              <NotificationSkeleton />
              <NotificationSkeleton />
              <NotificationSkeleton />
            </div>
          ) : notifications.length > 0 ? (
            <div className="space-y-4">
              {notifications
                .filter(
                  (n) =>
                    filter === "all" ||
                    (filter === "unread" && !n.read) ||
                    (filter === "high" && n.priority === "high") ||
                    filter === n.type
                )
                .map((notification) => (
                  <NotificationCard
                    key={notification.id}
                    notification={notification}
                    onMarkAsRead={markAsRead}
                    onDelete={deleteNotification}
                  />
                ))}
            </div>
          ) : (
            <EmptyNotifications />
          )}
        </TabsContent>

        <TabsContent value="unread" className="space-y-4">
          {isLoading ? (
            <div className="space-y-4">
              <NotificationSkeleton />
              <NotificationSkeleton />
            </div>
          ) : notifications.filter((n) => !n.read).length > 0 ? (
            <div className="space-y-4">
              {notifications
                .filter((n) => !n.read)
                .map((notification) => (
                  <NotificationCard
                    key={notification.id}
                    notification={notification}
                    onMarkAsRead={markAsRead}
                    onDelete={deleteNotification}
                  />
                ))}
            </div>
          ) : (
            <EmptyNotifications message="No unread notifications" />
          )}
        </TabsContent>

        <TabsContent value="diagnostic" className="space-y-4">
          {isLoading ? (
            <div className="space-y-4">
              <NotificationSkeleton />
            </div>
          ) : notifications.filter((n) => n.type === "diagnostic").length >
            0 ? (
            <div className="space-y-4">
              {notifications
                .filter((n) => n.type === "diagnostic")
                .map((notification) => (
                  <NotificationCard
                    key={notification.id}
                    notification={notification}
                    onMarkAsRead={markAsRead}
                    onDelete={deleteNotification}
                  />
                ))}
            </div>
          ) : (
            <EmptyNotifications message="No diagnostic notifications" />
          )}
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          {isLoading ? (
            <div className="space-y-4">
              <NotificationSkeleton />
            </div>
          ) : notifications.filter((n) => n.type === "system").length > 0 ? (
            <div className="space-y-4">
              {notifications
                .filter((n) => n.type === "system")
                .map((notification) => (
                  <NotificationCard
                    key={notification.id}
                    notification={notification}
                    onMarkAsRead={markAsRead}
                    onDelete={deleteNotification}
                  />
                ))}
            </div>
          ) : (
            <EmptyNotifications message="No system notifications" />
          )}
        </TabsContent>
      </Tabs>

      <Card>
        <CardHeader>
          <CardTitle>Notification Settings</CardTitle>
          <CardDescription>
            Manage how and when you receive notifications
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Diagnostic Results</Label>
              <p className="text-sm text-muted-foreground">
                Receive notifications when your diagnostic results are ready
              </p>
            </div>
            <Switch defaultChecked />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Blog Updates</Label>
              <p className="text-sm text-muted-foreground">
                Get notified about new blog posts and articles
              </p>
            </div>
            <Switch defaultChecked />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>System Updates</Label>
              <p className="text-sm text-muted-foreground">
                Receive notifications about system changes and improvements
              </p>
            </div>
            <Switch defaultChecked />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>Reminders</Label>
              <p className="text-sm text-muted-foreground">
                Get reminders for scheduled scans and follow-ups
              </p>
            </div>
            <Switch defaultChecked />
          </div>

          <Separator />

          <div className="space-y-2">
            <Label>Notification Delivery</Label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card className="border-dashed">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center">
                    <Bell className="h-4 w-4 mr-2" />
                    In-App Notifications
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Enabled</span>
                    <Switch defaultChecked />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-dashed">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base flex items-center">
                    <Mail className="h-4 w-4 mr-2" />
                    Email Notifications
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Enabled</span>
                    <Switch defaultChecked />
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <Button>Save Preferences</Button>
        </CardFooter>
      </Card>
    </div>
  );
}

// Notification card component
function NotificationCard({
  notification,
  onMarkAsRead,
  onDelete,
}: {
  notification: (typeof mockNotifications)[0]
  onMarkAsRead: (id: string) => void
  onDelete: (id: string) => void
}) {
  // Get icon based on notification type
  const getIcon = () => {
    switch (notification.type) {
      case "diagnostic":
        return <Brain className="h-5 w-5" />
      case "blog":
        return <FileText className="h-5 w-5" />
      case "reminder":
        return <Calendar className="h-5 w-5" />
      case "system":
        return <Settings className="h-5 w-5" />
      default:
        return <Info className="h-5 w-5" />
    }
  }

  // Get priority badge
  const getPriorityBadge = () => {
    switch (notification.priority) {
      case "high":
        return <Badge className="bg-red-500">High</Badge>
      case "medium":
        return <Badge className="bg-amber-500">Medium</Badge>
      case "low":
        return <Badge className="bg-green-500">Low</Badge>
      default:
        return null
    }
  }

  return (
    <Card className={notification.read ? "bg-card" : "bg-accent/5 border-accent/20"}>
      <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
        <div className="flex items-center gap-2">
          <div className={`rounded-full p-1 ${notification.read ? "bg-muted" : "bg-primary/10 text-primary"}`}>
            {getIcon()}
          </div>
          <div>
            <CardTitle className="text-base">{notification.title}</CardTitle>
            <CardDescription className="text-xs">{notification.date}</CardDescription>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {getPriorityBadge()}
          {!notification.read && (
            <Badge variant="secondary" className="ml-2">
              New
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm">{notification.description}</p>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="link" asChild className="px-0">
          <Link href={notification.action}>View Details</Link>
        </Button>
        <div className="flex gap-2">
          {!notification.read && (
            <Button variant="ghost" size="sm" onClick={() => onMarkAsRead(notification.id)}>
              <Check className="mr-2 h-4 w-4" />
              Mark as read
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={() => onDelete(notification.id)}>
            <Trash2 className="mr-2 h-4 w-4" />
            Delete
          </Button>
        </div>
      </CardFooter>
    </Card>
  )
}

// Empty notifications component
function EmptyNotifications({ message = "No notifications" }: { message?: string }) {
  return (
    <div className="text-center py-12">
      <BellOff className="mx-auto h-12 w-12 text-muted-foreground" />
      <h2 className="mt-4 text-lg font-medium">{message}</h2>
      <p className="mt-2 text-sm text-muted-foreground">
        You're all caught up! We'll notify you when there's something new.
      </p>
    </div>
  )
}

// Skeleton loader for notifications
function NotificationSkeleton() {
  return (
    <Card>
      <CardHeader className="flex flex-row items-start space-y-0 pb-2">
        <div className="flex items-center gap-2">
          <Skeleton className="h-10 w-10 rounded-full" />
          <div>
            <Skeleton className="h-5 w-40" />
            <Skeleton className="h-4 w-24 mt-2" />
          </div>
        </div>
      </CardHeader>
      <CardContent className="pb-2">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6 mt-2" />
      </CardContent>
      <CardFooter className="flex justify-between">
        <Skeleton className="h-9 w-28" />
        <Skeleton className="h-9 w-28" />
      </CardFooter>
    </Card>
  )
}

// Mail icon component
function Mail(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="20" height="16" x="2" y="4" rx="2" />
      <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
    </svg>
  )
}
