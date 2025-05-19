# Adeptus Vita

Adeptus Vita is an AI-powered platform for the diagnosis of Alzheimer's and dementia using MRI scans. The application provides a modern, responsive interface for uploading and analyzing brain scans, viewing diagnostic results, and accessing educational resources.

## Features

### Core Functionality

- **AI-Powered Diagnostic Model**: Upload MRI scans to receive automated analysis for signs of neurodegenerative diseases
- **Diagnostic History**: View and track previous scan results and analyses
- **Educational Blog**: Access articles and research on Alzheimer's, dementia, and diagnostic technologies
- **User Accounts**: Secure user authentication and profile management

### Technical Features

- **Modern UI/UX**: Clean, responsive interface built with Next.js and Tailwind CSS
- **Shadcn UI Components**: Consistent design system with accessible components
- **Dark/Light Mode**: Theme switching with system preference detection
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Breadcrumb Navigation**: Clear navigation path throughout the application
- **Search Functionality**: Search across blog posts and diagnostic results
- **Notification System**: In-app notifications for diagnostic results and updates
- **Settings Management**: Comprehensive user preference controls

## Pages

- **Home**: Landing page with overview of the platform's capabilities
- **Diagnostic Model**: Upload and analyze MRI scans
- **Blog**: Educational articles and research updates
- **Search**: Search functionality across the application
- **Settings**: User account and application preferences
- **Notifications**: System and diagnostic notifications
- **About**: Information about the platform and team

## Getting Started

### Prerequisites

- Node.js 18.0 or later
- npm or yarn

### Installation

1. Clone the repository:

   ```bash
   git clone <https://github.com/yourusername/adeptus-vita.git>
   cd adeptus-vita
   ```

2. Install dependencies:

   ```bash
   npm install

   # or

   yarn install
   ```

3. Run the development server:

   ```bash
   npm run dev

   # or

   yarn dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

## Usage Guide

### Diagnostic Model

1. Navigate to the Diagnostic Model page
2. Upload an MRI scan in JPG, PNG, or DICOM format
3. Click "Analyze Scan" to process the image
4. View the diagnostic results, including confidence level and recommendations
5. Access your diagnostic history in the "Prediction History" tab

### Blog

1. Browse articles on the Blog page
2. Click on an article to read the full content
3. Navigate between articles using the "Continue Reading" section

### Search

1. Use the search bar in the header or the dedicated Search page
2. Enter keywords related to diagnostics, research, or blog content
3. Filter and sort results by type, date, or relevance

### Settings

1. Access the Settings page from the header menu
2. Customize appearance preferences (theme, font size)
3. Manage notification preferences
4. Update account information and security settings

### Notifications

1. View notifications by clicking the bell icon in the header
2. Mark notifications as read or delete them
3. Adjust notification preferences in the Settings page

## Project Structure

```plaintext
adeptus-vita/
├── app/                  # Next.js App Router pages
│   ├── about/            # About page
│   ├── blog/             # Blog pages
│   ├── model/            # Diagnostic model page
│   ├── notifications/    # Notifications page
│   ├── search/           # Search page
│   ├── settings/         # Settings page
│   ├── layout.tsx        # Root layout
│   └── page.tsx          # Home page
├── components/           # React components
│   ├── layout/           # Layout components
│   ├── ui/               # UI components (shadcn)
│   └── ...               # Other components
├── hooks/                # Custom React hooks
├── lib/                  # Utility functions
├── public/               # Static assets
└── ...                   # Configuration files
```

## Customization

### Themes

The application supports light and dark modes, with a system preference option. Theme settings can be adjusted in the Settings page or via the theme toggle in the header.

### UI Components

UI components are built using the shadcn/ui library, which provides a consistent design system. Components can be customized by modifying the corresponding files in the `components/ui` directory.

### Styling

Styling is implemented using Tailwind CSS. Global styles are defined in `app/globals.css`, and component-specific styles are applied using Tailwind utility classes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
