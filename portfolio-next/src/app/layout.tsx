import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "../styles/globals.css";
import { ClientLayout } from "./client-layout";
import { getFolderStructure, getTagGroups } from "@/lib/content-navigation";
import { ThemeProvider } from "@/components/providers/theme-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Saik Anam Siam",
  description: "Portfolio and projects by Saik Anam Siam",
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const folderStructure = getFolderStructure();
  const tagGroups = getTagGroups();

  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
      >
        <ClientLayout folderStructure={folderStructure} tagGroups={tagGroups}>
          {children}
        </ClientLayout>
        </ThemeProvider>
      </body>
    </html>
  );
}
