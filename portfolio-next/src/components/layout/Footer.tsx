import { Github, Linkedin, Mail } from 'lucide-react'
import Link from 'next/link'

export default function Footer() {
  const currentYear = new Date().getFullYear()
  
  return (
    <footer className="border-t bg-background/50">
      <div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          {/* Left side - Copyright and Links */}
          <div className="flex flex-col items-center md:items-start gap-2">
            <p className="text-sm text-muted-foreground">
              © {currentYear} Saik Anam Siam. All rights reserved.
            </p>
            <p className="text-xs text-muted-foreground text-center md:text-left">
              Built with{" "}
              <a
                href="https://nextjs.org"
                target="_blank"
                rel="noreferrer"
                className="font-medium hover:text-foreground transition-colors"
              >
                Next.js
              </a>
              {" "}and{" "}
              <a
                href="https://ui.shadcn.com"
                target="_blank"
                rel="noreferrer"
                className="font-medium hover:text-foreground transition-colors"
              >
                shadcn/ui
              </a>
            </p>
          </div>
          
          {/* Right side - Social Links */}
          <div className="flex items-center gap-4">
            <Link
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
              aria-label="GitHub"
            >
              <Github className="h-5 w-5" />
            </Link>
            <Link
              href="https://linkedin.com"
              target="_blank"
              rel="noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors"
              aria-label="LinkedIn"
            >
              <Linkedin className="h-5 w-5" />
            </Link>
            <Link
              href="mailto:contact@saikanam.com"
              className="text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Email"
            >
              <Mail className="h-5 w-5" />
            </Link>
          </div>
        </div>
      </div>
    </footer>
  )
} 