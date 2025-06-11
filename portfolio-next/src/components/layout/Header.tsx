'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from '@/components/ui/navigation-menu'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { cn } from '@/lib/utils'
import { SidebarTrigger } from '@/components/animate-ui/radix/sidebar'
import { Button } from '@/components/ui/button'
import { Menu } from 'lucide-react'
import { useState } from 'react'

interface HeaderProps {
  title?: string
}

export default function Header({ title = "Saik Anam Siam" }: HeaderProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <motion.header 
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 50, delay: 0.2 }}
      className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
    >
      <div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Left side */}
          <div className="flex items-center gap-4">
            <SidebarTrigger className="-ml-1" />
            
            <Link href="/" className="flex items-center space-x-2">
              <span className="font-bold text-lg sm:text-xl bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/70">
                {title}
              </span>
            </Link>
          </div>

          {/* Right side - Desktop */}
          <div className="hidden md:flex items-center gap-4">
            <NavigationMenu>
              <NavigationMenuList>
                <NavigationMenuItem>
                  <Link href="/projects" legacyBehavior passHref>
                    <NavigationMenuLink className={cn(
                      navigationMenuTriggerStyle(), 
                      "text-foreground/70 hover:text-foreground transition-colors"
                    )}>
                      Projects
                    </NavigationMenuLink>
                  </Link>
                </NavigationMenuItem>
                <NavigationMenuItem>
                  <Link href="/about" legacyBehavior passHref>
                    <NavigationMenuLink className={cn(
                      navigationMenuTriggerStyle(), 
                      "text-foreground/70 hover:text-foreground transition-colors"
                    )}>
                      About
                    </NavigationMenuLink>
                  </Link>
                </NavigationMenuItem>
              </NavigationMenuList>
            </NavigationMenu>
            <ThemeToggle />
          </div>

          {/* Right side - Mobile */}
          <div className="flex md:hidden items-center gap-2">
            <ThemeToggle />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden"
            >
              <Menu className="h-5 w-5" />
            </Button>
          </div>
        </div>

        {/* Mobile menu */}
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="md:hidden py-4 border-t"
          >
            <nav className="flex flex-col space-y-2">
              <Link 
                href="/projects" 
                className="px-4 py-2 text-foreground/70 hover:text-foreground hover:bg-accent rounded-md transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                Projects
              </Link>
              <Link 
                href="/about" 
                className="px-4 py-2 text-foreground/70 hover:text-foreground hover:bg-accent rounded-md transition-colors"
                onClick={() => setMobileMenuOpen(false)}
              >
                About
              </Link>
            </nav>
          </motion.div>
        )}
      </div>
    </motion.header>
  )
} 