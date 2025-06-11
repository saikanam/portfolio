import React from 'react';
import { cn } from '@/lib/utils';

interface SidebarContentProps {
  children: React.ReactNode;
  className?: string;
}

export function SidebarContent({ children, className }: SidebarContentProps) {
  return (
    <aside 
      className={cn(
        "w-full space-y-8", 
        // Ensure proper dark mode inheritance
        "bg-background text-foreground",
        // Ensure borders and backgrounds work in dark mode
        "[&_.graph-container]:bg-card [&_.graph-container]:border-border",
        "[&_[data-slot='files']]:bg-card [&_[data-slot='files']]:border-border",
        className
      )}
    >
      {children}
    </aside>
  );
} 