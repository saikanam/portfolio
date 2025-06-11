"use client";

import { useEffect, useState, useRef } from "react";
import { Files, Folder, File } from "@/components/animate-ui/components/files";

interface Heading {
  id: string;
  text: string;
  level: number;
}

interface TableOfContentsFilesProps {
  content: string;
  className?: string;
}

export function TableOfContentsFiles({ content, className }: TableOfContentsFilesProps) {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const filesContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Parse headings from HTML content
    const parser = new DOMParser();
    const doc = parser.parseFromString(content, "text/html");
    const headingElements = doc.querySelectorAll("h1, h2, h3, h4, h5, h6");
    
    const extractedHeadings: Heading[] = Array.from(headingElements).map((heading, index) => {
      const level = parseInt(heading.tagName.charAt(1));
      const text = heading.textContent || "";
      let id = heading.id;
      
      // Generate ID if not present
      if (!id) {
        id = text
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/(^-|-$)/g, "");
      }
      
      return { id, text, level };
    });
    
    setHeadings(extractedHeadings);
  }, [content]);

  // Auto-scroll to active item in the Files component
  useEffect(() => {
    if (!activeId || !filesContainerRef.current) return;

    // Small delay to ensure the DOM is updated
    const timeoutId = setTimeout(() => {
      const container = filesContainerRef.current?.querySelector('[data-slot="files"]') as HTMLElement;
      if (!container) return;

      // Find the active element by looking for the element with matching text content
      const activeHeading = headings.find(h => h.id === activeId);
      if (!activeHeading) return;

      // Find all file buttons and look for the one with matching text
      const fileButtons = container.querySelectorAll('[data-slot="file-button"], [data-slot="folder-trigger"]');
      let activeElement: Element | null = null;

      fileButtons.forEach(button => {
        const textContent = button.textContent?.trim();
        if (textContent === activeHeading.text) {
          activeElement = button;
        }
      });

      if (activeElement && container) {
        const containerRect = container.getBoundingClientRect();
        const elementRect = (activeElement as HTMLElement).getBoundingClientRect();
        
        // Calculate the scroll position to center the active element
        const containerCenter = containerRect.height / 2;
        const elementCenter = elementRect.top - containerRect.top + elementRect.height / 2;
        const scrollOffset = elementCenter - containerCenter;
        
        // Smooth scroll to center the active element
        container.scrollTo({
          top: container.scrollTop + scrollOffset,
          behavior: 'smooth'
        });
      }
    }, 100);

    return () => clearTimeout(timeoutId);
  }, [activeId, headings]);

  useEffect(() => {
    if (headings.length === 0) return;

    // Add IDs to actual DOM headings
    const headingElements = document.querySelectorAll("h1, h2, h3, h4, h5, h6");
    
    headingElements.forEach((heading, index) => {
      if (headings[index]) {
        heading.id = headings[index].id;
      }
    });

    // Scroll-based tracking
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 100;
      
      let currentActiveId = "";
      const headingsWithElements = headings.map(heading => ({
        ...heading,
        element: document.getElementById(heading.id)
      })).filter(h => h.element);

      // Find the heading that's currently most relevant
      for (let i = headingsWithElements.length - 1; i >= 0; i--) {
        const heading = headingsWithElements[i];
        if (heading.element) {
          const rect = heading.element.getBoundingClientRect();
          const elementTop = rect.top + window.scrollY;
          
          if (elementTop <= scrollPosition) {
            currentActiveId = heading.id;
            break;
          }
        }
      }

      // If we're at the very top, use the first heading
      if (!currentActiveId && scrollPosition < 200 && headingsWithElements.length > 0) {
        currentActiveId = headingsWithElements[0].id;
      }

      if (currentActiveId !== activeId) {
        setActiveId(currentActiveId);
      }
    };

    // Add scroll listener
    window.addEventListener('scroll', handleScroll, { passive: true });
    
    // Set initial active heading
    handleScroll();

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [headings, activeId]);

  const scrollToHeading = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  };

  if (headings.length === 0) {
    return null;
  }

  // Build hierarchical structure
  const buildHierarchy = (headings: Heading[]) => {
    const result: any[] = [];
    const stack: any[] = [];

    headings.forEach((heading) => {
      const item = {
        ...heading,
        children: []
      };

      // Find the correct parent level
      while (stack.length > 0 && stack[stack.length - 1].level >= heading.level) {
        stack.pop();
      }

      if (stack.length === 0) {
        result.push(item);
      } else {
        stack[stack.length - 1].children.push(item);
      }

      stack.push(item);
    });

    return result;
  };

  const hierarchy = buildHierarchy(headings);

  const renderHeading = (heading: any): React.ReactNode => {
    const isActive = activeId === heading.id;
    
    if (heading.children.length > 0) {
      return (
        <Folder 
          key={heading.id} 
          name={heading.text}
          className={isActive ? "bg-primary/10 text-primary font-medium cursor-pointer" : "cursor-pointer hover:bg-muted/50"}
          onClick={() => scrollToHeading(heading.id)}
        >
          {heading.children.map(renderHeading)}
        </Folder>
      );
    } else {
      return (
        <File 
          key={heading.id}
          name={heading.text}
          className={isActive ? "bg-primary/10 text-primary font-medium cursor-pointer" : "cursor-pointer hover:bg-muted/50"}
          onClick={() => scrollToHeading(heading.id)}
        />
      );
    }
  };

  return (
    <div className={className}>
      <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-4">
        On This Page
      </h3>
      <div ref={filesContainerRef}>
        <Files className="max-h-96 overflow-auto" defaultOpen={hierarchy.map(h => h.text)}>
          {hierarchy.map(renderHeading)}
        </Files>
      </div>
    </div>
  );
} 