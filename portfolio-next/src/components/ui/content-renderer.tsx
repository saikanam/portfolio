"use client";

import React from 'react';
import parse, { Element, domToReact, HTMLReactParserOptions, DOMNode, attributesToProps } from 'html-react-parser';
import { Callout } from './callout';
import { extractCalloutData } from '@/lib/callout-processor';
import { H1, H2, H3, H4, H5, H6, P, List, Quote, InlineCode, Link } from './typography';

interface ContentRendererProps {
  content: string;
  className?: string;
}

// Void elements that cannot have children in React
const VOID_ELEMENTS = new Set([
  'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 
  'link', 'meta', 'param', 'source', 'track', 'wbr'
]);

export function ContentRenderer({ content, className }: ContentRendererProps) {
  const { html, callouts } = extractCalloutData(content);
  
  // Create a map of callout data for quick lookup
  const calloutMap = new Map(callouts.map(callout => [callout.id, callout]));
  
  // Options for html-react-parser to convert HTML elements to shadcn typography components
  const parseOptions: HTMLReactParserOptions = {
    replace: (domNode) => {
      if (domNode.type === 'tag' && domNode instanceof Element) {
        const { name, attribs, children } = domNode;
        
        // Convert 'class' to 'className' and other HTML attributes to React props
        const props = attributesToProps(attribs);
        
        // Ensure we handle any remaining 'class' attributes
        if ('class' in props && !('className' in props)) {
          props.className = props.class;
          delete props.class;
        }
        
        const childrenContent = domToReact(children as DOMNode[], parseOptions);

        // Convert heading elements to typography components
        switch (name) {
          case 'h1':
            return <H1 {...props}>{childrenContent}</H1>;
          case 'h2':
            return <H2 {...props}>{childrenContent}</H2>;
          case 'h3':
            return <H3 {...props}>{childrenContent}</H3>;
          case 'h4':
            return <H4 {...props}>{childrenContent}</H4>;
          case 'h5':
            return <H5 {...props}>{childrenContent}</H5>;
          case 'h6':
            return <H6 {...props}>{childrenContent}</H6>;
          case 'p':
            return <P {...props}>{childrenContent}</P>;
          case 'ul':
            return <List {...props}>{childrenContent}</List>;
          case 'blockquote':
            return <Quote {...props}>{childrenContent}</Quote>;
          case 'code':
            // Only convert if it's not inside a pre tag (inline code)
            if (!domNode.parent || (domNode.parent as Element).name !== 'pre') {
              return <InlineCode {...props}>{childrenContent}</InlineCode>;
            }
            break;
          case 'a':
            return <Link {...props}>{childrenContent}</Link>;
          case 'iframe':
            // Handle iframe attributes properly
            const iframeProps: any = { ...props };
            if ('frameborder' in iframeProps) {
              iframeProps.frameBorder = iframeProps.frameborder;
              delete iframeProps.frameborder;
            }
            if ('allowfullscreen' in iframeProps) {
              iframeProps.allowFullScreen = iframeProps.allowfullscreen;
              delete iframeProps.allowfullscreen;
            }
            return <iframe {...iframeProps}>{childrenContent}</iframe>;
          default:
            // Handle void elements that cannot have children
            if (VOID_ELEMENTS.has(name)) {
              return React.createElement(name, props);
            }
            // For other HTML elements, include children
            if (name && typeof name === 'string') {
              return React.createElement(name, props, childrenContent);
            }
        }
      }
      // Return undefined to use default behavior for other elements
      return undefined;
    }
  };
  
  // Function to process content with callouts and typography components
  const processContent = (htmlString: string): React.ReactNode[] => {
    const elements: React.ReactNode[] = [];
    
    // Split by callout placeholders
    const parts = htmlString.split(/(<div data-callout-placeholder="[^"]+"><\/div>)/);
    
    parts.forEach((part, index) => {
      // Check if this part is a callout placeholder
      const placeholderMatch = part.match(/<div data-callout-placeholder="([^"]+)"><\/div>/);
      
      if (placeholderMatch) {
        const calloutId = placeholderMatch[1];
        const calloutData = calloutMap.get(calloutId);
        
        if (calloutData) {
          elements.push(
            <Callout
              key={calloutId}
              variant={calloutData.variant}
              title={calloutData.title}
              collapsible={calloutData.collapsible}
              defaultOpen={calloutData.defaultOpen}
              className="my-4"
            >
              {parse(calloutData.content, parseOptions)}
            </Callout>
          );
        }
      } else if (part.trim()) {
        // Regular HTML content - parse and convert to typography components
        elements.push(
          <div key={`content-${index}`}>
            {parse(part, parseOptions)}
          </div>
        );
      }
    });
    
    return elements;
  };
  
  // If there are no callouts, parse the content directly
  if (callouts.length === 0) {
    return (
      <div className={className}>
        {parse(html, parseOptions)}
      </div>
    );
  }

  const processedElements = processContent(html);
  
  return (
    <div className={className}>
      {processedElements}
    </div>
  );
} 