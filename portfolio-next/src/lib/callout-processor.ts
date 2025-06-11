import { CalloutProps } from "@/components/ui/callout";

export interface ParsedCallout {
  type: CalloutProps['variant'];
  title?: string;
  content: string;
  collapsible: boolean;
  defaultOpen: boolean;
}

/**
 * Parses Obsidian-style callouts from markdown content
 * Uses a line-by-line approach for more reliable parsing
 */
export function parseCallouts(content: string): string {
  const lines = content.split('\n');
  const result: string[] = [];
  let i = 0;
  
  while (i < lines.length) {
    const line = lines[i];
    
    // Check if this line starts a callout
    const calloutMatch = line.match(/^>\s*\[!([a-zA-Z]+)\]([+-]?)\s*(.*?)$/);
    
    if (calloutMatch) {
      const [, type, collapsibleFlag, title] = calloutMatch;
      
      // Collect all subsequent lines that are part of this callout
      const calloutLines: string[] = [];
      i++; // Move to next line
      
      // Collect content lines (lines starting with >)
      while (i < lines.length && lines[i].startsWith('>')) {
        calloutLines.push(lines[i].replace(/^>\s?/, ''));
        i++;
      }
      
      // Parse collapsible state
      const collapsible = collapsibleFlag === '+' || collapsibleFlag === '-';
      const defaultOpen = collapsibleFlag !== '-';
      
      // Clean up content
      const cleanContent = calloutLines
        .filter(line => line.trim() !== '')
        .join('\n')
        .trim();
      
      // Map types
      const typeMapping: Record<string, CalloutProps['variant']> = {
        'note': 'note',
        'info': 'info',
        'tip': 'tip',
        'hint': 'tip',
        'important': 'important',
        'warning': 'warning',
        'caution': 'caution',
        'danger': 'danger',
        'error': 'danger',
        'success': 'success',
        'check': 'success',
        'question': 'question',
        'help': 'question',
        'faq': 'question',
        'quote': 'quote',
        'cite': 'quote',
        'example': 'example',
        'bug': 'bug',
        'abstract': 'abstract',
        'summary': 'abstract',
        'todo': 'todo',
      };
      
      const variant = typeMapping[type.toLowerCase()] || 'note';
      const cleanTitle = title.trim() || undefined;
      const calloutId = `callout-${Math.random().toString(36).substr(2, 9)}`;
      
      console.log('Processing callout:', { type, variant, title: cleanTitle, content: cleanContent.substring(0, 50) + '...' });
      
      // Add the callout marker
      result.push(`<div data-callout-id="${calloutId}" data-callout-variant="${variant}" data-callout-title="${cleanTitle || ''}" data-callout-collapsible="${collapsible}" data-callout-default-open="${defaultOpen}">
${cleanContent}
</div>`);
      
      // Don't increment i here as it's already been incremented in the while loop
    } else {
      // Regular line, add as-is
      result.push(line);
      i++;
    }
  }
  
  const processedContent = result.join('\n');
  console.log('Callout processing complete');
  
  return processedContent;
}

/**
 * Converts callout markers in HTML to React component props
 */
export function extractCalloutData(html: string): {
  html: string;
  callouts: Array<{
    id: string;
    variant: CalloutProps['variant'];
    title?: string;
    content: string;
    collapsible: boolean;
    defaultOpen: boolean;
  }>;
} {
  const callouts: Array<{
    id: string;
    variant: CalloutProps['variant'];
    title?: string;
    content: string;
    collapsible: boolean;
    defaultOpen: boolean;
  }> = [];
  
  // More flexible regex that handles potential HTML formatting
  const calloutRegex = /<div\s+data-callout-id="([^"]+)"\s+data-callout-variant="([^"]+)"\s+data-callout-title="([^"]*)"\s+data-callout-collapsible="([^"]+)"\s+data-callout-default-open="([^"]+)"[^>]*>([\s\S]*?)<\/div>/g;
  
  let match;
  while ((match = calloutRegex.exec(html)) !== null) {
    const [fullMatch, id, variant, title, collapsible, defaultOpen, content] = match;
    
    callouts.push({
      id,
      variant: variant as CalloutProps['variant'],
      title: title || undefined,
      content: content.trim(),
      collapsible: collapsible === 'true',
      defaultOpen: defaultOpen === 'true',
    });
  }
  
  const processedHtml = html.replace(calloutRegex, (match, id, variant, title, collapsible, defaultOpen, content) => {
    return `<div data-callout-placeholder="${id}"></div>`;
  });
  
  return {
    html: processedHtml,
    callouts,
  };
} 