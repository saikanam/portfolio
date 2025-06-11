import { visit } from 'unist-util-visit';
import type { Plugin } from 'unified';
import type { Root, Blockquote, Paragraph, Text } from 'mdast';

interface CalloutNode {
  type: 'callout';
  data: {
    hName: 'div';
    hProperties: {
      'data-callout-id': string;
      'data-callout-variant': string;
      'data-callout-title': string;
      'data-callout-collapsible': string;
      'data-callout-default-open': string;
    };
  };
  children: any[];
}

const remarkCallouts: Plugin<[], Root> = () => {
  return (tree) => {
    visit(tree, 'blockquote', (node: Blockquote, index, parent) => {
      // Check if this blockquote is a callout
      const firstChild = node.children[0];
      if (firstChild?.type !== 'paragraph') return;
      
      // Get the first text node directly
      const firstTextNode = firstChild.children.find(child => child.type === 'text') as Text;
      if (!firstTextNode) return;

      
      // Check if this looks like a callout - remove the 's' flag for compatibility
      const calloutMatch = firstTextNode.value.match(/^\[!([a-zA-Z]+)\]([+-]?)\s*(.*)/);
      if (!calloutMatch) {
        return;
      }
      
      const [, type, collapsibleFlag, titleAndContent] = calloutMatch;
      
      // Extract just the title (first line after the callout syntax)
      const titleMatch = titleAndContent.match(/^([^\r\n]*)/);
      const title = titleMatch ? titleMatch[1].trim() : '';
     
      // Map types
      const typeMapping: Record<string, string> = {
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
      const collapsible = collapsibleFlag === '+' || collapsibleFlag === '-';
      const defaultOpen = collapsibleFlag !== '-';
      const calloutId = `callout-${Math.random().toString(36).substr(2, 9)}`;
      
      // Remove the callout syntax from the first text node
      firstTextNode.value = firstTextNode.value.replace(/^\[!([a-zA-Z]+)\]([+-]?)\s*/, '');
      
      // Also remove the title line if it exists
      if (title) {
        firstTextNode.value = firstTextNode.value.replace(/^[^\r\n]*[\r\n]*/, '');
      }
      
      // If the text node is now empty, remove it
      if (!firstTextNode.value.trim()) {
        const textIndex = firstChild.children.indexOf(firstTextNode);
        if (textIndex !== -1) {
          firstChild.children.splice(textIndex, 1);
        }
      }
      
      // Create callout node
      const calloutNode: CalloutNode = {
        type: 'callout',
        data: {
          hName: 'div',
          hProperties: {
            'data-callout-id': calloutId,
            'data-callout-variant': variant,
            'data-callout-title': title,
            'data-callout-collapsible': collapsible.toString(),
            'data-callout-default-open': defaultOpen.toString(),
          },
        },
        children: node.children,
      };
      
      // Replace the blockquote with our callout node
      if (parent && typeof index === 'number') {
        parent.children[index] = calloutNode as any;
      }
      
     
    });
  };
};

export default remarkCallouts; 