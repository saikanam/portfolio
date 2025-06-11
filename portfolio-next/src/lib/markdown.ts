import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
import { remark } from 'remark'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkRehype from 'remark-rehype'
import rehypeSlug from 'rehype-slug'
import rehypeStringify from 'rehype-stringify'
import { unified } from 'unified'
import remarkCallouts from './remark-callouts'

const contentDirectory = path.join(process.cwd(), 'content')

export interface PostData {
  slug: string
  title: string
  date?: string
  tags: string[]
  content: string
  excerpt?: string
  frontmatter: { [key: string]: any }
}

export function getAllPostSlugs() {
  const getAllFiles = (dirPath: string, arrayOfFiles: string[] = []): string[] => {
    const files = fs.readdirSync(dirPath)

    files.forEach((file) => {
      const fullPath = path.join(dirPath, file)
      if (fs.statSync(fullPath).isDirectory()) {
        arrayOfFiles = getAllFiles(fullPath, arrayOfFiles)
      } else if (file.endsWith('.md')) {
        const relativePath = path.relative(contentDirectory, fullPath)
        arrayOfFiles.push(relativePath.replace(/\.md$/, '').replace(/\\/g, '/'))
      }
    })

    return arrayOfFiles
  }

  return getAllFiles(contentDirectory)
}

export async function getPostData(slug: string): Promise<PostData> {
  // Handle URL encoded slugs
  const decodedSlug = decodeURIComponent(slug)
  const fullPath = path.join(contentDirectory, `${decodedSlug}.md`)
  
  if (!fs.existsSync(fullPath)) {
    throw new Error(`Post not found: ${slug}`)
  }
  
  const fileContents = fs.readFileSync(fullPath, 'utf8')
  
  // Parse frontmatter
  const { data: frontmatter, content: rawContent } = matter(fileContents)
  
  // Process Obsidian-style embeds and links (but not callouts - handled by remark plugin)
  let processedRawContent = rawContent
    // Handle PDF embeds: ![[filename.pdf]] -> PDF viewer
    .replace(/!\[\[([^[\]]+\.pdf)\]\]/g, (match, filename) => {
      const pdfPath = `/${filename.replace(/\s+/g, '%20')}`
      return `<div class="pdf-embed">
        <iframe 
          src="${pdfPath}" 
          width="100%" 
          height="600px" 
          frameborder="0"
          title="${filename}"
        ></iframe>
        <p class="text-sm text-muted-foreground mt-2">
          <a href="${pdfPath}" target="_blank" rel="noopener noreferrer" class="underline">
            Open ${filename} in new tab
          </a>
        </p>
      </div>`
    })
    // Handle wiki links: [[link]] -> regular links
    .replace(/\[\[([^\]]+)\]\]/g, (match, linkText) => {
      const cleanText = linkText.trim()
      // For project files, create a properly encoded path
      const encodedPath = `/Projects/${encodeURIComponent(cleanText)}`
      return `[${cleanText}](${encodedPath})`
    })
  
  // Process markdown with proper heading ID generation
  const processedContent = await remark()
    .use(remarkGfm) // GitHub Flavored Markdown
    .use(remarkMath) // Math support
    .use(remarkCallouts) // Our custom callouts plugin
    .use(remarkRehype, { allowDangerousHtml: true }) // Convert to rehype (HTML AST)
    .use(rehypeSlug) // Add IDs to headings
    .use(rehypeStringify, { allowDangerousHtml: true }) // Convert back to HTML string
    .process(processedRawContent)
  
  const content = processedContent.toString()
  
  // Extract title from frontmatter or first heading
  let title = frontmatter.title || decodedSlug.split('/').pop() || decodedSlug
  if (!frontmatter.title && content.includes('<h1>')) {
    const h1Match = content.match(/<h1[^>]*>(.*?)<\/h1>/)
    if (h1Match) {
      title = h1Match[1].replace(/<[^>]*>/g, '') // Strip HTML tags
    }
  }
  
  // Generate excerpt
  const excerpt = frontmatter.excerpt || content
    .replace(/<[^>]*>/g, '') // Strip HTML
    .substring(0, 200) + '...'
  
  return {
    slug: decodedSlug,
    title,
    date: frontmatter.date,
    tags: frontmatter.tags || [],
    content,
    excerpt,
    frontmatter
  }
}

export async function getAllPosts(): Promise<PostData[]> {
  const slugs = getAllPostSlugs()
  const posts = await Promise.all(
    slugs.map(async (slug) => await getPostData(slug))
  )
  
  // Sort posts by date (newest first)
  return posts.sort((a, b) => {
    if (a.date && b.date) {
      return new Date(b.date).getTime() - new Date(a.date).getTime()
    }
    return 0
  })
}

export function getPostsByTag(tag: string): Promise<PostData[]> {
  return getAllPosts().then(posts => 
    posts.filter(post => post.tags?.includes(tag))
  )
}

// Graph view functionality
export interface GraphNode {
  id: string
  label: string
  title: string
  slug: string
  tags: string[]
  isCurrentPage?: boolean
}

export interface GraphLink {
  from: string
  to: string
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphLink[]
}

function extractWikiLinks(content: string): string[] {
  const wikiLinkRegex = /\[\[([^\]]+)\]\]/g
  const links: string[] = []
  let match
  
  while ((match = wikiLinkRegex.exec(content)) !== null) {
    const linkText = match[1].trim()
    // Convert to the same format as our file structure
    if (linkText) {
      links.push(linkText)
    }
  }
  
  return links
}

export async function getGraphData(currentSlug?: string): Promise<GraphData> {
  const posts = await getAllPosts()
  const nodes: GraphNode[] = []
  const edges: GraphLink[] = []
  
  // Create a map of post titles to slugs for link resolution
  const titleToSlug = new Map<string, string>()
  
  // First pass: create nodes and build title mapping
  for (const post of posts) {
    const nodeId = post.slug
    
    nodes.push({
      id: nodeId,
      label: post.title,
      title: post.title,
      slug: post.slug,
      tags: post.tags || [],
      isCurrentPage: currentSlug === post.slug
    })
    
    // Map title to slug for link resolution
    titleToSlug.set(post.title, post.slug)
    
    // Also map the base filename without extension
    const baseFilename = post.slug.split('/').pop() || post.slug
    titleToSlug.set(baseFilename, post.slug)
  }
  
  // Second pass: extract links and create edges
  for (const post of posts) {
    // Re-read the raw content to extract wiki links
    const fullPath = path.join(contentDirectory, `${post.slug}.md`)
    
    if (fs.existsSync(fullPath)) {
      const fileContents = fs.readFileSync(fullPath, 'utf8')
      const { content: rawContent } = matter(fileContents)
      
      const wikiLinks = extractWikiLinks(rawContent)
      
      for (const linkText of wikiLinks) {
        // Try to resolve the link to a known post
        let targetSlug = titleToSlug.get(linkText)
        
        // If direct match fails, try with Projects/ prefix
        if (!targetSlug && !linkText.startsWith('Projects/')) {
          targetSlug = titleToSlug.get(`Projects/${linkText}`)
        }
        
        // Also try exact filename match
        if (!targetSlug) {
          const exactMatch = posts.find(p => 
            p.slug === linkText || 
            p.slug === `Projects/${linkText}` ||
            p.title === linkText
          )
          if (exactMatch) {
            targetSlug = exactMatch.slug
          }
        }
        
        if (targetSlug && targetSlug !== post.slug) {
          edges.push({
            from: post.slug,
            to: targetSlug
          })
        }
      }
    }
  }
  
  return { nodes, edges }
} 