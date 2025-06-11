import { getAllPostSlugs, getPostData, getGraphData } from "@/lib/markdown";
import { notFound } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { H1, P } from "@/components/ui/typography";
import { SidebarContent } from "@/components/ui/sidebar-content";
import { ArrowLeft } from "lucide-react";
import Graph from "@/components/Graph";
import { TableOfContentsFiles } from "@/components/ui/table-of-contents-files";
import { ScrollProgress } from "@/components/animate-ui/components/scroll-progress";
import { ContentRenderer } from "@/components/ui/content-renderer";

interface PageProps {
  params: Promise<{
    slug: string[];
  }>;
}

export async function generateStaticParams() {
  const slugs = getAllPostSlugs();
  return slugs.map((slug) => ({
    slug: slug.split('/'),
  }));
}

export default async function PostPage({ params }: PageProps) {
  const { slug: slugArray } = await params;
  // Decode each part of the slug properly
  const decodedSlugArray = slugArray.map(part => decodeURIComponent(part));
  const slug = decodedSlugArray.join('/');
  
  try {
    const post = await getPostData(slug);
    const globalGraphData = await getGraphData(slug);
    
    // Filter to show only nodes connected to current page (local graph)
    const connectedNodeIds = new Set([slug]);
    globalGraphData.edges.forEach(edge => {
      if (edge.from === slug) connectedNodeIds.add(edge.to);
      if (edge.to === slug) connectedNodeIds.add(edge.from);
    });
    
    const localGraphData = {
      nodes: globalGraphData.nodes.filter(node => connectedNodeIds.has(node.id)),
      edges: globalGraphData.edges.filter(edge => 
        connectedNodeIds.has(edge.from) && connectedNodeIds.has(edge.to)
      )
    };
    
    return (
      <>
        <ScrollProgress />
        <div className="w-full max-w-7xl mx-auto">
        <div className="mb-8">
            <Button variant="ghost" size="sm" asChild className="mb-6 -ml-1">
            <Link href={slug.startsWith('Projects/') ? '/projects' : '/'}>
              <ArrowLeft className="mr-2 h-4 w-4" />
                Back to {slug.startsWith('Projects/') ? 'Projects' : 'Home'}
            </Link>
          </Button>
          
            <H1 className="text-3xl md:text-4xl font-bold mb-4">{post.title}</H1>
          
            <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
          {post.date && (
                <time dateTime={post.date}>
                  {new Date(post.date).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                  })}
                </time>
          )}
          
          {post.tags && post.tags.length > 0 && (
                <>
                  <span className="text-muted-foreground/50">•</span>
                  <div className="flex flex-wrap gap-2">
              {post.tags.map((tag) => (
                <span
                  key={tag}
                        className="inline-flex items-center rounded-md bg-secondary px-2.5 py-0.5 text-xs font-medium text-secondary-foreground"
                >
                  {tag}
                </span>
              ))}
            </div>
                </>
          )}
            </div>
        </div>
        
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 lg:gap-8">
            {/* Main content */}
            <article className="xl:col-span-2">
              <ContentRenderer 
                content={post.content}
                className="prose prose-lg dark:prose-invert max-w-none 
                  prose-headings:scroll-mt-20 prose-headings:font-bold
                  prose-h1:text-3xl prose-h1:mb-4
                  prose-h2:text-2xl prose-h2:mt-8 prose-h2:mb-4
                  prose-p:text-base prose-p:leading-7 prose-p:mb-4
                  prose-a:text-primary prose-a:no-underline hover:prose-a:underline
                  prose-ul:my-4 prose-ol:my-4
                  prose-li:text-base prose-li:leading-7
                  prose-pre:bg-muted prose-pre:border prose-pre:border-border
                  prose-code:text-sm prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded
                  prose-img:rounded-lg prose-img:shadow-md"
              />
            </article>
            
            {/* Sidebar */}
            <aside className="xl:col-span-1">
              <div className="sticky top-20 space-y-6">
                {/* Graph Component */}
                <div className="w-full">
                  <Graph 
                    localData={localGraphData} 
                    globalData={globalGraphData}
                    currentSlug={slug}
                  />
                </div>
                
                {/* Table of Contents */}
                <div className="rounded-lg border bg-card p-4">
                  <div className="max-h-96 overflow-y-auto">
                    <TableOfContentsFiles content={post.content} />
                  </div>
                </div>
              </div>
            </aside>
          </div>
        </div>
      </>
    );
  } catch (error) {
    console.error('Error loading post:', error);
    notFound();
  }
} 