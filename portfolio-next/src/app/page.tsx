import { getPostData, getGraphData } from "@/lib/markdown";
import { Button } from "@/components/ui/button";
import { H2 } from "@/components/ui/typography";
import Graph from "@/components/Graph";
import { TableOfContentsFiles } from "@/components/ui/table-of-contents-files";
import { ScrollProgress } from "@/components/animate-ui/components/scroll-progress";
import { ContentRenderer } from "@/components/ui/content-renderer";

export default async function Home() {
  const indexData = await getPostData('index');
  const globalGraphData = await getGraphData('index');
  
  // Filter to show only nodes connected to index page (local graph)
  const connectedNodeIds = new Set(['index']);
  globalGraphData.edges.forEach(edge => {
    if (edge.from === 'index') connectedNodeIds.add(edge.to);
    if (edge.to === 'index') connectedNodeIds.add(edge.from);
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
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 lg:gap-8">
          {/* Main content */}
          <div className="xl:col-span-2">
            <ContentRenderer 
              content={indexData.content}
              className="prose prose-lg dark:prose-invert max-w-none 
                prose-headings:scroll-mt-20 prose-headings:font-bold
                prose-h1:text-3xl prose-h1:mb-4
                prose-h2:text-2xl prose-h2:mt-8 prose-h2:mb-4
                prose-p:text-base prose-p:leading-7 prose-p:mb-4
                prose-a:text-primary prose-a:no-underline hover:prose-a:underline
                prose-ul:my-4 prose-ol:my-4
                prose-li:text-base prose-li:leading-7"
            />
            
            {/* Quick Actions - Better positioned */}
            <div className="mt-12 pt-8 border-t border-border">
              <H2 className="text-2xl font-bold mb-6 text-center lg:text-left">Quick Actions</H2>
              <div className="flex flex-wrap gap-3 justify-center lg:justify-start">
                <Button size="lg" asChild>
                  <a href="/projects">View Projects</a>
                </Button>
                <Button size="lg" variant="outline" asChild>
                  <a href="mailto:contact@saikanam.com">Contact Me</a>
                </Button>
                <Button size="lg" variant="secondary" asChild>
                  <a href="/resume.pdf" target="_blank">Download Resume</a>
                </Button>
              </div>
            </div>
          </div>
          
          {/* Sidebar */}
          <aside className="xl:col-span-1">
            <div className="sticky top-20 space-y-6">
              {/* Graph Component */}
              <div className="w-full">
                <Graph 
                  localData={localGraphData} 
                  globalData={globalGraphData}
                  currentSlug="index"
                />
              </div>
              
              {/* Table of Contents */}
              <div className="rounded-lg border bg-card p-4">
                <div className="max-h-96 overflow-y-auto">
                  <TableOfContentsFiles content={indexData.content} />
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </>
  );
}
