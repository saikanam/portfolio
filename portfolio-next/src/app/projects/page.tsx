import { getAllPosts } from "@/lib/markdown";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { H1, P } from "@/components/ui/typography";
import { ArrowRight, Calendar, Tag } from "lucide-react";

export default async function ProjectsPage() {
  const posts = await getAllPosts();
  const projects = posts.filter(post => post.slug.startsWith('Projects/'));
  
  return (
    <div className="w-full max-w-7xl mx-auto">
      <div className="mb-12">
        <H1 className="text-3xl md:text-4xl font-bold mb-4">Projects</H1>
        <P className="text-muted-foreground text-lg max-w-3xl">
          A collection of my work in machine learning, computer vision, and software development.
        </P>
      </div>
      
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {projects.map((project) => (
          <Card key={project.slug} className="h-full flex flex-col hover:shadow-lg transition-shadow duration-200 border-border/50">
            <CardHeader className="pb-3">
              <CardTitle className="line-clamp-2 text-xl leading-tight">{project.title}</CardTitle>
              {project.date && (
                <CardDescription className="flex items-center gap-1 text-sm">
                  <Calendar className="h-3 w-3" />
                  {new Date(project.date).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short'
                  })}
                </CardDescription>
              )}
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              <p className="text-muted-foreground mb-4 flex-1 line-clamp-3 text-sm leading-relaxed">
                {project.excerpt}
              </p>
              {project.tags && project.tags.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-4">
                  {project.tags.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="inline-flex items-center gap-1 rounded-md bg-secondary/50 px-2 py-1 text-xs font-medium text-secondary-foreground"
                    >
                      <Tag className="h-2.5 w-2.5" />
                      {tag}
                    </span>
                  ))}
                  {project.tags.length > 3 && (
                    <span className="inline-flex items-center rounded-md bg-muted px-2 py-1 text-xs text-muted-foreground">
                      +{project.tags.length - 3} more
                    </span>
                  )}
                </div>
              )}
              <Button asChild className="mt-auto w-full group" variant="outline">
                <Link href={`/${project.slug}`} className="flex items-center justify-center gap-2">
                  Read More
                  <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                </Link>
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
      
      {projects.length === 0 && (
        <div className="text-center py-12">
          <p className="text-muted-foreground">No projects found.</p>
        </div>
      )}
    </div>
  );
} 