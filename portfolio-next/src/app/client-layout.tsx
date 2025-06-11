'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  Folder, 
  FolderOpen, 
  FileText, 
  Tag, 
  Hash,
  ChevronRight,
  ChevronDown,
  Home,
  Layers,
  ChevronsUpDown,
  MoreHorizontal
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { 
  Sidebar,
  SidebarProvider, 
  SidebarInset, 
  SidebarTrigger, 
  SidebarHeader,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarMenuSub,
  SidebarMenuSubItem,
  SidebarMenuSubButton,
  SidebarMenuBadge
} from "@/components/animate-ui/radix/sidebar";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/animate-ui/radix/collapsible';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/animate-ui/radix/dropdown-menu';
import { useIsMobile } from '@/hooks/use-mobile';
import { type FolderStructure, type TagGroup, type ContentItem } from "@/lib/content-navigation";
import Header from "@/components/layout/Header";
import Footer from "@/components/layout/Footer";

interface ClientLayoutProps {
  children: React.ReactNode;
  folderStructure: FolderStructure;
  tagGroups: TagGroup[];
}

export function ClientLayout({ children, folderStructure, tagGroups }: ClientLayoutProps) {
  const [viewMode, setViewMode] = useState<'folders' | 'tags'>('folders');
  const pathname = usePathname();
  const isMobile = useIsMobile();

  const isActive = (slug: string) => {
    if (slug === '' || slug === 'index') {
      return pathname === '/';
    }
    return pathname === `/${slug}`;
  };

  const renderFolderItem = (item: ContentItem) => (
    <SidebarMenuSubItem key={item.slug}>
      <SidebarMenuSubButton asChild isActive={isActive(item.slug)}>
        <Link href={item.slug === 'index' ? '/' : `/${item.slug}`}>
          <FileText className="h-4 w-4" />
          <span>{item.title}</span>
        </Link>
      </SidebarMenuSubButton>
    </SidebarMenuSubItem>
  );

  const renderFolder = (folder: FolderStructure, level: number = 0) => {
    const hasContent = folder.children.length > 0 || folder.files.length > 0;

    if (!hasContent) {
      return (
        <SidebarMenuItem key={folder.path}>
          <SidebarMenuButton>
            <Folder className="h-4 w-4" />
            <span>{folder.name}</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      );
    }

    return (
      <Collapsible 
        key={folder.path}
        defaultOpen={false}
        className="group/collapsible"
      >
        <SidebarMenuItem>
          <CollapsibleTrigger asChild>
            <SidebarMenuButton tooltip={folder.name}>
              <ChevronRight className="h-4 w-4 transition-transform duration-300 group-data-[state=open]/collapsible:rotate-90" />
              <Folder className="h-4 w-4" />
              <span>{folder.name}</span>
            </SidebarMenuButton>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <SidebarMenuSub>
              {folder.files.map(renderFolderItem)}
              {folder.children.map(child => renderFolder(child, level + 1))}
            </SidebarMenuSub>
          </CollapsibleContent>
        </SidebarMenuItem>
      </Collapsible>
    );
  };

  const renderTagGroup = (tagGroup: TagGroup) => {
    return (
      <Collapsible 
        key={tagGroup.tag}
        defaultOpen={false}
        className="group/collapsible"
      >
        <SidebarMenuItem>
          <CollapsibleTrigger asChild>
            <SidebarMenuButton tooltip={tagGroup.tag}>
              <ChevronRight className="h-4 w-4 transition-transform duration-300 group-data-[state=open]/collapsible:rotate-90" />
              <Hash className="h-4 w-4" />
              <span>{tagGroup.tag}</span>
              <SidebarMenuBadge>{tagGroup.count}</SidebarMenuBadge>
            </SidebarMenuButton>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <SidebarMenuSub>
              {tagGroup.items.map(item => (
                <SidebarMenuSubItem key={item.slug}>
                  <SidebarMenuSubButton asChild isActive={isActive(item.slug)}>
                    <Link href={item.slug === 'index' ? '/' : `/${item.slug}`}>
                      <FileText className="h-4 w-4" />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuSubButton>
                </SidebarMenuSubItem>
              ))}
            </SidebarMenuSub>
          </CollapsibleContent>
        </SidebarMenuItem>
      </Collapsible>
    );
  };

  return (
    <SidebarProvider>
      <div className="flex h-screen overflow-hidden w-full">
        <Sidebar 
          className="border-r debug-sidebar" 
          collapsible="offcanvas" 
          animateOnHover={false}
        >
          <SidebarHeader>
            {/* Portfolio Title */}
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton size="lg">
                  <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                    <Home className="size-4" />
                  </div>
                  <div className="grid flex-1 text-left text-sm leading-tight">
                    <span className="truncate font-semibold">Portfolio</span>
                    <span className="truncate text-xs">Saik Anam Siam</span>
                  </div>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>

            {/* View Mode Selector - Only dropdown here */}
            <SidebarMenu>
              <SidebarMenuItem>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <SidebarMenuButton>
                      {viewMode === 'folders' ? <Layers className="size-4" /> : <Tag className="size-4" />}
                      <span>{viewMode === 'folders' ? 'Browse by Folders' : 'Browse by Tags'}</span>
                      <ChevronsUpDown className="ml-auto size-4" />
                    </SidebarMenuButton>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent
                    className="w-[--radix-dropdown-menu-trigger-width] min-w-56 rounded-lg"
                    align="start"
                    side={isMobile ? 'bottom' : 'right'}
                    sideOffset={4}
                  >
                    <DropdownMenuLabel className="text-xs text-muted-foreground">
                      View Mode
                    </DropdownMenuLabel>
                    <DropdownMenuItem
                      onClick={() => setViewMode('folders')}
                      className="gap-2 p-2"
                    >
                      <Layers className="size-4" />
                      Folders
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => setViewMode('tags')}
                      className="gap-2 p-2"
                    >
                      <Tag className="size-4" />
                      Tags
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarHeader>

          <SidebarContent>
            {viewMode === 'folders' ? (
              <SidebarGroup>
                <SidebarGroupLabel>Content Structure</SidebarGroupLabel>
                <SidebarGroupContent>
                  <SidebarMenu>
                    {/* Root files */}
                    {folderStructure.files.map((item) => (
                      <SidebarMenuItem key={item.slug}>
                        <SidebarMenuButton asChild isActive={isActive(item.slug)} tooltip={item.title}>
                          <Link href={item.slug === 'index' ? '/' : `/${item.slug}`}>
                            <FileText className="h-4 w-4" />
                            <span>{item.title}</span>
                          </Link>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    ))}
                    
                    {/* Folders as collapsibles */}
                    {folderStructure.children.map(folder => renderFolder(folder))}
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            ) : (
              <SidebarGroup>
                <SidebarGroupLabel>Tags ({tagGroups.length})</SidebarGroupLabel>
                <SidebarGroupContent>
                  <SidebarMenu>
                    {tagGroups.map(renderTagGroup)}
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            )}
          </SidebarContent>
        </Sidebar>
        
        <SidebarInset className="flex-1 flex flex-col overflow-hidden">
          <Header />
          <div className="flex-1 overflow-auto scrollable-content">
            <main className="w-full px-4 sm:px-6 lg:px-8 py-8 lg:py-12 min-w-0">
              <div className="w-full min-w-0 flex-1">
                {children}
              </div>
            </main>
            <Footer />
          </div>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
} 