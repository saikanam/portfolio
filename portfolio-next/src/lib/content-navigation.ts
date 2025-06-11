import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

export interface ContentItem {
  title: string;
  slug: string;
  fullPath: string;
  tags: string[];
  draft: boolean;
  folder: string;
}

export interface FolderStructure {
  name: string;
  path: string;
  children: FolderStructure[];
  files: ContentItem[];
}

export interface TagGroup {
  tag: string;
  count: number;
  items: ContentItem[];
}

const contentDir = path.join(process.cwd(), 'content');

export function getAllContent(): ContentItem[] {
  const items: ContentItem[] = [];
  
  function scanDirectory(dir: string, relativePath: string = '') {
    const entries = fs.readdirSync(dir);
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        scanDirectory(fullPath, path.join(relativePath, entry));
      } else if (entry.endsWith('.md')) {
        try {
          const content = fs.readFileSync(fullPath, 'utf-8');
          const { data } = matter(content);
          
          const slug = path.join(relativePath, entry.replace('.md', ''));
          const folder = relativePath || 'root';
          
          items.push({
            title: data.title || entry.replace('.md', ''),
            slug: slug.replace(/\\/g, '/'), // Normalize path separators
            fullPath,
            tags: data.tags || [],
            draft: data.draft || false,
            folder
          });
        } catch (error) {
          console.error(`Error parsing ${fullPath}:`, error);
        }
      }
    }
  }
  
  scanDirectory(contentDir);
  return items.filter(item => !item.draft);
}

export function getFolderStructure(): FolderStructure {
  const items = getAllContent();
  const root: FolderStructure = {
    name: 'Content',
    path: '',
    children: [],
    files: []
  };
  
  const folderMap = new Map<string, FolderStructure>();
  folderMap.set('', root);
  
  // Create folder structure
  items.forEach(item => {
    const folderPath = item.folder === 'root' ? '' : item.folder;
    const parts = folderPath.split('/').filter(Boolean);
    
    let currentPath = '';
    let parent = root;
    
    // Create folder hierarchy
    parts.forEach(part => {
      const fullPath = currentPath ? `${currentPath}/${part}` : part;
      
      if (!folderMap.has(fullPath)) {
        const newFolder: FolderStructure = {
          name: part,
          path: fullPath,
          children: [],
          files: []
        };
        
        parent.children.push(newFolder);
        folderMap.set(fullPath, newFolder);
      }
      
      parent = folderMap.get(fullPath)!;
      currentPath = fullPath;
    });
    
    // Add file to appropriate folder
    if (folderPath === 'root' || folderPath === '') {
      root.files.push(item);
    } else {
      const folder = folderMap.get(folderPath);
      if (folder) {
        folder.files.push(item);
      }
    }
  });
  
  return root;
}

export function getTagGroups(): TagGroup[] {
  const items = getAllContent();
  const tagMap = new Map<string, ContentItem[]>();
  
  items.forEach(item => {
    item.tags.forEach(tag => {
      if (!tagMap.has(tag)) {
        tagMap.set(tag, []);
      }
      tagMap.get(tag)!.push(item);
    });
  });
  
  return Array.from(tagMap.entries())
    .map(([tag, items]) => ({
      tag,
      count: items.length,
      items
    }))
    .sort((a, b) => b.count - a.count);
} 