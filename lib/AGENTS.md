# Lib Guide

## Overview
Utility functions and data fetching for blog posts.

## Files
| File | Purpose | Use In |
|------|---------|--------|
| `posts.ts` | Read/parse Markdown posts | Pages (server-side) |
| `utils.ts` | General utilities (cn, slugify) | Components/Pages |
| `utils-client.ts` | Client-safe utilities | Components |
| `portfolio.ts` | Portfolio data (if used) | Portfolio pages |

## Key Functions

### posts.ts (Server-Side Only)
```typescript
// Get all posts sorted by date
getSortedPostsData(): PostMeta[]

// Get paths for static generation
getAllPostIds(): { params: { year, month, day, slug } }[]

// Get single post content
getPostData(year, month, day, slug): Promise<PostData>

// Filter by category
getPostsByCategory(category): PostMeta[]
getAllCategories(): string[]
```

### utils.ts
```typescript
cn(...inputs)           // Merge class names (uses clsx)
slugify(text)           // URL-safe string
capitalizeFirst(text)   // Capitalize first letter
truncate(text, length)  // Truncate with ellipsis
readingTime(content)    // Estimate reading time
createExcerpt(content)  // Strip markdown, truncate
```

## Patterns

### ✅ DO: Import dynamically in getStaticProps
```typescript
export const getStaticProps: GetStaticProps = async () => {
  const { getSortedPostsData } = await import('@/lib/posts')
  // ...
}
```

### ❌ DON'T: Import posts.ts in client components
`posts.ts` uses Node.js `fs` module - server-side only!

## Types
Key interfaces in `posts.ts`:
- `PostMeta` - Post metadata (slug, title, date, categories)
- `PostData` - Full post with HTML content
