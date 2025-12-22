# Pages Guide

## Overview
Next.js Pages Router. Each `.tsx` file becomes a route.

## Route Structure
```
pages/
├── index.tsx                    # /
├── about.tsx                    # /about/
├── archive.tsx                  # /archive/
├── blog/index.tsx               # /blog/
├── categories/
│   ├── index.tsx                # /categories/
│   └── [category].tsx           # /categories/:category/
├── [year]/[month]/[day]/[slug].tsx  # /:year/:month/:day/:slug/
├── _app.tsx                     # App wrapper (global styles)
└── _document.tsx                # HTML document structure
```

## Creating Pages

### ✅ DO: Use Layout component
```typescript
import Layout from '@/components/Layout'

export default function NewPage() {
  return (
    <Layout title="Page Title" description="SEO description">
      {/* Page content */}
    </Layout>
  )
}
```

### ✅ DO: Use getStaticProps for data
```typescript
import { GetStaticProps } from 'next'

export const getStaticProps: GetStaticProps = async () => {
  const { getSortedPostsData } = await import('@/lib/posts')
  const posts = getSortedPostsData()
  return { props: { posts } }
}
```

### ✅ DO: Use getStaticPaths for dynamic routes
```typescript
// See pages/[year]/[month]/[day]/[slug].tsx for full pattern
export async function getStaticPaths() {
  const { getAllPostIds } = await import('@/lib/posts')
  return { paths: getAllPostIds(), fallback: false }
}
```

## Key Files
| File | Purpose |
|------|---------|
| `index.tsx` | Homepage with recent posts |
| `[year]/[month]/[day]/[slug].tsx` | Individual blog post |
| `blog/index.tsx` | All posts listing |
| `categories/[category].tsx` | Posts filtered by category |

## URL Convention
- Blog posts: `/:year/:month/:day/:slug/` (matches Jekyll)
- All URLs have trailing slash (see `next.config.js`)
