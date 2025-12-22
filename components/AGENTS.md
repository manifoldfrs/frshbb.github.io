# Components Guide

## Overview
React UI components for the blog. Each component lives in its own folder with an `index.tsx` export.

## Structure Pattern
```
components/
├── ComponentName/
│   └── index.tsx    # Main component + export
```

## Creating Components

### ✅ DO: Follow existing folder pattern
```typescript
// components/NewComponent/index.tsx
import { ReactNode } from 'react'

interface NewComponentProps {
  children: ReactNode
  className?: string
}

export default function NewComponent({ children, className = '' }: NewComponentProps) {
  return <div className={`... ${className}`}>{children}</div>
}
```

### ✅ DO: Use path aliases
```typescript
import Layout from '@/components/Layout'
import { PostMeta } from '@/lib/posts'
```

### ❌ DON'T: Create flat files
```typescript
// BAD: components/Button.tsx
// GOOD: components/Button/index.tsx
```

## Key Components
| Component | Purpose | Example |
|-----------|---------|---------|
| `Layout/` | Page wrapper with Header/Footer/SEO | See `pages/index.tsx` |
| `Post/PostCard.tsx` | Blog post preview card | See `pages/blog/index.tsx` |
| `SEO/` | Meta tags for pages | Used via Layout |
| `Header/`, `Footer/` | Site navigation | Auto-included by Layout |

## Styling Rules
- Use Tailwind classes only
- Use Nord semantic colors: `text-text-primary`, `bg-background-dark`
- Copy patterns from `styles/globals.css` for reusable classes

## Touch Points
- Layout wrapper: `Layout/index.tsx`
- Post display: `Post/PostCard.tsx`, `Post/index.tsx`
- Navigation: `Header/index.tsx`, `Navigation/index.tsx`
