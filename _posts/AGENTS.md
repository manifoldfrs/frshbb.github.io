# Blog Posts Guide

## Overview
Markdown blog posts with YAML frontmatter. Processed by `lib/posts.ts`.

## File Naming Convention
```
YYYY-MM-DD-title-slug.md
```
Examples:
- `2024-01-28-neo-sparta.md`
- `2024-05-26-diet-and-fitness.md`

## Frontmatter Template
```yaml
---
layout: post
title: "Your Post Title"
date: YYYY-MM-DD HH:MM:SS -0800
categories:
  - CategoryName
---
```

### Valid Categories
Found in existing posts:
- `Posts` (general)
- `dev` (technical)
- `diet` / `fitness`
- `religion`
- `talks`

## Content Guidelines

### Images
```markdown
![Alt text](/images/your-image.png)
```
- Place images in `public/images/`
- Reference as `/images/filename.ext`

### Code Blocks
````markdown
```typescript
const example = 'code here'
```
````

### Links
```markdown
[Link text](https://example.com)
[Internal link](/about/)
```

## URL Generated
File `2024-01-28-neo-sparta.md` becomes:
`/2024/01/28/neo-sparta/`

## Quick Commands
```bash
# List all posts
ls _posts/*.md

# Find posts by category
grep -l "Posts" _posts/*.md

# Create new post (example)
touch "_posts/$(date +%Y-%m-%d)-new-post-title.md"
```
