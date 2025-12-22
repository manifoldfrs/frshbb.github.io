# AGENTS.md

## Project Snapshot
- **Type**: Single-project Next.js 14 blog (Pages Router)
- **Stack**: TypeScript (strict), React 18, Tailwind CSS, Markdown content
- **Deploy**: Static export to GitHub Pages
- **Sub-guides**: See `components/`, `pages/`, `lib/`, `_posts/` for detailed guidance

## Quick Commands
```bash
npm install          # Install dependencies
npm run dev          # Dev server at localhost:3000
npm run build        # Production build (static export to ./out)
npm run lint         # ESLint check
```

## Universal Conventions

### TypeScript
- Strict mode enabled (`tsconfig.json`)
- Use path aliases: `@/components/*`, `@/lib/*`, `@/styles/*`
- Export types alongside implementations

### Styling
- **Tailwind CSS only** - no inline styles or CSS modules
- **Nord color palette** - use semantic tokens from `tailwind.config.js`:
  - `bg-background-dark`, `text-text-primary`, `text-links`, `text-primary`
- Never hardcode hex colors

### Code Style
- Single quotes (Prettier)
- ESLint extends `next/core-web-vitals`
- Functional components only

## Security & Secrets
- No API keys in this static site
- Never commit `.env*` files (see `.gitignore`)
- No user data collection

## JIT Index

### Directory Map
- **Pages**: `pages/` → [pages/AGENTS.md](pages/AGENTS.md)
- **Components**: `components/` → [components/AGENTS.md](components/AGENTS.md)
- **Utilities**: `lib/` → [lib/AGENTS.md](lib/AGENTS.md)
- **Blog Posts**: `_posts/` → [_posts/AGENTS.md](_posts/AGENTS.md)

### Quick Find
```bash
# Find a component
grep -rn "export default function" components/

# Find a page
ls pages/**/*.tsx

# Find a utility function
grep -rn "export function" lib/

# Find posts by category
grep -l "categories:" _posts/*.md
```

## Definition of Done
Before PR:
1. `npm run lint` passes
2. `npm run build` succeeds (generates `./out/`)
3. No TypeScript errors
4. Tested locally with `npm run dev`
