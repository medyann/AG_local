# Comparison Report: baoyu-slide-deck vs paper-slide-deck

Two Claude Code skills for generating professional slide deck images from content. This report provides a structured side-by-side comparison.

---

## 1. Overview

| Aspect | **baoyu-slide-deck** | **paper-slide-deck** |
|--------|---------------------|---------------------|
| **Repository** | [JimLiu/baoyu-skills](https://github.com/JimLiu/baoyu-skills) | [luwill/research-skills](https://github.com/luwill/research-skills) |
| **Skill Location** | `skills/baoyu-slide-deck/` | `paper-slide-deck/` |
| **Primary Purpose** | General-purpose slide deck generation from any content | Academic paper-focused slide deck generation with figure extraction |
| **Target Users** | Content creators, marketers, educators, business professionals | Researchers, academics, conference presenters |
| **Part of Ecosystem** | Yes -- one of 14+ skills (infographics, comics, cover images, social media, etc.) | Yes -- one of 3 research skills (literature review, slide deck, research proposal) |
| **Total Codebase Size** | ~3,688 lines (skill only) | ~4,985 lines (skill only) |
| **License** | MIT | MIT |

---

## 2. Input Sources & Content Types

| Feature | **baoyu-slide-deck** | **paper-slide-deck** |
|---------|---------------------|---------------------|
| Markdown files | Yes | Yes |
| Pasted text | Yes | Yes |
| PDF papers | No native support | Yes -- first-class support with figure detection |
| URL content | Via separate `baoyu-url-to-markdown` skill | Not mentioned |
| Academic papers | No special handling | Auto-detects figures, tables, citations |
| Content length guidance | Detailed table (< 1000 words to > 5000 words) | Mapped to talk duration (5 min to 30+ min) |

**Verdict**: paper-slide-deck wins for academic content (PDF ingestion, figure detection). baoyu-slide-deck is more versatile for general content types.

---

## 3. Style System

### 3.1 Number of Styles

| Aspect | **baoyu-slide-deck** | **paper-slide-deck** |
|--------|---------------------|---------------------|
| Built-in preset styles | 16 | 17 (16 shared + `academic-paper`) |
| Custom style support | Yes -- 4-dimension system | No explicit dimension system |
| Possible custom combinations | 450 (5 x 6 x 5 x 3) | Limited to presets |

### 3.2 Style Dimensions (baoyu-slide-deck exclusive)

baoyu-slide-deck has a sophisticated 4-dimension style system:

| Dimension | Options |
|-----------|---------|
| **Texture** | clean, grid, organic, pixel, paper |
| **Mood** | professional, warm, cool, vibrant, dark, neutral |
| **Typography** | geometric, humanist, handwritten, editorial, technical |
| **Density** | minimal, balanced, dense |

paper-slide-deck uses the same presets but does **not** expose the underlying dimension system for custom combinations.

### 3.3 Shared Preset Styles (16)

Both skills share these identical presets: `blueprint` (default), `chalkboard`, `corporate`, `minimal`, `sketch-notes`, `watercolor`, `dark-atmospheric`, `notion`, `bold-editorial`, `editorial-infographic`, `fantasy-animation`, `intuition-machine`, `pixel-art`, `scientific`, `vector-illustration`, `vintage`.

### 3.4 Unique Styles

| **baoyu-slide-deck** | **paper-slide-deck** |
|---------------------|---------------------|
| (none unique -- all 16 are shared) | `academic-paper` -- Clean professional style for conference talks, thesis defense, with precise charts, equation formatting, citation markers |

### 3.5 Auto Style Selection

Both use keyword-based auto-selection from content signals. paper-slide-deck adds academic keywords:

| Content Signals (paper-slide-deck only) | Selected Style |
|----------------------------------------|----------------|
| paper, thesis, defense, conference, ieee, acm, icml, neurips, cvpr, acl, aaai, iclr | `academic-paper` |

**Verdict**: baoyu-slide-deck wins on customizability (450 combinations). paper-slide-deck wins on academic domain specialization.

---

## 4. Layout System

| Aspect | **baoyu-slide-deck** | **paper-slide-deck** |
|--------|---------------------|---------------------|
| **Slide-specific layouts** | 10 | 10 (same) |
| **Infographic-derived layouts** | 14 | 14 (same) |
| **Academic-specific layouts** | None | 8 unique academic layouts |

### Academic-Specific Layouts (paper-slide-deck only)

| Layout | Description | Best For |
|--------|-------------|----------|
| `paper-title` | Title, authors, affiliations, venue | Conference paper cover |
| `outline-agenda` | Numbered section list with highlights | Talk structure overview |
| `methods-diagram` | Central architecture/pipeline diagram | Methods, system design |
| `results-chart` | Chart area + data annotations | Quantitative results |
| `equation-focus` | Centered equation + variable definitions | Mathematical derivations |
| `qualitative-grid` | 2x2 or 3x2 image comparison grid | Visual results, ablations |
| `references-list` | Numbered citation list | Key references slide |
| `contributions` | Numbered contribution points | Contributions summary |

**Verdict**: paper-slide-deck wins with 8 additional academic layouts. The shared 24 general layouts are identical.

---

## 5. Workflow & User Interaction

### 5.1 Workflow Steps

| Step | **baoyu-slide-deck** | **paper-slide-deck** |
|------|---------------------|---------------------|
| 1 | Setup & Analyze (load preferences, analyze, check existing) | Analyze Content (+ auto figure detection for PDFs) |
| 2 | Confirmation (5 interactive questions, 2 rounds) | Generate 3 Outline Variants |
| 3 | Generate Outline | User Confirmation (single AskUserQuestion) |
| 4 | Review Outline (conditional) | Generate Prompts |
| 5 | Generate Prompts | Image Generation Method Selection |
| 5.5 | -- | Process IMAGE_SOURCE (auto figure extraction) |
| 6 | Review Prompts (conditional) | Generate Images |
| 7 | Generate Images | Merge to PPTX/PDF |
| 8 | Merge to PPTX/PDF | Output Summary |
| 9 | Output Summary | -- |

### 5.2 User Confirmation Approach

| Aspect | **baoyu-slide-deck** | **paper-slide-deck** |
|--------|---------------------|---------------------|
| Confirmation rounds | 2 rounds (Round 1 always, Round 2 if custom) | 1 round |
| Questions asked | 5 (style, audience, slide count, outline review, prompt review) | 1-2 (style variant + optional language) |
| Outline variants | 1 generated, user reviews | 3 variants generated, user picks one |
| Outline review | Optional (user chooses in confirmation) | Implicit (user selects variant) |
| Prompt review | Optional (user chooses in confirmation) | Not offered |

### 5.3 Conflict Handling

| Aspect | **baoyu-slide-deck** | **paper-slide-deck** |
|--------|---------------------|---------------------|
| Existing directory | Interactive 4-option dialog (regenerate outline, regenerate images, backup + regenerate, exit) | Auto-append timestamp to directory name |

**Verdict**: baoyu-slide-deck offers more granular user control with 5 confirmation questions and review gates. paper-slide-deck is more streamlined by generating 3 variants upfront and letting the user pick.

---

## 6. Image Generation

| Feature | **baoyu-slide-deck** | **paper-slide-deck** |
|---------|---------------------|---------------------|
| **Primary method** | Via `baoyu-image-gen` skill (supports OpenAI, Google, DashScope) | Gemini API via Python script (`generate-slides.py`) |
| **Secondary method** | `baoyu-danger-gemini-web` (reverse-engineered browser API) | Gemini Web via `baoyu-danger-gemini-web` |
| **Dedicated generation script** | No (delegates to image-gen skill) | Yes -- `generate-slides.py` with retry logic, skip-already-done |
| **Session ID for consistency** | Yes | Yes |
| **Auto-retry on failure** | Once | 3 retries with exponential backoff |
| **Skip completed slides** | Not mentioned | Yes (files > 10KB skipped) |
| **Figure extraction from PDF** | No | Yes -- `detect-figures.ts`, `extract-figure.ts`, `apply-template.ts` |
| **Figure container template** | No | Yes -- detailed specs (1920x1080 or 4K, colors, typography) |

### Figure Extraction Pipeline (paper-slide-deck only)

```
PDF → detect-figures.ts → figures.json → extract-figure.ts → raw PNG → apply-template.ts → slide image
```

With PyMuPDF fallback for complex PDFs.

**Verdict**: paper-slide-deck has a significantly more robust image generation pipeline, especially for academic content. The figure extraction pipeline is unique and valuable. baoyu-slide-deck benefits from a broader ecosystem of generation backends.

---

## 7. Scripts & Tooling

| Script | **baoyu-slide-deck** | **paper-slide-deck** |
|--------|---------------------|---------------------|
| `merge-to-pptx.ts` | Yes (with prompt notes embedded) | Yes |
| `merge-to-pdf.ts` | Yes | Yes |
| `generate-slides.py` | No | Yes -- Python, Gemini API, retry logic |
| `detect-figures.ts` | No | Yes -- PDF figure/table detection via pdfjs-dist |
| `extract-figure.ts` | No | Yes -- PDF page rendering to PNG via canvas |
| `apply-template.ts` | No | Yes -- Academic figure container template |
| `package.json` | No (uses npx) | Yes -- with pdfjs-dist, canvas, pptxgenjs dependencies |

| Language | **baoyu-slide-deck** | **paper-slide-deck** |
|----------|---------------------|---------------------|
| TypeScript | 2 scripts | 5 scripts |
| Python | 0 scripts | 1 script |

**Verdict**: paper-slide-deck has 4 additional scripts, giving it a more complete automation pipeline.

---

## 8. Configuration & Extensibility

| Feature | **baoyu-slide-deck** | **paper-slide-deck** |
|---------|---------------------|---------------------|
| **Preferences file** | `EXTEND.md` (project + user level) | `EXTEND.md` (project + user level) |
| **Preferences schema** | Documented (`preferences-schema.md`) | Not documented separately |
| **Custom styles** | Reusable custom dimension combos in EXTEND.md | Via EXTEND.md (override defaults) |
| **First-time setup** | Interactive preference setup wizard | Not mentioned |
| **Plugin marketplace** | Yes -- installable via `npx skills add` or `/plugin` | Manual copy to skills directory |

### EXTEND.md Paths

| Level | **baoyu-slide-deck** | **paper-slide-deck** |
|-------|---------------------|---------------------|
| Project | `.baoyu-skills/baoyu-slide-deck/EXTEND.md` | `.paper-skills/paper-slide-deck/EXTEND.md` |
| User | `~/.baoyu-skills/baoyu-slide-deck/EXTEND.md` | `~/.paper-skills/paper-slide-deck/EXTEND.md` |

**Verdict**: baoyu-slide-deck has a more polished configuration system with documented schema, first-time setup wizard, and plugin marketplace integration.

---

## 9. Partial Workflows & Regeneration

| Feature | **baoyu-slide-deck** | **paper-slide-deck** |
|---------|---------------------|---------------------|
| `--outline-only` | Yes | Yes |
| `--prompts-only` | Yes | Not mentioned |
| `--images-only` | Yes | Not mentioned |
| `--regenerate N` | Yes (single or multiple slides: `3` or `2,5,8`) | Not mentioned |
| Slide modification guide | Detailed (edit, add, delete with renumbering) | Identical guide |

**Verdict**: baoyu-slide-deck offers significantly more partial workflow options, making it easier to iterate on specific parts of a deck.

---

## 10. Shared / Identical Components

The following files are **identical or near-identical** between the two skills:

| Component | Status |
|-----------|--------|
| `references/analysis-framework.md` (Sections 1-6) | Identical |
| `references/content-rules.md` | Identical |
| `references/modification-guide.md` | Identical |
| `references/base-prompt.md` (core sections) | Nearly identical |
| 16 shared style files (`references/styles/`) | Likely identical |
| `merge-to-pptx.ts` logic | Similar |
| `merge-to-pdf.ts` logic | Similar |
| Outline template (core structure) | Similar |

This suggests paper-slide-deck was **forked from or heavily inspired by** baoyu-slide-deck, with academic-specific extensions layered on top.

---

## 11. Unique Strengths

### baoyu-slide-deck Unique Strengths

1. **4-Dimension Custom Style System** -- 450 possible style combinations vs. fixed presets
2. **5-Question Interactive Confirmation** -- Granular control over every aspect
3. **Partial Workflows** -- `--prompts-only`, `--images-only`, `--regenerate N`
4. **Multi-Provider Image Generation** -- OpenAI, Google Gemini, DashScope via `baoyu-image-gen`
5. **Preferences Schema** -- Documented, with first-time setup wizard and reusable custom styles
6. **Plugin Marketplace** -- Easy install/update via `npx skills add` or `/plugin`
7. **Rich Ecosystem** -- Part of 14+ complementary skills (infographics, comics, social posting, etc.)
8. **Design Guidelines** -- Separate document with audience-specific principles, font recommendations, dimension combination guide
9. **Multilingual Font Pairing** -- English/Chinese font recommendations

### paper-slide-deck Unique Strengths

1. **PDF Figure Detection & Extraction** -- `detect-figures.ts` automatically finds figures/tables in PDFs
2. **Smart Figure-to-Slide Mapping** -- Auto-maps extracted figures to appropriate slides via caption analysis
3. **Figure Container Template** -- Professional academic template (`apply-template.ts`) with precise layout specs
4. **`academic-paper` Style** -- Dedicated style for conference talks with proper equation formatting, citations, chart standards
5. **8 Academic-Specific Layouts** -- `paper-title`, `methods-diagram`, `results-chart`, `equation-focus`, `qualitative-grid`, `contributions`, `references-list`, `outline-agenda`
6. **3-Variant Outline Generation** -- Generates 3 style variants for user to compare and choose
7. **Academic Analysis Framework** -- Section 7 (paper structure to slide mapping) and Section 8 (automatic figure detection) in analysis framework
8. **Gemini API Generation Script** -- Dedicated Python script with idempotent retry, exponential backoff, skip-already-done
9. **Academic Slide Templates** -- Cover with authors/affiliations, methods/architecture, quantitative/qualitative results, contributions, references
10. **PyMuPDF Fallback** -- Robust PDF handling for complex academic papers

---

## 12. Summary Comparison Matrix

| Category | **baoyu-slide-deck** | **paper-slide-deck** | Winner |
|----------|:---:|:---:|--------|
| General content versatility | 5 | 3 | baoyu-slide-deck |
| Academic paper support | 1 | 5 | paper-slide-deck |
| Style customization depth | 5 | 3 | baoyu-slide-deck |
| Layout variety (general) | 4 | 4 | Tie |
| Layout variety (academic) | 1 | 5 | paper-slide-deck |
| User interaction polish | 5 | 3 | baoyu-slide-deck |
| Image generation flexibility | 4 | 4 | Tie |
| Figure extraction from PDF | 0 | 5 | paper-slide-deck |
| Partial workflow support | 5 | 2 | baoyu-slide-deck |
| Automation scripts | 3 | 5 | paper-slide-deck |
| Configuration system | 5 | 3 | baoyu-slide-deck |
| Ecosystem integration | 5 | 3 | baoyu-slide-deck |
| Documentation quality | 5 | 4 | baoyu-slide-deck |

*(Scale: 0 = not supported, 1 = minimal, 3 = average, 5 = excellent)*

---

## 13. Recommendation

**Choose `baoyu-slide-deck` if you:**
- Create slides from general content (blog posts, articles, marketing material, tutorials)
- Want maximum style customization (450 combinations)
- Need fine-grained control over the generation process
- Want to iterate on specific slides (`--regenerate`)
- Already use or plan to use the broader baoyu-skills ecosystem
- Need multi-provider image generation support

**Choose `paper-slide-deck` if you:**
- Primarily create slides from academic papers (PDFs)
- Need automatic figure and table extraction from PDFs
- Present at conferences (ICML, NeurIPS, CVPR, ACL, etc.)
- Need academic-specific layouts (methods diagrams, results charts, qualitative grids)
- Want the `academic-paper` visual style with proper citation and equation formatting
- Prefer a streamlined workflow with 3 outline variants to pick from

**Ideal scenario**: Use **both** -- baoyu-slide-deck for general content and paper-slide-deck for academic papers. The skills are complementary and share enough DNA that switching between them is frictionless.
