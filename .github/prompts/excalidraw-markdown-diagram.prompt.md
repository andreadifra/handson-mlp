---
description: "Create, update, or embed Excalidraw diagrams in Markdown notes."
name: "Excalidraw Markdown Diagram"
argument-hint: "Describe the diagram, the target Markdown file, and whether this is a new diagram, an update, or an embed."
agent: "agent"
---

Use the `excalidraw-diagram-generator` skill as the primary workflow.

## Inputs needed
- Target Markdown file
- Task type: new diagram / update existing / embed only
- Diagram purpose and detail level
- Target renderer (Obsidian Excalidraw plugin, GitHub, generic Markdown)

Ask one clarifying question only if a blocking input is missing.

## Workflow (use separate subagent invocations)
1. **Discover** — inspect the target file and neighbouring assets; detect existing naming conventions (`.excalidraw`, `.excalidraw.svg`, `.excalidraw.png`)
2. **Read existing** — if updating, read the current diagram before changing anything; extremely important to invoke subagent to avoid polluting main agent's context with large JSON content.
3. **Generate** — create or update the diagram using `excalidraw-diagram-generator`; keep it  aligned with local conventions
4. **Review** — independently check accuracy, layout, labels, and consistency with note text; fix issues before finishing

## File and embedding rules
- `.excalidraw` JSON is the editable source of truth — never rename it to `.png` or `.svg`
- Default: embed an exported sidecar (`.excalidraw.svg` for scalability, `.excalidraw.png` for broadest compatibility)
- Direct `.excalidraw` embeds only when the renderer is confirmed to support them (e.g. Obsidian Excalidraw plugin) — state this assumption explicitly
- Match existing local conventions; use relative Markdown links; replace stale embeds rather than duplicating them

## Validation (required before finishing)
Confirm: source `.excalidraw` exists · sidecar export exists · Markdown link points to correct relative path · embedded image is inspectable. If live preview is unavailable, say so explicitly.

## Response format
State: files created/modified · embedding method chosen and why · renderer assumptions · any follow-up work needed.