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
- Target renderer (VS Code Excalidraw extension, Obsidian Excalidraw plugin, GitHub, generic Markdown)

Ask one clarifying question only if a blocking input is missing.

## Workflow (use separate subagent invocations)
1. **Discover** — inspect the target file and neighbouring assets; detect existing naming conventions (`.excalidraw`, `.excalidraw.svg`, `.excalidraw.png`)
2. **Read existing** — if updating, read the current diagram before changing anything; extremely important to invoke subagent to avoid polluting main agent's context with large JSON content.
3. **Generate** — create or update the editable `.excalidraw` source using `excalidraw-diagram-generator`; keep it aligned with local conventions
4. **Materialize image sidecar** — when Markdown needs an image artifact and the official VS Code Excalidraw extension workflow is available, duplicate and rename the file to `.excalidraw.svg` or `.excalidraw.png`, then open that file in the Excalidraw editor and save it so the extension exports the actual image with embedded scene
5. **Review** — independently check accuracy, layout, labels, and consistency with note text; fix issues before finishing

## File and embedding rules
- `.excalidraw` JSON is the editable source of truth.
- Never hand-author, patch, or manually synthesize a `.excalidraw.svg` or `.excalidraw.png` sidecar. Those files must be produced by Excalidraw itself, either through the official VS Code extension save flow or an official Excalidraw export API.
- Default: keep the editable `.excalidraw` source, then create a sidecar image artifact only after the export step has actually happened.
- If image export is not available in the current environment, do not fake the sidecar. Either keep a direct `.excalidraw` embed when the renderer supports it, or fall back to a normal Markdown link and state the limitation explicitly.
- Direct `.excalidraw` embeds only when the renderer is confirmed to support them (e.g. Obsidian Excalidraw plugin) — state this assumption explicitly
- Match existing local conventions; use relative Markdown links; replace stale embeds rather than duplicating them
- A `.excalidraw.svg` intended to reopen in Excalidraw must be an actual Excalidraw export with embedded Excalidraw payload metadata. A plain standalone SVG with that filename is invalid.

## Validation (required before finishing)
Confirm: source `.excalidraw` exists · any `.excalidraw.svg` or `.excalidraw.png` sidecar was actually saved by Excalidraw after the rename or duplicate step · sidecar and source correspond to the same diagram state · the image artifact can be opened as an image and, if relevant, reopened in the Excalidraw editor · Markdown link points to correct relative path · embedded image is inspectable. If live preview or export tooling is unavailable, say so explicitly and do not fabricate the sidecar.

## Response format
State: files created/modified · whether the official VS Code Excalidraw save flow or another official Excalidraw export path was used · embedding method chosen and why · renderer assumptions · any follow-up work needed.