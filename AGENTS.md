# Notebook Agent Guide

This file applies to the whole repository.

This repository is a fork of a textbook codebase. Much of the upstream material is
still stored as Jupyter notebooks, but for personal notes, exercises, and
experiments, the primary working format in this workspace is marimo notebooks in
`notes/`.

Unless the user asks otherwise:

- Prefer editing or creating marimo notebooks in `notes/*.py`.
- Treat `*.ipynb` files as upstream reference material, compatibility artifacts, or
  snapshots rather than the main target for new work.
- Avoid hand-editing generated marimo state files in `__marimo__/` or
  `notes/__marimo__/` unless the task is explicitly about those files.

## Project intent

The target quality bar is a middle ground:

- Better than a messy scratch notebook full of ad hoc globals and unclear cells.
- Less abstract than production application code with unnecessary layers, deep
  class hierarchies, or helper functions nested everywhere.
- Strong on readability, reproducibility, documentation, naming, and types.
- Optimized for learning, exploration, and durable personal notes.

When there is a tradeoff, prefer clear notebook code over over-engineering.

## Marimo fundamentals

Marimo is not Jupyter. Agents working here must respect marimo's execution model.

- Cells are reactive and form a directed acyclic graph.
- Variables cannot be redeclared across cells.
- A change to a variable can rerun dependent cells.
- UI elements are reactive, and their values are read with `.value`.
- Do not read a UI element's value in the same cell where the element is created.
- The last expression in a cell is automatically displayed.
- Avoid hidden state and side effects that make reruns confusing.

## Primary notebook conventions

Use these conventions by default for notebooks in `notes/`.

- Keep imports, shared constants, and notebook-wide setup in `with app.setup:`.
- Preserve the standard marimo structure when editing an existing notebook:
  `import marimo`, `app = marimo.App(...)`, decorators, and generated return
  signatures.
- Use `@app.cell(hide_code=True)` for section headings or short explanatory
  markdown when it improves notebook flow.
- Keep each code cell focused on one conceptual step: load data, define controls,
  transform data, train a model, visualize results, or summarize findings.
- Prefer a small number of coherent cells over many tiny fragmented cells that make
  the notebook hard to follow.
- Also avoid giant omnibus cells that mix setup, transformation, plotting, and
  interpretation all at once.

## Abstraction level

This repo does not want two extremes:

- Not messy prototype notebooks with dozens of weakly named variables and no
  structure.
- Not "productionized" notebooks full of abstractions that obscure the lesson.

Default approach:

- Write straightforward notebook code first.
- Extract a helper function when logic is reused, conceptually important, or much
  easier to understand when named.
- Introduce a small class or dataclass only when it genuinely clarifies state or
  groups related behavior.
- Prefer top-level helpers or `@app.function` for reusable logic.
- Avoid deep inheritance, factories, service layers, and heavy framework patterns
  unless the user explicitly asks for them.
- Avoid nested helper functions unless locality is clearly more readable than a
  top-level definition.

## Documentation and typing

Code should aim for good engineering hygiene even when the notebook is exploratory.

- Use descriptive variable names.
- Add type hints for non-trivial helper functions, classes, and important data
  structures.
- Add docstrings to helpers whose purpose or inputs are not obvious.
- Use short comments to explain reasoning, assumptions, tensor shapes, or tricky
  implementation details.
- Do not add filler comments that merely restate the code.
- Prefer named constants over unexplained magic numbers.
- When a result depends on a shape, dtype, seed, or device assumption, say so in
  code or markdown.

## Data science and ML guidance

- Keep experiments reproducible when practical: set seeds when comparisons matter.
- Centralize device selection and reuse a shared device variable rather than
  scattering `"cuda"` and `"cpu"` literals.
- Keep preprocessing, training, evaluation, and visualization stages distinct.
- Prefer small, readable experiment scaffolding over generic training frameworks for
  one-off chapter exercises.
- If an experiment writes outputs, keep them in sensible repo locations and avoid
  creating clutter.
- Do not add heavyweight dependencies or large new datasets unless the task calls
  for them.

## Visualization and UI

- Use notebook UI only when it materially improves understanding or exploration.
- Group related controls together and keep the interaction model obvious.
- For matplotlib, prefer returning the axes object, such as `plt.gca()`, rather
  than calling `plt.show()`.
- For plotly or altair, return the figure or chart object directly.
- Label plots clearly and choose defaults that support explanation, not visual
  novelty.
- Prefer layouts that aid comparison or teaching over dashboard-style complexity.

## File-specific guidance

- `notes/`: primary home for marimo notebooks, markdown notes, and exercises.
- `notes/images/`: figures and exported visuals used by notes.
- `notes/layouts/`: marimo layout metadata; preserve it when it is part of the
  notebook workflow.
- `notes/__marimo__/` and root `__marimo__/`: generated marimo artifacts; do not
  manually refactor or tidy these unless asked.
- Upstream textbook files outside `notes/` should generally be left alone unless
  the task clearly requires modifying them.

## What to avoid

- Do not silently rewrite upstream Jupyter notebooks into marimo.
- Do not refactor notebook exercises into package-style architectures without a
  concrete benefit.
- Do not introduce indirection purely to make code look more "professional".
- Do not split logic so aggressively across cells that the narrative becomes hard
  to read.
- Do not keep all logic in a single scratch cell just because the notebook is
  exploratory.
- Do not break marimo reactivity by redeclaring names across cells or creating
  circular dependencies.

## Preferred workflow

When adding or updating a notebook:

1. Keep the notebook runnable end-to-end.
2. Organize it as a readable narrative with short markdown guidance where useful.
3. Use helpers only where they clearly improve comprehension or reuse.
4. Keep the code clean enough to revisit later without turning it into a library.
5. Preserve marimo semantics and generated structure.

## Rule of thumb

Aim for notebooks that feel like well-kept technical study notes:

- cleaner than a prototype,
- lighter than production code,
- and fully aligned with marimo's reactive model.
