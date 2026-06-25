# Devcontainer notes

## Codex CLI

This repository is configured to run Codex inside the devcontainer with:

- `default_permissions = ":danger-full-access"`
- `approval_policy = "on-request"`

Rationale:

- the devcontainer is the intended isolation boundary for local development;
- Codex's inner Linux sandbox depends on nested namespace support;
- many Docker/devcontainer setups block that path even when `bubblewrap` is
  installed, which leads to errors such as `bwrap: No permissions to create new
  namespace`.

This is configured in [.codex/config.toml](../.codex/config.toml), so it works
immediately in the current container and after future rebuilds without extra
machine-local setup.

If you need Codex's inner sandbox instead of relying on the devcontainer
boundary, start from OpenAI's secure devcontainer guidance and ensure the outer
container runtime allows the namespace and security settings required by
`bubblewrap`.

## Codex VS Code extension

If the Codex sidebar hangs or stays on a spinner inside the devcontainer:

- prefer the `openai.chatgpt` extension for this workspace;
- do not auto-install `GitHub.copilot-chat` from the devcontainer config;
- if Copilot Chat or GitHub Copilot is enabled in the remote window, disable it
  for this workspace/container and reload the VS Code window before retrying
  Codex.

Rationale:

- this repository has seen Codex activate successfully while the remote
  extension host simultaneously logs a Copilot startup `PendingMigrationError`;
- OpenAI issue reports show Codex can get stuck on the same spinner when a
  competing chat/agent extension breaks the extension host or chat surface.
