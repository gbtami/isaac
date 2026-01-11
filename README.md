# 🍏 Isaac

Isaac is a coding client and agent implementing the [Agent Client Protocol (ACP)](https://agentclientprotocol.com/).

> Since Newton discovered gravity, everything's been going downhill.

To run the client with the agent: `uv run python -m isaac.client uv run isaac`

## OAuth login

Isaac supports OAuth logins for OpenAI Codex and Code Assist:

- OpenAI Codex: run `/login openai`, then switch to `openai-codex:gpt-5.2-codex`.
- Code Assist (gemini-cli style): run `/login code-assist`, then switch to `code-assist:gemini-2.5-flash`.

OAuth tokens are stored at `~/.config/isaac/oauth_tokens.json` (OpenAI Codex) and `~/.config/isaac/code_assist.json` (Code Assist).
