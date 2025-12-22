# 🍏 Isaac

Isaac is a coding client and agent implementing the [Agent Client Protocol (ACP)](https://agentclientprotocol.com/).

> Since Newton discovered gravity, everything's been going downhill.

## Quick start

To run the client with the agent: `uv run python -m isaac.client uv run isaac`

## Reusable ACP core

Isaac exposes a reusable ACP core in `isaac.acp` so other agents can plug in their
own tools, slash commands, prompt strategies, and system prompts.

```python
from isaac.acp import ACPAgentCore, build_isaac_config, apply_overrides

config = build_isaac_config(
    agent_name="my-agent",
    agent_title="My ACP Agent",
)

# Override anything you need.
config = apply_overrides(
    config,
    slash_commands=None,
    plan_support=None,
)

agent = ACPAgentCore(config)
```

If you want a thinner starting point, construct `ACPAgentConfig` directly and
provide only the components your agent supports.

```python
from acp.schema import PromptResponse
from isaac.acp import ACPAgentConfig, ACPAgentCore, PromptStrategy, NullSessionStore


class EchoStrategy(PromptStrategy):
    async def init_session(self, session_id, toolsets, system_prompt=None):
        return None

    async def set_session_model(self, session_id, model_id, toolsets, system_prompt=None):
        return None

    async def handle_prompt(self, session_id, prompt_text, cancel_event):
        return PromptResponse(stop_reason="end_turn")

    def model_id(self, session_id):
        return "echo"


config = ACPAgentConfig(
    agent_name="echo-agent",
    agent_title="Echo Agent",
    agent_version="0.1.0",
    prompt_strategy=EchoStrategy(),
    session_store=NullSessionStore(),
)

agent = ACPAgentCore(config)
```

## Wheel install (cross-client testing)

To test Isaac with other ACP clients without bumping the version, build and
install a wheel to your user site:

```
uv build --wheel
python -m pip install --user --no-deps --force-reinstall dist/isaac-*.whl
```
