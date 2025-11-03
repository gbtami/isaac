import argparse
import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from acp import (
    Agent,
    AgentSideConnection,
    AuthenticateRequest,
    AuthenticateResponse,
    CancelNotification,
    InitializeRequest,
    InitializeResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
    session_notification,
    stdio_streams,
    text_block,
    update_agent_message,
    PROTOCOL_VERSION,
)
from acp.schema import AgentCapabilities, AgentMessageChunk, Implementation

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
BASE_URL = "https://api.cerebras.ai/v1"
MODEL_NAME = "qwen-3-235b-a22b-instruct-2507"


async def run_simple_agent(agent: PydanticAgent):
    """A minimal interactive mode using basic stdin/stdout."""
    while True:
        try:
            prompt = input(">>> ")
            if prompt.lower() in ["exit", "quit"]:
                break
            response = await agent.run(prompt)
            if response and response.output:
                print(response.output)
        except (EOFError, KeyboardInterrupt):
            break
    print("\nExiting simple interactive agent.")


def setup_acp_logging():
    """Set up a file logger for the ACP server to avoid interfering with stdio."""
    log_dir = Path.home() / ".isaac"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "acp_server.log"

    logger = logging.getLogger("acp_server")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
    return logger


logger = logging.getLogger("acp_server")


class ACPAgent(Agent):
    def __init__(self, conn: AgentSideConnection, agent: PydanticAgent) -> None:
        self._conn = conn
        self.agent = agent
        self._sessions: set[str] = set()

    async def _send_agent_message(self, session_id: str, content: Any) -> None:
        update = (
            content
            if isinstance(content, AgentMessageChunk)
            else update_agent_message(content)
        )
        await self._conn.sessionUpdate(session_notification(session_id, update))

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        logger.info("Received initialize request")
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(),
            agentInfo=Implementation(
                name="example-agent", title="Example Agent", version="0.1.0"
            ),
        )

    async def authenticate(
        self, params: AuthenticateRequest
    ) -> AuthenticateResponse | None:
        logger.info("Received authenticate request %s", params.methodId)
        return AuthenticateResponse()

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        logger.info("Received new session request")
        session_id = str(uuid.uuid4())
        self._sessions.add(session_id)
        return NewSessionResponse(sessionId=session_id, modes=None)

    async def loadSession(
        self, params: LoadSessionRequest
    ) -> LoadSessionResponse | None:
        logger.info("Received load session request %s", params.sessionId)
        self._sessions.add(params.sessionId)
        return LoadSessionResponse()

    async def setSessionMode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse | None:
        logger.info(
            "Received set session mode request %s -> %s",
            params.sessionId,
            params.modeId,
        )
        return SetSessionModeResponse()

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        """
        Handles a new prompt from the client.
        """
        try:
            logger.info(f"Received prompt request for session: {params.sessionId}")
            # Extract prompt text from the request
            prompt_text = "".join(
                block.text
                for block in params.prompt
                if block.type == "text" and hasattr(block, "text") and block.text
            )

            if not prompt_text:
                logger.warning("Prompt text is empty. Ending turn.")
                return PromptResponse(stopReason="end_turn")

            logger.info(f"Extracted prompt text: {prompt_text}")

            logger.info(
                f"Using agent: {type(self.agent).__name__} for session: {params.sessionId}"
            )

            # Run the agent with the prompt
            logger.info("Running agent with prompt...")
            response = await self.agent.run(prompt_text)
            logger.info(
                f"Agent response received: {response.output if response else 'None'}"
            )

            # Send the response back to the client as a text block update
            if response and response.output:
                logger.info("Sending response to client.")
                await self._conn.sessionUpdate(
                    session_notification(
                        params.sessionId,
                        update_agent_message(text_block(response.output)),
                    )
                )

            # Signal the end of the turn
            logger.info("Ending turn.")
            return PromptResponse(stopReason="end_turn")
        except Exception as e:
            logger.error(f"An error occurred in the prompt method: {e}", exc_info=True)
            return PromptResponse(stopReason="refusal")

    async def cancel(self, params: CancelNotification) -> None:
        logger.info("Received cancel notification for session %s", params.sessionId)

    async def extMethod(self, method: str, params: dict) -> dict:
        logger.info("Received extension method call: %s", method)
        return {"example": "response"}

    async def extNotification(self, method: str, params: dict) -> None:
        logger.info("Received extension notification: %s", method)


async def run_acp_agent(agent: PydanticAgent):
    """Run the ACP server."""
    setup_acp_logging()
    logger.info("Starting ACP server on stdio")

    reader, writer = await stdio_streams()
    conn = AgentSideConnection(lambda conn: ACPAgent(conn, agent), writer, reader)
    await conn.wait()


async def main():
    parser = argparse.ArgumentParser(description="Simple ACP agent")

    parser.add_argument("--acp", action="store_true", help="Run in ACP mode")
    args = parser.parse_args()

    model = OpenAIChatModel(
        MODEL_NAME,
        provider=OpenAIProvider(base_url=BASE_URL, api_key=CEREBRAS_API_KEY),
    )
    agent = PydanticAgent(model)

    if args.acp:
        await run_acp_agent(agent)
    else:
        await run_simple_agent(agent)


def main_entry():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    main_entry()
