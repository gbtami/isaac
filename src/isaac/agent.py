import argparse
import asyncio
import logging
import os
import uuid
from pathlib import Path

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
    ToolCall,
    session_notification,
    stdio_streams,
    text_block,
    tool_result,
    update_agent_message,
    PROTOCOL_VERSION,
)
from acp.schema import AgentCapabilities, Implementation, Tool, ToolParameter, ToolCall

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import tools
from .tools.list_directory import list_files
from .tools.read_file import read_file
from .tools import get_tools

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
BASE_URL = "https://api.cerebras.ai/v1"
MODEL_NAME = "qwen-3-235b-a22b-instruct-2507"


model = OpenAIChatModel(
    MODEL_NAME,
    provider=OpenAIProvider(base_url=BASE_URL, api_key=CEREBRAS_API_KEY),
)
pydantic_agent = PydanticAgent(model)

# Register tools with the agent
@pydantic_agent.tool()
def tool_list_files(directory: str = ".", recursive: bool = True):
    """List files in a directory."""
    return list_files(directory, recursive)

@pydantic_agent.tool()
def tool_read_file(file_path: str, start_line: int = None, num_lines: int = None):
    """Read a file with optional line range."""
    return read_file(file_path, start_line, num_lines)

async def run_simple_agent():
    """A minimal interactive mode using basic stdin/stdout."""
    while True:
        try:
            prompt = input(">>> ")
            if prompt.lower() in ["exit", "quit"]:
                break
            response: StreamedRunResult = await pydantic_agent.run(prompt)
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

    # Force re-configuration by removing all handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )

    logger = logging.getLogger("acp_server")
    return logger


logger = logging.getLogger("acp_server")


class ACPAgent(Agent):
    def __init__(self, conn: AgentSideConnection) -> None:
        self._conn = conn
        self._sessions: set[str] = set()

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        logger.info("Received initialize request")
        return InitializeResponse(
            protocolVersion=PROTOCOL_VERSION,
            agentCapabilities=AgentCapabilities(tools=get_tools()),
            agentInfo=Implementation(name="isaac", title="Isaac ACP Agent", version="0.1.0"),
        )

    async def authenticate(self, params: AuthenticateRequest) -> AuthenticateResponse | None:
        logger.info("Received authenticate request %s", params.methodId)
        return AuthenticateResponse()

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        logger.info("Received new session request")
        session_id = str(uuid.uuid4())
        self._sessions.add(session_id)
        return NewSessionResponse(sessionId=session_id, modes=None)

    async def loadSession(self, params: LoadSessionRequest) -> LoadSessionResponse | None:
        logger.info("Received load session request %s", params.sessionId)
        self._sessions.add(params.sessionId)
        return LoadSessionResponse()

    async def setSessionMode(self, params: SetSessionModeRequest) -> SetSessionModeResponse | None:
        logger.info(
            "Received set session mode request %s -> %s",
            params.sessionId,
            params.modeId,
        )
        return SetSessionModeResponse()

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        """Handles a new prompt from the client."""
        logger.info(f"Received prompt request for session: {params.sessionId}")
        
        # First, check if there are any tool calls in the prompt history
        for block in params.prompt:
            if hasattr(block, 'toolCall') and block.toolCall:
                tool_call = block.toolCall
                function_name = tool_call.function
                arguments = tool_call.arguments or {}
                
                logger.info(f"Handling tool call: {function_name} with args: {arguments}")
                
                # Dispatch to appropriate tool
                if function_name == "tool_list_files":
                    result = await list_files(
                        directory=arguments.get("directory", "."),
                        recursive=arguments.get("recursive", True)
                    )
                elif function_name == "tool_read_file":
                    result = await read_file(
                        file_path=arguments.get("file_path"),
                        start_line=arguments.get("start_line"),
                        num_lines=arguments.get("num_lines")
                    )
                else:
                    error_msg = f"Unknown tool function: {function_name}"
                    logger.error(error_msg)
                    result = {"error": error_msg, "content": None}
                
                # Send tool result back to client
                await self._conn.sessionUpdate(
                    session_notification(
                        params.sessionId,
                        tool_result(tool_call_id=tool_call.toolCallId, content=result)
                    )
                )
                
                return PromptResponse(stopReason="tool_calls")
        
        # If no tool calls, process as normal text prompt
        prompt_text = "".join(
            block.text
            for block in params.prompt
            if block.type == "text" and hasattr(block, "text") and block.text
        )

        if not prompt_text:
            logger.warning("Prompt text is empty. Ending turn.")
            return PromptResponse(stopReason="end_turn")

        try:
            logger.info(f"Running agent with prompt: '{prompt_text}'")
            response: StreamedRunResult = await pydantic_agent.run(prompt_text)
            logger.info(f"Agent finished running. Response: '{response.output}'")
            if response and response.output:
                await self._conn.sessionUpdate(
                    session_notification(
                        params.sessionId,
                        update_agent_message(text_block(response.output)),
                    )
                )
            return PromptResponse(stopReason="end_turn")
        except Exception as e:
            logger.error(f"Error in prompt: {e}", exc_info=True)
            return PromptResponse(stopReason="refusal")

    async def cancel(self, params: CancelNotification) -> None:
        logger.info("Received cancel notification for session %s", params.sessionId)


async def run_acp_agent():
    """Run the ACP server."""
    setup_acp_logging()
    logger.info("Starting ACP server on stdio")

    reader, writer = await stdio_streams()
    AgentSideConnection(lambda conn: ACPAgent(conn), writer, reader)
    await asyncio.Event().wait()


async def main():
    parser = argparse.ArgumentParser(description="Simple ACP agent")
    parser.add_argument("--acp", action="store_true", help="Run in ACP mode")
    args = parser.parse_args()

    if args.acp:
        await run_acp_agent()
    else:
        await run_simple_agent()


def main_entry():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    main_entry()
