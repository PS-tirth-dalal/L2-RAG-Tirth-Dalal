# agent_fun.py
# Weekend Wizard -- a cheerful CLI agent that plans your weekend.
# Uses a ReAct-style agentic loop with one-shot reflection.
# Connects to server_fun.py via MCP (stdio transport) and uses Groq API for LLM.
# Features: Rich terminal UI, conversation memory, and personalized responses.

import asyncio
import json
import os
import re
import sys
from typing import Dict, Any, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.status import Status
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule

from memory import ConversationMemory

# ──────────────────────────────────────────────
# Groq client setup
# ──────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"  # Groq's current flagship large model

client = Groq(api_key=GROQ_API_KEY)
console = Console()


# ──────────────────────────────────────────────
# System prompt -- ReAct style
# ──────────────────────────────────────────────
SYSTEM = (
    "You are Weekend Wizard, a cheerful weekend helper.\n"
    "You can call MCP tools to fetch real data.\n"
    "Decide step-by-step (ReAct style). At each step output ONLY valid JSON.\n"
    "\n"
    "To call a tool, output:\n"
    '{"action":"<tool_name>","args":{...}}\n'
    "\n"
    "When you have enough info to answer the USER's ENTIRE request, output:\n"
    '{"action":"final","answer":"<your friendly response>"}\n'
    "\n"
    "Rules:\n"
    "- If the user is greeting you or making casual conversation (e.g. 'hello', "
    "'good morning', 'help me plan my weekend'), respond directly with a friendly "
    "message using the 'final' action. Do NOT call any tools for greetings or chitchat.\n"
    "- If the user asks a simple factual or math question that does not need any tool "
    "(e.g. 'what is 2+2'), answer directly using 'final'. Do NOT call a tool.\n"
    "- CRITICAL: Call ONE tool per step. If the user asks for MULTIPLE things (e.g. a joke AND "
    "a dog photo), you MUST call the first tool, wait for the result, then call the second "
    "tool, and so on. Do NOT output 'final' until you have gathered data for ALL parts of the request.\n"
    "- Strictly follow the required tool argument schemas (e.g. get_weather requires "
    "latitude and longitude floats; do not pass city names. If a user asks for a city, "
    "estimate its latitude/longitude coordinates).\n"
    "- If a user asks for weather in a clearly invalid or nonsense location, you should STILL attempt "
    "to call get_weather using default coordinates (e.g. 0.0, 0.0) so you fulfill the tool requirement.\n"
    "- TRIVIA RULE: When the trivia tool returns a 'trivia_card', you MUST copy that trivia_card text into your final answer EXACTLY as-is. Do NOT answer the trivia question. Do NOT summarize it into a fact. Do NOT add your own knowledge. Just relay the card.\n"
    "- When providing a dog photo or video, you MUST include the actual URL in your final answer.\n"
    "- Use real data from tool results in your final answer.\n"
    "- Keep answers concise, upbeat, and formatted nicely.\n"
    "- Never invent data you haven't fetched from a tool.\n"
    "- Output ONLY the JSON object, no extra text or markdown fences.\n"
)


# ──────────────────────────────────────────────
# LLM helper -- calls Groq API, returns parsed JSON
# ──────────────────────────────────────────────
def llm_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Ask the LLM for a JSON action and parse the result."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    txt = resp.choices[0].message.content.strip()

    # First attempt: direct parse
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass

    # Second attempt: extract JSON from markdown fences or surrounding text
    match = re.search(r"\{.*\}", txt, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Third attempt: ask LLM to repair
    fix = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No explanation."},
            {"role": "user", "content": txt},
        ],
        temperature=0,
        max_tokens=1024,
    )
    fix_txt = fix.choices[0].message.content.strip()
    try:
        return json.loads(fix_txt)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", fix_txt, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse LLM output as JSON:\n{fix_txt}")


# ──────────────────────────────────────────────
# One-shot reflection
# ──────────────────────────────────────────────
def reflect(answer: str) -> str:
    """Ask a second LLM pass to catch mistakes or improve the answer.
    
    Conservative: only replaces if the review is a short, targeted fix.
    Skips reflection for short conversational answers to avoid over-editing.
    """
    # Skip reflection for short answers (greetings, trivia evaluations, etc.)
    if len(answer) < 300:
        return answer

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant evaluating the draft response of a weekend planner agent. "
                    "If the draft response is acceptable, reply EXACTLY AND ONLY with the word: PASS\n"
                    "Only if the draft contains obvious factual contradictions or broken formatting, "
                    "reply with a corrected version that is similar in length to the original."
                ),
            },
            {"role": "user", "content": answer},
        ],
        temperature=0,
        max_tokens=1024,
    )
    review = resp.choices[0].message.content.strip()
    if "PASS" in review:
        return answer
    # Only accept the correction if it's not way longer than the original
    if len(review) > len(answer) * 1.5:
        return answer
    return review


# ──────────────────────────────────────────────
# Display helpers (Rich UI)
# ──────────────────────────────────────────────
def show_welcome_banner(tool_names: List[str], memory: ConversationMemory) -> None:
    """Display the welcome banner with connected tools and memory status."""
    if memory.is_returning_user():
        sessions = memory.data.get("session_count", 0)
        subtitle = f"Welcome back! This is session #{sessions}"
    else:
        subtitle = "Your chill weekend planner"

    banner = Panel(
        f"[bold]Powered by Groq ({MODEL})[/bold]\n"
        f"Connected tools: [cyan]{', '.join(tool_names)}[/cyan]\n\n"
        f'Type a request, "help" for commands, or "exit" to quit.',
        title="[bold]Weekend Wizard[/bold]",
        subtitle=subtitle,
        border_style="bright_blue",
        padding=(1, 2),
    )
    console.print(banner)
    console.print()


def show_agent_answer(answer: str) -> None:
    """Display the agent's final answer in a styled panel."""
    try:
        md = Markdown(answer)
        panel = Panel(
            md,
            title="[bold]Agent[/bold]",
            border_style="green",
            padding=(1, 2),
        )
    except Exception:
        panel = Panel(
            answer,
            title="[bold]Agent[/bold]",
            border_style="green",
            padding=(1, 2),
        )
    console.print(panel)
    console.print()


def show_tool_call(tname: str, args: Dict) -> None:
    """Display a tool call notification."""
    console.print(f"  [cyan]Calling tool:[/cyan] [bold]{tname}[/bold]({args})")


def show_tool_result(char_count: int) -> None:
    """Display a tool result notification."""
    console.print(f"  [dim]Result received ({char_count} chars)[/dim]")


def show_error(msg: str) -> None:
    """Display an error message."""
    console.print(f"  [red bold]Error:[/red bold] {msg}")


def show_help() -> None:
    """Display available commands."""
    help_panel = Panel(
        "[bold]Available commands:[/bold]\n\n"
        "  [cyan]help[/cyan]          -- Show this help message\n"
        "  [cyan]clear memory[/cyan]  -- Reset conversation memory\n"
        "  [cyan]exit / quit[/cyan]   -- Exit the wizard\n\n"
        "[bold]Example queries:[/bold]\n\n"
        '  "What is the weather in Mumbai?"\n'
        '  "Recommend sci-fi books"\n'
        '  "Tell me a joke and show me a dog photo"\n'
        '  "Give me a trivia question"',
        title="[bold]Help[/bold]",
        border_style="yellow",
        padding=(1, 2),
    )
    console.print(help_panel)
    console.print()


# ──────────────────────────────────────────────
# Main agent loop
# ──────────────────────────────────────────────
async def main():
    # Validate API key
    if not GROQ_API_KEY:
        console.print(Panel(
            "[bold]GROQ_API_KEY not set![/bold]\n\n"
            "Set it via environment variable:\n"
            '  $env:GROQ_API_KEY = "your-key-here"  (PowerShell)\n'
            "  set GROQ_API_KEY=your-key-here       (cmd)",
            title="[bold]Configuration Error[/bold]",
            border_style="red",
            padding=(1, 2),
        ))
        sys.exit(1)

    server_path = sys.argv[1] if len(sys.argv) > 1 else "server_fun.py"

    # ── Initialize conversation memory ──
    memory = ConversationMemory()
    memory.start_session()

    # ── Connect to MCP server via stdio ──
    exit_stack = AsyncExitStack()
    stdio = await exit_stack.enter_async_context(
        stdio_client(
            StdioServerParameters(
                command=sys.executable,  # use the same Python from venv
                args=[server_path],
            )
        )
    )
    r_in, w_out = stdio
    session = await exit_stack.enter_async_context(ClientSession(r_in, w_out))
    await session.initialize()

    # ── Discover tools ──
    tools = (await session.list_tools()).tools
    tool_index = {t.name: t for t in tools}
    tool_names = list(tool_index.keys())

    show_welcome_banner(tool_names, memory)

    tool_defs = []
    for t in tools:
        tool_defs.append(f"- {t.name}: {t.description}\n  Schema: {json.dumps(t.inputSchema)}")
    tool_defs_str = "\n".join(tool_defs)

    # Inject memory context into system prompt
    memory_context = memory.get_summary()
    memory_section = f"\n\nUser context from previous sessions:\n{memory_context}" if memory_context else ""
    system_prompt = f"{SYSTEM}\n\nAvailable Tools and their JSON schemas:\n{tool_defs_str}{memory_section}"

    history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    try:
        while True:
            try:
                user = Prompt.ask("[bold bright_blue]You[/bold bright_blue]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nSee you next weekend!")
                break

            if not user or user.lower() in {"exit", "quit"}:
                console.print(Rule("See you next weekend!", style="bright_blue"))
                break

            # ── Built-in commands ──
            if user.lower() == "help":
                show_help()
                continue

            if user.lower() == "clear memory":
                memory.clear()
                console.print(Panel(
                    "Conversation memory has been cleared.",
                    border_style="yellow",
                    padding=(0, 2),
                ))
                console.print()
                continue

            history.append({"role": "user", "content": user})

            # ── Agentic ReAct loop (max 8 steps for safety) ──
            max_steps = 8
            final_answer = ""
            for step in range(max_steps):
                try:
                    with Status("[bold cyan]Thinking...[/bold cyan]", console=console, spinner="dots"):
                        decision = llm_json(history)
                except Exception as e:
                    show_error(f"JSON parse error: {e}")
                    history.append(
                        {
                            "role": "assistant",
                            "content": '{"action":"final","answer":"Sorry, I hit a hiccup. Could you rephrase?"}',
                        }
                    )
                    show_agent_answer("Sorry, I hit a hiccup. Could you rephrase?")
                    break

                # ── Final answer ──
                if decision.get("action") == "final":
                    answer = decision.get("answer", "")
                    # One-shot reflection
                    with Status("[bold cyan]Reviewing...[/bold cyan]", console=console, spinner="dots"):
                        answer = reflect(answer)
                    show_agent_answer(answer)
                    history.append({"role": "assistant", "content": answer})
                    final_answer = answer
                    break

                # ── Tool call ──
                tname = decision.get("action", "")
                args = decision.get("args", {})

                if tname not in tool_index:
                    msg = f"Unknown tool '{tname}', available: {tool_names}"
                    show_error(msg)
                    history.append({"role": "user", "content": f"System error: unknown tool '{tname}'. Please choose from {tool_names} or output final answer."})
                    continue

                show_tool_call(tname, args)
                try:
                    with Status("[bold cyan]Fetching data...[/bold cyan]", console=console, spinner="dots"):
                        result = await session.call_tool(tname, args)
                        payload = (
                            result.content[0].text
                            if result.content
                            else result.model_dump_json()
                        )
                except Exception as e:
                    payload = json.dumps({"error": str(e)})

                # ── Trivia shortcut: display card directly, bypass LLM ──
                if tname == "trivia":
                    try:
                        trivia_data = json.loads(payload)
                        card = trivia_data.get("trivia_card", "")
                        if card:
                            show_agent_answer(card)
                            history.append({"role": "assistant", "content": card})
                            # Store the correct answer so the AI can check guesses later
                            correct = trivia_data.get("correct_answer", "")
                            if correct:
                                history.append({"role": "user", "content": f"(System note: The correct answer to the trivia question above is '{correct}'. When the user guesses, tell them if they are right or wrong and reveal the correct answer. Do NOT call the trivia tool again unless the user explicitly asks for a NEW question.)"})
                            final_answer = card
                            break
                    except (json.JSONDecodeError, KeyError):
                        pass  # fall through to normal LLM handling

                # Feed observation back into history as user/system prompt
                observation = f"Tool result for {tname}: {payload}\nNow output the final response to the user using 'action': 'final'."
                show_tool_result(len(payload))
                history.append({"role": "user", "content": observation})
            else:
                # Ran out of steps
                show_agent_answer("I've done a lot of research! Let me wrap up with what I have.")
                history.append(
                    {
                        "role": "assistant",
                        "content": "I've gathered enough data, let me summarize.",
                    }
                )

            # ── Save exchange to memory ──
            if final_answer:
                memory.add_exchange(user, final_answer)

    finally:
        await exit_stack.aclose()


if __name__ == "__main__":
    asyncio.run(main())
