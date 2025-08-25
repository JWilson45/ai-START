import logging
from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.graph import build_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Quiet noisy third-party logs while keeping our own DEBUG/INFO as configured
for name, level in {"httpx": logging.WARNING, "openai": logging.WARNING, "langchain": logging.INFO}.items():
    logging.getLogger(name).setLevel(level)


# --- Interrupt helpers (clean, testable) ---

def _latest_ai_message(messages):
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None


def _print_ai(out) -> bool:
    last_ai = _latest_ai_message(out.get("messages", []))
    if last_ai and str(getattr(last_ai, "content", "")).strip():
        print("AI:", str(last_ai.content), "\n")
        return True
    return False


def _resume(app, config, payload):
    logger.debug("Resuming with payload: %s", payload)
    return app.invoke(Command(resume=payload), config)


def handle_interrupts(app, config, out):
    """Process any pending interrupts in a single place.

    Returns (out, printed_ai) where `out` is the updated graph output and
    `printed_ai` indicates whether an AI reply was printed as part of handling.
    """
    printed_ai = False
    while "__interrupt__" in out:
        payload = out["__interrupt__"][0].value
        logger.debug("Handling interrupt: %s", payload)
        if isinstance(payload, dict) and payload.get("tool") == "run_bash":
            cmd = payload.get("command", "")
            print(f"Tool request: run_bash\n$ {cmd}")
            yn = input("Approve this single command? (y/n): ").strip().lower()
            if yn == "y":
                out = _resume(app, config, {"approve": True})
                # Print tool output (latest ToolMessage from run_bash)
                for m in reversed(out.get("messages", [])):
                    if isinstance(m, ToolMessage) and getattr(m, "name", "") == "run_bash":
                        print("\n―――― TOOL OUTPUT ――――")
                        print(m.content)
                        print("―――― END OUTPUT ――――\n")
                        break
                printed_ai = _print_ai(out)
            else:
                reason = input("Why deny? ").strip()
                out = _resume(app, config, {"approve": False, "reason": reason})
                # Print tool response (denied message from ToolMessage)
                for m in reversed(out.get("messages", [])):
                    if isinstance(m, ToolMessage) and getattr(m, "name", "") == "run_bash":
                        print("\n―――― TOOL RESPONSE ――――")
                        print(m.content)
                        print("―――― END RESPONSE ――――\n")
                        break
                printed_ai = _print_ai(out)
        else:
            # Non-run_bash interrupts: resume once and exit loop
            out = _resume(app, config, {"approve": False, "reason": "Non-run_bash interrupt"})
            break
    return out, printed_ai

# --- CLI ---
def main():
    print("Welcome — pirate CLI. Type 'exit' to quit.")
    app = build_app()
    config = {"configurable": {"thread_id": "default"}}

    while True:
        user_input = input("You: ")
        logger.debug("User input: %s", user_input)
        if user_input.lower() in {"exit", "quit"}:
            break

        out = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        logger.debug("Graph output: %s", out)
        printed_ai = False

        # Handle any pending interrupts (tool approvals) in one place
        out, printed_ai = handle_interrupts(app, config, out)

        if not printed_ai:
            if not _print_ai(out):
                print("AI:", "Ahoy! How can I assist ye today?", "\n")

if __name__ == "__main__":
    main()