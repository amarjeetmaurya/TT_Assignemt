# tools/my_tools.py
import datetime
from langchain.tools import tool

def get_current_time() -> str:
    """Return current date and time as YYYY-MM-DD HH:MM:SS."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Exposed tool name will be "get_current_time"
get_current_time = tool(get_current_time)

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

@tool
def fake_search(query: str) -> str:
    """Return a short canned search summary for the given query."""
    return f"FakeSearchResult for '{query}': top result summary."

@tool("web_search")  # custom exposed name
def search(query: str) -> str:
    """Search the web for information (fake demo)."""
    return f"Results for: {query}"
