from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from tools.my_tools import add_numbers, get_current_time, fake_search, search

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools([add_numbers, get_current_time, fake_search, search])

# Build a registry mapping exposed tool names -> StructuredTool objects
IMPORTED_TOOLS = (add_numbers, get_current_time, fake_search, search)
TOOLS = {t.name: t for t in IMPORTED_TOOLS}

# Debug: show what names the LLM can call (based on the wrapped tools)
print("Available tools:", list(TOOLS.keys()))

messages = [
    SystemMessage(content="Your name is Rize. Reply short, few words.")
]

while True:
    question = input("You: ")
    messages.append(HumanMessage(content=question))

    # 1) First invoke
    response = llm_with_tools.invoke(messages)

    # Debug prints
    print("DEBUG first invoke content:", repr(response.content))
    print("DEBUG first invoke tool_calls:", response.tool_calls)

    # 2) Append the model response object itself so tool-call metadata is preserved
    messages.append(response)

    # 3) If the model requested tools, execute them and append ToolMessage(s)
    if response.tool_calls:
        for call in response.tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args", {})

            tool = TOOLS.get(tool_name)
            if tool:
                try:
                    # Execute the underlying Python function
                    result = tool.func(**tool_args) if tool_args else tool.func()
                except TypeError as e:
                    # Argument mismatch or other invocation error
                    result = f"Tool invocation error: {e}"
                except Exception as e:
                    # Any other runtime error inside the tool
                    result = f"Tool runtime error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            # Append the tool result as a ToolMessage tied to the tool_call id
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            print("DEBUG executed tool", tool_name, "result:", result)

        # 4) Second invoke so the model can produce final text
        final_response = llm_with_tools.invoke(messages)
        print("DEBUG final_response.content:", repr(final_response.content))

        # messages.append(AIMessage(content=final_response.content))
        messages.append(final_response)
        print("Rize:", final_response.content)
    else:
        # No tool was called; model already returned text
        messages.append(AIMessage(content=response.content))
        print("Rize:", response.content)
