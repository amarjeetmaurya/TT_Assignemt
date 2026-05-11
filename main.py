from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from tools.my_tools import add_numbers, get_current_time

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools([add_numbers, get_current_time])

messages = [
    SystemMessage(content="Your name is Rize. You only reply in short sentences, few words only.")
]

while True:
    question = input("You: ")
    messages.append(HumanMessage(content=question))

    response = llm_with_tools.invoke(messages)

    # If the model called a tool
    if response.tool_calls:
        for call in response.tool_calls:
            if call["name"] == "add_numbers":
                result = add_numbers.func(**call["args"])   # use .func to run the underlying Python function
            elif call["name"] == "get_current_time":
                result = get_current_time.func()            # same here
            else:
                result = "Unknown tool"

            messages.append(AIMessage(content=str(result)))
            print("Rize (tool):", result)
    else:
        messages.append(AIMessage(content=response.content))
        print("Rize:", response.content)
