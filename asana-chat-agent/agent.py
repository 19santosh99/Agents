import os
import asana
from datetime import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
model = "gpt-4o-mini"

@tool
def hello():
    """ This is a simple tool that returns a greeting message """
    print("Hello from the tool")
    pass

def get_openai_response(messages, nested_calls=0):
    if nested_calls > 3:
        raise Exception("Too many nested calls")
    
    # list of all available tools
    tools = [hello]
    
    # Create a client and bind the tools to it
    client = ChatOpenAI(model=model)
    client_with_tools = client.bind_tools(tools)
    
    # Invoke the client with the messages
    ai_response = client_with_tools.invoke(messages)
    
    # Count the number of tool calls in the response
    tool_calls = len(ai_response.tool_calls)

    # If there are tool calls, call the tools and add the results to the messages list
    if tool_calls > 0:
        available_functions = {
            "hello": hello
        }
        
        # Add the openai response(Tool call response) to the messages list
        messages.append(ai_response)
        
        print("Tool calls: ")
        for tool_call in ai_response.tool_calls:
            function_name = tool_call['name'].lower()
            tool_call_id = tool_call['id']
            tool_call_function = available_functions.get(function_name)
            tool_output = tool_call_function.invoke(tool_call["args"])
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call_id))
            print(f"Tool call: {tool_call['name']} with args: {tool_call['args']}")
            
        # Call the AI again so it can produce a response with the result of calling the tool(s)
        ai_response = get_openai_response(messages, nested_calls + 1)      
    
    return ai_response

def main():
    # Initialize messages list with system message
    messages = [
            SystemMessage(content=f"You are a helpful assistant who manages asana tasks. Todays date is {datetime.now()}"),
        ]
        
    while True:
        # Get user input
        input_message = input("Enter your message: ").strip()
        if input_message.lower() == "quit":
            break
        
        # Add user input to messages list
        messages.append(HumanMessage(content=input_message))
        
        # Get response from OpenAI
        openai_response = get_openai_response(messages)
        print(openai_response.content)
        
        # Add response to messages list to be used in next iteration
        messages.append(openai_response)

if __name__ == "__main__":
    main()