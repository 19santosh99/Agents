import os
import asana
import json
import streamlit as st
from datetime import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Set page config
st.set_page_config(page_title="Asana Chat Agent", page_icon="📋", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=f"You are a helpful assistant who manages asana tasks. Todays date is {datetime.now()}. If you need any details from the user, ask for them before creating tasks. If you dont know about anything just say so."),
    ]

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
model = "gpt-4o-mini"

configuration = asana.Configuration()
configuration.access_token = os.getenv('ASANA_ACCESS_TOKEN', '')
api_client = asana.ApiClient(configuration)

@tool
def create_asana_task(task_name, assignee_email, due_on="today"):
    """
    Creates a task in Asana with the given task name, assignee email, and due date.

    Args:
        task_name (str): The name of the task to be created.
        assignee_email (str): The email of the person to whom the task will be assigned.
        due_on (str, optional): The due date for the task. Defaults to "today".

    Returns:
        str: A message indicating the success or failure of the task creation.
    """
    if due_on == "today":
        due_on = str(datetime.now().date())

    tasks_api_instance = asana.TasksApi(api_client)

    task_body = {
        "data": {
            "name": task_name,
            "due_on": due_on,
            "assignee_email": assignee_email,
            "projects": [os.getenv("ASANA_PROJECT_ID", "")]
        }
    }
    try:
        api_response = tasks_api_instance.create_task(task_body, {})
        return f"Task created successfully with id: {api_response}"
    except asana.ApiException as e:
        return f"Exception when calling TasksApi->create_task: {e}"

def get_openai_response(messages, nested_calls=0):
    try:
        if nested_calls > 3:
            raise Exception("Too many nested calls")
        
        # list of all available tools
        tools = [create_asana_task]
        
        # Create a client with streaming enabled and bind the tools to it
        client = ChatOpenAI(
            model=model,
            api_key=api_key,
            streaming=True
        )
        client_with_tools = client.bind_tools(tools)
        
        # Create a generator for streaming responses
        response_generator = client_with_tools.stream(messages)
        
        # Initialize variables for handling streaming
        current_content = ""
        tool_calls = []
        
        # Process each chunk in the stream
        for chunk in response_generator:
            # Update content if present in chunk
            if chunk.content:
                current_content += chunk.content
                yield "content", current_content
            
            # Collect tool calls if present
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
        
        # Create final AI response
        ai_response = AIMessage(content=current_content, tool_calls=tool_calls)
        
        # If there are tool calls, process them
        if tool_calls:
            available_functions = {
                "create_asana_task": create_asana_task
            }
            
            # Add the AI response to messages
            messages.append(ai_response)
            
            print("Tool calls: ")
            for tool_call in tool_calls:
                function_name = tool_call['name'].lower()
                tool_call_id = tool_call['id']
                tool_call_function = available_functions.get(function_name)
                tool_output = tool_call_function.invoke(tool_call["args"])
                messages.append(ToolMessage(tool_output, tool_call_id=tool_call_id))
                print(f"Tool call: {tool_call['name']} with args: {tool_call['args']}")
                
            # Call the AI again for final response after tool calls
            for content_type, content in get_openai_response(messages, nested_calls + 1):
                yield content_type, content
        else:
            yield "final", ai_response
            
    except Exception as e:
        yield "error", AIMessage(content=f"An error occurred: {str(e)}")

def main():
    st.title("Asana Chat Agent")
    st.markdown("---")

    # Display chat messages
    for message in st.session_state.messages:
        message_json = json.loads(message.json())
        message_type = message_json["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message_json["content"])        

    # Chat input
    if prompt := st.chat_input("What would you like me to do?"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)

        # Create a placeholder for the streaming response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            current_response = ""

            # Stream the AI response
            for content_type, content in get_openai_response(st.session_state.messages):
                if content_type == "content":
                    current_response = content
                    response_placeholder.markdown(current_response + "▌")
                elif content_type == "final":
                    response_placeholder.markdown(content.content)
                    st.session_state.messages.append(content)
                elif content_type == "error":
                    response_placeholder.markdown(content.content)
                    st.session_state.messages.append(content)

if __name__ == "__main__":
    main()
