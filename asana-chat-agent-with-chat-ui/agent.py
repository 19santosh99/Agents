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
        
        # Create a client and bind the tools to it
        client = ChatOpenAI(
            model=model,
            api_key=api_key
        )
        client_with_tools = client.bind_tools(tools)
        
        # Invoke the client with the messages
        ai_response = client_with_tools.invoke(messages)
        
        # Count the number of tool calls in the response
        tool_calls = len(ai_response.tool_calls)

        # If there are tool calls, call the tools and add the results to the messages list
        if tool_calls > 0:
            available_functions = {
                "create_asana_task": create_asana_task
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
    except Exception as e:
        return AIMessage(content=f"An error occurred: {str(e)}")

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

        # Get AI response
        with st.spinner("Thinking..."):
            response = get_openai_response(st.session_state.messages)
            st.session_state.messages.append(response)
            st.chat_message("assistant").write(response.content)
        
        # Rerun to update the chat history
        st.rerun()

if __name__ == "__main__":
    main()