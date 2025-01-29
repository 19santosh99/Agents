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

configuration = asana.Configuration()
configuration.access_token = os.getenv('ASANA_ACCESS_TOKEN', '')
api_client = asana.ApiClient(configuration)

@tool
def hello():
    """ This is a simple tool that returns a greeting message """
    print("Hello from the tool")
    pass

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
    if nested_calls > 3:
        raise Exception("Too many nested calls")
    
    # list of all available tools
    tools = [hello, create_asana_task]
    
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
            "hello": hello,
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

def main():
    # Initialize messages list with system message
    messages = [
            SystemMessage(content=f"You are a helpful assistant who manages asana tasks. Todays date is {datetime.now()}. If you need any details from the user, ask for them before creating tasks"),
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