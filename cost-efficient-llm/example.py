import asyncio
from llm_router import LLMRouter, Message
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

async def main():
    # Verify required API keys
    required_keys = ["GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")

    # Initialize the router
    router = LLMRouter()
    
    # Simulate a conversation
    conversation = [
        Message(role="user", content="Hi, can you help me understand machine learning?"),
        Message(role="assistant", content="Of course! Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience. What specific aspect would you like to know more about?"),
        Message(role="user", content="Can you explain neural networks?"),
        Message(role="assistant", content="Neural networks are computing systems inspired by biological neural networks in human brains. They consist of interconnected nodes (neurons) that process and transmit information. Would you like me to explain their basic structure?"),
        Message(role="user", content="Yes, and can you also provide a Python code example of implementing a simple neural network for image classification?")
    ]
    
    # Process the conversation
    print("Processing conversation...")
    print("-" * 50)
    
    # Get response using the entire conversation history
    # (but model selection will only use last 3 messages)
    response, selected_model = router.get_response(conversation)
    
    # Print conversation flow and model selection
    for msg in conversation:
        print(f"{msg.role.capitalize()}: {msg.content}")
    
    print("-" * 50)
    print(f"Selected Model: {selected_model.value}")
    print(f"Assistant's Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())