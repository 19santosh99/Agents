# Cost-Efficient LLM Router

This project implements an intelligent router system that uses Llama (via Groq) to make informed decisions about which LLM to use for a given conversation. The system analyzes the last 3 messages of the conversation to select the most appropriate model, then processes the entire conversation history with the selected model.

## How It Works

1. When processing a conversation:
   - The router extracts the last 3 messages for context
   - Sends these messages to Llama (through Groq's API) for model selection
   - Llama analyzes the conversation context and recommends the most suitable model
   - The entire conversation history is then processed using the selected model

2. Supported Models:
   - GPT-3.5 Turbo: For general, simple conversations
   - GPT-4: For complex reasoning and coding tasks
   - Claude: For long context and analysis
   - Llama 2: For non-sensitive, batch processing tasks

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
LLAMA_MODEL_PATH=path_to_llama_model  # Optional, for local Llama model
```

## Usage

```python
import asyncio
from llm_router import LLMRouter, Message

async def main():
    # Initialize the router
    router = LLMRouter()

    # Create conversation history
    conversation = [
        Message(role="user", content="Hi, can you help me understand machine learning?"),
        Message(role="assistant", content="Of course! What would you like to know?"),
        Message(role="user", content="Can you explain neural networks?")
    ]
    
    # Get response (uses last 3 messages for model selection)
    response, selected_model = await router.get_response(conversation)
    print(f"Selected model: {selected_model}")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Model Selection Process

1. **Context Analysis**:
   - Takes the last 3 messages from the conversation
   - Sends them to Llama through Groq's API
   - Llama analyzes the conversation complexity and requirements

2. **Model Selection**:
   - Based on conversation context, selects the most appropriate model
   - Considers factors like complexity, technical content, and context length

3. **Response Generation**:
   - Passes the entire conversation history to the selected model
   - Returns both the model's response and the selected model type

## Running the Example

```bash
# Set up environment variables
export GROQ_API_KEY=your_groq_api_key
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key

# Run the example
python example.py
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License