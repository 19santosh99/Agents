import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import groq
import openai
import anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelType(Enum):
    GPT_3_5 = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    CLAUDE = "claude-2"
    LLAMA = "llama-2"

@dataclass
class Message:
    role: str
    content: str

class LLMRouter:
    def __init__(self):
        # Initialize clients
        self.groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize Llama model if path is provided
        llama_path = os.getenv("LLAMA_MODEL_PATH")
        if llama_path:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_path)
            self.llama_model = AutoModelForCausalLM.from_pretrained(llama_path)
        else:
            self.llama_tokenizer = None
            self.llama_model = None

    def get_model_recommendation(self, recent_messages: List[Message]) -> ModelType:
        """
        Use Llama through Groq to determine the best model based on recent messages.
        """
        system_prompt = """
        You are a model selection expert. Based on the recent conversation messages,
        recommend the most suitable model from these options:
        
        1. GPT-3.5-Turbo ($0.002/1k tokens): Best for general, simple tasks
        2. GPT-4 ($0.03/1k tokens): Best for complex reasoning, coding
        3. Claude-2 ($0.01/1k tokens): Best for long context, analysis
        4. Llama-2 (Free/self-hosted): Best for non-sensitive, batch tasks
        
        Analyze the nature and complexity of the conversation to determine the most suitable model.
        Respond with ONLY the model name as: GPT-3.5-TURBO, GPT-4, CLAUDE-2, or LLAMA-2
        """
        
        # Format recent messages for context
        messages_context = "\n".join([
            f"{msg.role}: {msg.content}" for msg in recent_messages
        ])
        
        user_prompt = f"""
        Recent conversation:
        {messages_context}
        
        Based on this conversation context, which model would be most appropriate?
        just give the model name as: GPT-3.5-TURBO, GPT-4, CLAUDE-2, or LLAMA-2 only no other text is required.
        """
        
        response = self.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=10
        )
        
        model_name = response.choices[0].message.content.strip()
        print(f"Model recommendation: {model_name}")
        return {
            "GPT-3.5-TURBO": ModelType.GPT_3_5,
            "GPT-4": ModelType.GPT_4,
            "CLAUDE-2": ModelType.CLAUDE,
            "LLAMA-2": ModelType.LLAMA
        }.get(model_name, ModelType.GPT_3_5)

    def get_response(self, messages: List[Message]) -> Tuple[str, ModelType]:
        """
        Get a response using the most appropriate model based on recent conversation context.
        
        Args:
            messages: List of all conversation messages
        
        Returns:
            Tuple of (response text, selected model type)
        """
        # Get last 3 messages for model selection
        recent_messages = messages[-3:] if len(messages) >= 3 else messages
        
        # Get model recommendation based on recent context
        selected_model = self.get_model_recommendation(recent_messages)
        
        # Format messages for the selected model
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        # Get response from selected model
        if selected_model in [ModelType.GPT_3_5, ModelType.GPT_4]:
            response = self.openai_client.chat.completions.create(
                model=selected_model.value,
                messages=formatted_messages
            )
            return response.choices[0].message.content, selected_model
            
        elif selected_model == ModelType.CLAUDE:
            response = self.anthropic_client.messages.create(
                model=selected_model.value,
                messages=formatted_messages
            )
            return response.content[0].text, selected_model
            
        elif selected_model == ModelType.LLAMA:
            if self.llama_model is None:
                # Fallback to GPT-3.5 if Llama is not available
                response = self.openai_client.chat.completions.create(
                    model=ModelType.GPT_3_5.value,
                    messages=formatted_messages
                )
                return response.choices[0].message.content, ModelType.GPT_3_5
            else:
                # Use local Llama model
                input_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
                inputs = self.llama_tokenizer(input_text, return_tensors="pt")
                outputs = self.llama_model.generate(**inputs, max_length=1000)
                response = self.llama_tokenizer.decode(outputs[0])
                return response, selected_model