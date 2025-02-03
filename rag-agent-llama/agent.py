"""
RAG-enabled chatbot using LangChain and Streamlit.
Allows users to query documents using a LLaMA model and vector similarity search.
"""

from datetime import datetime
from typing import List, Optional

import json
import os
import logging
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from model_router import ModelRouter, ModelType
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Application configuration settings."""
    model_name: str = os.getenv('LLM_MODEL', 'meta-llama/Meta-Llama-3.1-405B-Instruct')
    docs_directory: str = os.getenv('DIRECTORY', 'meetingNotes')
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_new_tokens: int = 1024
    embedding_model: str = "all-MiniLM-L6-v2"
    k_similar_docs: int = 5

class DocumentProcessor:
    """Handles document loading and processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

    def load_and_split_documents(self) -> List:
        """
        Load documents from directory and split into chunks.
        
        Returns:
            List: List of document chunks
        """
        try:
            loader = DirectoryLoader(self.config.docs_directory)
            documents = loader.load()
            docs = self.text_splitter.split_documents(documents)
            logger.info(f"Loaded and split {len(documents)} documents into {len(docs)} chunks")
            return docs
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

class VectorStore:
    """Manages vector storage and similarity search."""
    
    def __init__(self, config: Config):
        self.config = config
        self.db = self._initialize_chroma()

    @st.cache_resource
    def _initialize_chroma(self) -> Chroma:
        """
        Initialize Chroma vector store with document embeddings.
        
        Returns:
            Chroma: Initialized Chroma instance
        """
        try:
            doc_processor = DocumentProcessor(self.config)
            docs = doc_processor.load_and_split_documents()
            
            embedding_function = SentenceTransformerEmbeddings(
                model_name=self.config.embedding_model
            )
            
            return Chroma.from_documents(docs, embedding_function)
        except Exception as e:
            logger.error(f"Error initializing Chroma: {str(e)}")
            raise

    def query_documents(self, question: str) -> List[str]:
        """
        Search for relevant document chunks based on query.
        
        Args:
            question: User's question
            
        Returns:
            List[str]: Formatted list of relevant document chunks with sources
        """
        try:
            similar_docs = self.db.similarity_search(
                question, 
                k=self.config.k_similar_docs
            )
            
            return [
                f"Source: {doc.metadata.get('source', 'NA')}\n"
                f"Content: {doc.page_content}"
                for doc in similar_docs
            ]
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            raise

class ChatBot:
    """Manages chat interactions and LLM responses."""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = VectorStore(config)
        self.llm = self._initialize_llm()

    @st.cache_resource
    def _initialize_llm(self) -> HuggingFaceEndpoint:
        """
        Initialize the language model.
        
        Returns:
            HuggingFaceEndpoint: Initialized LLM instance
        """
        try:
            return HuggingFaceEndpoint(
                repo_id=self.config.model_name,
                task="text-generation",
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

    def get_ai_response(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Generate AI response based on chat history and relevant documents.
        
        Args:
            messages: List of chat messages
            
        Returns:
            AIMessage: AI's response
        """
        try:
            user_prompt = messages[-1].content
            retrieved_context = self.vector_store.query_documents(user_prompt)
            
            formatted_prompt = (
                f"Context for answering the question:\n{retrieved_context}\n"
                f"Question/user input:\n{user_prompt}"
            )
            
            doc_chatbot = ChatHuggingFace(llm=self.llm)
            return doc_chatbot.invoke(
                messages[:-1] + [HumanMessage(content=formatted_prompt)]
            )
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            raise

class StreamlitUI:
    """Manages Streamlit user interface."""
    
    def __init__(self, config: Config):
        self.config = config
        self.chatbot = ChatBot(config)

    def initialize_session(self) -> None:
        """Initialize chat session state."""
        if "messages" not in st.session_state:
            system_message = (
                "You are a personal assistant who answers questions based on "
                "the context provided if the provided context can answer the "
                "question. You only provide the answer to the question/user "
                f"input and nothing else. The current date is: {datetime.now().date()}"
            )
            st.session_state.messages = [SystemMessage(content=system_message)]

    def display_chat_history(self) -> None:
        """Display chat message history."""
        for message in st.session_state.messages:
            message_json = json.loads(message.json())
            message_type = message_json["type"]
            
            if message_type in ["human", "ai", "system"]:
                with st.chat_message(message_type):
                    st.markdown(message_json["content"])

    def handle_user_input(self) -> None:
        """Process user input and generate response."""
        if prompt := st.chat_input("What questions do you have?"):
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append(HumanMessage(content=prompt))
            
            # Generate and display AI response
            with st.chat_message("assistant"):
                try:
                    ai_response = self.chatbot.get_ai_response(
                        st.session_state.messages
                    )
                    st.markdown(ai_response.content)
                    st.session_state.messages.append(ai_response)
                except Exception as e:
                    error_msg = "An error occurred while generating the response."
                    logger.error(f"{error_msg}: {str(e)}")
                    st.error(error_msg)

def main() -> None:
    """Main application entry point."""
    try:
        st.title("Chat with Local Documents")
        
        config = Config()
        ui = StreamlitUI(config)
        
        ui.initialize_session()
        ui.display_chat_history()
        ui.handle_user_input()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()