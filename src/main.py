import streamlit as st
import os
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Custom CSS for chat interface
def local_css():
    st.markdown("""
        <style>
        .css-1s3ycrg {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .chat-container {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #2e2e2e;
            color: white;
            text-align: right;
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .assistant-message {
            background-color: #404040;
            color: white;
            text-align: left;
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .stTextInput input {
            border-radius: 0.5rem;
            border: 1px solid #4a4a4a;
            background-color: #2e2e2e;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

class BankingChatbot:
    def __init__(self):
        # Get API key from environment variable
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        self.groq_chat = ChatGroq(
            groq_api_key=self.api_key,
            model_name='llama3-8b-8192'
        )
        
        self.system_prompt = """You are an AI Banking Assistant. Your role is to:
        1. Help users with banking-related queries
        2. Provide information about account services
        3. Assist with transaction inquiries
        4. Guide users through banking processes
        
        Maintain a professional yet friendly tone. Be clear and concise in your responses.
        For security, never share sensitive information or account details.
        If you're unsure about something, recommend speaking with a human banker."""
        
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])
        
        self.conversation = LLMChain(
            llm=self.groq_chat,
            prompt=self.prompt,
            verbose=False,
            memory=self.memory
        )
    
    def get_response(self, user_input):
        try:
            response = self.conversation.predict(human_input=user_input)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = BankingChatbot()
        except ValueError as e:
            st.error(str(e))
            st.stop()

def main():
    st.set_page_config(page_title="AI Banking Assistant", layout="wide")
    local_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Chat interface
    st.title("💬 AI Banking Assistant")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">You: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">Assistant: {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    with st.container():
        if prompt := st.chat_input("Ask me anything about banking:"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get chatbot response
            response = st.session_state.chatbot.get_response(prompt)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update chat display
            st.rerun()

if __name__ == "__main__":
    main()