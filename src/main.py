import streamlit as st
import os
from datetime import datetime
import uuid
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

class BankingMode:
    CHAT = "chat"
    ACCOUNT = "account"
    TRANSACTION = "transaction"

def local_css():
    st.markdown("""
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            color: white;
        }
        .user-message {
            background-color: #1a1a1a;
            text-align: right;
        }
        .assistant-message {
            background-color: #2e2e2e;
            text-align: left;
        }
        .feature-button {
            padding: 1rem;
            margin: 0.5rem;
            border-radius: 0.5rem;
            background-color: #2e2e2e;
            color: white;
            text-align: center;
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)

class BankingAssistant:
    def __init__(self):
        if 'mode' not in st.session_state:
            st.session_state.mode = None
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'account_data' not in st.session_state:
            st.session_state.account_data = {}
        if 'chatbot' not in st.session_state:
            try:
                st.session_state.chatbot = self.initialize_chatbot()
            except Exception as e:
                st.error(f"Error initializing chatbot: {str(e)}")

    def initialize_chatbot(self):
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,
            model_name='llama3-8b-8192'
        )
        
        system_prompt = """You are a Banking Assistant, a specialized AI chatbot designed to assist users with specific banking features. You can only help with and discuss the following services:

        1. Financial Transactions:
        - Deposits and withdrawals
        - Bank account creation and management
        - Transaction status checking

        2. User Management:
        - Agent creation and management

        3. Complaint Management:
        - Creating complaints
        - Tracking complaint status
        - Deleting complaints

        Important Guidelines:
        - Only provide information about the features listed above
        - If asked about anything outside these services, politely explain that you can only assist with the listed features
        - Always maintain a helpful and professional tone
        - Suggest appropriate services based on user queries
        - Guide users through the available options
        - Never share sensitive account information
        - Always clarify if you need more information to assist properly

        Example responses:
        - For account queries: "I can help you create a new bank account or manage your existing one. Would you like to proceed with either of these?"
        - For transactions: "I can assist you with deposits, withdrawals, or checking transaction status. Which service do you need?"
        - For complaints: "I can help you create a new complaint, track an existing one, or delete a complaint. What would you like to do?"

        Remember: You are specifically designed to handle these banking services and should not provide information about other banking features or services."""
        
        memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])
        
        return LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=False,
            memory=memory
        )

    def add_message(self, content: str, is_user: bool = False):
        st.session_state.messages.append({
            "content": content,
            "is_user": is_user,
            "timestamp": datetime.now().strftime("%H:%M")
        })

    def show_feature_selection(self):
        st.write("Please select what you'd like to do:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💬 Chat with Assistant", help="Ask questions about banking"):
                st.session_state.mode = BankingMode.CHAT
                self.add_message("How can I help you with your banking questions?", is_user=False)
                st.rerun()
                
        with col2:
            if st.button("🏦 Create Account", help="Create a new bank account"):
                st.session_state.mode = BankingMode.ACCOUNT
                self.add_message("Let's create your account! What's your full name?", is_user=False)
                st.rerun()
                
        with col3:
            if st.button("💳 Make Transaction", help="Make a deposit or withdrawal"):
                st.session_state.mode = BankingMode.TRANSACTION
                self.add_message("Would you like to make a deposit or withdrawal?", is_user=False)
                st.rerun()

    def handle_chat(self):
        user_input = st.chat_input("Ask me anything about banking...")
        
        if user_input:
            self.add_message(user_input, is_user=True)
            try:
                response = st.session_state.chatbot.predict(human_input=user_input)
                self.add_message(response, is_user=False)
                st.rerun()
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")

        if st.button("⬅️ Back to Main Menu"):
            st.session_state.mode = None
            st.rerun()

    def handle_account_creation(self):
        if 'account_step' not in st.session_state:
            st.session_state.account_step = 'name'
            st.session_state.account_data = {}

        if st.session_state.account_step == 'name':
            name = st.text_input("Full Name")
            if st.button("Continue"):
                if name:
                    st.session_state.account_data['name'] = name
                    st.session_state.account_step = 'email'
                    self.add_message(f"Thanks {name}! Please enter your email:", is_user=False)
                    st.rerun()

        elif st.session_state.account_step == 'email':
            email = st.text_input("Email Address")
            if st.button("Continue"):
                if email and '@' in email:
                    st.session_state.account_data['email'] = email
                    st.session_state.account_step = 'bank_details'
                    self.add_message("Please enter your banking details:", is_user=False)
                    st.rerun()

        elif st.session_state.account_step == 'bank_details':
            col1, col2 = st.columns(2)
            with col1:
                bank_id = st.text_input("Bank ID")
                daily_limit = st.number_input("Daily Transaction Limit", min_value=0.0, step=1000.0)
                ifsc_code = st.text_input("IFSC Code")

            with col2:
                upi_id = st.text_input("UPI ID")
                login_id = st.text_input("Login ID")
                agent_id = st.text_input("Agent ID")

            if st.button("Continue"):
                if bank_id and daily_limit > 0 and ifsc_code:
                    st.session_state.account_data.update({
                        'bank_id': bank_id,
                        'daily_limit': daily_limit,
                        'ifsc_code': ifsc_code,
                        'upi_id': upi_id,
                        'login_id': login_id,
                        'agent_id': agent_id
                    })
                    st.session_state.account_step = 'security'
                    self.add_message("Please set up your security credentials:", is_user=False)
                    st.rerun()

        elif st.session_state.account_step == 'security':
            col1, col2 = st.columns(2)
            with col1:
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")

            with col2:
                trxn_password = st.text_input("Transaction Password", type="password")
                otp_access = st.checkbox("Enable OTP Access")

            if st.button("Complete Account Creation"):
                if username and password and trxn_password:
                    account_number = str(uuid.uuid4())[:8]
                    qrcode_url = f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={account_number}"
                    
                    st.session_state.account_data.update({
                        'account_number': account_number,
                        'username': username,
                        'password': password,
                        'trxn_password': trxn_password,
                        'otp_access': otp_access,
                        'qrcode_url': qrcode_url
                    })

                    success_message = f"""Account created successfully! 🎉
                    Account Number: {account_number}
                    Name: {st.session_state.account_data['name']}
                    Email: {st.session_state.account_data['email']}
                    Bank ID: {st.session_state.account_data['bank_id']}
                    Daily Limit: ${st.session_state.account_data['daily_limit']:,.2f}
                    IFSC Code: {st.session_state.account_data['ifsc_code']}
                    UPI ID: {st.session_state.account_data['upi_id']}
                    Username: {username}"""

                    self.add_message(success_message, is_user=False)
                    st.session_state.account_step = None
                    st.session_state.account_data = {}
                    st.session_state.mode = None
                    st.rerun()

        if st.button("⬅️ Back to Main Menu"):
            st.session_state.mode = None
            st.session_state.account_step = None
            st.session_state.account_data = {}
            st.rerun()

    def handle_transaction(self):
        if 'transaction_step' not in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Deposit"):
                    st.session_state.transaction_step = 'amount'
                    st.session_state.transaction_type = 'deposit'
                    self.add_message("How much would you like to deposit?", is_user=False)
                    st.rerun()
            with col2:
                if st.button("Withdraw"):
                    st.session_state.transaction_step = 'amount'
                    st.session_state.transaction_type = 'withdraw'
                    self.add_message("How much would you like to withdraw?", is_user=False)
                    st.rerun()
        
        elif st.session_state.transaction_step == 'amount':
            amount = st.number_input("Amount", min_value=0.0, step=100.0)
            if st.button("Confirm Transaction"):
                success_message = f"""Transaction successful! 🎉
                Type: {st.session_state.transaction_type.title()}
                Amount: ${amount:,.2f}"""
                self.add_message(success_message, is_user=False)
                st.session_state.transaction_step = None
                st.session_state.mode = None
                st.rerun()

        if st.button("⬅️ Back to Main Menu"):
            st.session_state.mode = None
            st.session_state.transaction_step = None
            st.rerun()

def main():
    st.set_page_config(page_title="Banking Assistant", layout="wide")
    local_css()

    st.title("💬 Banking Assistant")

    assistant = BankingAssistant()

    # Display chat history
    for message in st.session_state.messages:
        message_class = "user-message" if message["is_user"] else "assistant-message"
        st.markdown(
            f"""<div class="chat-message {message_class}">
                {'You' if message["is_user"] else 'Assistant'}: {message["content"]}
            </div>""",
            unsafe_allow_html=True
        )

    # Handle different modes
    if st.session_state.mode is None:
        assistant.show_feature_selection()
    elif st.session_state.mode == BankingMode.CHAT:
        assistant.handle_chat()
    elif st.session_state.mode == BankingMode.ACCOUNT:
        assistant.handle_account_creation()
    elif st.session_state.mode == BankingMode.TRANSACTION:
        assistant.handle_transaction()

if __name__ == "__main__":
    main()