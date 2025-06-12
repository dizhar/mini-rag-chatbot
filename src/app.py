"""Streamlit app for the Mini RAG Chatbot."""

import streamlit as st
import os
from utils import process_all_pdfs
from rag import RAGChatbot


# Page config
st.set_page_config(
    page_title="Policy RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 MarkAIting Pro Policy Chatbot")
st.markdown("Ask questions about our company policies and get answers with source citations!")

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
    st.session_state.initialized = False

# Sidebar for initialization
with st.sidebar:
    st.header("📋 Documents")
    
    if not st.session_state.initialized:
        with st.spinner("Loading and processing documents..."):
            try:
                # Define PDF paths
                pdf_paths = [
                    "data/security_policy.pdf",
                    "data/expense_policy.pdf", 
                    "data/remote_work_policy.pdf"
                ]
                
                # Check if files exist
                missing_files = [path for path in pdf_paths if not os.path.exists(path)]
                if missing_files:
                    st.error(f"Missing PDF files: {missing_files}")
                    st.info("Please ensure the PDF files are in the data/ folder")
                    st.stop()
                
                # Process PDFs
                chunks = process_all_pdfs(pdf_paths)
                
                # Initialize chatbot
                chatbot = RAGChatbot()
                chatbot.load_documents(chunks)
                
                st.session_state.chatbot = chatbot
                st.session_state.initialized = True
                
                st.success("✅ Documents loaded successfully!")
                
            except Exception as e:
                st.error(f"Error initializing chatbot: {e}")
                st.info("Make sure your OPENAI_API_KEY is set in .env file")
                st.stop()
    else:
        st.success("✅ Ready to answer questions!")
    
    # Show document info
    if st.session_state.initialized:
        st.markdown("**Loaded documents:**")
        st.markdown("• Security Policy")
        st.markdown("• Expense & Reimbursement Policy") 
        st.markdown("• Remote Work Policy")

# Main chat interface
if st.session_state.initialized:
    # Example questions
    st.markdown("### 💡 Try asking:")
    example_questions = [
        "What's the meal allowance for international travel?",
        "Do I need MFA for remote work?", 
        "What happens if I violate the security policy?",
        "How much can I expense for lodging?",
        "What are the core hours for remote work?"
    ]
    
    cols = st.columns(len(example_questions))
    for i, question in enumerate(example_questions):
        if cols[i].button(f"'{question[:30]}...'", key=f"example_{i}"):
            st.session_state.current_question = question
    
    # Question input
    question = st.text_input(
        "Ask a question about the policies:",
        value=st.session_state.get("current_question", ""),
        placeholder="e.g., What's the expense limit for meals?"
    )
    
    if question:
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Get answer from chatbot
                result = st.session_state.chatbot.ask(question)
                
                # Display answer
                st.markdown("### 🤖 Answer:")
                st.markdown(result["answer"])
                
                # Display sources
                if result["sources"]:
                    st.markdown("### 📑 Sources:")
                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(f"📄 {source['source']} (Page {source['page']}) - Relevance: {source['relevance_score']:.3f}"):
                            st.markdown(f"**Text snippet:**")
                            st.markdown(f"*{source['text']}*")
                
            except Exception as e:
                st.error(f"Error processing question: {e}")
                st.info("Please check your API key and try again")

else:
    st.info("👆 Initializing the chatbot in the sidebar...")