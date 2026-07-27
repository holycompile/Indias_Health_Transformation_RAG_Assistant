import streamlit as st
import os

# setting page config
st.set_page_config(page_title="RAG Q&A Assistant", layout="centered")

# Hide Streamlit brandings and menu options
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            [data-testid="stHeader"] {display: none;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# import db and ask function from other file
from history_retrieval_convo_skill import db, ask_question

# init session state variables for ui mode and history
if "ui_mode" not in st.session_state:
    st.session_state.ui_mode = "basic" # basic or advanced

if "history_list" not in st.session_state:
    st.session_state.history_list = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

# sync chat history with the backend convo file
import history_retrieval_convo_skill
history_retrieval_convo_skill.chat_history = st.session_state.history_list

# Title as instrcuted
st.title("India's Health Transformation RAG Assistant")

# simple toggle button at the top to switch between modes
if st.session_state.ui_mode == "basic":
    if st.button("Switch to Advanced Conversational Mode"):
        st.session_state.ui_mode = "advanced"
        st.session_state.history_list = []
        history_retrieval_convo_skill.chat_history = []
        st.rerun()
else:
    if st.button("Switch to Basic Single-Turn Mode"):
        st.session_state.ui_mode = "basic"
        st.session_state.history_list = []
        st.session_state.last_answer = None
        st.session_state.last_docs = []
        history_retrieval_convo_skill.chat_history = []
        st.rerun()

st.write("---")

# ----------------- BASIC SINGLE-TURN MODE -----------------
if st.session_state.ui_mode == "basic":
    
    # Text Box
    user_query = st.text_input("Ask your question...", key="basic_query_input")
    
    # Button
    if st.button("Ask", key="basic_ask_button"):
        if not user_query.strip():
            st.warning("Please enter a question first!")
        else:
            # clear previous history to make it strict single-turn
            st.session_state.history_list = []
            history_retrieval_convo_skill.chat_history = []
            
            with st.spinner("Searching and generating answer..."):
                # fetch relevant chunks from chroma db to show sources
                retriever = db.as_retriever(search_kwargs={"k": 5})
                retrieved_docs = retriever.invoke(user_query)
                
                # generate final answer
                final_answer = ask_question(user_query)
                
                # store in session state so it doesn't clear on page reload
                st.session_state.last_answer = final_answer
                st.session_state.last_docs = retrieved_docs
                
    # Display Answer
    if st.session_state.last_answer is not None:
        st.write("### Answer")
        st.write(st.session_state.last_answer)
        
        # Display Retrieved Context (Expandable)
        st.write("### Retrieved Context (Expandable)")
        for idx, doc in enumerate(st.session_state.last_docs, 1):
            page_num = doc.metadata.get("page", 0) + 1
            with st.expander(f"Chunk {idx} (PDF Page {page_num})"):
                st.write(doc.page_content)

# ----------------- ADVANCED CONVERSATIONAL MODE -----------------
else:
    st.write("### Chat Interface")
    
    if st.button("Clear Conversation"):
        st.session_state.history_list = []
        history_retrieval_convo_skill.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()
        
    # show chat messages from history using custom icons
    from langchain_core.messages import HumanMessage, AIMessage
    
    for msg in st.session_state.history_list:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.write(f"**What is asked:** {msg.content}")
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg.content)
                
    # chat input for user
    chat_query = st.chat_input("Ask a follow-up question...")
    
    if chat_query:
        # show user message immediately
        with st.chat_message("user", avatar="👤"):
            st.write(f"**What is asked:** {chat_query}")
            
        with st.spinner("Generating response..."):
            # run query with history context
            final_answer = ask_question(chat_query)
            
        # show assistant response
        with st.chat_message("assistant", avatar="🤖"):
            st.write(final_answer)
            
        st.rerun()
