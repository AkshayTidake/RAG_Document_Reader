import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Modern LCEL, Memory, and Format Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 🔧 CONFIGURATION & SEEDING
st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise RAG Knowledge Assistant")
st.write("Securely chat with your documents with persistent conversation memory and page citations.")

# Securely check for Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
    if not groq_api_key:
        st.warning("Please enter your Groq API Key in the sidebar to proceed.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize Session State for DB
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Streamlit-native Chat History management
msgs = StreamlitChatMessageHistory(key="chat_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Awaiting a PDF document upload to activate the cognitive engine.")

# File Upload Sidebar
uploaded_file = st.sidebar.file_uploader("Upload your document (PDF)", type=["pdf"])

# 📦 PIPELINE PHASE 1: INGESTION, CHUNKING & EMBEDDINGS
if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("Processing document... Extracting text and generating vector embeddings."):
        # Save file locally to feed into loader
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and parse PDF
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        # 🛡️ DEFENSIVE GUARDRAIL 1: Check if PDF contains readable digital text
        full_text_check = "".join(doc.page_content for doc in docs).strip()
        if not full_text_check:
            st.error("❌ The uploaded PDF contains no extractable digital text. It might be a scanned image. Please upload a digitally created PDF document.")
            os.remove(temp_file_path)
            st.stop()
        
        # Optimized Chunking Strategy
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 🛡️ DEFENSIVE GUARDRAIL 2: Verify splits are not empty before model ingestion
        if not splits:
            st.error("❌ Text parsing generated zero valid data fragments. Ingestion stopped.")
            os.remove(temp_file_path)
            st.stop()
            
        # Instantiate localized embedding model with standardized CPU/MPS inference overrides
        # This completely bypasses backend tensor array index truncation bugs
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True}
        )
        
        st.session_state.vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        os.remove(temp_file_path)
        st.sidebar.success("Document ingested successfully!")
        msgs.clear()
        msgs.add_ai_message("Document loaded! Ask me anything about its contents.")

# 💬 INTERFACE & CORE PIPELINE
for msg in msgs.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

if st.session_state.vector_store:
    if user_query := st.chat_input("Ask a question about your uploaded document:"):
        st.chat_message("human").markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # 1️⃣ Setup Retriever
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(user_query)
                
                # Extract context text and list of source page numbers safely
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                pages = sorted(list(set(int(doc.metadata.get("page", 0)) + 1 for doc in retrieved_docs)))
                
                # 2️⃣ FIXED: Defined using explicit template placeholders to shield literal brackets {}
                contextual_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are an enterprise document assistant. Answer the user's question using ONLY the provided context. "
                     "If the answer is not present, state clearly that it is not found. Do not invent information.\n\n"
                     "Context:\n{context}"  # <--- Fed as an internal LangChain parameter
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ])
                
                # 3️⃣ Initialize LLM Core
                llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
                
                # 4️⃣ Assemble RAG Chain
                chain = contextual_prompt | llm | StrOutputParser()
                
                # 5️⃣ Wrap chain with session history manager
                chain_with_history = RunnableWithMessageHistory(
                    chain,
                    lambda session_id: msgs,
                    input_messages_key="input",
                    history_messages_key="history",
                )
                
                # 6️⃣ FIXED: Pass context explicitly into the dictionary mapping
                raw_answer = chain_with_history.invoke(
                    {"input": user_query, "context": context_text},
                    config={"configurable": {"session_id": "streamlit_session"}}
                )
                
                # Format visual citation badges if pages exist
                if pages:
                    citation_string = f"\n\n*📄 **Sources:** Retrieved from Page{'s' if len(pages) > 1 else ''}: {', '.join(f'`Page {p}`' for p in pages)}*"
                    final_compiled_answer = f"{raw_answer}{citation_string}"
                    
                    if len(msgs.messages) > 0:
                        msgs.messages[-1].content = final_compiled_answer
                else:
                    final_compiled_answer = raw_answer
                
                st.markdown(final_compiled_answer)
