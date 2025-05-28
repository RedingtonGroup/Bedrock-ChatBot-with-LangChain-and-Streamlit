import random
from io import BytesIO
from typing import List, Tuple, Union, Dict, Any
import re
import os # Added for path operations

import streamlit as st
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

# Imports for document loading and splitting (used for indexing)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader # Add loaders as needed
from PIL import Image, UnidentifiedImageError # Still needed for image handling if you re-introduce it. For now, commented out relevant parts.
import pdfplumber # Still needed for PDF processing if you re-introduce it. For now, commented out relevant parts.
import docx # For docx processing

from dotenv import load_dotenv

from config import config
from models import ChatModel
from role_prompt import role_prompt
# REMOVED: from bedrock_embedder import search_index # search_index will now be defined directly or integrated.

# Load the env variables
load_dotenv()

INIT_MESSAGE = {
    "role": "assistant",
    "content": "Hi there! I'm the Redington AI Bot, built to support you. How may I assist you today?",
    "llm_content": "Hi there! I'm the Redington AI Bot, built to support you. How may I assist you today?",
    "user_prompt": "",  # For storing clean user prompts
    "has_context": False  # Flag to indicate if message has RAG/search context
}

# --- Indexing Functions (Moved from bedrock_embedder.py and integrated) ---

@st.cache_resource
def load_bedrock_embeddings():
    """
    Load and cache Bedrock Embeddings.
    """
    return BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

def search_index(query: str, index_path: str) -> List[Any]:
    """
    Performs a similarity search on the FAISS index.
    """
    try:
        embeddings = load_bedrock_embeddings()
        # Set allow_dangerous_deserialization to True
        allow_dangerous = True
        db = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=allow_dangerous
        )
        docs = db.similarity_search(query)
        return docs
    except Exception as e:
        st.error(f"Error loading or searching FAISS index: {e}")
        return [f"Error: {e}"] # Return an error message to be handled by caller


def perform_indexing(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], index_path: str = "faiss_index") -> bool:
    """
    Processes uploaded files, creates text chunks, embeds them, and updates the FAISS index.
    """
    if not uploaded_files:
        st.warning("No files uploaded for indexing.")
        return False

    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) # Defined here

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_content = BytesIO(uploaded_file.getvalue())

        try:
            if file_extension == ".pdf":
                with pdfplumber.open(file_content) as pdf:
                    text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
                docs = text_splitter.create_documents([text])
                st.info(f"Processed PDF: {uploaded_file.name} - {len(docs)} chunks")
            elif file_extension == ".txt":
                text = file_content.getvalue().decode('utf-8')
                docs = text_splitter.create_documents([text])
                st.info(f"Processed TXT: {uploaded_file.name} - {len(docs)} chunks")
            elif file_extension == ".csv":
                # CSVLoader requires a file path, so we'll save it temporarily
                temp_file_path = f"/tmp/{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                loader = CSVLoader(file_path=temp_file_path)
                docs = loader.load_and_split(text_splitter)
                os.remove(temp_file_path) # Clean up temp file
                st.info(f"Processed CSV: {uploaded_file.name} - {len(docs)} chunks")
            elif file_extension == ".docx":
                # Docx2txtLoader requires a file path, so we'll save it temporarily
                temp_file_path = f"/tmp/{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                loader = Docx2txtLoader(file_path=temp_file_path)
                docs = loader.load_and_split(text_splitter)
                os.remove(temp_file_path) # Clean up temp file
                st.info(f"Processed DOCX: {uploaded_file.name} - {len(docs)} chunks")
            else:
                st.warning(f"Unsupported file type for indexing: {uploaded_file.name}")
                continue
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            continue

    if not all_docs:
        st.error("No valid documents could be processed for indexing.")
        return False

    try:
        embeddings = load_bedrock_embeddings()
        # Check if the index directory exists and contains FAISS files
        if os.path.exists(index_path) and any(f.endswith(".faiss") for f in os.listdir(index_path)):
            # Load existing index and add new documents
            allow_dangerous = True
            st.info(f"Loading existing FAISS index from {index_path}...")
            existing_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=allow_dangerous)
            st.info("Adding new documents to existing index...")
            existing_db.add_documents(all_docs)
            db = existing_db
        else:
            # Create a new index
            st.info("Creating a new FAISS index...")
            db = FAISS.from_documents(all_docs, embeddings)

        db.save_local(index_path)
        st.success(f"Indexing complete! Added {len(all_docs)} chunks to index at '{index_path}'.")
        return True
    except Exception as e:
        st.error(f"Error during embedding or saving FAISS index: {e}")
        return False

# --- End Indexing Functions ---

def set_page_config() -> None:
    """
    Set the Streamlit page configuration.
    """
    st.set_page_config(page_title="🤖 Chat with Redington AI Bot", layout="wide")
    st.title("🤖 Chat with Redington AI Bot")


def render_sidebar() -> Tuple[Dict, str, str, bool]:
    """
    Render the sidebar UI and return the inference parameters.
    """
    with st.sidebar:
        model_name_select = st.selectbox(
            "Model",
            list(config["models"].keys()),
            key=f"{st.session_state['widget_key']}_Model_Id",
        )

        role_select = st.selectbox(
            "Role",
            ["Custom"] + list(role_prompt.keys()),
            key=f"{st.session_state['widget_key']}_role_Id",
        )
        # Set the initial value of the text area based on the selected role
        role_prompt_text = (
            "" if role_select == "Custom" else role_prompt.get(role_select, "")
        )
        st.session_state["model_name"] = model_name_select

        model_config = config["models"][model_name_select]

        system_prompt = st.text_area(
            "System Prompt",
            value=role_prompt_text,
            key=f"{st.session_state['widget_key']}_System_Prompt",
        )

        # File uploader for indexing (now in sidebar for combined file)
        st.subheader("Document Indexing")
        if "indexing_uploader_key" not in st.session_state:
            st.session_state["indexing_uploader_key"] = 0

        st.session_state.indexing_uploaded_files = st.file_uploader(
            "Upload files for RAG Indexing",
            type=["pdf", "txt", "csv", "docx"], # Types suitable for indexing
            accept_multiple_files=True,
            key=f"indexing_file_uploader_{st.session_state['indexing_uploader_key']}",
            help="Upload documents (PDF, TXT, CSV, DOCX) to build or update the RAG index."
        )

        if st.session_state.indexing_uploaded_files:
            if st.button("Build/Update Index", key="build_index_button"):
                with st.spinner("Processing files and updating index..."):
                    if perform_indexing(st.session_state.indexing_uploaded_files, index_path="faiss_index"):
                        st.success("FAISS Index updated successfully!")
                    else:
                        st.error("FAISS Index update failed.")
                # Clear the uploader after processing to prevent re-indexing on refresh
                st.session_state["indexing_uploader_key"] = random.randint(1, 1000000)
                st.rerun() # Rerun to clear the uploader widget

        web_local = st.selectbox(
            "Options",
            ("RAG", "Local"), # Removed "Web" if SerpAPI is not configured/needed
            key=f"{st.session_state['widget_key']}_Options",
        )

        # Add debug mode toggle
        debug_mode = st.checkbox(
            "Debug Mode (Show RAG Context)",
            value=False,
            key=f"{st.session_state['widget_key']}_Debug_Mode",
            help="Enable to see RAG search results and backend processing"
        )

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                top_p = st.slider(
                    "Top-P",
                    min_value=0.0,
                    max_value=1.0,
                    value=model_config.get("top_p", 1.0),
                    step=0.01,
                    key=f"{st.session_state['widget_key']}_Top-P",
                )
            with col2:
                top_k = st.slider(
                    "Top-K",
                    min_value=1,
                    max_value=model_config.get("max_top_k", 500),
                    value=model_config.get("top_k", 500),
                    step=5,
                    key=f"{st.session_state['widget_key']}_Top-K",
                )
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=model_config.get("temperature", 1.0),
                    key=f"{st.session_state['widget_key']}_Temperature",
                )
            with col2:
                max_tokens = st.slider(
                    "Max Token",
                    min_value=0,
                    max_value=4096,
                    value=model_config.get("max_tokens", 4096),
                    step=8,
                    key=f"{st.session_state['widget_key']}_Max_Token",
                )

    model_kwargs = {
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    return model_kwargs, system_prompt, web_local, debug_mode


def extract_reasoning_and_text(input: Any) -> str:
    """
    Extracts reasoning content and normal text from the LLM's output.
    Processes streaming responses and yields text chunks.

    Args:
        input: The LLM's output stream

    Returns:
        Yields text chunks for the stream
    """
    # For streaming responses
    in_reasoning_block = False
    current_text = ""
    display_text = ""
    
    for chunk in input:
        content = chunk.content if hasattr(chunk, "content") else chunk
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "reasoning_content":
                    reasoning_text = item.get("reasoning_content", {}).get("text", "")
                    if reasoning_text:
                        if not in_reasoning_block:
                            display_text += "```thinking\n"
                            yield "```thinking\n"
                            in_reasoning_block = True
                        display_text += reasoning_text
                        yield reasoning_text
                elif item.get("type") == "text" and (text := item.get("text")):
                    if in_reasoning_block:
                        display_text += "\n```\n"
                        yield "\n```\n"
                        in_reasoning_block = False
                    display_text += text
                    current_text += text
                    yield text
        else:
            if in_reasoning_block:
                display_text += "\n```\n"
                yield "\n```\n"
                in_reasoning_block = False
            display_text += content
            current_text += content
            yield content
            
    if in_reasoning_block:
        display_text += "\n```"
        yield "\n```"
        
    # Store the clean text for LLM history
    st.session_state["current_llm_text"] = current_text
    st.session_state["current_display_text"] = display_text


def store_message(role: str, content: str, user_prompt: str = "", has_context: bool = False, images: List[str] = None) -> None:
    """
    Store a message in the session state for display purposes.
    
    Args:
        role: The role of the message sender ('user' or 'assistant')
        content: The message content (may include RAG context for backend)
        user_prompt: The clean user prompt (what user actually typed)
        has_context: Whether the message includes RAG/search context
        images: Optional list of image IDs
    """
    message = {"role": role, "has_context": has_context}
    
    if role == "assistant" and "current_display_text" in st.session_state:
        # For assistant responses
        message["content"] = st.session_state["current_display_text"]
        if "current_llm_text" in st.session_state:
            message["llm_content"] = st.session_state["current_llm_text"]
    else:
        message["content"] = content
        # For user messages, clean any thinking blocks to be safe
        if role == "user":
            message["llm_content"] = re.sub(r'```thinking.*?```', '', content, flags=re.DOTALL)
            message["user_prompt"] = user_prompt  # Store the clean user prompt
        else:
            message["llm_content"] = content
        
    if images:
        message["images"] = images
        
    st.session_state.messages.append(message)


def init_runnablewithmessagehistory(
    system_prompt: str, chat_model: ChatModel
) -> RunnableWithMessageHistory:
    """
    Initialize the RunnableWithMessageHistory with the given parameters.
    """
    # Use a standard message history
    msgs = StreamlitChatMessageHistory()
    # Clear any existing messages
    msgs.clear()
    
    # Create the conversation chain
    conversation = (
        ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="query"),
            ]
        )
        | chat_model.llm,
        lambda session_id: msgs,
        input_messages_key="query",
        history_messages_key="chat_history",
    )
    | extract_reasoning_and_text

    # Store LLM generated responses for display
    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]
    if "current_llm_text" not in st.session_state:
        st.session_state.current_llm_text = ""
    if "msgs" not in st.session_state:
        st.session_state.msgs = msgs

    return conversation


def generate_response(
    conversation: RunnableWithMessageHistory, input: Union[str, List[dict]]
) -> str:
    """
    Generate a response from the conversation chain with the given input.
    """
    # Get the message history
    msgs = st.session_state.msgs
    
    # Clear the standard history to replace with our cleaned one
    msgs.clear()
    
    # Add all previous messages to history with reasoning removed from assistant responses
    # But exclude the current user message which will be sent as "query"
    for i, msg in enumerate(st.session_state.messages[:-1]):  # Skip the last message (current user prompt)
        if i == 0:  # Skip the initial greeting
            continue
            
        if msg["role"] == "user":
            # Use the original content (with RAG context) for LLM processing
            msgs.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            # Remove thinking blocks from assistant messages
            clean_msg = re.sub(r'```thinking.*?```', '', msg["content"], flags=re.DOTALL)
            clean_msg = clean_msg.strip()
            if clean_msg:  # Only add if there's content after removal
                msgs.add_ai_message(clean_msg)
    
    # Format input as a chat message
    if isinstance(input, str):
        # Remove any thinking blocks
        clean_input = re.sub(r'```thinking.*?```', '', input, flags=re.DOTALL)
        formatted_input = [{"role": "user", "content": clean_input}]
    else:
        formatted_input = input

    # For streaming responses
    return st.write_stream(
        conversation.stream(
            {"query": formatted_input},
            config={"configurable": {"session_id": "streamlit_chat"}}
        )
    )


def new_chat() -> None:
    """
    Reset the chat session and initialize a new RunnableWithMessageHistory.
    """
    # Clear display messages
    st.session_state["messages"] = [INIT_MESSAGE]
    
    # Clear LangChain message history
    if "msgs" in st.session_state:
        st.session_state.msgs.clear()
    
    # Reset file uploader for indexing
    st.session_state["indexing_uploader_key"] = random.randint(1, 1000000)
    
    # Clear any other chat-related state
    if "current_llm_text" in st.session_state:
        del st.session_state["current_llm_text"]
    if "current_display_text" in st.session_state:
        del st.session_state["current_display_text"]


def display_chat_messages(
    # No uploaded_files parameter needed here for general chat display
    debug_mode: bool = False
) -> None:
    """
    Display chat messages in the Streamlit app.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # No display of images from chat-context uploader here
            # if "images" in message and message["images"]:
            #     display_images(message["images"]) # This would require re-adding display_images and handling image sources

            if message["role"] == "user":
                display_user_message(message, debug_mode)

            if message["role"] == "assistant":
                display_assistant_message(message["content"])

# Removed display_images and display_uploaded_files if they were solely for chat-context file uploads.
# If you want to re-add image display for other reasons, these functions will need to be properly defined
# and imported. For now, we assume removal of chat-context file uploads.

def display_user_message(message: dict, debug_mode: bool = False) -> None:
    """
    Display user message in the chat message.
    Shows clean user prompt by default, full content in debug mode.
    """
    if debug_mode and message.get("has_context", False):
        # In debug mode, show both the clean prompt and the RAG context
        user_prompt = message.get("user_prompt", "")
        if user_prompt:
            st.markdown("**Your Question:**")
            st.markdown(user_prompt)
            
            # Show RAG context in an expander
            with st.expander("🔍 RAG Context (Debug)", expanded=False):
                # Extract and display the RAG context
                full_content = message["content"]
                if "<search>" in full_content and "</search>" in full_content:
                    rag_content = full_content.split("<search>")[1].split("</search>")[0]
                    st.markdown("**Retrieved Context:**")
                    st.text(rag_content.strip())
    else:
        # Production mode: show only the clean user prompt
        user_prompt = message.get("user_prompt", "")
        if user_prompt:
            st.markdown(user_prompt)
        else:
            # Fallback to parsing content if user_prompt is not available
            message_content = message["content"]
            if isinstance(message_content, str):
                # Remove RAG context from display
                clean_content = message_content.split("</search>\n\n", 1)[-1]
                st.markdown(clean_content)
            elif isinstance(message_content, dict):
                # Handle cases where message_content might be a dict (e.g., from multimodal)
                # Assuming the main text content is at message_content["input"][0]["content"][0]["text"]
                if (isinstance(message_content, dict) and "input" in message_content and
                    isinstance(message_content["input"], list) and len(message_content["input"]) > 0 and
                    isinstance(message_content["input"][0], dict) and "content" in message_content["input"][0] and
                    isinstance(message_content["input"][0]["content"], list) and len(message_content["input"][0]["content"]) > 0 and
                    isinstance(message_content["input"][0]["content"][0], dict) and "text" in message_content["input"][0]["content"][0]):
                    message_text = message_content["input"][0]["content"][0]["text"]
                    clean_content = message_text.split("</search>\n\n", 1)[-1]
                    st.markdown(clean_content)
                else:
                    st.markdown("_(Complex user input)_") # Fallback for unexpected formats
            else:
                st.markdown(message_content[0]["text"]) # Assuming it's a list of dicts with text

def display_assistant_message(message_content: Union[str, dict]) -> None:
    """
    Display assistant message in the chat message.
    """
    if isinstance(message_content, str):
        st.markdown(message_content)
    elif "response" in message_content:
        st.markdown(message_content["response"])


def rag_search(prompt: str) -> tuple[str, str]:
    """
    Perform RAG search and return both the enhanced prompt and RAG context.
    
    Returns:
        tuple: (enhanced_prompt_with_context, rag_context_only)
    """
    # Perform the search using the search_index function (now local)
    docs = search_index(prompt, "faiss_index")
    # Check if an error message was returned
    if isinstance(docs[0], str) and docs[0].startswith("Error:"):
        st.warning(f"RAG search error: {docs[0]}")
        return prompt, "Error retrieving RAG context"
    
    # Format the RAG context
    rag_context = "\n\n".join(doc.page_content for doc in docs)
    
    # Format the enhanced prompt
    rag_content = (
        "Here are the RAG search results: \n\n<search>\n\n"
        + rag_context
        + "\n\n</search>\n\n"
    )
    enhanced_prompt = rag_content + prompt
    
    return enhanced_prompt, rag_context


def web_or_local(prompt: str, web_local_rag: str) -> tuple[str, bool]:
    """
    Process prompt with web search or RAG, return enhanced prompt and context flag.
    
    Returns:
        tuple: (enhanced_prompt, has_context)
    """
    has_context = False
    
    if web_local_rag == "Web":
        search = SerpAPIWrapper()
        try:
            search_text = search.run(prompt)
            web_content = (
                "Here is the web search result: \n\n<search>\n\n"
                + search_text
                + "\n\n</search>\n\n"
            )
            prompt = web_content + prompt
            has_context = True
        except Exception as e:
            st.warning(f"Web search failed (check SerpAPI key/service): {e}")
            # Fallback to no context if web search fails
            pass
    elif web_local_rag == "RAG":
        prompt, _ = rag_search(prompt) # The actual RAG context is for debug display, not direct prompt
        has_context = True
        
    return prompt, has_context


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    set_page_config()

    # Generate a unique widget key only once
    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))
    
    # Ensure indexing_uploaded_files is initialized if it's not set from sidebar already
    if "indexing_uploaded_files" not in st.session_state:
        st.session_state["indexing_uploaded_files"] = []

    # Add a button to start a new chat
    st.sidebar.button("New Chat", on_click=new_chat, type="primary")

    model_kwargs, system_prompt, web_local, debug_mode = render_sidebar()
    chat_model = ChatModel(st.session_state["model_name"], model_kwargs)
    runnable_with_messagehistory = init_runnablewithmessagehistory(
        system_prompt, chat_model
    )

    # Display chat messages (no uploaded_files parameter for this function anymore)
    display_chat_messages(debug_mode=debug_mode)

    # User-provided prompt
    prompt = st.chat_input()

    # Process the user prompt
    if prompt:
        # Store the original user prompt
        original_prompt = prompt
        
        # Enhance prompt with RAG/Web search if needed
        formatted_prompt, has_context = web_or_local(prompt, web_local)
        
        # Store and display user message with both original and enhanced versions
        store_message("user", formatted_prompt, user_prompt=original_prompt, has_context=has_context)
        
        with st.chat_message("user"):
            # Create a temporary message dict for display
            temp_message = {
                "content": formatted_prompt,
                "user_prompt": original_prompt,
                "has_context": has_context
            }
            display_user_message(temp_message, debug_mode)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response = generate_response(
                runnable_with_messagehistory,
                formatted_prompt
            )
            # Store the assistant message (content is already captured in state during streaming)
            store_message("assistant", response)


if __name__ == "__main__":
    main()
