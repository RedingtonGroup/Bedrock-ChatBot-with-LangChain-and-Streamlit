import random
from io import BytesIO
from typing import List, Tuple, Union, Dict, Any
import re
import datetime

import streamlit as st
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper # Keep if you intend to use web search later
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from PIL import Image, UnidentifiedImageError
import pdfplumber

from dotenv import load_dotenv

from config import config
from models import ChatModel
from role_prompt import role_prompt
from bedrock_embedder import search_index # Assuming this is correctly implemented

# Load the env variables
load_dotenv()

# --- Simple User Storage (In-memory for demonstration) ---
# In a real application, you would use a database (like Firebase, PostgreSQL, etc.)
# for persistent user storage.
REGISTERED_USERS = {
    "admin@redaibot.com": "adminpass"
}

INIT_MESSAGE = {
    "role": "assistant",
    "content": "Hi there! I'm the Redington AI Bot, built to support you. How may I assist you today?",
    "llm_content": "Hi there! I'm the Redington AI Bot, built to support you. How may I assist you today?",
    "user_prompt": "",
    "has_context": False,
    "token_count": 0 # Added token_count to INIT_MESSAGE
}

def set_page_config() -> None:
    """
    Set the Streamlit page configuration.
    """
    st.set_page_config(page_title="🤖 Chat with Redington AI Bot", layout="wide")
    st.title("🤖 Chat with Redington AI Bot")

def init_session_state() -> None:
    """
    Initialize Streamlit session state variables.
    """
    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "user_id" not in st.session_state: # Stores the logged-in user's identifier (e.g., email)
        st.session_state["user_id"] = None
    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]
    if "current_llm_text" not in st.session_state:
        st.session_state.current_llm_text = ""
    if "msgs" not in st.session_state:
        st.session_state.msgs = StreamlitChatMessageHistory() # LangChain's in-memory history for current session
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

def login_user(email, password):
    """
    Authenticates a user against the hardcoded REGISTERED_USERS.
    """
    if email in REGISTERED_USERS and REGISTERED_USERS[email] == password:
        st.session_state.logged_in = True
        st.session_state.user_id = email # Use email as user_id for simplicity
        st.success(f"Logged in as {email}")
        st.rerun() # Rerun to update UI after login
    else:
        st.error("Invalid email or password.")

def register_user(email, password):
    """
    A placeholder for user registration. For this simple demo, it just acknowledges.
    In a real app, you would securely store new user credentials.
    """
    if email in REGISTERED_USERS:
        st.error("User already exists. Please log in.")
    else:
        # In a real application, you would add the user to a persistent store (e.g., database)
        # For this demo, we'll just acknowledge and instruct to use the hardcoded user.
        st.success(f"User {email} registered successfully! For this demo, please use 'admin@redaibot.com' and 'adminpass' to log in.")


def logout_user():
    """
    Logs out the current user and clears session state.
    """
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.messages = [INIT_MESSAGE] # Reset chat messages
    st.session_state.msgs.clear() # Clear LangChain history
    st.success("Logged out successfully.")
    st.rerun() # Rerun to update UI after logout

def render_sidebar_auth_and_params() -> Tuple[Dict, str, str, bool]:
    """
    Renders the authentication section and model parameters in the sidebar.
    Returns model parameters and debug mode setting.
    """
    with st.sidebar:
        st.header("Authentication")
        if not st.session_state.logged_in:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", use_container_width=True):
                    login_user(email, password)
            with col2:
                if st.button("Sign Up", use_container_width=True):
                    register_user(email, password)
            st.info("For this demo, use 'admin@redaibot.com' and 'adminpass' to log in.")
            # If not logged in, we return default empty values for model params,
            # as the main chat UI won't be rendered.
            return {}, "", "Local", False
        else:
            st.write(f"Logged in as: {st.session_state.user_id}")
            st.button("Logout", on_click=logout_user, type="secondary", use_container_width=True)

            st.markdown("---")
            st.button("New Chat", on_click=new_chat, type="primary") # Button to reset current chat

            st.markdown("---")
            st.header("Model Parameters")

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

            web_local = st.selectbox(
                "Options",
                ("RAG", "Local"), # Removed "Web" as SerpAPIWrapper might not be configured
                key=f"{st.session_state['widget_key']}_Options",
            )

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
    """
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
        
    st.session_state["current_llm_text"] = current_text
    st.session_state["current_display_text"] = display_text


def store_message(role: str, content: str, user_prompt: str = "", has_context: bool = False, images: List[str] = None, token_count: int = 0) -> None:
    """
    Store a message in the session state for display purposes.
    (No Firestore saving in this simplified version)
    """
    message = {"role": role, "has_context": has_context}
    
    # If it's an assistant message, use the content from current_display_text
    # and store the token_count.
    if role == "assistant":
        message["content"] = st.session_state.get("current_display_text", "")
        message["llm_content"] = st.session_state.get("current_llm_text", "")
        message["token_count"] = token_count
    else: # For user messages
        message["content"] = content
        if role == "user":
            message["llm_content"] = re.sub(r'```thinking.*?```', '', content, flags=re.DOTALL)
            message["user_prompt"] = user_prompt
        else:
            message["llm_content"] = content # Fallback for other roles if any
            
    if images:
        message["images"] = images
            
    st.session_state.messages.append(message)


def init_runnablewithmessagehistory(
    system_prompt: str, chat_model: ChatModel
) -> RunnableWithMessageHistory:
    """
    Initialize the RunnableWithMessageHistory with the given parameters.
    """
    msgs = st.session_state.msgs # Use the StreamlitChatMessageHistory from session state
    # No need to clear msgs here, as new_chat() already handles it for a fresh start.

    conversation_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="query"),
            ]
        )
        | chat_model.llm
    )
    
    runnable_with_history = RunnableWithMessageHistory(
        conversation_chain,
        lambda session_id: msgs, # Use the StreamlitChatMessageHistory from session state
        input_messages_key="query",
        history_messages_key="chat_history",
    )
    
    final_runnable = runnable_with_history | extract_reasoning_and_text

    return final_runnable


def generate_response(
    conversation: RunnableWithMessageHistory, input: Union[str, List[dict]]
) -> str:
    """
    Generate a response from the conversation chain with the given input.
    Accumulates streamed text and returns the full content (llm_content).
    """
    if isinstance(input, str):
        clean_input = re.sub(r'```thinking.*?```', '', input, flags=re.DOTALL)
        formatted_input = [{"role": "user", "content": clean_input}]
    else:
        formatted_input = input

    session_id_for_langchain = st.session_state.user_id if st.session_state.user_id else "default_session"

    # Create an empty container to stream the response
    response_placeholder = st.empty() 

    # Iterate through the streamed chunks and display them
    for chunk in conversation.stream(
        {"query": formatted_input},
        config={"configurable": {"session_id": session_id_for_langchain}}
    ):
        # The extract_reasoning_and_text function (which is part of the runnable)
        # updates st.session_state["current_display_text"] and st.session_state["current_llm_text"]
        # as it processes chunks. We just need to display the current_display_text.
        response_placeholder.markdown(st.session_state.get("current_display_text", ""))
    
    # After streaming is complete, return the raw LLM content for token counting
    return st.session_state.get("current_llm_text", "")


def new_chat() -> None:
    """
    Resets the current chat session.
    """
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["msgs"].clear() # Clear LangChain's internal history
    st.session_state["file_uploader_key"] = random.randint(1, 100) # Reset file uploader
    
    if "current_llm_text" in st.session_state:
        del st.session_state["current_llm_text"]
    if "current_display_text" in st.session_state:
        del st.session_state["current_display_text"]
    # Removed st.rerun() as it's not needed and causes the "no-op" warning.
    # Streamlit will automatically re-run due to session state changes.


def display_chat_messages(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    debug_mode: bool = False
) -> None:
    """
    Display chat messages and uploaded images in the Streamlit app.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if uploaded_files and "images" in message and message["images"]:
                display_images(message["images"], uploaded_files)

            if message["role"] == "user":
                display_user_message(message, debug_mode)

            if message["role"] == "assistant":
                display_assistant_message(message["content"])
                # Display token count if available
                if "token_count" in message and message["token_count"] > 0:
                    st.caption(f"Tokens used: {message['token_count']}")


def display_images(
    image_ids: List[str],
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
) -> None:
    """
    Display uploaded images in the chat message.
    """
    num_cols = 10
    cols = st.columns(num_cols)
    i = 0

    for image_id in image_ids:
        for uploaded_file in uploaded_files:
            if image_id == uploaded_file.file_id:
                if uploaded_file.type.startswith("image/"):
                    img = Image.open(uploaded_file)

                    with cols[i]:
                        st.image(img, caption="", width=75)
                        i += 1

                    if i >= num_cols:
                        i = 0
                elif uploaded_file.type in [
                    "text/plain",
                    "text/csv",
                    "text/x-python-script",
                ]:
                    if uploaded_file.type == "text/x-python-script":
                        st.write(f"🐍 Uploaded Python file: {uploaded_file.name}")
                    else:
                        st.write(f"📄 Uploaded text file: {uploaded_file.name}")
                elif uploaded_file.type == "application/pdf":
                    # This part needs to be updated to handle PDF content extraction.
                    # For now, it will just write the file name.
                    st.write(f"📑 Uploaded PDF file: {uploaded_file.name}")


def rag_search(prompt: str) -> tuple[str, str]:
    """
    Perform RAG search and return both the enhanced prompt and RAG context.
    """
    # Ensure search_index is properly defined and returns a valid index or path
    # If search_index is meant to load the FAISS index, it should return the loaded index.
    # If it returns a path, FAISS.load_local should be called with that path.
    
    # Assuming search_index returns the path to the FAISS index
    index_path = "faiss_index" # This should be where your FAISS index is saved

    try:
        embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
        # Load the FAISS index
        db = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        
        docs = db.similarity_search(prompt)

        rag_context = "\n\n".join(doc.page_content for doc in docs)
        
        rag_content = (
            "Here are the RAG search results: \n\n<search>\n\n"
            + rag_context
            + "\n\n</search>\n\n"
        )
        enhanced_prompt = rag_content + prompt
        
        return enhanced_prompt, rag_context
    except Exception as e:
        st.error(f"RAG Search Error: {e}. Ensure 'faiss_index' exists and 'bedrock_embedder.py' is correctly configured.")
        return prompt, "Error retrieving RAG context"


def web_or_local(prompt: str, web_local_rag: str) -> tuple[str, bool]:
    """
    Process prompt with web search or RAG, return enhanced prompt and context flag.
    """
    has_context = False
    
    if web_local_rag == "Web":
        # Ensure SerpAPIWrapper is configured with API key if used
        # For now, keeping it as a placeholder as it might not be configured
        # search = SerpAPIWrapper()
        # search_text = search.run(prompt)
        # web_content = (
        #     "Here is the web search result: \n\n<search>\n\n"
        #     + search_text
        #     + "\n\n</search>\n\n"
        # )
        # prompt = web_content + prompt
        # has_context = True
        st.warning("Web search functionality is not fully configured in this demo.")
        pass # Fallback to no context or handle differently
    elif web_local_rag == "RAG":
        # Note: rag_search will currently fail because 'db' is not defined (Firebase removed)
        # You'll need to ensure your FAISS index loading is independent of Firebase if you use RAG.
        prompt, _ = rag_search(prompt)
        has_context = True
        
    return prompt, has_context


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    set_page_config()
    init_session_state()

    # Render authentication and get model parameters
    model_kwargs, system_prompt, web_local, debug_mode = render_sidebar_auth_and_params()
    
    # Stop execution if user is not logged in (render_sidebar_auth_and_params returns early)
    if not st.session_state.logged_in:
        return

    # --- Main Chat Application (only runs if logged in) ---
    chat_model = ChatModel(st.session_state["model_name"], model_kwargs)
    runnable_with_messagehistory = init_runnablewithmessagehistory(
        system_prompt, chat_model
    )

    model_config = config["models"][st.session_state["model_name"]]
    image_upload_disabled = (
        True if model_config.get("input_format") == "text" else False
    )
    uploaded_files = st.file_uploader(
        "Choose a file",
        type=["jpg", "jpeg", "png", "txt", "pdf", "csv", "py"],
        accept_multiple_files=True,
        key=st.session_state["file_uploader_key"],
        disabled=image_upload_disabled,
    )

    display_chat_messages(uploaded_files, debug_mode)

    prompt = st.chat_input()

    message_images_list = [
        image_id
        for message in st.session_state.messages
        if message["role"] == "user" and "images" in message and message["images"]
        for image_id in message["images"]
    ]

    if prompt:
        original_prompt = prompt
        
        # Note: If RAG or Web search is selected, this will use the functions above.
        # Ensure your RAG setup (bedrock_embedder.py and FAISS index loading)
        # is independent of Firebase if you use it.
        formatted_prompt, has_context = web_or_local(prompt, web_local)
        
        with st.chat_message("user"):
            temp_message = {
                "content": formatted_prompt,
                "user_prompt": original_prompt,
                "has_context": has_context
            }
            display_user_message(temp_message, debug_mode)

        store_message("user", formatted_prompt, user_prompt=original_prompt, has_context=has_context)
        
        with st.chat_message("assistant"):
            # generate_response now returns the full response text (llm_content)
            response_text_for_tokens = generate_response(
                runnable_with_messagehistory,
                formatted_prompt
            )
            
            # Calculate token count for the generated response
            # Using a simple word count as a proxy for tokens
            token_count = len(response_text_for_tokens.split()) 
            
            # store_message needs the 'display_text' for 'content' and 'llm_content' for 'llm_content'
            # The 'content' for display should come from st.session_state["current_display_text"]
            # The 'llm_content' for internal use is now passed as response_text_for_tokens
            
            store_message(
                "assistant", 
                st.session_state.get("current_display_text", ""), # Use display text for content
                token_count=token_count
            )


if __name__ == "__main__":
    main()
