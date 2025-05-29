import base64
import random
from io import BytesIO
from typing import List, Tuple, Union

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import AIMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from PIL import Image

# --- Simple User Storage (In-memory for demonstration) ---
# In a real application, you would use a database (like Firebase, PostgreSQL, etc.)
# for persistent user storage and proper password hashing.
REGISTERED_USERS = {
    "admin@redaibot.com": "adminpass"
}

CLAUDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input"),
    ]
)

INIT_MESSAGE = {
    "role": "assistant",
    "content": "Hi there! I'm the Redington AI Bot, built to support you. How may I assist you today?",
}

class StreamHandler(BaseCallbackHandler):
    """
    Callback handler to stream the generated text to Streamlit.
    """

    def __init__(self, container: st.container) -> None:
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """
        Append the new token to the text and update the Streamlit container.
        """
        self.text += token
        self.container.markdown(self.text)


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
    if "user_email" not in st.session_state: # Store user's email if logged in
        st.session_state["user_email"] = None
    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]
    if "langchain_messages" not in st.session_state:
        st.session_state["langchain_messages"] = [] # Used by this app's specific logic
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0


def login_user(email, password):
    """
    Authenticates a user against the hardcoded REGISTERED_USERS.
    """
    if email in REGISTERED_USERS and REGISTERED_USERS[email] == password:
        st.session_state.logged_in = True
        st.session_state.user_email = email
        st.success(f"Logged in as {email}")
        # Reset chat history for the new session after login
        new_chat() # This will also rerun
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
        st.success(f"User {email} registered successfully! For this demo, please use 'test@example.com' and 'password123' to log in.")


def logout_user():
    """
    Logs out the current user and clears session state.
    """
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.session_state.messages = [INIT_MESSAGE] # Reset chat messages
    st.session_state["langchain_messages"] = [] # Clear LangChain history
    st.success("Logged out successfully.")
    st.rerun() # Rerun to update UI after logout


def get_sidebar_params() -> Tuple[float, float, int, int, int, str, str]:
    """
    Get inference parameters from the sidebar, conditionally rendering based on login status.
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
            st.info("For this demo, use 'test@example.com' and 'password123' to log in.")
            # Return default/dummy values if not logged in, as chat UI won't be active
            return 1.0, 1.0, 500, 4096, 10, "", "anthropic.claude-3-sonnet-20240229-v1:0"
        else:
            st.write(f"Logged in as: {st.session_state.user_email}")
            st.button("Logout", on_click=logout_user, type="secondary", use_container_width=True)

            st.markdown("---")
            st.button("New Chat", on_click=new_chat, type="primary") # Button to reset current chat session

            st.markdown("---")
            st.markdown("## Inference Parameters")
            model_id_select = st.selectbox(
                'Model',
                ('Claude 3 Sonnet', 'Claude 3 Haiku', 'Mistral Large'),
                key=f"{st.session_state['widget_key']}_Model_Id",
            )

            model_map = {
                "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
                "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
                "Mistral Large": "mistral.mistral-large-2402-v1:0"
            }

            model_id = model_map.get(model_id_select)

            if model_id_select == "Mistral Large":
                system_prompt = st.text_area(
                    "System Prompt",
                    "",
                    key=f"{st.session_state['widget_key']}_System_Prompt",
                    disabled = True
                )
            else:
                system_prompt = st.text_area(
                    "System Prompt",
                    "You're a cool assistant, love to respond with emoji.",
                    key=f"{st.session_state['widget_key']}_System_Prompt",
                )

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                key=f"{st.session_state['widget_key']}_Temperature",
            )
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    top_p = st.slider(
                        "Top-P",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.00,
                        step=0.01,
                        key=f"{st.session_state['widget_key']}_Top-P",
                    )
                with col2:
                    if model_id_select == "Mistral Large":
                        top_k = st.slider(
                            "Top-K",
                            min_value=1,
                            max_value=200,
                            value=200,
                            step=5,
                            key=f"{st.session_state['widget_key']}_Top-K",
                        )
                    else:
                        top_k = st.slider(
                            "Top-K",
                            min_value=1,
                            max_value=500,
                            value=500,
                            step=5,
                            key=f"{st.session_state['widget_key']}_Top-K",
                        )
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    max_tokens = st.slider(
                        "Max Token",
                        min_value=0,
                        max_value=4096,
                        value=4096,
                        step=8,
                        key=f"{st.session_state['widget_key']}_Max_Token",
                    )
                with col2:
                    memory_window = st.slider(
                        "Memory Window",
                        min_value=0,
                        max_value=10,
                        value=10,
                        step=1,
                        key=f"{st.session_state['widget_key']}_Memory_Window",
                    )

            return temperature, top_p, top_k, max_tokens, memory_window, system_prompt, model_id


def init_conversationchain(
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    memory_window: int,
    system_prompt: str,
    model_id: str
) -> ConversationChain:
    """
    Initialize the ConversationChain with the given parameters.
    """
    model_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }

    if system_prompt != "":
        model_kwargs["system"] = system_prompt

    llm = BedrockChat(model_id=model_id, model_kwargs=model_kwargs, streaming=True)

    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferWindowMemory(
            k=memory_window,
            ai_prefix="Assistant",
            chat_memory=StreamlitChatMessageHistory(key="langchain_messages"), # Use a distinct key for LangChain's memory
            return_messages=True,
        ),
        prompt=CLAUDE_PROMPT,
    )

    # Store LLM generated responses (initial message handled by init_session_state)
    # This block is redundant if init_session_state handles it.
    # if "messages" not in st.session_state:
    #     st.session_state.messages = [INIT_MESSAGE]

    return conversation


def generate_response(
    conversation: ConversationChain, input: Union[str, List[dict]]
) -> str:
    """
    Generate a response from the conversation chain with the given input.
    """
    # The input to conversation.invoke needs to be a dictionary matching the prompt's variable names.
    # CLAUDE_PROMPT has 'input' and 'history'.
    # The `input` parameter here is the user's current prompt.
    response = conversation.invoke(
        {"input": input}, {"callbacks": [StreamHandler(st.empty())]}
    )
    # The response from conversation.invoke will be a dictionary, e.g., {'response': '...'}.
    # We need to extract the actual text.
    return response['response'] if isinstance(response, dict) and 'response' in response else str(response)


def new_chat() -> None:
    """
    Reset the chat session and initialize a new conversation chain.
    """
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = [] # Clear LangChain's internal history
    st.session_state["file_uploader_key"] = random.randint(1, 100) # Reset file uploader key
    st.rerun() # Rerun to clear chat display


def display_chat_messages(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> None:
    """
    Display chat messages and uploaded images in the Streamlit app.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display uploaded files if they exist and are associated with the message
            if "images" in message and message["images"]:
                num_cols = 10
                cols = st.columns(num_cols)
                i = 0
                for image_id in message["images"]:
                    for uploaded_file in uploaded_files:
                        if image_id == uploaded_file.file_id:
                            img = Image.open(uploaded_file)
                            with cols[i]:
                                st.image(img, caption="", width=75)
                                i += 1
                            if i >= num_cols:
                                i = 0
                            break # Found the image, move to next image_id

            # Display message content based on role and type
            if message["role"] == "user":
                # User message content can be string or a list of dicts (for multimodal)
                if isinstance(message["content"], str):
                    st.markdown(message["content"])
                elif isinstance(message["content"], list): # Multimodal input
                    # Assuming the first item in the list is the text part
                    text_content = next((item["text"] for item in message["content"] if item.get("type") == "text"), "")
                    st.markdown(text_content)
                else:
                    # Fallback for unexpected content structure
                    st.markdown(str(message["content"]))
            elif message["role"] == "assistant":
                # Assistant message content is expected to be a string
                st.markdown(message["content"])


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    set_page_config()
    init_session_state()

    # Render authentication and get model parameters
    # This function now handles the conditional rendering of auth vs chat params
    temperature, top_p, top_k, max_tokens, memory_window, system_prompt, model_id = get_sidebar_params()
    
    # --- Main Chat Application (only runs if logged in) ---
    if st.session_state.logged_in:
        conv_chain = init_conversationchain(temperature, top_p, top_k, max_tokens, memory_window, system_prompt, model_id)

        # Image uploader
        # The logic for enabling/disabling based on model_id_select is now handled directly here
        image_upload_disabled = (model_id == "mistral.mistral-large-2402-v1:0")
        uploaded_files = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=st.session_state["file_uploader_key"],
            disabled=image_upload_disabled,
        )

        # Display chat messages
        display_chat_messages(uploaded_files)

        # User-provided prompt
        prompt = st.chat_input()

        # Get images from previous messages (for display purposes)
        message_images_list = [
            image_id
            for message in st.session_state.messages
            if message["role"] == "user"
            and "images" in message
            and message["images"]
            for image_id in message["images"]
        ]

        # Process the user prompt and images
        if prompt:
            current_uploaded_file_ids = []
            content_for_llm = []

            # Handle uploaded images for the current prompt
            if uploaded_files and len(message_images_list) < len(uploaded_files):
                # Only process newly uploaded images
                for uploaded_file in uploaded_files:
                    if uploaded_file.file_id not in message_images_list:
                        current_uploaded_file_ids.append(uploaded_file.file_id)
                        img = Image.open(uploaded_file)
                        with BytesIO() as output_buffer:
                            img.save(output_buffer, format=img.format)
                            content_image_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf8")
                        
                        # Add image to LLM content
                        content_for_llm.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": content_image_base64},
                        })
            
            # Add text prompt to LLM content
            content_for_llm.append({"type": "text", "text": prompt})

            # Store user message in session state for display
            st.session_state.messages.append(
                {"role": "user", "content": content_for_llm, "images": current_uploaded_file_ids}
            )

            # Display the user message immediately
            with st.chat_message("user"):
                # Display text part
                st.markdown(prompt)
                # Display images
                if current_uploaded_file_ids:
                    num_cols = 10
                    cols = st.columns(num_cols)
                    i = 0
                    for image_id in current_uploaded_file_ids:
                        for uploaded_file in uploaded_files:
                            if image_id == uploaded_file.file_id:
                                img = Image.open(uploaded_file)
                                with cols[i]:
                                    st.image(img, caption="", width=75)
                                    i += 1
                                if i >= num_cols:
                                    i = 0
                                break # Found the image, move to next image_id


            # Generate a new response if last message is not from assistant
            # Use the formatted content_for_llm for the LLM input
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    response_text = generate_response(
                        conv_chain, {"input": content_for_llm} # Pass dict with 'input' key
                    )
                message = {"role": "assistant", "content": response_text}
                st.session_state.messages.append(message)
    else:
        # Message to display when not logged in
        st.info("Please log in to start chatting with the Redington AI Bot.")


if __name__ == "__main__":
    main()
