import streamlit as st
import os
import tempfile
import PyPDF2
import io
import json
import requests

# we can make this configurable later on
LLM_MODEL = "mistral:latest"

# Configure the page with professional settings
st.set_page_config(
    page_title="HR Policy Reasoning System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Existing styles remain unchanged */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 3px solid #3b82f6;
    }
    /* ... (rest of your CSS unchanged) ... */
</style>
""", unsafe_allow_html=True)

# Backend API configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002")


### --- NEW CODE START ---
# -------------------------------
# SESSION STATE (Auth-related)
# -------------------------------
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "show_register" not in st.session_state:
    st.session_state.show_register = False
### --- NEW CODE END ---


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location and return path"""
    try:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


def ingest_document(uploaded_file, department):
    """Upload file to backend /ingest endpoint"""
    try:
        file_bytes = uploaded_file.getvalue()
        file_obj = io.BytesIO(file_bytes)
        files = {"file": (uploaded_file.name, file_obj, uploaded_file.type or "application/octet-stream")}
        data = {"department": department}
        headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}
        response = requests.post(
            f"{BACKEND_URL}/ingest",
            files=files,
            data=data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.HTTPError:
        return False, f"Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def chat_with_assistant(user_query, department, model=LLM_MODEL):
    """Call the backend chat API and handle streamed responses."""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}

        # Enable streaming mode to handle StreamingResponse properly
        with requests.post(
            f"{BACKEND_URL}/chat",
            json={"query": user_query, "k": 4, "model": model, "role": "u-employee"},
            headers=headers,
            stream=True,
            timeout=60,
        ) as response:

            if response.status_code != 200:
                return False, f"Error {response.status_code}: {response.text}"

            # Collect streamed chunks
            full_text = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    decoded = chunk.decode("utf-8")
                    full_text += decoded
                    # (Optional) If you want to show live updates in Streamlit:
                    # st.write(decoded)

            return True, {"message": full_text}

    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"



### --- NEW CODE START ---
# -------------------------------
# AUTH PAGES
# -------------------------------
def register_page():
    st.title("🧾 Create an Account")

    with st.form("register_form"):
        # username = st.text_input("Username")
        email = st.text_input("Email ID")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Register")

        if submit:
            if not email or not password:
                st.warning("Please fill in all fields.")
            else:
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/register/user",
                        json={"email_id": email, "password": password, "hashed_password": None},
                        timeout=15
                    )
                    if res.status_code == 200:
                        st.success("✅ Registration successful! You can now log in.")
                        st.session_state.show_register = False
                        st.rerun()
                    else:
                        st.error(f"❌ Registration failed: {res.text}")
                except Exception as e:
                    st.error(f"Registration error: {str(e)}")

    if st.button("⬅️ Back to Login"):
        st.session_state.show_register = False
        st.rerun()


def login_page():
    st.title("🔐 Login to Continue")

    with st.form("login_form"):
        email = st.text_input("Email ID")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/token",
                    data={"username": email, "password": password},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=15
                )
                if response.status_code == 200:
                    token_data = response.json()
                    st.session_state.auth_token = token_data["access_token"]
                    st.session_state.username = email
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")
            except Exception as e:
                st.error(f"Login failed: {str(e)}")

    if st.button("📝 Register"):
        st.session_state.show_register = True
        st.rerun()


def logout():
    st.session_state.auth_token = None
    st.session_state.username = None
    st.session_state.show_register = False
    st.rerun()
### --- NEW CODE END ---


### --- NEW CODE START ---
# -------------------------------
# AUTHENTICATION GATE
# -------------------------------
if not st.session_state.auth_token:
    if st.session_state.show_register:
        register_page()
    else:
        login_page()
    st.stop()
### --- NEW CODE END ---


def render_ingest_page():
    """Render the document ingestion page"""
    st.markdown('<div class="section-header">📤 Document Ingestion</div>', unsafe_allow_html=True)
    # (existing content unchanged)
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    department = st.selectbox("Department", ["finance", "hr", "legal", "operations"])

    if uploaded_file and st.button("🚀 Upload & Ingest", type="primary", use_container_width=True):
        success, result = ingest_document(uploaded_file, department)
        if success:
            st.success(f"✅ {uploaded_file.name} ingested successfully.")
        else:
            st.error(result)


def render_chat_page():
    """Render the chat page"""
    st.markdown("""
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-top: 1rem;
        }

        .chat-message {
            max-width: 75%;
            padding: 0.8rem 1rem;
            border-radius: 1rem;
            line-height: 1.5;
            word-wrap: break-word;
        }

        /* Assistant on left */
        .chat-message.assistant {
            background-color: #2d2d2d;
            color: #ffffff;
            align-self: flex-start;
            border-top-left-radius: 0;
        }

        /* User on right */
        .chat-message.user {
            background-color: #0078d4;
            color: white;
            align-self: flex-end;
            border-top-right-radius: 0;
        }

        .section-header {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">💬 Policy Assistant Chat</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_department = st.selectbox("Your Department", ["finance", "hr", "legal", "operations"], key="chat_dept")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for user_msg, assistant_msg in st.session_state.chat_history:
        st.markdown(f"<div class='chat-message user'>{user_msg}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-message assistant'>{assistant_msg}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.text_area("Ask a question:", placeholder="Type here...")
    if st.button("📤 Send"):
        if not user_input.strip():
            st.warning("Please enter a question.")
        else:
            chat_box = st.empty()  # Placeholder for live updates
            full_response = ""

            # Streamed version of chat_with_assistant
            headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}
            with requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"query": user_input, "k": 4, "model": LLM_MODEL, "role": "u-employee"},
                    headers=headers,
                    stream=True,
                    timeout=60,
            ) as response:

                if response.status_code != 200:
                    st.error(f"Error {response.status_code}: {response.text}")
                else:
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            decoded = chunk.decode("utf-8")
                            full_response += decoded
                            chat_box.markdown(
                                f"<div class='chat-message-assistant'><b>Assistant:</b> {full_response}</div>",
                                unsafe_allow_html=True,
                            )

            st.session_state.chat_history.append((user_input, full_response))
            st.rerun()


def render_about_page():
    """Render about page"""
    st.markdown('<div class="section-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    ## This RAG-based assistant helps you interact with company policy documents.
    - 📤 Upload and process policies  
    - 💬 Chat for contextual answers  
    - 🔐 Department-based access controls  
    """)


def main():
    """Main application function"""
    st.markdown('<div class="main-header" style="color: lightblue;">🏢 HR Terms & Conditions Policy Assistant</div>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.markdown("## 📋 Navigation")
    st.sidebar.write(f"👋 Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        logout()

    if st.sidebar.button("📤 Ingest Documents", use_container_width=True):
        st.session_state.page = "ingest"
    if st.sidebar.button("💬 Chat with Assistant", use_container_width=True):
        st.session_state.page = "chat"
    if st.sidebar.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "about"
        st.rerun()

    if "page" not in st.session_state:
        st.session_state.page = "ingest"

    if st.session_state.page == "ingest":
        render_ingest_page()
    elif st.session_state.page == "chat":
        render_chat_page()
    elif st.session_state.page == "about":
        render_about_page()

    st.markdown("""
    <div class="footer">
        <p>🔐 Secure RAG T&C Policy Assistant | Powered by FastAPI & Streamlit</p>
        <p style="font-size: 0.8rem; color: #9ca3af;">
            All conversations are logged for compliance.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
