import base64

import streamlit as st
import os
import tempfile
import PyPDF2
import io
import json
import requests
from PIL import Image

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
    
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #e5e7eb;
    }
    
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
        background: black;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 22px 22px 22px 6px;
        margin: 0.5rem 0;
        border: 1px solid #d1d5db;
        max-width: 85%;
        margin-right: auto;
    }
    
    .you-label {
        color: #7fffd4;
        font-weight: 900;
    }

    /* User on right */
    .chat-message.user {
        background: black; /* Changed from linear-gradient */
        color: white;    /* Added for readability against a black background */
        padding: 1rem 1.5rem;
        border-radius: 22px 22px 22px 6px;
        margin: 0.5rem 0;
        border: 1px solid #bfdbfe;
        max-width: 45%;
        margin-left: auto;
    }
    
    .assistant-label {
        color: #0066ff;
        font-weight: 900;
    }
       
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #e5e7eb;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    
    .sidebar-nav {
        padding: 1rem 0;
    }

    .nav-button {
        width: 100%;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        border: none;
        background: #f8fafc;
        color: #374151;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }

    .nav-button:hover {
        background: #e2e8f0;
    }

    .nav-button.active {
        background: #3b82f6;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    textarea {
        width: 100% !important;   /* Full width of container */
        min-height: 40px !important;  /* Adjust height */
        color: #f2f2f2 !important;    /* Custom text color (DodgerBlue) */
        font-size: 16px !important;   /* Optional: font size */
    }
    /* ... (rest of your CSS unchanged) ... */
</style>
""", unsafe_allow_html=True)

# Backend API configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002")
API_PREFIX = "/api/v1"

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

# authentication pages
def load_captcha(captcha_type="register"):
    """
    Load captcha from backend and store in session state

    Args:
        captcha_type (str): Type of captcha - "register" or "login"
    """
    try:
        res = requests.get(f"{BACKEND_URL}/{API_PREFIX}/captcha/init", timeout=10)
    except Exception as e:
        st.error(f"Unable to contact backend for captcha: {e}")
        return False

    if res.status_code != 200:
        st.error(f"Unable to load captcha: {res.text}")
        return False

    # Store captcha data with the specified type prefix
    st.session_state[f"{captcha_type}_captcha_id"] = res.headers.get("X-Captcha-ID")
    st.session_state[f"{captcha_type}_captcha_img"] = res.content
    return True


def register_page():
    st.title("🧾 Create an Account")
    # ------------------------------------------------------
    # Show persistent error if exists (after rerun)
    # ------------------------------------------------------
    if "register_error" in st.session_state and st.session_state["register_error"]:
        st.error(st.session_state.pop("register_error"))
    # ------------------------------------------------------
    # Registration form
    # ------------------------------------------------------
    with st.form("register_form"):
        email = st.text_input("Email ID")
        password = st.text_input("Password", type="password")

        # --------------------------------------------------
        # Conditionally show captcha ONLY after both fields filled
        # --------------------------------------------------
        if email and password:

            # Ensure captcha is loaded in session state
            if "register_captcha_id" not in st.session_state:
                if not load_captcha("register"):
                    st.stop()

            captcha_id = st.session_state.get("register_captcha_id")
            captcha_img_bytes = st.session_state.get("register_captcha_img")

            # Display captcha
            try:
                captcha_img = Image.open(io.BytesIO(captcha_img_bytes))
                st.image(captcha_img, caption="Please enter the text shown above")
            except:
                st.warning("Failed to render captcha image.")

            # Captcha input
            captcha_text = st.text_input("Captcha", placeholder="Enter the text")

            # Refresh captcha button
            if st.form_submit_button("🔄 Refresh Captcha"):
                st.session_state.pop("register_captcha_id", None)
                st.session_state.pop("register_captcha_img", None)
                load_captcha("register")
                st.rerun()

        else:
            captcha_text = None

        # Submit button
        submit = st.form_submit_button("Register")

        # ------------------------------------------------------
        # Submit logic
        # ------------------------------------------------------
        if submit:

            # Basic validation
            if not email or not password:
                st.warning("Please fill in both email and password.")
                return

            if not captcha_text:
                st.warning("Please complete the captcha.")
                return

            params = {
                "captcha_id": st.session_state.get("register_captcha_id"),
                "captcha_text": captcha_text
            }

            # Send registration request
            try:
                res = requests.post(
                    f"{BACKEND_URL}/register/user",
                    json={"email_id": email, "password": password},
                    params=params,
                    timeout=15
                )
            except Exception as e:
                st.error(f"Registration request failed: {e}")
                return

            # Parse body if possible
            try:
                resp_json = res.json()
            except:
                resp_json = None

            if res.status_code == 200:
                # Backend-level success/failure
                if resp_json and resp_json.get("status") == "success":
                    st.success("🎉 Registration successful! You can now log in.")

                    # cleanup state and go back to login
                    st.session_state.pop("register_captcha_id", None)
                    st.session_state.pop("register_captcha_img", None)
                    st.session_state.show_register = False
                    st.rerun()

                else:
                    # registration failed — show error and refresh captcha
                    msg = (
                        resp_json.get("message")
                        if resp_json else res.text
                    )
                    st.session_state["register_error"] = f"Registration failed: {msg}"

                    st.session_state.pop("register_captcha_id", None)
                    st.session_state.pop("register_captcha_img", None)
                    st.rerun()

            else:
                # Non-200 status → show error & refresh captcha
                msg = (
                    resp_json.get("detail") or
                    resp_json.get("message") if resp_json else
                    res.text
                )
                st.session_state["register_error"] = f"Registration failed: {msg}"

                st.session_state.pop("register_captcha_id", None)
                st.session_state.pop("register_captcha_img", None)
                st.rerun()

    # ------------------------------------------------------
    # Back to login
    # ------------------------------------------------------
    if st.button("⬅️ Back to Login"):
        st.session_state.show_register = False
        # cleanup captcha
        st.session_state.pop("register_captcha_id", None)
        st.session_state.pop("register_captcha_img", None)
        st.rerun()


def login_page():
    st.title("🔐 Login to Continue")
    # ------------------------------------------------------
    # Show any previous error stored across reruns
    # ------------------------------------------------------
    if "login_error" in st.session_state and st.session_state["login_error"]:
        st.error(st.session_state.pop("login_error"))

    # ------------------------------------------------------
    # Login form
    # ------------------------------------------------------
    with st.form("login_form"):
        email = st.text_input("Email ID")
        password = st.text_input("Password", type="password")

        # --------------------------------------------------
        # Conditionally show captcha ONLY if both fields filled
        # --------------------------------------------------
        if email and password:

            # ensure captcha exists
            if "login_captcha_id" not in st.session_state:
                if not load_captcha("login"):
                    st.stop()

            captcha_id = st.session_state.get("login_captcha_id")
            captcha_img_bytes = st.session_state.get("login_captcha_img")

            # Show captcha image
            try:
                captcha_img = Image.open(io.BytesIO(captcha_img_bytes))
                st.image(captcha_img, caption="Please enter the text shown above")
            except:
                st.warning("Failed to render captcha image.")

            # Captcha input box
            captcha_text = st.text_input("Captcha", placeholder="Enter the text")

            # Refresh button
            if st.form_submit_button("🔄 Refresh Captcha"):
                st.session_state.pop("login_captcha_id", None)
                st.session_state.pop("login_captcha_img", None)
                load_captcha("login")
                st.rerun()

        else:
            captcha_text = None  # not required yet

        # Final submit button
        submit = st.form_submit_button("Login")

        # --------------------------------------------------
        # Login submit logic
        # --------------------------------------------------
        if submit:
            if not email or not password:
                st.warning("Please fill in both email and password.")
                return

            # If captcha is required but empty
            if email and password and not captcha_text:
                st.warning("Please complete the captcha.")
                return

            # Build params only if captcha was shown
            params = {}
            if email and password:
                params = {
                    "captcha_id": st.session_state.get("login_captcha_id"),
                    "captcha_text": captcha_text
                }

            # Perform login request
            try:
                res = requests.post(
                    f"{BACKEND_URL}/api/v1/token",
                    data={"username": email, "password": password},
                    params=params,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=15,
                )
            except Exception as e:
                st.error(f"Login failed: {e}")
                return

            # Parse response
            if res.status_code == 200:
                token_data = res.json()
                st.session_state.auth_token = token_data["access_token"]
                st.session_state.username = email

                # Cleanup captcha
                st.session_state.pop("login_captcha_id", None)
                st.session_state.pop("login_captcha_img", None)

                st.success("✅ Login successful!")
                st.rerun()
            else:
                # Capture error and refresh captcha
                try:
                    body = res.json()
                    msg = body.get("detail") or body.get("message") or str(body)
                except:
                    msg = res.text or "Invalid login."

                st.session_state["login_error"] = f"Login failed: {msg}"

                # Regenerate captcha next rerun
                st.session_state.pop("login_captcha_id", None)
                st.session_state.pop("login_captcha_img", None)

                st.rerun()

    # ------------------------------------------------------
    # Register button
    # ------------------------------------------------------
    if st.button("📝 Register"):
        # Cleanup captcha when leaving page
        st.session_state.pop("login_captcha_id", None)
        st.session_state.pop("login_captcha_img", None)
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
            background: black;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 22px 22px 22px 6px;
            margin: 0.5rem 0;
            border: 1px solid #d1d5db;
            max-width: 85%;
            margin-right: auto;
        }

        .you-label {
            color: #7fffd4;
            font-weight: 900;
        }

        /* User on right */
        .chat-message.user {
            background: black;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 22px 22px 22px 6px;
            margin: 0.5rem 0;
            border: 1px solid #bfdbfe;
            max-width: 45%;
            margin-left: auto;
        }

        .assistant-label {
            color: #0066ff;
            font-weight: 900;
        }

        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 0.9rem;
            margin-top: 3rem;
            padding: 2rem 0;
            border-top: 1px solid #e5e7eb;
        }

        .stSelectbox > div > div {
            border-radius: 8px;
            border: 1px solid #d1d5db;
        }

        .section-header {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #ffffff;
        }

        textarea {
            width: 100% !important;
            min-height: 40px !important;
            color: #f2f2f2 !important;
            font-size: 16px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="section-header">💬 Policy Assistant Chat</div>', unsafe_allow_html=True)

    # Session state initialization
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Department selector
    chat_department = st.selectbox("Your Department", ["finance", "hr", "legal", "operations"], key="chat_dept")

    # Clear history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Display chat messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for user_msg, assistant_msg in st.session_state.chat_history:
        st.markdown(
            f"""
            <div class='chat-message user'>
                <div class='you-label'>You</div>
                <div>{user_msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='chat-message assistant'>
                <div class='assistant-label'>Assistant</div>
                <div>{assistant_msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- INPUT FORM ---
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask a question:",
            placeholder="Type here...",
            label_visibility="collapsed",
            key="user_input_text"
        )
        submitted = st.form_submit_button("📤 Send")

    if submitted:
        if not user_input.strip():
            st.warning("Please enter a question.")
        else:
            chat_box = st.empty()
            full_response = ""

            headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}

            try:
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
                                    f"""
                                    <div class='chat-message assistant'>
                                        <div class='assistant-label'>Assistant</div>
                                        <div>{full_response}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                # Save history
                st.session_state.chat_history.append((user_input, full_response))
                st.rerun()

            except requests.exceptions.RequestException as e:
                st.error(f"Network error: {e}")


def render_about_page():
    """Render about page"""
    st.markdown('<div class="section-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    ## This RAG-based assistant helps you interact with company policy documents.
    - 📤 Upload and process policies  
    - 💬 Chat for contextual answers  
    - 🔐 Department-based access controls  
    """)


def render_sidebar_logo():
    """Render logo and app title in the sidebar (ChatGPT-style)"""
    # Encode your local image to base64
    def get_base64_image(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string

    # Replace "path/to/your/logo.png" with your actual file path
    logo_base64 = get_base64_image("images/logo_v3.png")

    st.sidebar.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 0.5rem 0.5rem 1rem 0.5rem;
            border-bottom: 4px solid #374151;
        ">
            <img src="data:image/png;base64,{logo_base64}"
                 alt="🏢 Klar"
                 width="40"
                 style="border-radius: 1px;">
            <h2 style="color: white; font-weight: 700; font-size: 1.2rem; margin: 0;">
                HR Policy Assistant
            </h2>
        </div>
    """, unsafe_allow_html=True)



def main():
    """Main application function"""
    st.markdown(
        '<div class="main-header" style="color: lightblue;">🏢 HR Terms & Conditions Policy Assistant</div>',
        unsafe_allow_html=True
    )

    # Sidebar navigation
    render_sidebar_logo()
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
