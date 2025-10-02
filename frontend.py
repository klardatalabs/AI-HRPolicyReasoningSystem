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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 3px solid #3b82f6;
    }

    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1.5rem;
        padding: 0.5rem 0;
        border-left: 4px solid #3b82f6;
        padding-left: 1rem;
    }

    .info-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .success-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #bbf7d0;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .error-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #fecaca;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .chat-message-user {
        background: black; /* Changed from linear-gradient */
        color: white;    /* Added for readability against a black background */
        padding: 1rem 1.5rem;
        border-radius: 22px 22px 22px 6px;
        margin: 0.5rem 0;
        border: 1px solid #bfdbfe;
        max-width: 85%;
        margin-left: auto;
    }
    
    .you-label {
        color: #7fffd4;
        font-weight: 900;
    }

    .chat-message-assistant {
        background: black;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 22px 22px 22px 6px;
        margin: 0.5rem 0;
        border: 1px solid #d1d5db;
        max-width: 85%;
        margin-right: auto;
    }
    
    .assistant-label {
        color: #0066ff;
        font-weight: 900;
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

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #e5e7eb;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }

    textarea {
        width: 100% !important;   /* Full width of container */
        min-height: 40px !important;  /* Adjust height */
        color: #f2f2f2 !important;    /* Custom text color (DodgerBlue) */
        font-size: 16px !important;   /* Optional: font size */
    }
</style>
""", unsafe_allow_html=True)

# Backend API configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002")

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
    """
    Upload file to backend /ingest which expects:
      - file: UploadFile = File(...)
      - department: str = Form(...)
    """
    try:
        # Make a file-like object from the uploaded file bytes
        file_bytes = uploaded_file.getvalue()          # bytes
        file_obj = io.BytesIO(file_bytes)              # file-like

        # Build the tuple as (filename, fileobj, content_type)
        files = {
            "file": (uploaded_file.name, file_obj, uploaded_file.type or "application/octet-stream")
        }
        data = {"department": department}

        # Do NOT set headers['Content-Type'] manually
        response = requests.post(
            f"{BACKEND_URL}/ingest",
            files=files,
            data=data,
            timeout=120  # give enough time for ingest + embeddings
        )

        # raise_for_status will raise HTTPError for 4xx/5xx
        response.raise_for_status()
        return True, response.json()

    except requests.exceptions.HTTPError:
        # include server returned body for debugging
        return False, f"Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"



def chat_with_assistant(user_query, department, model=LLM_MODEL):
    """Call the backend chat API"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "user_id": "u-employee",
                "query": user_query,
                "k": 4,
                "model": model
            },
            timeout=30
        )

        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Error {response.status_code}: {response.text}"

    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def render_ingest_page():
    """Render the document ingestion page"""
    st.markdown('<div class="section-header">📤 Document Ingestion</div>', unsafe_allow_html=True)

    # Info card
    st.markdown("""
    <div class="info-card">
        <h4 style="color: #374151; margin-bottom: 0.5rem;">📋 Upload Policy Documents</h4>
        <p style="color: #6b7280; margin: 0;">
            Upload PDF or TXT documents to add them to the knowledge base. 
            Select the appropriate department for access control.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    department = st.selectbox("Department", ["finance", "hr", "legal", "operations"])

    if uploaded_file is not None:
        st.markdown("##### File Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size:,} bytes")
        with col3:
            st.metric("File Type", uploaded_file.type)

        if st.button("🚀 Upload & Ingest", type="primary", use_container_width=True):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            data = {"department": department}
            try:
                response = requests.post(f"{BACKEND_URL}/ingest", files=files, data=data, timeout=120)
                if response.status_code == 200:
                    st.markdown(f"""
                    <div class="success-card">
                        <h4 style="color: #16a34a; margin-bottom: 0.5rem;">✅ Processing Complete</h4>
                        <p style="color: #15803d; margin: 0;">
                            File '{uploaded_file.name}' uploaded and ingested successfully!
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-card">
                        <h4 style="color: #dc2626; margin-bottom: 0.5rem;">❌ Processing Failed</h4>
                        <p style="color: #7f1d1d; margin: 0;">{response.text}</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="error-card">
                    <h4 style="color: #dc2626; margin-bottom: 0.5rem;">❌ Connection Error</h4>
                    <p style="color: #7f1d1d; margin: 0;">{e}</p>
                </div>
                """, unsafe_allow_html=True)


def chat_stream(user_query, user_id="u-employee"):
    """
    Generator that yields incremental (cumulative) text from the backend /chat endpoint.
    Keeps the original 2-arg signature: (user_query, user_id).
    Handles:
      - streaming text/plain chunks (RAG LLM output)
      - non-streaming application/json responses (refusals / short messages)
    Yields strings (not bytes). On errors, yields a string starting with ❌.
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"user_id": user_id, "query": user_query, "k": 4, "model": LLM_MODEL},
            stream=True,
            timeout=120
        )

        # Raise early for non-2xx responses
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        # If backend returned JSON (e.g. refusal message), parse & yield that once
        if "application/json" in content_type:
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    if "message" in payload and isinstance(payload["message"], str):
                        yield payload["message"]
                    else:
                        # yield the first string value found in the dict (fallback)
                        for v in payload.values():
                            if isinstance(v, str):
                                yield v
                                break
                        else:
                            yield json.dumps(payload)
                else:
                    yield str(payload)
            except ValueError:
                # not valid JSON for some reason
                yield response.text
            return

        # Otherwise treat as streaming plain text. Yield cumulative text so the UI
        # placeholder can show the full growing answer each iteration.
        collected = ""
        for chunk in response.iter_content(chunk_size=128, decode_unicode=True):
            if not chunk:
                continue
            collected += chunk
            # Normalize occasional carriage returns that can break markdown display
            yield collected

    except requests.exceptions.HTTPError as e:
        # try to extract a helpful error body
        error_detail = "Unknown error"
        if hasattr(e, "response") and e.response is not None:
            try:
                body = e.response.json()
                if isinstance(body, dict):
                    if "detail" in body:
                        error_detail = body["detail"]
                    elif "message" in body:
                        error_detail = body["message"]
                    else:
                        error_detail = json.dumps(body)
                else:
                    error_detail = str(body)
            except Exception:
                try:
                    error_detail = e.response.text
                except Exception:
                    error_detail = str(e)
        yield f"❌ **Error {getattr(e.response, 'status_code', 'Unknown')}:** {error_detail}"

    except requests.exceptions.RequestException as e:
        yield f"❌ **Connection Error:** {str(e)}"

    except Exception as e:
        yield f"❌ **Unexpected Error:** {str(e)}"


def render_chat_page():
    st.markdown('<div class="section-header">💬 Policy Assistant Chat</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize a counter to force text area recreation
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    chat_department = st.selectbox("Your Department", ["finance", "hr", "legal", "operations"], key="chat_dept")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Display chat history
    for user_msg, assistant_msg in st.session_state.chat_history:
        st.markdown(
            f"<div class='chat-message-user'><span class='you-label'>You:</span> \n\n{user_msg}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='chat-message-assistant'><span class='assistant-label'>Assistant:</span> \n\n{assistant_msg}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Use a dynamic key that changes after submission to force widget recreation
    user_input = st.text_area(
        label="Ask anything:",  # still required internally
        placeholder="Ask anything...",
        height=45,
        key=f"query_input_{st.session_state.input_key}",
        label_visibility="collapsed"
    )

    # if st.button("📤 Send Message"):
    #     if user_input.strip():
    #         placeholder = st.empty()
    #         full_answer = ""
    #         for partial in chat_stream(user_input):
    #             placeholder.markdown(
    #                 f"<div class='chat-message-assistant'><strong>Assistant:</strong> {partial}</div>",
    #                 unsafe_allow_html=True
    #             )
    #             full_answer = partial
    #         st.session_state.chat_history.append((user_input, full_answer))
    #
    #         # Increment the key counter to force a new text area widget
    #         st.session_state.input_key += 1
    #         st.rerun()
    #     else:
    #         st.warning("⚠️ Please enter a question before sending.")

    if st.button("📤 Send Message"):
        if user_input.strip():
            placeholder = st.empty()
            full_answer = ""
            for partial in chat_stream(user_input):
                # Check if the response is an error (starts with ❌)
                if partial.startswith("❌"):
                    placeholder.markdown(
                        f"<div class='error-card'>{partial}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    placeholder.markdown(
                        f"<div class='chat-message-assistant'><span class='assistant-label'>Assistant:</span> \n\n{partial}</div>",
                        unsafe_allow_html=True
                    )
                full_answer = partial

            st.session_state.chat_history.append((user_input, full_answer))
            st.session_state.input_key += 1
            st.rerun()
        else:
            st.warning("⚠️ Please enter a question before sending.")



def render_about_page():
    st.markdown('<div class="section-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    ## ****This RAG based app helps you interact with your company's policy documents.****
    
    Features:
    - 📤 ****Document ingestion****
    - 💬 ****AI-powered chat with contextual answers****  
    - 🔐 ****Department-based access controls****  
    """)


def main():
    """Main application function"""
    st.markdown(
        '<div class="main-header" style="color: lightblue; font-family: sans-serif;">🏢 HR Terms & Conditions Policy Assistant</div>',
        unsafe_allow_html=True
    )

    # Sidebar navigation
    st.sidebar.markdown("## 📋 Navigation")

    # Navigation buttons
    if st.sidebar.button("📤 Ingest Documents", use_container_width=True):
        st.session_state.page = "ingest"

    if st.sidebar.button("💬 Chat with Assistant", use_container_width=True):
        st.session_state.page = "chat"
    if st.sidebar.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "about"
        st.rerun()

    # Initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "ingest"

    # Render the selected page
    if st.session_state.page == "ingest":
        render_ingest_page()
    elif st.session_state.page == "chat":
        render_chat_page()
    elif st.session_state.page == "about":
        render_about_page()

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🔐 Secure RAG T&C Policy Assistant | Powered by FastAPI & Streamlit</p>
        <p style="font-size: 0.8rem; color: #9ca3af;">
            All conversations are logged for compliance. Access is controlled by department permissions.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()