<<<<<<< HEAD
<<<<<<< HEAD
# AI-HRPolicyResoningSystem
Our AI work and open projects
=======
📘 Secure RAG Travel & Expense Policy Assistant – Full Documentation
====================================================================

Overview
--------

This project implements a **FastAPI-based proof-of-concept (POC) service** for securely answering employee queries about a company’s Travel & Expense (T&E) policies using **Retrieval Augmented Generation (RAG)**.

It is designed with **enterprise-grade security features**:

*   🔐 Role-Based Access Control (RBAC)
    
*   🛡️ Prompt Injection Detection
    
*   📜 PII Redaction
    
*   🚦 Output Guardrails
    
*   📝 Comprehensive Audit Logging
    

The service uses **FAISS** for vector similarity search, **Sentence Transformers** for embeddings, and **Ollama** as the LLM backend.

Application Entry Point
-----------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   secure_rag = SecureRAGApp()  app = secure_rag.app   `

*   **Entry point**: The SecureRAGApp class creates and orchestrates all components.
    
*   **Exported app**: app is the FastAPI instance.
    
*   uvicorn app:app --reload --port 8000
    

High-Level Architecture
-----------------------

The system is composed of 6 major components:

1.  **UserManager** – handles authentication & RBAC.
    
2.  **SecurityManager** – applies PII redaction, prompt injection detection, and output filtering.
    
3.  **DocumentProcessor** – chunks, embeds, indexes, and retrieves policy documents.
    
4.  **LLMClient** – builds prompts and queries Ollama models.
    
5.  **AuditLogger** – logs all interactions for compliance & monitoring.
    
6.  **SecureRAGApp** – integrates all of the above into FastAPI endpoints.
    

Configuration
-------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   OLLAMA_URL      = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")  EMBED_MODEL     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  AUDIT_LOG_PATH  = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")  INDEX_PATH      = os.getenv("INDEX_PATH", "index/faiss.index")  DOCSTORE_PATH   = os.getenv("DOCSTORE_PATH", "index/docstore.json")   `

*   Allows overriding default config via environment variables.
    
*   Creates logs/ and index/ directories if they don’t exist.
    
*   Defines the **system prompt**:
    
    *   Constrains answers to context only.
        
    *   Prohibits exposing policies verbatim or developer/system instructions.
        

1\. User Management & RBAC – UserManager
----------------------------------------

### Purpose

Provides a **proof-of-concept role-based access control system**.In production, this would be replaced with a **user directory, database, and token-based authentication**.

### Users

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   USERS = {      "u-employee": {          "role": "employee",          "allowed_departments": ["finance"]      },      "u-contractor": {          "role": "contractor",          "allowed_departments": []      },      "u-finance": {          "role": "finance-analyst",          "allowed_departments": ["finance"]      },  }   `

### Key Method

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   get_user(user_id: str) -> Dict[str, Any]   `

*   Returns the user profile (role, allowed\_departments).
    
*   Raises 403 Forbidden if user is unknown.
    

2\. Security & Privacy – SecurityManager
----------------------------------------

### Purpose

Protects against **data leaks, prompt injection attacks, and unsafe outputs**.

### Features

*   **PII Redaction**:Uses regex to replace emails/phone numbers with \[REDACTED\_EMAIL\] / \[REDACTED\_PHONE\].
    
*   **Prompt Injection Detection**:Blocks queries containing patterns like "ignore previous instructions", "show the policy verbatim", "exfiltrate", etc.
    
*   **Output Filtering**:Prohibits banned content in answers ("social security number", "credit card number", "password:").
    

3\. Document Processing – DocumentProcessor
-------------------------------------------

### Purpose

Handles **document ingestion, embedding, indexing, and retrieval**.

### Workflow

1.  **Chunking**:Splits documents into overlapping word chunks (600 words per chunk, 80 words overlap).→ Ensures relevant info isn’t split across chunks.
    
2.  **Embedding**:Uses SentenceTransformer (default: all-MiniLM-L6-v2) to create vector embeddings.
    
3.  **Indexing**:Stores embeddings in **FAISS IndexFlatIP** (inner product, equivalent to cosine similarity with normalized vectors).Metadata is stored in a parallel **docstore.json**.
    
4.  **Search**:Retrieves the top-k chunks most relevant to the query.Results are filtered using **RBAC** (user’s allowed departments).
    

### Key Methods

*   chunk\_text(text, chunk\_size=600, overlap=80) → \[chunks\]
    
*   build\_index(chunks) → Builds FAISS index.
    
*   search\_index(query, k, allowed\_departments) → Returns filtered results.
    
*   ingest\_file(path, department) → Reads, chunks, embeds, indexes.
    
*   load\_index() / \_save\_index() → Persist/reload FAISS index.
    

4\. LLM Integration – LLMClient
-------------------------------

### Purpose

Handles **communication with Ollama** and **prompt construction**.

### Methods

1.  call\_ollama(prompt, model="llama3:8b", temperature=0.1, max\_tokens=350)
    
    *   Sends JSON payload to Ollama REST API.
        
    *   Returns generated text.
        
2.  build\_prompt(user\_query, contexts)
    
    *   Builds final LLM prompt:
        
        *   Includes SYSTEM\_PROMPT (rules & constraints).
            
        *   Adds retrieved document chunks with source attribution.
            
        *   Appends user’s question.
            

5\. Audit Logging – AuditLogger
-------------------------------

### Purpose

Provides **tamper-resistant audit logging** for compliance & monitoring.

### Features

*   Anonymizes user IDs using **SHA-256 hash truncated to 12 chars**.
    
*   Logs in **JSON Lines format (JSONL)** for easy parsing.
    
*   Adds **timestamps (epoch seconds)** to each event.
    

### Example Logged Events

*   prompt\_injection\_blocked
    
*   chat\_no\_results
    
*   chat\_success
    

6\. FastAPI Application – SecureRAGApp
--------------------------------------

### Purpose

Orchestrates all components and defines **API endpoints**.

### Lifecycle

*   **Startup**: Loads FAISS index + docstore if available.
    
*   **Shutdown**: (No cleanup needed, but hook is in place).
    

### Endpoints

#### 1\. /ingest (POST)

*   { "path": "docs/travel\_policy.txt", "department": "finance"}
    
*   **Process**:Reads file → chunks → embeds → indexes.
    
*   { "status": "success", "message": "Document ingested successfully", "total\_chunks": 42}
    

#### 2\. /chat (POST)

*   { "user\_id": "u-employee", "query": "Can I expense taxi rides to the airport?", "k": 4, "model": "llama3:8b"}
    
*   **Process**:
    
    1.  Authenticate user via UserManager.
        
    2.  Block if prompt injection detected.
        
    3.  Redact PII.
        
    4.  Retrieve relevant chunks with RBAC filtering.
        
    5.  Build prompt and call Ollama.
        
    6.  Apply output guardrails.
        
    7.  Log interaction via AuditLogger.
        
*   { "answer": "Yes, taxi rides to the airport are reimbursable if related to business travel.", "sources": \[ { "source": "docs/travel\_policy.txt", "department": "finance", "score": 0.82 } \]}
    

Request Flow Summary
--------------------

1.  **User sends a query** → /chat.
    
2.  **Authentication & RBAC** → UserManager.get\_user().
    
3.  **Prompt Injection Check** → SecurityManager.looks\_like\_injection().
    
4.  **PII Redaction** → SecurityManager.redact\_pii().
    
5.  **Semantic Search** → DocumentProcessor.search\_index().
    
6.  **Prompt Build & LLM Call** → LLMClient.build\_prompt() + call\_ollama().
    
7.  **Output Guardrails** → SecurityManager.violates\_output\_policy().
    
8.  **Audit Logging** → AuditLogger.write\_audit\_log().
    
9.  **Response returned** → JSON with answer + sources.
    

Key Security Measures
---------------------

*   🚫 **RBAC filtering** ensures contractors don’t see finance docs.
    
*   🔎 **Prompt injection detection** blocks malicious queries.
    
*   ✂️ **PII redaction** removes sensitive user info.
    
*   ⚠️ **Output guardrails** prevent leaking secrets.
    
*   📝 **Audit logs** provide a full trace of all interactions.
>>>>>>> 82f2bc9 (Initial commit)
=======
# AI-HRPolicyResoningSystem
Our AI work and open projects
>>>>>>> 625ce6cf4fd2c50e99d1ddd8b704e60a8673d3c9
