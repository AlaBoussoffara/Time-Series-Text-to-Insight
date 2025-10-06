Time-Series-Text-To-Insight

Setup steps:
run "pip install -r requirements" (or "conda env create -f environment.yml")
run "chainlit create-secret"
paste output in .env as CHAINLIT_AUTH_SECRET
.env file: set up USE_MODEL from ["mistral-cloud","mistral-ollama"]
              - mistral-cloud: setup MISTRAL_CLOUD_MODEL and MISTRAL_API_KEY
              - mistral-ollama: setup MISTRAL_LOCAL_MODEL

Conversation history:
- Start the app with `chainlit run app.py` and log in.
- Each new chat automatically appears in the left sidebar; click any entry to reopen that conversation.
- Use the "+ New chat" button (top of the sidebar) to begin a fresh thread; prior chats stay listed for quick switching.
- Sidebar items use the first user message as a title. If you need a custom name, rename it directly from the Chainlit UI.
- Conversation metadata is stored locally in `.chainlit_memory/chat_data.db`; delete the file if you ever want a clean slate.
- Configure memory scope via `CHAT_MEMORY_SCOPE` in `.env`: keep the default `conversation` to restrict context to the active thread, or set to `user` (alias `all`) to aggregate every stored message under your login before each reply. Restart the app after changing this setting.
