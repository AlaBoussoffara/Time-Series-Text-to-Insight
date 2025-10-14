# Time-Series-Text-To-Insight

## Install Dependencies
- Create a virtual environment, then run `pip install -r requirements.txt` or `conda env create -f environment.yml`.
- Run `chainlit create-secret`, then add the generated value to `.env` as `CHAINLIT_AUTH_SECRET`.

## Configure Environment Variables
Create a `.env` file at the project root and include the variables required by `llm.py`:

```ini
# Core settings
CHAINLIT_AUTH_SECRET=...
USE_PROVIDER="aws"          # aws | mistral | ollama
USE_MODEL="anthropic.claude-sonnet-4-5-20250929-v1:0"

# Provider-specific keys (set only what you need)
MISTRAL_API_KEY=          # required if USE_PROVIDER=mistral
```

`llm.py` defaults to the AWS Bedrock provider with the Claude 3.5 Sonnet model (`anthropic.claude-sonnet-4-5-20250929-v1:0`). To switch providers, update `USE_PROVIDER` and `USE_MODEL` accordingly:
- `aws`: choose any Bedrock model available to your account (e.g., `anthropic.claude-3-haiku-20240307-v1:0` or `anthropic.claude-3-5-sonnet-20241022-v2:0`).
- `mistral`: specify a hosted model supported by Mistral’s API and supply `MISTRAL_API_KEY`.
- `ollama`: make sure your Ollama server is running locally with the requested model pulled.

## Connect AWS CLI to Bedrock
[Install AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and confirm with `aws --version`.

## Run the App
- Start Chainlit with `chainlit run app.py`.
- Log in using your Chainlit credentials; each new chat is stored locally in `.chainlit_memory/chat_data.db`.
- Conversation history is available from the sidebar; use "+ New chat" to start fresh threads.
- Control the memory scope via `CHAT_MEMORY_SCOPE` in `.env` (`conversation` for per-thread context, `user`/`all` to aggregate history). Restart the app after changing this setting.
