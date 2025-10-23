# Time-Series-Text-To-Insight

## Install Dependencies
- Create a virtual environment, then run `pip install -r requirements.txt` or `conda env create -f environment.yml`.
- Run `chainlit create-secret`, then add the generated value to `.env` as `CHAINLIT_AUTH_SECRET`.

## Configure Environment Variables
Create a `.env` file at the project root and include the variables required by `utils/llm.py` plus the PostgreSQL connection string:

```ini
# Core settings
CHAINLIT_AUTH_SECRET=...
USE_PROVIDER="aws"          # aws | mistral | ollama
USE_MODEL="anthropic.claude-sonnet-4-5-20250929-v1:0"
POSTGRES_DSN=postgresql://ts_user:strong_password@localhost:5432/ts_text_to_insight

# Provider-specific keys (set only what you need)
MISTRAL_API_KEY=          # required if USE_PROVIDER=mistral
```

`utils/llm.py` defaults to the AWS Bedrock provider with the Claude 3.5 Sonnet model (`anthropic.claude-sonnet-4-5-20250929-v1:0`). To switch providers, update `USE_PROVIDER` and `USE_MODEL` accordingly:
- `aws`: choose any Bedrock model available to your account (e.g., `anthropic.claude-3-haiku-20240307-v1:0` or `anthropic.claude-3-5-sonnet-20241022-v2:0`).
- `mistral`: specify a hosted model supported by Mistral’s API and supply `MISTRAL_API_KEY`.
- `ollama`: make sure your Ollama server is running locally with the requested model pulled.

## Set Up the Database (first run)
1. **Install PostgreSQL**  
   - Ubuntu: `sudo apt install postgresql postgresql-contrib`  
   - macOS (Homebrew): `brew install postgresql@16`

2. **Create a dedicated role and database**
   ```bash
   sudo -u postgres psql -c "CREATE ROLE ts_user WITH LOGIN PASSWORD 'strong_password';"
   createdb -U ts_user ts_text_to_insight
   ```
   Adjust names/passwords to match your security requirements.

3. **Expose the DSN**  
   - Add `POSTGRES_DSN=postgresql://ts_user:strong_password@localhost:5432/ts_text_to_insight` to `.env`.  
   - Alternatively export it in your shell before running scripts:  
     `export POSTGRES_DSN=postgresql://ts_user:strong_password@localhost:5432/ts_text_to_insight`

4. **Install the PostgreSQL driver** (if it isn’t already in your environment)  
   `pip install psycopg[binary]`

5. **Seed the database from the CSV dumps**  
   ```bash
   POSTGRES_DSN=postgresql://ts_user:strong_password@localhost:5432/ts_text_to_insight \
     python3 utils/bootstrap_postgres.py
   ```
   The script creates the required tables and bulk-loads every CSV in `database/`.

6. **Sanity check** (optional)  
   ```bash
   psql "$POSTGRES_DSN" -c "SELECT COUNT(*) FROM raw_measurements;"
   ```
   You should see ~443 k rows if the import succeeded.

## Connect AWS CLI to Bedrock
[Install AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and confirm with `aws --version`.

## Run the App
- Start Chainlit with `make run` (sets `PYTHONPATH` to the project root and runs `chainlit run ui/app.py -w` so package imports like `ui.*` resolve).
- Log in using your Chainlit credentials; each new chat is stored locally in `.chainlit_memory/chat_data.db`.
- Conversation history is available from the sidebar; use "+ New chat" to start fresh threads.
- Control the memory scope via `CHAT_MEMORY_SCOPE` in `.env` (`conversation` for per-thread context, `user`/`all` to aggregate history). Restart the app after changing this setting.
