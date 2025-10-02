Time-Series-Text-To-Insight

Setup steps:
run "pip install -r requirements" (or "conda env create -f environment.yml")
run "chainlit create-secret"
paste output in .env as CHAINLIT_AUTH_SECRET
.env file: set up USE_MODEL from ["mistral-cloud","mistral-ollama"]
              - mistral-cloud: setup MISTRAL_CLOUD_MODEL and MISTRAL_API_KEY
              - mistral-ollama: setup MISTRAL_LOCAL_MODEL