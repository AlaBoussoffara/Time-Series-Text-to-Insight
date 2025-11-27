.PHONY: run chainlit

run: chainlit

chainlit:
	PYTHONPATH="$(CURDIR)" .venv/bin/chainlit run ui/app.py -w
