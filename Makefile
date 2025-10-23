.PHONY: run chainlit

run: chainlit

chainlit:
	PYTHONPATH="$(CURDIR)" chainlit run ui/app.py -w
