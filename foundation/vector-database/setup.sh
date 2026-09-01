#/usr/bin/sh -e

python3 -m venv venv
source venv/bin/activate
pip install langchain-postgres psycopg[binary] langchain-google-genai
pip install pypdf
pip install "unstructured[pdf]"
pip install pdfplumber

#Python: Select Interpreter