## 1. Start ollama
[ollama docs](https://docs.ollama.com/quickstart)
```bash
#start ollama service
sudo systemctl start ollama

#pull llama3.2:3b
ollama pull llama3.2:3b

```
## 2. Install libs
```
python3 -m venv venv
source env/bin/activate
pip3 install -U langchain langchain-ollama
```
