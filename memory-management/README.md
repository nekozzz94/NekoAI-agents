## Setup run local:  
### 1. Start ollama
[ollama docs](https://docs.ollama.com/quickstart)
```bash
#start ollama service
sudo systemctl start ollama

#pull llama3.2:3b
ollama pull llama3.2:3b

```
### 2. Install libs
```
python3 -m venv venv
source venv/bin/activate
pip3 install -U langchain langchain-ollama
```

## Memory management techniques:  
### 1. Named-entity recognition (NER):  
Think of Named-Entity Recognition (NER) as an automated highlighter for computers.  

When a human reads a sentence, our brains instantly pick out important pieces of information—like names of people, companies, dates, or prices. Computers, however, just see a flat string of characters.  

NER is the technology that teaches a computer to scan a piece of text and "highlight" or tag specific categories, such as:
```
Names of People: (e.g., "Neko Chan")
Organizations: (e.g., "Lazzy farm" or "Happy land")
Locations: (e.g., "Earth" or "Moon")
Dates & Times: (e.g., "October 5th")
Money: (e.g., "1 fish")
```

#### **USE CASE 1:**  
Integrating Named-Entity Recognition (NER) into an HR Agent Chat transforms it from a generic chatbot into an active assistant that can safely handle backend operations like booking time off, updating employee files, or answering policy questions.  

In an HR chat context, NER acts as the Perception Layer. When an employee types a message, the NER parser extracts critical variables (like dates, benefit plans, or locations) so the agent knows precisely what database or API tool to call.

*Step-by-Step Integration Architecture:*  
- User Message: The employee types a request in the chat interface.

- NER Extraction Layer: The agent intercepts the message and forces a structured JSON extraction (using a local model like Llama 3.2 or Claude) to identify specific HR entities.

- Intent & Policy Guardrails: The agent evaluates the extracted entities to check permissions (e.g., Does this user have enough PTO balance for these dates?).

- Action Execution: The clean structured data triggers an API call to your HRIS (like Workday, BambooHR, or Gusto).

[Python code](./HR-NER.py)