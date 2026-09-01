![](./img/Agent-general.png)

Before starting to implement an agent, let's go through the essential terms.

## TERMS

| # | TERM | EXPLANATION |
|---|------|-------------|
| 1 | **Agent** | An autonomous program that perceives its environment, reasons over observations, and takes actions to achieve a goal. In AI systems, an agent wraps an LLM in a loop so it can plan, act, and adapt across multiple steps rather than answering a single prompt. |
| 2 | **LLM** | *Large Language Model.* A neural network trained on massive text corpora that can generate, understand, and reason over natural language. The LLM is the "brain" of an AI agent — it decides what to think and what to do next. |
| 3 | **MCP** | *Model Context Protocol.* An open standard that defines how LLMs discover and call external tools, resources, and prompts in a uniform way. MCP lets you plug any tool into any compatible agent without writing custom integration code each time. |
| 4 | **Embedding** | A dense numeric vector that encodes the semantic meaning of text (or other data). Similar meanings produce vectors that are close together in high-dimensional space, which enables semantic search, retrieval, and long-term memory in AI systems. |
| 5 | **Tools** | Functions or APIs that an agent can invoke to interact with the outside world — web search, code execution, database queries, file I/O, and more. Tools are what turn an LLM from a text generator into a system that can act on real data. |
| 6 | **Memory** | The mechanism by which an agent retains and retrieves information. *Short-term memory* is the context window — fast, coherent, and erased when the session ends. *Long-term memory* is stored externally (e.g., a vector database) and retrieved via semantic search across sessions. |
| 7 | **Prompt** | The structured input fed to an LLM, combining the system instruction, conversation history, tool definitions, retrieved memories, and the user's current request. Because tool definitions live in the prompt, clear and precise prompt engineering directly determines how reliably the agent uses its tools. |

---

## WHAT IS AN "AI" AGENT AND WHY IS IT DIFFERENT FROM A TRADITIONAL AGENT OR AUTOMATION BOT?

**Traditional automation bots** (e.g., RPA scripts, rule-based chatbots) follow a fixed, pre-programmed sequence of steps. They are fast and predictable, but brittle: any input that deviates from the expected format breaks the script. A button moves on the screen, a field changes its name, and the bot fails.

**AI agents** replace that fixed script with an LLM reasoning loop. Instead of matching patterns, the agent interprets intent. Instead of following a predetermined path, it plans its own steps based on what it discovers at runtime.

| Dimension | Traditional Bot / Automation | AI Agent |
|-----------|------------------------------|----------|
| Decision logic | Hard-coded rules and conditions | LLM reasoning in natural language |
| Handling novel input | Fails or requires reprogramming | Adapts by reasoning about what to do |
| Step sequence | Fixed and predetermined | Determined dynamically based on observations |
| Tool selection | Explicit in the script | The model chooses which tool to call and when |
| Failure recovery | Explicit retry logic | The model can re-reason after a failed action |
| Maintenance | Update code for every new case | Update the prompt or add a tool |

In short: a traditional bot executes a recipe you wrote in advance. An AI agent writes its own recipe on the fly, using the LLM as both the chef and the decision-maker.

---

## WHAT CAN ONLY AN "AI" AGENT HELP WITH?

AI agents are the right tool when **the steps to complete a task are unknown or variable until runtime** — when you cannot write the script in advance because the answer depends on what the agent discovers along the way.

**Open-ended research**
The agent searches multiple sources, follows the most relevant leads, synthesizes conflicting information, and produces a structured answer — adapting its search strategy based on what it finds rather than following a fixed query list.

**Iterative debugging and code generation**
Write code → run it → read the error → reason about the root cause → fix → run again. Each step depends on the previous result. No hard-coded script can enumerate every possible error and its fix; the LLM can reason about them semantically.

**Multi-source data analysis**
Pull records from a database, fetch supplementary data from an API, reconcile inconsistencies, identify anomalies, and generate a narrative summary — with the agent deciding which data to pull next based on what it already knows.

**Personalized, context-sensitive workflows**
Tasks like travel planning, report generation, or customer support require understanding nuance, adapting to constraints expressed in natural language, and making judgment calls. Rule-based systems require you to enumerate every case; an AI agent infers the intent and handles cases you never explicitly coded.

**Tasks with ambiguous or evolving goals**
When the user's request is high-level ("make this codebase easier to understand"), the agent must interpret the goal, break it into sub-tasks, execute them, and determine when it is done — something a script cannot do without a human specifying every sub-step in advance.

---

## HOW AI AGENTS WORK: THE REASONING LOOP

The core mechanism is straightforward: **keep calling the LLM until the task is done**. This pattern is called **ReAct** (Reason + Act):

```
while task_not_done:
    thought  = llm.reason(context)     # What should I do next?
    action   = llm.choose_tool(thought)
    result   = tool.run(action)
    context += [thought, action, result]  # observe and remember
```

Each iteration, the model sees everything that has happened so far — the original goal, every thought, every action, every result — and decides what to do next. When it judges the task complete, it stops.

```
Goal: find the cheapest flight to Tokyo next month.

[Step 1] Thought: I should search for flights.
         Action:  web_search("flights to Tokyo next month")
         Result:  [list of flights and prices]

[Step 2] Thought: I have prices. I should filter for the lowest.
         Action:  filter(results, sort_by="price")
         Result:  ANA, ¥89,000, departing July 14

[Step 3] Thought: I have the answer.
         Action:  return("ANA on July 14, ¥89,000")
```

### Memory in the Loop

| Memory type | Where it lives | Lifespan | Use case |
|-------------|---------------|----------|----------|
| Short-term | Context window | Current session only | Reasoning over the current task |
| Long-term | Vector database | Persists across sessions | User preferences, past decisions, domain knowledge |

Long-term memory works by storing text as embeddings and retrieving the most semantically relevant chunks when needed — as a tool call, not magic.

### Common Failure Modes

- **Hallucinated tool calls** — the model invents plausible but incorrect arguments. Mitigate with strict output validation and structured error feedback.
- **Infinite loops** — the agent keeps retrying without making progress. Mitigate with hard iteration caps and loop-state logging.
- **Prompt injection** — external content (a web page, a user message) contains instructions designed to hijack the agent's behavior. Treat all tool results as untrusted data.
- **Reward hacking** — the agent satisfies the stated goal while violating unstated intent (e.g., deletes failing tests instead of fixing them). Mitigate with precise goal definitions and negative constraints.

---

## WHEN TO USE AN AGENT VS. A PIPELINE

| | Pipeline | Agent |
|---|----------|-------|
| Steps known upfront? | Yes | No |
| Steps depend on previous results? | Predictably | Unpredictably |
| Failure handling | Explicit retry logic | Model decides |
| Cost | Predictable | Variable |
| Debuggability | High | Lower |

If you can express your workflow as a directed acyclic graph with known nodes, write it as a pipeline. Use an agent when the number or order of steps is genuinely unknown until runtime.
