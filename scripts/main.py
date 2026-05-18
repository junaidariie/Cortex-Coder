from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from scripts.load_model import get_model
from typing import TypedDict, Literal, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from scripts.prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    CODE_AGENT_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT
)

# =========================================================
# LLM
# =========================================================
llm = get_model()


# =========================================================
# STATE
# =========================================================
class ChatState(TypedDict):
    user_input: Annotated[list[BaseMessage], add_messages]
    intent: Optional[Literal["coding", "general_chat"]]
    task_type: Optional[Literal["code_generation", "debugging", "explanation"]]
    language: Optional[Literal["python", "javascript", "cpp", "sql", "java", "unknown"]]
    generated_output: Optional[str]
    is_code: Optional[bool]
    retry_count: int


# =========================================================
# HELPERS
# =========================================================
def get_last_user_message(state: ChatState) -> str:
    messages = state["user_input"]

    if not messages:
        return ""

    return messages[-1].content


# =========================================================
# INTENT CLASSIFIER
# =========================================================
def classify_intent(state: ChatState):
    query = get_last_user_message(state).lower()

    coding_keywords = [
        "code", "python", "javascript", "java", "c++", "sql",
        "bug", "debug", "fix", "function", "algorithm",
        "class", "implement", "build", "api", "query",
        "optimize", "refactor", "script", "program"
    ]

    intent = "coding" if any(word in query for word in coding_keywords) else "general_chat"

    return {"intent": intent}


def route_intent(state: ChatState):
    if state["intent"] == "general_chat":
        return "general_chat"

    return "coding"


# =========================================================
# GENERAL CHAT
# =========================================================
def handle_general_chat(state: ChatState):
    query = get_last_user_message(state)

    messages = [
        SystemMessage(content=GENERAL_CHAT_SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]

    response = llm.invoke(messages)

    return {
        "generated_output": response.content
    }


# =========================================================
# TASK CLASSIFIER
# =========================================================
def classify_task(state: ChatState):
    query = get_last_user_message(state).lower()

    debugging_keywords = [
        "fix", "debug", "error", "broken",
        "issue", "not working", "exception", "traceback"
    ]

    explanation_keywords = [
        "explain", "what is", "how does", "why", "difference between"
    ]

    if any(word in query for word in debugging_keywords):
        task = "debugging"

    elif any(word in query for word in explanation_keywords):
        task = "explanation"

    else:
        task = "code_generation"

    return {"task_type": task}


# =========================================================
# LANGUAGE DETECTOR
# =========================================================
def detect_language(state: ChatState):
    query = get_last_user_message(state).lower()

    if any(k in query for k in ["python", "def ", "import ", "print("]):
        lang = "python"

    elif any(k in query for k in ["javascript", "js", "function ", "console.log", "const ", "let "]):
        lang = "javascript"

    elif any(k in query for k in ["c++", "#include", "std::", "cout"]):
        lang = "cpp"

    elif any(k in query for k in ["java", "public class", "system.out"]):
        lang = "java"

    elif any(k in query for k in ["sql", "select ", "insert ", "update ", "delete ", "join "]):
        lang = "sql"

    else:
        lang = "unknown"

    return {"language": lang}


# =========================================================
# GENERATOR
# =========================================================
def generate_code(state: ChatState):
    query = get_last_user_message(state)

    task = state["task_type"]
    language = state["language"]

    messages = [
        SystemMessage(content=CODE_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Task Type: {task}
Requested Language: {language}

User Request:
{query}
""")
    ]

    response = llm.invoke(messages)

    return {
        "generated_output": response.content
    }


# =========================================================
# OUTPUT CLASSIFIER
# =========================================================
def classify_output(state: ChatState):
    output = (state["generated_output"] or "").lower()

    code_markers = [
        "def ", "class ", "function ", "const ", "let ",
        "#include", "std::", "public class", "system.out",
        "select ", "insert ", "update ", "delete ",
        "console.log", "print(", "fn "
    ]

    is_code = any(marker in output for marker in code_markers)

    return {
        "is_code": is_code
    }


# =========================================================
# ROUTER
# =========================================================
def route_output(state: ChatState):
    if state["task_type"] == "explanation":
        return "final"

    if state["is_code"]:
        return "final"

    if state["retry_count"] >= 2:
        return "final"

    return "repair"


# =========================================================
# REPAIR
# =========================================================
def repair_code(state: ChatState):
    bad_output = state["generated_output"] or ""
    language = state["language"]

    messages = [
        SystemMessage(content=REPAIR_SYSTEM_PROMPT),
        HumanMessage(content=f"""
The previous response was expected to be executable code but was not.

Requested language:
{language}

Bad output:
{bad_output}

Return ONLY executable code.
No explanation.
No markdown.
""")
    ]

    response = llm.invoke(messages)

    return {
        "generated_output": response.content,
        "retry_count": state["retry_count"] + 1
    }


# =========================================================
# GRAPH
# =========================================================
checkpointer = MemorySaver()
builder = StateGraph(ChatState)

builder.add_node("classify_intent", classify_intent)
builder.add_node("general_chat", handle_general_chat)
builder.add_node("classify_task", classify_task)
builder.add_node("detect_language", detect_language)
builder.add_node("generate_code", generate_code)
builder.add_node("classify_output", classify_output)
builder.add_node("repair", repair_code)

builder.set_entry_point("classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "general_chat": "general_chat",
        "coding": "classify_task"
    }
)

builder.add_edge("general_chat", END)

builder.add_edge("classify_task", "detect_language")
builder.add_edge("detect_language", "generate_code")
builder.add_edge("generate_code", "classify_output")

builder.add_conditional_edges(
    "classify_output",
    route_output,
    {
        "final": END,
        "repair": "repair"
    }
)

builder.add_edge("repair", "classify_output")

graph = builder.compile(checkpointer=checkpointer)


# =========================================================
# STREAM
# =========================================================
def stream_chat_response(user_message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_input": [HumanMessage(content=user_message)],
        "intent": None,
        "task_type": None,
        "language": None,
        "generated_output": None,
        "is_code": None,
        "retry_count": 0
    }

    for chunk in graph.stream(
        initial_state,
        config=config,
        stream_mode="messages"
    ):
        yield chunk


def get_chat_metadata(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    state = graph.get_state(config)
    values = state.values

    return {
        "intent": values.get("intent"),
        "task_type": values.get("task_type"),
        "language": values.get("language"),
        "is_code": values.get("is_code"),
        "retry_count": values.get("retry_count")
    }