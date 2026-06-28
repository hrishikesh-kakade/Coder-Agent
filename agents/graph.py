from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_groq.chat_models import ChatGroq
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from agents.prompts import *
from agents.states import *
from agents.tools import write_file, read_file, get_current_directory, list_files

_ = load_dotenv()

set_debug(True)
set_verbose(True)

llm = ChatGroq(model="openai/gpt-oss-120b") # Or your preferred model

# agents/graph.py

# agents/graph.py

def planner_agent(state: dict) -> dict:
    """Converts user prompt into a structured Plan."""
    
    # 1. If we have a Task Plan or Coder State, we are way past planning.
    if state.get("task_plan") or state.get("coder_state"):
        print("✅ Plan/TaskPlan exists. Skipping Planner.")
        # Pass existing plan forward to keep state consistent
        return {"plan": state.get("plan")}

    # 2. Standard Idempotency
    if state.get("plan"):
        print("✅ Plan exists. Passing forward.")
        return {"plan": state["plan"]}

    # 3. Generate Plan
    user_prompt = state["user_prompt"]
    resp = llm.with_structured_output(Plan).invoke(
        planner_prompt(user_prompt)
    )
    if resp is None:
        raise ValueError("Planner did not return a valid response.")
    return {"plan": resp}


def architect_agent(state: dict) -> dict:
    """Creates TaskPlan from Plan."""

    # 1. STRICT GUARD: If coding has started, DO NOT generate a new plan.
    # This prevents the "Adding more tasks" / "Regenerating" issue.
    if state.get("coder_state"):
        print("✅ Coder is active. Skipping Architect to prevent overwrite.")
        return {"task_plan": state.get("task_plan")}

    # 2. Idempotency: If task plan exists, just pass it through.
    if state.get("task_plan"):
        print("✅ Task Plan exists. Passing forward.")
        return {"task_plan": state["task_plan"]}

    # 3. Generate Task Plan
    plan = state.get("plan")
    if not plan:
        raise ValueError("Architect cannot run: 'plan' is missing.")

    plan_str = plan.model_dump_json() if hasattr(plan, 'model_dump_json') else str(plan)
    resp = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan_str)
    )
    if resp is None:
        raise ValueError("Architect did not return a valid response.")

    return {"task_plan": resp}


def coder_agent(state: dict) -> dict:
    """LangGraph tool-using coder agent."""
    
    # Safely retrieve coder_state
    coder_state = state.get("coder_state")
    
    # Initialize ONLY if it doesn't exist
    if coder_state is None:
        if not state.get("task_plan"):
             raise ValueError("Coder cannot run: 'task_plan' is missing.")
        # Start at 0
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    
    # Check if we are already done
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    # --- EXECUTE TASK ---
    current_task = steps[coder_state.current_step_idx]
    print(f"🔹 Executing Step {coder_state.current_step_idx + 1}/{len(steps)}: {current_task.filepath}")

    existing_content = read_file.invoke(current_task.filepath)

    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    coder_tools = [read_file, write_file, list_files, get_current_directory]
    react_agent = create_react_agent(llm, coder_tools)

    react_agent.invoke({"messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": user_prompt}]})

    # --- INCREMENT STEP ---
    coder_state.current_step_idx += 1
    
    # Return the updated state
    return {"coder_state": coder_state}


# --- GRAPH DEFINITION ---
graph = StateGraph(dict)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")

def should_continue(state):
    # Check if we are done
    if state.get("status") == "DONE":
        return END
    return "coder"

graph.add_conditional_edges(
    "coder",
    should_continue,
    {"END": END, "coder": "coder"}
)

graph.set_entry_point("planner")
agent = graph.compile()


if __name__ == "__main__":
    result = agent.invoke({"user_prompt": ""},
                          {"recursion_limit": 100})
    print("Final State:", result)