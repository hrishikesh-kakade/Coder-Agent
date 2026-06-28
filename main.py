import streamlit as st
import os
import uuid
from agents.graph import agent  # Import your compiled graph
from agents.states import TaskPlan  # Import your Pydantic models

# --- Page Config ---
st.set_page_config(page_title="Engineering Project Planner", layout="wide")
st.title("🏗️ Human-in-the-Loop Project Planner")

# --- Initialize Session State ---
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {}  # Start empty
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "plan_approved" not in st.session_state:
    st.session_state.plan_approved = False

# --- Helper to Config ---
config = {"recursion_limit": 100, "configurable": {"thread_id": st.session_state.thread_id}}

# --- 1. Initial Project Setup ---
if not st.session_state.graph_state.get("plan"):
    st.header("Step 1: Define Project")
    user_prompt = st.text_area("Enter your project request:", 
                               placeholder="Build a modern Todo App in HTML/CSS/JS", 
                               height=100)
    
    if st.button("Generate Plan & Tasks"):
        with st.spinner("Planner & Architect are working..."):
            # We run the graph targeted ONLY at the planner and architect first
            # Note: In LangGraph, we can run until a specific node, or we can just check state.
            # Here, we invoke the full agent but will rely on interrupt logic OR 
            # we can trust the graph flows. Since your graph flows Planner -> Architect -> Coder,
            # we need to artificially stop it or check state. 
            
            # BETTER APPROACH for Streamlit: Run the first two agents manually or update the graph.
            # Assuming your graph is compiled as 'agent', let's run it.
            # To stop before Coder, we can use LangGraph's interrupt_before=["coder"] if configured,
            # OR we can just check the output if we modify the graph.
            
            # For this UI, let's assume we run the full thing BUT we want to stop.
            # Hack: We can invoke just the planner/architect functions if exposed, 
            # but using the graph is cleaner.
            
            # Let's run the graph up to the point where 'task_plan' is populated.
            # We can do this by running the graph and adding a conditional edge that stops if we want approval.
            # However, adapting your EXISTING graph:
            
            initial_state = {"user_prompt": user_prompt}
            
            # We use stream to stop after architect
            curr_state = initial_state
            for output in agent.stream(initial_state, config):
                for key, value in output.items():
                    curr_state.update(value)
                    if key == "architect":
                        # STOP HERE
                        st.session_state.graph_state = curr_state
                        st.rerun()

# --- 2. Review & Execute Tasks ---
else:
    st.header("Step 2: Implementation Mode")
    
    # --- FIX: Retrieve ALL state variables first ---
    plan = st.session_state.graph_state.get("plan")
    task_plan = st.session_state.graph_state.get("task_plan")
    coder_state = st.session_state.graph_state.get("coder_state")  # <--- This line was missing
    
    # Display the High Level Plan
    if plan:
        with st.expander("📄 View High-Level Project Plan", expanded=False):
            # Check for Pydantic v2 (model_dump) vs v1 (dict)
            if hasattr(plan, "model_dump"):
                st.json(plan.model_dump())
            else:
                st.json(plan.dict())

    steps = task_plan.implementation_steps if task_plan else []
    
    # Safely determine current index
    if coder_state:
        # Check if it's a Pydantic object or a dict (LangGraph can return either)
        if hasattr(coder_state, "current_step_idx"):
            current_idx = coder_state.current_step_idx
        else:
            current_idx = coder_state.get("current_step_idx", 0)
    else:
        current_idx = 0

    total_steps = len(steps)
    
    # Progress Bar
    progress = min(current_idx / total_steps, 1.0) if total_steps > 0 else 0
    st.progress(progress, text=f"Progress: {current_idx}/{total_steps} Tasks Completed")

    # --- COMPLETE STATE ---
    if current_idx >= total_steps:
        st.success("🎉 All implementation tasks are complete!")
        if st.button("Reset Project"):
            st.session_state.graph_state = {}
            st.rerun()

    # --- ACTIVE TASK STATE ---
    else:
        current_step = steps[current_idx]
        
        # Display ONLY the Current Task
        st.info(f"### 🚩 Current Task #{current_idx + 1}")
        
        with st.container(border=True):
            st.markdown(f"**Target File:** `{current_step.filepath}`")
            
            # --- EDITABLE INSTRUCTIONS ---
            new_instructions = st.text_area(
                "**Task Instructions (Editable):**",
                value=current_step.task_description,
                height=200,
                key=f"task_desc_{current_idx}" 
            )
            
            # UPDATE STATE: If user changed text
            if new_instructions != current_step.task_description:
                st.session_state.graph_state["task_plan"].implementation_steps[current_idx].task_description = new_instructions
        
        st.write("---")
        
        # Execution Controls
        col_manual, col_auto = st.columns([1, 1])
        
        # OPTION A: Execute Single Step
        with col_manual:
            if st.button(f"▶️ Execute Task #{current_idx + 1}", type="primary", use_container_width=True):
                with st.spinner(f"Coding {current_step.filepath}..."):
                    
                    # Pass the FULL state. The Planner/Architect will now auto-skip 
                    # because we added the checks in graph.py.
                    # The Coder will pick up exactly where 'coder_state.current_step_idx' says.
                    
                    for event in agent.stream(st.session_state.graph_state, config):
                        for k, v in event.items():
                            st.session_state.graph_state.update(v)
                        
                        # We break the loop as soon as the 'coder' node finishes ONE run.
                        if "coder" in event:
                            break 
                    
                    st.rerun()

        # OPTION B: Run All Remaining
        with col_auto:
            if st.button("⏩ Auto-Execute Remaining", use_container_width=True):
                with st.status("Agent working...", expanded=True) as status:
                    for event in agent.stream(st.session_state.graph_state, config):
                        for k, v in event.items():
                            st.session_state.graph_state.update(v)
                            # Log progress
                            if "coder" in event:
                                # Safe access to new index
                                c_state = st.session_state.graph_state.get("coder_state")
                                idx = 0
                                if c_state:
                                     idx = c_state.current_step_idx if hasattr(c_state, "current_step_idx") else c_state.get("current_step_idx", 0)
                                status.write(f"✅ Completed Task #{idx}")
                                
                    status.update(label="All Tasks Finished!", state="complete")
                st.rerun()

    # --- Optional: Sidebar Task List ---
    with st.sidebar:
        st.subheader("Task Roadmap")
        for i, step in enumerate(steps):
            icon = "✅" if i < current_idx else ("🚩" if i == current_idx else "⏳")
            st.markdown(f"{icon} **{i+1}.** `{step.filepath}`")