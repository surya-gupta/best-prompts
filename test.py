import json
from typing import Dict, List, Tuple, TypedDict, Annotated, Sequence
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# Define our state schema
class AuditState(TypedDict):
    context: str  # JSON context string
    audit_checklists: List[str]  # Original list of audit checklist prompts
    batch_size: int  # Size of each batch (n)
    batches: List[List[str]]  # Batched prompts
    batch_results: Dict[int, str]  # Results for each batch, keyed by batch index
    batch_status: Dict[int, bool]  # Status of each batch (pass/fail)
    retry_counts: Dict[int, int]  # Number of retries for each batch
    final_result: str  # Aggregated final result
    max_retries: int  # Maximum number of retries allowed


# Step 1: Initialize the workflow
def initialize(state: AuditState) -> AuditState:
    """Initialize the workflow state."""
    context = state.get("context", "")
    audit_checklists = state.get("audit_checklists", [])
    batch_size = state.get("batch_size", 3)
    max_retries = state.get("max_retries", 3)
    
    # Create batches
    batches = []
    for i in range(0, len(audit_checklists), batch_size):
        batch = audit_checklists[i:i+batch_size]
        batches.append(batch)
    
    return {
        "context": context,
        "audit_checklists": audit_checklists,
        "batch_size": batch_size,
        "batches": batches,
        "batch_results": {},
        "batch_status": {},
        "retry_counts": {i: 0 for i in range(len(batches))},
        "final_result": "",
        "max_retries": max_retries
    }


# Step 2: Process batches in parallel
async def process_batches(state: AuditState) -> AuditState:
    """Process all batches in parallel using OpenAI."""
    context = state["context"]
    batches = state["batches"]
    batch_results = state.get("batch_results", {})
    batch_status = state.get("batch_status", {})
    
    # Create LLM instance
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Process batches that don't have results yet or failed evaluation
    tasks = []
    batch_indices_to_process = []
    
    for i, batch in enumerate(batches):
        # Skip if batch already passed evaluation
        if batch_status.get(i) == True:
            continue
            
        # Combine prompts in this batch
        combined_prompt = "\n\n".join([f"Task {j+1}: {prompt}" for j, prompt in enumerate(batch)])
        
        # Create the task for this batch
        prompt_template = PromptTemplate.from_template(
            "You are an audit assistant. Please review the following context and respond to the tasks:\n\n"
            "CONTEXT:\n{context}\n\n"
            "TASKS:\n{tasks}\n\n"
            "Provide a detailed response for each task."
        )
        
        chain = prompt_template | llm | StrOutputParser()
        
        task = chain.ainvoke({"context": context, "tasks": combined_prompt})
        tasks.append(task)
        batch_indices_to_process.append(i)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Store results
    for idx, result in zip(batch_indices_to_process, results):
        batch_results[idx] = result
    
    return {**state, "batch_results": batch_results}


# Step 3: Evaluate batch results
async def evaluate_results(state: AuditState) -> AuditState:
    """Evaluate the results of each batch."""
    batch_results = state["batch_results"]
    batch_status = state.get("batch_status", {})
    
    # Create evaluator LLM
    evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    evaluation_tasks = []
    batch_indices = []
    
    for batch_idx, result in batch_results.items():
        # Skip if already evaluated and passed
        if batch_status.get(batch_idx) == True:
            continue
            
        # Create evaluation task
        eval_prompt = PromptTemplate.from_template(
            "You are a quality control evaluator for audit responses. "
            "Evaluate if the following response is sufficient and high-quality for an audit:\n\n"
            "RESPONSE TO EVALUATE:\n{response}\n\n"
            "Provide your evaluation as a single word: 'PASS' or 'FAIL'. "
            "Use 'PASS' if the response is thorough, accurate, and addresses all tasks adequately. "
            "Use 'FAIL' if the response is incomplete, inaccurate, or insufficient."
        )
        
        eval_chain = eval_prompt | evaluator_llm | StrOutputParser()
        task = eval_chain.ainvoke({"response": result})
        evaluation_tasks.append(task)
        batch_indices.append(batch_idx)
    
    # Wait for all evaluations
    evaluations = await asyncio.gather(*evaluation_tasks)
    
    # Update batch status based on evaluations
    for idx, evaluation in zip(batch_indices, evaluations):
        batch_status[idx] = "PASS" in evaluation.upper()
    
    return {**state, "batch_status": batch_status}


# Step 4: Check if we need to retry any batches
def should_retry(state: AuditState) -> str:
    """Determine if any batches need to be retried."""
    batch_status = state["batch_status"]
    retry_counts = state["retry_counts"]
    max_retries = state["max_retries"]
    
    for batch_idx, passed in batch_status.items():
        if not passed and retry_counts[batch_idx] < max_retries:
            # Increment retry count for this batch
            retry_counts[batch_idx] += 1
            return "retry"
    
    # If all batches passed or reached max retries, we're done
    return "aggregate"


# Step 5: Aggregate final results
def aggregate_results(state: AuditState) -> AuditState:
    """Aggregate results from all batches into a final result."""
    batch_results = state["batch_results"]
    batches = state["batches"]
    
    # Combine all results in order of original batches
    final_result = ""
    for i in range(len(batches)):
        if i in batch_results:
            final_result += f"\n\n--- BATCH {i+1} RESULTS ---\n\n"
            final_result += batch_results[i]
    
    return {**state, "final_result": final_result.strip()}


# Now let's build our graph
def build_audit_workflow():
    """Build and return the LangGraph workflow for audit processing."""
    # Create our state graph
    workflow = StateGraph(AuditState)
    
    # Add nodes
    workflow.add_node("initialize", initialize)
    workflow.add_node("process_batches", process_batches)
    workflow.add_node("evaluate_results", evaluate_results)
    workflow.add_node("aggregate_results", aggregate_results)
    
    # Add edges
    workflow.add_edge("initialize", "process_batches")
    workflow.add_edge("process_batches", "evaluate_results")
    
    # Conditional routing after evaluation
    workflow.add_conditional_edges(
        "evaluate_results",
        should_retry,
        {
            "retry": "process_batches",
            "aggregate": "aggregate_results"
        }
    )
    
    workflow.add_edge("aggregate_results", END)
    
    # Compile the graph
    return workflow.compile()


# Example usage
async def run_audit_workflow(context_json: str, audit_checklists: List[str], batch_size: int = 3, max_retries: int = 3):
    """Run the audit workflow with the given inputs."""
    workflow = build_audit_workflow()
    
    # Initialize the state
    initial_state = {
        "context": context_json,
        "audit_checklists": audit_checklists,
        "batch_size": batch_size,
        "max_retries": max_retries
    }
    
    # Execute the workflow
    final_state = await workflow.ainvoke(initial_state)
    
    return final_state["final_result"]


# Example execution
if __name__ == "__main__":
    # Sample context and audit checklists
    sample_context = json.dumps({
        "company": "Acme Corp",
        "year": 2023,
        "financial_data": {
            "revenue": 10000000,
            "expenses": 7500000,
            "profit": 2500000
        },
        "compliance_status": {
            "tax_filings": True,
            "regulatory_reports": True,
            "internal_controls": "passed"
        }
    })
    
    sample_audit_checklists = [
        "Verify if the company's revenue matches supporting documentation",
        "Check if expenses are properly categorized and recorded",
        "Confirm profit calculation is accurate",
        "Assess if tax filings were completed on time",
        "Evaluate the effectiveness of internal controls",
        "Verify if financial statements comply with accounting standards",
        "Check if proper approval processes were followed for major expenses",
        "Confirm if regulatory reporting requirements were met",
    ]
    
    import asyncio
    
    # Run the workflow
    result = asyncio.run(run_audit_workflow(
        sample_context, 
        sample_audit_checklists, 
        batch_size=3, 
        max_retries=3
    ))
    
    print("Final Audit Result:")
    print(result)
