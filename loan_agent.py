import os
import json
import sqlite3
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Define the state for the Loan Approval Workflow
class LoanState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    applicant_data: dict
    guardrail_status: str
    credit_status: str
    emi_burden: float
    decision: str
    reasoning: str

# 1. Node: Gather Information
def gather_data_node(state: LoanState):
    """Extracts applicant data from the conversation."""
    messages = state.get("messages", [])
    last_message = messages[-1].content
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = (
        "Extract the following loan application details from the user's message as a JSON object: "
        "monthly_income, loan_amount, credit_score. "
        "If they provide annual income, divide it by 12 to get monthly_income. "
        "Return ONLY valid JSON. If information is missing, put null."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message)
    ])
    
    try:
        data = json.loads(response.content.strip('```json\n').strip('```'))
    except json.JSONDecodeError:
        data = {"monthly_income": None, "loan_amount": None, "credit_score": None}
        
    return {"applicant_data": data}

# 2. Node: Input Guardrails
def guardrail_node(state: LoanState):
    """Validates the extracted data against common sense constraints."""
    data = state.get("applicant_data", {})
    issues = []
    
    cs = data.get("credit_score")
    if cs is not None and (cs < 300 or cs > 900):
        issues.append(f"Invalid credit score: {cs}. Must be between 300 and 900.")
        
    inc = data.get("monthly_income")
    if inc is not None and inc <= 0:
        issues.append("Monthly income must be greater than 0.")
        
    amt = data.get("loan_amount")
    if amt is not None and amt <= 0:
        issues.append("Loan amount must be greater than 0.")
        
    if issues:
        return {"guardrail_status": "FAILED", "reasoning": "Guardrail Check Failed: " + "; ".join(issues), "decision": "REJECTED"}
    
    return {"guardrail_status": "PASSED"}

# 3. Node: Credit Check Policy
def credit_check_node(state: LoanState):
    """Evaluates the credit score based on policy."""
    if state.get("guardrail_status") == "FAILED":
        return {}
        
    data = state.get("applicant_data", {})
    score = data.get("credit_score")
    
    if score is None:
        status = "UNKNOWN"
    elif score >= 750:
        status = "EXCELLENT"
    elif score >= 650:
        status = "FAIR"
    else:
        status = "POOR"
        
    return {"credit_status": status}

# 4. Node: Affordability Policy
def affordability_node(state: LoanState):
    """Calculates EMI Burden based on policy."""
    if state.get("guardrail_status") == "FAILED":
        return {}
        
    data = state.get("applicant_data", {})
    income = data.get("monthly_income")
    loan_amount = data.get("loan_amount")
    
    emi_burden = 100.0
    if income and loan_amount:
        # Mock EMI Calculation: Assuming roughly 2% of loan amount as monthly EMI (approx 5 yrs)
        estimated_emi = loan_amount * 0.02
        emi_burden = (estimated_emi / income) * 100
            
    return {"emi_burden": emi_burden}

# 5. Node: Decision Logic based on Policy Rules
def decision_node(state: LoanState):
    """Makes the loan approval decision strictly following the policy document."""
    if state.get("guardrail_status") == "FAILED":
        return {}
        
    credit_status = state.get("credit_status")
    emi_burden = state.get("emi_burden")
    data = state.get("applicant_data")
    
    decision = ""
    reasoning = ""

    if data.get("monthly_income") is None or data.get("loan_amount") is None or data.get("credit_score") is None:
        decision = "INCOMPLETE_DATA"
        reasoning = "Cannot make a decision due to missing application data."

    # Reject Rules
    elif credit_status == "POOR" or emi_burden > 50:
        decision = "REJECTED"
        reasoning = f"Policy Reject: Credit Score is {data.get('credit_score')} and EMI Burden is {emi_burden:.1f}%."

    # Manual Review Rules
    elif credit_status == "FAIR" or (35 < emi_burden <= 50):
        decision = "MANUAL_REVIEW"
        reasoning = f"Policy Manual Review: Credit Score {data.get('credit_score')}, EMI Burden {emi_burden:.1f}%."

    # Approve Rules
    elif credit_status == "EXCELLENT" and emi_burden <= 35:
        decision = "APPROVED"
        reasoning = f"Policy Approve: Excellent Credit ({data.get('credit_score')}) and low EMI Burden ({emi_burden:.1f}%)."
    
    else:
        decision = "MANUAL_REVIEW"
        reasoning = f"Catch-all (Credit: {credit_status}, EMI Burden: {emi_burden:.1f}%)."
        
    return {"decision": decision, "reasoning": reasoning}

# 6. Node: Human Review
def human_review_node(state: LoanState):
    """Handles the manual review outcome provided by the human."""
    decision = state.get("decision")
    reasoning = state.get("reasoning")
    
    final_message = AIMessage(content=f"**Final Decision (After Human Review):** {decision}\n**Reasoning:** {reasoning}")
    return {"messages": [final_message]}

# Automatic output generation
def output_generation_node(state: LoanState):
    decision = state.get("decision")
    reasoning = state.get("reasoning")
    final_message = AIMessage(content=f"**Automated Decision:** {decision}\n**Reasoning:** {reasoning}")
    return {"messages": [final_message]}

# 7. Build the Graph
def route_after_guardrail(state: LoanState):
    if state["guardrail_status"] == "FAILED":
        return "output_generation"
    return "credit_check"

def route_after_decision(state: LoanState):
    if state["decision"] == "MANUAL_REVIEW":
        return "human_review"
    return "output_generation"

def create_loan_workflow():
    workflow = StateGraph(LoanState)
    
    workflow.add_node("gather_data", gather_data_node)
    workflow.add_node("guardrails", guardrail_node)
    workflow.add_node("credit_check", credit_check_node)
    workflow.add_node("affordability", affordability_node)
    workflow.add_node("decision_maker", decision_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("output_generation", output_generation_node)
    
    workflow.add_edge(START, "gather_data")
    workflow.add_edge("gather_data", "guardrails")
    
    workflow.add_conditional_edges("guardrails", route_after_guardrail)
    
    workflow.add_edge("credit_check", "affordability")
    workflow.add_edge("affordability", "decision_maker")
    
    workflow.add_conditional_edges("decision_maker", route_after_decision)
    
    workflow.add_edge("human_review", END)
    workflow.add_edge("output_generation", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["human_review"])

# Example Usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = create_loan_workflow()
    
    test_cases = [
        "Hi, I want $800000 loan. My monthly income is $100000 and credit score is 790.", # Ex 1: Approve
        "Hi, I want a $1500000 loan. My monthly income is $90000, credit score is 610.", # Ex 2: Reject
        "I need a $1100000 loan. My monthly income is $120000, credit score is 700.",    # Ex 3: Manual Review
        "I need a $50000 loan. My monthly income is $5000, credit score is 9999.",       # Guardrail Fail
    ]
    
    for i, user_input in enumerate(test_cases):
        print("-" * 50)
        print(f"User Input: {user_input}")
        
        config = {"configurable": {"thread_id": f"thread_eval_{i}"}}
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        agent.invoke(initial_state, config=config)
        
        state = agent.get_state(config)
        
        if state.next and "human_review" in state.next:
            print(f"[PAUSED] WORKFLOW PAUSED: Human In The Loop Triggered.")
            print(f"Reasoning: {state.values['reasoning']}")
            human_decision = input(">>> Type 'APPROVE' or 'REJECT' (or press Enter to skip override): ").strip().upper()
            
            if human_decision in ["APPROVE", "REJECT"]:
                agent.update_state(config, {"decision": human_decision, "reasoning": state.values['reasoning'] + f" [OVERRIDE: {human_decision} by Human]"}, as_node="decision_maker")
            else:
                print(">>> HUMAN ACTION: Skipped override.")
            
            result = agent.invoke(None, config=config)
            print("[RESUMED] WORKFLOW RESUMED.")
            print(result["messages"][-1].content)
        else:
            print("[SUCCESS] WORKFLOW COMPLETED AUTOMATICALLY.")
            print(state.values.get("messages", [AIMessage(content="Error")])[-1].content)
