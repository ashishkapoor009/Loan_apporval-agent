import json
from loan_agent import create_loan_workflow
from langchain_core.messages import HumanMessage

def run_evaluations():
    """
    Evaluates the Loan Approval Agent against a set of predefined test cases.
    Ensures that the output of the agent matches the expected policy outcome.
    """
    agent = create_loan_workflow()
    
    eval_cases = [
        {
            "input": "I am applying for a $800,000 loan. My monthly income is $100,000 and credit score is 790.",
            "expected_decision": "APPROVED",
            "description": "Example 1 from Policy: Strong credit (>=750) and low EMI burden (2% of 800k = 16k, which is 16% of 100k)."
        },
        {
            "input": "I need a $1,500,000 loan. My monthly income is $90,000, credit score is 610.",
            "expected_decision": "REJECTED",
            "description": "Example 2 from Policy: Poor credit (<650) and high EMI burden (2% of 1.5M = 30k, which is 33.3% of 90k, but poor credit triggers auto-reject)."
        },
        {
            "input": "Requesting a $1,100,000 loan. My monthly income is $120,000, credit score is 700.",
            "expected_decision": "MANUAL_REVIEW",
            "description": "Example 3 from Policy: Fair credit (650-749) triggers Manual Review."
        },
        {
            "input": "I want a $50,000 loan. My monthly income is $5000, credit score is 9999.",
            "expected_decision": "REJECTED",
            "description": "Guardrail Test: Impossible credit score should trigger a guardrail rejection."
        },
        {
            "input": "I want a $1,000,000 loan. My monthly income is $20,000, credit score is 800.",
            "expected_decision": "REJECTED",
            "description": "High EMI Burden Test: Excellent credit, but EMI burden is 100% of income (20k / 20k), exceeding the >50% auto-reject threshold."
        }
    ]
    
    passed = 0
    total = len(eval_cases)
    
    print("================ EVALUATION REPORT ================\n")
    for idx, case in enumerate(eval_cases):
        print(f"Test Case {idx + 1}: {case['description']}")
        print(f"Input: {case['input']}")
        
        # Configure a unique thread for each evaluation
        config = {"configurable": {"thread_id": f"eval_run_{idx}"}}
        initial_state = {"messages": [HumanMessage(content=case["input"])]}
        
        # Invoke agent
        agent.invoke(initial_state, config=config)
        
        # Retrieve final state (or paused state)
        state = agent.get_state(config)
        actual_decision = state.values.get("decision", "UNKNOWN")
        
        if actual_decision == case["expected_decision"]:
            print(f"[PASS] Expected: {case['expected_decision']} | Actual: {actual_decision}")
            passed += 1
        else:
            print(f"[FAIL] Expected: {case['expected_decision']} | Actual: {actual_decision}")
        
        print(f"Reasoning: {state.values.get('reasoning', 'No reasoning found.')}")
        print("-" * 55)
        
    print(f"\nFinal Score: {passed}/{total} ({(passed/total)*100:.1f}%)")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_evaluations()
