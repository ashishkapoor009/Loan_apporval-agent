import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loan_agent import create_loan_workflow
from langchain_core.messages import HumanMessage
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Loan Approval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = create_loan_workflow()

class EvaluateRequest(BaseModel):
    message: str

class OverrideRequest(BaseModel):
    thread_id: str
    decision: str  # "APPROVE" or "REJECT"

@app.post("/api/evaluate")
def evaluate_loan(req: EvaluateRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"messages": [HumanMessage(content=req.message)]}
    
    agent.invoke(initial_state, config=config)
    state = agent.get_state(config)
    
    if state.next and "human_review" in state.next:
        return {
            "status": "PAUSED",
            "thread_id": thread_id,
            "reasoning": state.values.get("reasoning", "Needs Manual Review"),
            "decision": state.values.get("decision", "MANUAL_REVIEW")
        }
        
    return {
        "status": "COMPLETED",
        "thread_id": thread_id,
        "result": state.values["messages"][-1].content,
        "decision": state.values.get("decision", "")
    }

@app.post("/api/override")
def override_loan(req: OverrideRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    state = agent.get_state(config)
    
    if not state.next or "human_review" not in state.next:
        raise HTTPException(status_code=400, detail="Workflow is not paused for human review.")
        
    if req.decision in ["APPROVE", "REJECT"]:
        override_decision = "APPROVED" if req.decision == "APPROVE" else "REJECTED"
        agent.update_state(
            config, 
            {
                "decision": override_decision, 
                "reasoning": state.values['reasoning'] + f" [OVERRIDE: {override_decision} by Human]"
            }, 
            as_node="decision_maker"
        )
    
    # Resume
    agent.invoke(None, config=config)
    new_state = agent.get_state(config)
    
    return {
        "status": "COMPLETED",
        "thread_id": req.thread_id,
        "result": new_state.values.get("messages", [{}])[-1].content,
        "decision": new_state.values.get("decision", "")
    }

# Serve React build if available
if os.path.isdir("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
