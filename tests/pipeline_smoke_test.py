
from core.state import initial_state
from core.providers.gemini_provider import GeminiProvider
from core.agents.topic_agent import run as run_topic
from core.agents.summary_agent import run as run_summary
from core.agents.action_agent import run as run_action
from core.agents.owner_agent import run as run_owner
from core.agents.priority_agent import run as run_priority

# -----------------------------------------------
# Initialize provider
# -----------------------------------------------

provider = GeminiProvider()

# -----------------------------------------------
# Sample transcript
# -----------------------------------------------

transcript = """
John: We need to improve customer onboarding this quarter.
Sarah: The login authentication bug is still affecting retention.
Mike: John should coordinate with the backend team to fix authentication by Friday.
Sarah: We also need a hiring plan for two new engineers.
Mike: Hiring is important but authentication is highest priority.
"""

# -----------------------------------------------
# Initialize pipeline state
# -----------------------------------------------

state = initial_state(raw_transcript=transcript)

# Normally preprocessing fills this
state["cleaned_transcript"] = transcript

# -----------------------------------------------
# Run agents sequentially
# -----------------------------------------------

state = run_topic(state, provider)
state = run_summary(state, provider)
state = run_action(state, provider)
state = run_owner(state, provider)
state = run_priority(state, provider)

# -----------------------------------------------
# Print results
# -----------------------------------------------

print("\n===== TOPICS =====")
print(state["topics"])

print("\n===== SUMMARY =====")
print(state["summary"])

print("\n===== ACTION ITEMS =====")
for item in state["action_items"]:
    print(item)

print("\n===== ERRORS =====")
print(state["errors"])