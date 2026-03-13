from tinydb import Query
from optimization.PlaybookRefactorer import PlaybookRefactorer
from utils.util import call_llm, dbToString, execute_tool_call
import uuid


class HypothesesRefiner:
    def __init__(self, client, model, hypothesesDb):
        self.client = client
        self.model = model
        self.hypothesesDb = hypothesesDb
        self.query = Query()
        self.TOOL_MAPPING = {
            "ADD": self.add,
            "REMOVE": self.remove,
            "MODIFY": self.modify,
        }

    def run(self, trajectory, debug=False):
        prompt = self.generateInitialPrompt(trajectory)

        msg = call_llm(self.client, self.model, prompt, self.getTools(), debug)

        if msg and msg.tool_calls:
            if debug:
                print(f"== Num_Tools: {len( msg.tool_calls)}")
            for tool_call in msg.tool_calls:
                tool_response = execute_tool_call(self.TOOL_MAPPING, tool_call, debug)

        playbook_refactorer = PlaybookRefactorer(self.client, self.model, self.hypothesesDb)
        playbook_refactorer.run(debug=debug)

        return msg.content

    def add(self, section, content):
        newEntry = {"id": str(uuid.uuid4()), "section": section, "content": content}
        self.hypothesesDb.insert(newEntry)

    def modify(self, bullet_id, content):
        self.hypothesesDb.update({"content": content}, self.query.id == bullet_id)

    def remove(self, bullet_id):
        self.hypothesesDb.remove(self.query.id == bullet_id)

    def getTools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "ADD",
                    "description": "Adds a new entry into the hypotheses playbook to help future navigation tasks. Keep ADD atomic and only add on bulletpoint per operation!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section": {
                                "type": "string",
                                "description": "The section of the hypotheses to which the new entry should be added. If there is no appropriate section, it will create a new section with a relevant title. If you can please reuse existing sections.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the new entry to be added to the hypotheses. This could be a theory, an observation about the environment.",
                            },
                        },
                        "required": ["section", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "REMOVE",
                    "description": "Removes an existing entry from the hypotheses playbook that was falsified.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bullet_id": {
                                "type": "string",
                                "description": "The identifier of the bulletpoint to be removed from the hypotheses playbook.",
                            }
                        },
                        "required": ["bullet_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "MODIFY",
                    "description": "Modifies an existing entry in the hypotheses to improve its clarity, accuracy, or relevance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bullet_id": {
                                "type": "string",
                                "description": "The identifier of the bulletpoint to be modified.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content which will overwrite the existing entry.",
                            },
                        },
                        "required": ["bullet_id", "content"],
                    },
                },
            },
        ]

    def generateInitialPrompt(self, trajectory):
        initial_prompt = [
            {
                "role": "system",
                "content": 
f"""
You are a scientist analyzing a dynamic or non-deterministic 2D environment using navigation trajectories.
Your ONLY task is to maintain a hypotheses playbook describing how the ENVIRONMENT works.
The hypotheses must describe environmental mechanics, not the behavior or reasoning of the agent that generated the trajectory.

# Inputs

## Current Hypotheses about the Environment:
{dbToString(self.hypothesesDb) if self.hypothesesDb is not None and dbToString(self.hypothesesDb) else 'No hypotheses available yet!'}

## Trajectory of a navigation agent that traversed the environment:
{trajectory}

# Strict Scope Rule (Critical)

You are modeling the ENVIRONMENT ONLY.

Valid environmental topics include:
- cell or object types
- object properties
- interaction rules
- cause-effect mechanics
- movement constraints
- termination conditions
- environment boundaries
- environmental transformations or dynamics

You MUST ignore anything about the agent itself, including:
- planning
- search
- lookahead
- value estimates
- policies
- strategies
- decision processes

If an observation only describes the agent's behavior and reveals nothing about the environment, IGNORE it.

# Understanding the Hypotheses Playbook Structure

The playbook contains two kinds of entries:

## Live Hypotheses
Normal entries that describe an active environmental mechanic. These have an id, section, and content.
These entries can be ADDED, MODIFIED, or REMOVED.

## Contradiction Entries
Entries in the "Contradictions" section that flag an unresolved conflict between two prior hypotheses. They follow this format:

  CONFLICT — <id-A> stated: '<claim A>' | <id-B> stated: '<claim B>' — Resolution required.

CRITICAL rules about Contradiction entries:

- The IDs referenced INSIDE a CONFLICT string (id-A, id-B) are HISTORICAL references. Those hypotheses may have already been removed from the playbook. Do NOT attempt to MODIFY or REMOVE hypotheses using those historical IDs — they may no longer exist.
- NEVER create a contradiction between a Contradictions entry and any other hypothesis. Contradictions entries are bookkeeping records, not active claims about the environment.
- NEVER create a contradiction between two Contradictions entries.
- Only live hypotheses (non-Contradictions section entries) can logically conflict with each other.
- A Contradiction entry can itself be REMOVED once the conflict it describes is resolved.

# Updating Process

## Step 1 — Systematically extract environmental observations

Do not skim the trajectory. Read it event by event and run the following checklist. For each item, write down what you observe before moving to Step 2.

### Termination signals
- Did the episode end early (before reaching the goal)? What was the agent's position and what cell type was it on? → This reveals what cells cause termination.
- Did the episode end successfully? What cell type triggered success? → This reveals the goal mechanic.

### Movement and boundary signals  
- Did the agent attempt a move and end up in the same position it started? → This reveals either a wall/blocked cell or a boundary constraint.
- Did the agent reach the edge of the grid and attempt to move further? If its position did not change, the environment blocks out-of-bounds movement.
- Was the agent ever blocked from entering a cell? What type was that cell?

### Cell and object signals
- What distinct cell types or symbols appear in the trajectory (e.g., H, G, W, empty)?
- What happened each time the agent entered each symbol type? Was there a consistent outcome?

### Cause-effect signals
- Are there any repeated patterns where action X on cell type Y always produces outcome Z?
- Did any cell type appear multiple times with a consistent effect?

After completing this checklist, you should have a list of concrete observations such as:
- "Agent entered H at step N → episode terminated"
- "Agent attempted to move left from column 0 → position unchanged → boundary blocks movement"
- "Agent reached G → episode ended successfully"

Do NOT proceed to Step 2 until you have exhausted the checklist. Every signal in the trajectory is evidence.

## Step 2 — Resolve pre-existing contradictions (highest priority)

Scan for entries in the "Contradictions" section. For each:

1. Read both claims stated in the CONFLICT string.
2. Check whether the current trajectory supports one claim, both, or neither.

Resolution actions:

If the trajectory supports claim A and refutes claim B:
- If hypothesis B still exists as a live entry → REMOVE it
- If hypothesis A still exists → MODIFY it if wording should improve
- REMOVE the contradiction entry itself

If the trajectory supports claim B and refutes claim A:
- If hypothesis A still exists as a live entry → REMOVE it
- If hypothesis B still exists → MODIFY it if wording should improve
- REMOVE the contradiction entry itself

If both claims can simultaneously be true (they describe different objects or different contexts):
- REMOVE the contradiction entry (it was a false conflict)
- Keep any still-existing underlying hypotheses untouched

If the trajectory provides no new evidence either way:
- Leave the contradiction entry unchanged. Do not guess.

Never invent a resolution without direct evidence from the trajectory.

## Step 3 — Compare observations with existing hypotheses

Take each observation from your Step 1 checklist and compare it against every live hypothesis:

If it CONFIRMS an existing live hypothesis → do nothing.

If it CONTRADICTS an existing live hypothesis → MODIFY or REMOVE that hypothesis.

If it reveals a NEW environmental mechanic not yet captured in any live hypothesis → ADD a hypothesis.

If the mechanic is already covered by a live hypothesis (even if worded differently) → do NOT add a duplicate.

If after completing the checklist no observation is new or contradictory → perform NO updates.

Doing nothing is correct when no new knowledge is gained.

## Step 4 — Detect NEW contradictions (rare)

Only flag a new contradiction when two LIVE (non-Contradictions section) hypotheses cannot both be true simultaneously in the same context.

Valid contradiction:
"X does A." vs "X does NOT do A."

NOT a contradiction:
"X terminates the episode." and "Y ends the episode successfully." — different objects, independently true.

NOT a contradiction:
Any comparison involving a Contradictions entry — those are records, not claims.

Before flagging any contradiction ask:
"Can both statements be true at the same time in the same context?"

If YES → they are NOT contradictory. Do not flag them.

If a genuine unresolvable contradiction is found, ADD a contradiction entry:

  CONFLICT — <id-A> stated: '<exact content of live hypothesis A>' | <id-B> stated: '<exact content of live hypothesis B>' — Resolution required.

## Step 5 — Deduplication check

Before adding any hypothesis verify:
- No existing live hypothesis already captures the same mechanic
- The fact is not represented in another entry under different wording
- The rule is not being split across multiple entries

One mechanic = one hypothesis.

Prefer MODIFY over ADD when improving an existing hypothesis.

Example of a duplicate to avoid:
"Entering H terminates the episode."
"The episode ends when the agent enters H."

These are the same rule. Only one should exist.

## Step 6 — Final review

Before making any update verify:

- The hypothesis describes an environmental rule, not agent behavior
- The hypothesis is not redundant with an existing entry
- The hypothesis is directly supported by a specific event in the trajectory
- Any contradiction flagged involves two live hypotheses and is logically valid
- No Contradiction entry is being treated as a live environmental claim

# Guidelines

DO:
- describe object types and properties
- describe environmental interaction rules
- describe termination conditions
- describe movement constraints and boundaries
- describe what happens at the edges of the environment

DO NOT:
- describe agent reasoning, planning, or strategy
- describe value estimates or search processes
- flag Contradictions entries as conflicting with live hypotheses
- reference historical IDs found inside CONFLICT strings as if they are live hypotheses

Write hypotheses as general environmental rules, not trajectory-specific observations.

Good: "Entering cell H immediately terminates the episode."
Bad: "The agent died when it stepped on H."

Good: "The environment has finite boundaries; movement attempts beyond the grid edge are blocked and the agent remains in its current position."
Bad: "The agent tried to move left but stayed in place."

# Output
Your output MUST contain ONLY tool calls.
Do NOT include explanations or analysis.
If the trajectory produces no new environmental knowledge and resolves no contradictions, output NOTHING (zero tool calls).
""",
            }
        ]
        return initial_prompt