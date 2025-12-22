import gymnasium as gym

from utils.util import dbToString
from environments.FrozenLakeActions import FrozenLakeActions
from environments.Prompts import Prompts
    
class FrozenLakeEnvironment(FrozenLakeActions, Prompts):
    def __init__(self, env = gym.make("FrozenLake-v1", render_mode="ansi",  desc=None, map_name="4x4", is_slippery=True, success_rate=2.0/3.0, reward_schedule=(1, 0, 0)), policyDb = None, hypothesesDb = None):
        super().__init__(env)
        self.policyDb = policyDb
        self.hypothesesDb = hypothesesDb
    
    def getGeneratorPrompt(self) -> str:
        return [
            {
                "role": "system",
                "content": f"""
                     You are a multi-turn LLM Agent Navigator in a dynamic 2D environment. Your goal: reach position G from current position [] using the provided tools for movement.

                     ## Multi-Turn Operation
                     You receive a NAVIGATION TRAJECTORY containing:
                     - All your previous decisions and reasoning
                     - Environment responses after each action
                     - Current state resulting from your last move

                     Your output becomes input for your next iteration. Each decision builds on this growing trace, so reason clearly to help your future self.

                     ## Core Task
                     Read the playbook and the reflection -> Apply rules, knowledge and strategies retrieved from those documents -> Decide the next one move from the given context

                     ## Decision Process
                     1. **Apply Learning**
                        - PLAYBOOK: Proven strategies for similar situations, common failure patterns, environment dynamics
                        - REFLECTION: What worked and what failed, corrected approaches

                     2. **Construct your next move** (from trajectory)
                        - Use the learned strategies to reason

                     3. **Execute Single Move**
                        - Reason concisely (this persists in your trace)
                        - Call exactly ONE tool: move_left, move_right, move_up, move_down
                        - Environment responds → new state → next iteration begins

                     ## Critical Constraints
                     ✓ Remember you are navigating a dynamic 2D environment where the reaction of the environment to your moves is non-deterministic.
                     ✓ One tool call per turn (you're multi-turn, not multi-action)
                     ✓ State only updates after environment processes your move
                     ✓ Your reasoning propagates forward—be clear and useful
                     ✗ Never call multiple movement tools
                     ✗ Don't assume behaviors not observed in the trajectory

                     ## Input Context
                     PLAYBOOK_START
                     {dbToString(self.policyDb) if self.policyDb is not None else 'No playbook available'}
                     PLAYBOOK_END

                     HYPOTHESES_START
                     {dbToString(self.hypothesesDb) if self.hypothesesDb is not None else 'No environment description available'}
                     HYPOTHESES_END

                     STATE_START
                     {self.getState()}
                     STATE_END
                     """,
            }
        ]
    def getGeneratorTools(self) -> str:
        return [
            {
                "type": "function",
                "function": {
                    "name": "move_left",
                    "description": "Moves the player left in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated."
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_right",
                    "description": "Moves the player right in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated."
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_up",
                    "description": "Moves the player up in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated."
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_down",
                    "description": "Moves the player down in the 2D environment. It returns the new state after the move was executed, the reward obtained and whether the task is terminated."
                }
            }
        ]
        
    def getReflectorPrompt(self, trajectory) -> str:
        return [
            {
                "role": "system",
                "content": f'''
                You are an expert reflection agent analyzing a multi-turn navigation trajectory. The navigation agent operated iteratively: it received a trace, made ONE decision using guidance from a PLAYBOOK, got environment feedback, then was called again with the updated trace. Your job is to diagnose what went wrong across these iterations AND evaluate the playbook's effectiveness.
                Remind yourself: the environment the agent navigated is dynamic and non-deterministic. The agent must learn from its growing trace to adapt its strategy over time.

                ## Understanding the Multi-Turn Trace
                The NAVIGATION_TRAJECTORY shows:
                - Agent's reasoning at each turn (based on accumulated context and playbook)
                - Single action executed per turn
                - Environment's response after each action
                - How errors compounded or were corrected across iterations

                ## Your Analysis Task
                1. **Trace the Decision Chain**: Follow how the agent interpreted its growing trace at each step
                2. **Analyze Playbook Usage**: 
                   - Which playbook strategies did the agent apply or ignore?
                   - Were strategies applied correctly in context?
                   - Did playbook guidance lead to success or failure?
                3. **Identify Breaking Points**: Where did reasoning diverge from optimal strategy?
                4. **Diagnose Root Causes**:
                   - Misinterpretation of previous feedback in the trace
                   - Incorrect state extraction from trajectory
                   - Misapplied or ignored playbook strategies
                   - Tool usage errors or format mismatches
                5. **Evaluate Each Playbook Bulletpoint**: Tag as 'helpful', 'harmful', or dont tag (skip) it if it was neutral based on:
                   - Did it guide the agent toward correct decisions?
                   - Did it cause errors or confusion?
                   - Was it irrelevant to the observed failure?

                ## Critical Constraints
                ✓ Explicitly assess each playbook item's impact on navigation
                ✓ Focus on strategy and tool usage patterns (not environment layout—it changes)
                ✓ Ground analysis in actual environment feedback, not assumptions
                ✓ Provide actionable corrections applicable to future multi-turn episodes
                ✗ Don't learn specific positions or obstacle locations

                ## Inputs
                NAVIGATION_TRAJECTORY_BEGIN
                {trajectory if trajectory is not None else ''}
                NAVIGATION_TRAJECTORY_END

                PLAYBOOK_START
                {dbToString(self.policyDb) if self.policyDb is not None else 'No playbook available'}
                PLAYBOOK_END

                ## Required Output (JSON only)
                {{
                  "reasoning": "[Trace the multi-turn decision chain: what did the agent observe, decide, and receive as feedback at each iteration? Which playbook strategies were applied? Where did the reasoning break down?]",
                  "error_identification": "[Specific mistakes in trace interpretation, playbook application, tool usage, or strategy selection across turns]",
                  "root_cause_analysis": "[Why did the agent fail to learn from its growing trace? What concept about multi-turn operation was misunderstood? Were playbook strategies misleading or misapplied?]",
                  "correct_approach": "[Step-by-step: how should the agent have processed the trace, applied playbook guidance, and made decisions differently?]",
                  "key_insight": "[Generalizable strategy for multi-turn navigation: principles for trace interpretation, feedback integration, playbook usage, or tool usage]",
                  "bullet_tags": [{{"id": "example_id", "tag": "helpful|harmful"}}]
                }}
                '''
            }
        ]
    
    def getCuratorPrompt(self, trajectory, reflection) -> str:
        return [
            {
                "role": "system",
                "content": f'''
                    You are a master curator of 2D navigation knowledge. Your job is to maintain a clean, concise playbook by analyzing reflections and updating existing entries strategically.
                    The playbook guides future navigation agents in reaching goals efficiently while avoiding common pitfalls. You are provided with the playbook, the navigation trajectory generated by the agent, and the reflection analysis of its performance.
                    Remind yourself: the environment the agent navigated is dynamic and non-deterministic.

                    ## Core Principles
                    - **Quality over quantity**: Keep the playbook small and focused
                    - **Merge over add**: Consolidate similar insights instead of duplicating
                    - **Remove harmful content**: Delete bulletpoints that were often tagged as 'harmful'
                    - **One insight per tool call**: Each ADD/MODIFY/DELETE operation handles ONE bulletpoint

                    ## Your Curation Process

                    ### 1. Analyze Reflection Tags
                    - **Harmful tags**: Use REMOVE to delete these bulletpoints if they were often tagged as 'harmful'
                    - **Helpful tags**: Keep these, but check if they can be improved by merging with others
                    - **New insights**: Extract from reflection's key_insight and correct_approach

                    ### 2. Consolidation Strategy (CRITICAL)
                    Before adding ANY new content, check existing playbook:
                    - **If similar content exists**: Use MODIFY to merge/enhance the existing bulletpoint
                    - **If content is redundant**: Skip adding it entirely
                    - **Only use ADD if**: The insight is genuinely novel and cannot be merged

                    ### 3. Execution Rules
                    - **One bulletpoint per tool call**: Never add multiple points in a single ADD operation
                    - **Always prefer MODIFY over ADD**: Merge similar strategies into existing entries
                    - **Clean format**: Each bulletpoint must be a single, clear statement (no sub-bullets or lists within)
                    - **If no changes needed**: Don't call any tools

                    ## Inputs

                    CURRENT_PLAYBOOK_BEGIN
                    {dbToString(self.policyDb) if self.policyDb is not None else 'No playbook available'}
                    CURRENT_PLAYBOOK_END

                    NAVIGATION_TRAJECTORY_BEGIN
                    {trajectory if trajectory is not None else ''}
                    NAVIGATION_TRAJECTORY_END

                    REFLECTION_BEGIN
                    {reflection if reflection is not None else ''}
                    REFLECTION_END

                    ## Your Task
                    1. Analyze the reflection about the given navigation trajectory
                    2. Analyze the given navigation trajectory with the reflection in mind
                    3. Analyze the current playbook
                    4. Identify truly NEW insights from reflection's key_insight and correct_approach
                    5. For each new insight:
                       - Check if similar advice already exists in playbook
                       - If yes: MODIFY the existing entry to merge them
                       - If no: ADD as a new entry (one tool call per bulletpoint)
                    6. If the playbook is already comprehensive, do nothing
                    7. If a lot of entries were tagged 'harmful', REMOVE them

                    Remember: A tight, well-curated playbook of 5-10 bulletpoints is far better than a bloated one with 20+ redundant entries.
                '''
            }
        ]
    
    def getCuratorTools(self) -> str:
        return [
            {
                "type": "function",
                "function": {
                    "name": "ADD",
                    "description": "Adds a new entry into the playbook to help future navigation tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section": {
                                "type": "string",
                                "description": "The section of the playbook to which the new entry should be added. If there is no appropriate section, it will create a new section with a relevant title. If you can please reuse existing sections.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The content of the new entry to be added to the playbook. This could be a strategy, tip, common mistake to avoid, or any other relevant information that would help in future navigation tasks."
                            }
                        },
                        "required": ["section", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "REMOVE",
                    "description": "Removes an existing entry from the playbook that is deemed unhelpful or redundant for future navigation tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bullet_id": {
                                "type": "string",
                                "description": "The identifier of the bulletpoint to be removed from the playbook."
                            }
                        },
                        "required": ["bullet_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "MODIFY",
                    "description": "Modifies an existing entry in the playbook to improve its clarity, accuracy, or relevance for future navigation tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bullet_id": {
                                "type": "string",
                                "description": "The identifier of the bulletpoint to be modified to the playbook."
                            },
                            "content": {
                                "type": "string",
                                "description": "The content which will overwrite the existing entry in the playbook."
                            }
                        },
                        "required": ["bullet_id", "content"]
                    }
                }
            }
        ]