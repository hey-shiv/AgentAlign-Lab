"""AgentAlign v2: batched on-policy rollouts, step-level verifier pairs, RFT + DPO.

v1 trained on prompts of the form "Task: <id>" with whole trajectories (including
environment observations) as the completion, which does not match what the agent sees
at inference. v2 records the exact prompt and raw completion of every step and trains
on those, so training and inference use identical inputs.
"""
