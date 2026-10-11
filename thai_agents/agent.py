"""Entry point for the ADK tools: `adk web` and `adk run` look for `root_agent` in this module."""
from thai_agents.agents import build_agents
from thai_agents.env import load_env
from thai_agents.retrieval import LazyKnowledgeBase

load_env()
root_agent = build_agents(LazyKnowledgeBase())
