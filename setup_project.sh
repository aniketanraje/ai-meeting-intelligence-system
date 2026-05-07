#!/bin/bash

# --------------------------------------------------
# AI Meeting Intelligence System
# Project Skeleton Setup Script
# Usage:
#   chmod +x setup_project.sh
#   ./setup_project.sh
# --------------------------------------------------

set -e

echo "--------------------------------------------------"
echo " Creating AI Meeting Intelligence System Skeleton "
echo "--------------------------------------------------"

# --------------------------------------------------
# Directory Structure
# --------------------------------------------------

mkdir -p app

mkdir -p core/agents
mkdir -p core/graph
mkdir -p core/prompts
mkdir -p core/utils
mkdir -p core/providers

mkdir -p voice
mkdir -p db

mkdir -p data/sample_transcripts
mkdir -p outputs
mkdir -p tests
mkdir -p docs

# --------------------------------------------------
# Root-Level Files
# --------------------------------------------------

touch README.md
touch requirements.txt
touch .env.example
touch .gitignore
touch Dockerfile
touch .dockerignore
touch run.sh
touch LICENSE

# --------------------------------------------------
# Core Package
# --------------------------------------------------

touch core/__init__.py
touch core/state.py
touch core/config.py
touch core/models.py

# --------------------------------------------------
# Agents
# --------------------------------------------------

touch core/agents/__init__.py

touch core/agents/topic_agent.py
touch core/agents/summary_agent.py
touch core/agents/action_agent.py
touch core/agents/owner_agent.py
touch core/agents/priority_agent.py

# --------------------------------------------------
# LangGraph Pipeline
# --------------------------------------------------

touch core/graph/__init__.py
touch core/graph/pipeline.py

# --------------------------------------------------
# Prompt Templates
# --------------------------------------------------

touch core/prompts/topic_prompt.txt
touch core/prompts/summary_prompt.txt
touch core/prompts/action_prompt.txt
touch core/prompts/owner_prompt.txt
touch core/prompts/priority_prompt.txt

# --------------------------------------------------
# Utilities
# --------------------------------------------------

touch core/utils/__init__.py

touch core/utils/preprocessing.py
touch core/utils/validation.py
touch core/utils/logger.py

# --------------------------------------------------
# LLM Providers
# --------------------------------------------------

touch core/providers/__init__.py

touch core/providers/base_provider.py
touch core/providers/gemini_provider.py
touch core/providers/openai_provider.py
touch core/providers/claude_provider.py

# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

touch app/__init__.py
touch app/streamlit_app.py

# --------------------------------------------------
# Voice / Speech-to-Text
# --------------------------------------------------

touch voice/__init__.py
touch voice/stt.py

# --------------------------------------------------
# Database Layer
# --------------------------------------------------

touch db/__init__.py
touch db/sqlite_handler.py

# --------------------------------------------------
# Tests
# --------------------------------------------------

touch tests/__init__.py

touch tests/test_preprocessing.py
touch tests/test_validation.py
touch tests/test_agents.py
touch tests/test_pipeline.py

# --------------------------------------------------
# Sample Transcript Files
# --------------------------------------------------

touch data/sample_transcripts/clean_meeting.txt
touch data/sample_transcripts/noisy_meeting.txt
touch data/sample_transcripts/no_action_items.txt
touch data/sample_transcripts/multiple_owners.txt
touch data/sample_transcripts/urgent_priorities.txt
touch data/sample_transcripts/missing_ownership.txt

# --------------------------------------------------
# Final Output
# --------------------------------------------------

echo ""
echo "--------------------------------------------------"
echo " Project skeleton created successfully."
echo "--------------------------------------------------"

echo ""
echo "Recommended next steps:"
echo "1. Configure .env.example"
echo "2. Populate requirements.txt"
echo "3. Implement core/config.py"
echo "4. Implement core/state.py"
echo "5. Implement provider abstraction layer"
echo ""