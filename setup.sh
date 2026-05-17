#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Setting up Stuttgart Agent...${NC}"

# Install dependencies
if command -v uv &> /dev/null; then
    echo -e "${BLUE}Installing dependencies with uv...${NC}"
    uv sync
else
    echo -e "${BLUE}Installing dependencies with pip...${NC}"
    pip install -e .
fi

# Create .env if it doesn't exist yet (first-time setup on a new machine)
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}.env not found — copying from .env.example${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Fill in your credentials in .env before running the agent.${NC}"
else
    echo -e "${GREEN}.env already exists — keeping it as-is${NC}"
fi

echo -e "${GREEN}Done. Run: python main.py${NC}"
