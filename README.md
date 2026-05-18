ALL CODE IS DISTRIBUTED ON THIS ORGANIZATION's 3 REPOSITORIES. See the other two repositories in this organization for the hydration layer built with MOSS and the custom map builder we created to provide an indoor map of OAK airport to the agent.

<img width="956" height="536" alt="Screenshot 2026-05-17 at 8 31 20 PM" src="https://github.com/user-attachments/assets/05d2e96a-a416-480f-b86b-6beb0e1c6bff" />

DeepIndoors

DeepIndoors is an AI-powered indoor navigation agent that helps users navigate complex indoor environments such as airports, stadiums, and college campuses through natural conversation.

The platform combines real-time indoor mapping, personalized recommendations, cross-context memory, and phone accessibility to create a dynamic and inclusive navigation experience for everyone.

Overview

DeepIndoors enables users to interact with indoor spaces conversationally through voice or phone calls. Instead of relying on static maps or confusing signage, users can simply ask questions like:

“How to get to my gate?”

The system dynamically integrates with indoor infrastructure through Moss.dev and user preferences with Supermemory to provide real-time, personalized assistance to all users at indoor spaces.

Features
Indoor Navigation
Navigate airports, stadiums, and campuses conversationally
Real-time routing through indoor schematics
Dynamic rerouting based on changing conditions
Phone-Based Accessibility
Accessible through a standard telephone number using AgentPhone
Designed for:
Elderly users
International travelers
Users without smartphones
Users requiring accessibility accommodations
Ultra-Low Latency Integrations
Powered by a Moss.dev hydration layer
Sub-10 ms propagation latency for integrated systems
Persistent Cross-Context Memory

Using Supermemory, DeepIndoors remembers user preferences across:

Calls
Airports
Stadiums
Campuses
Indoor environments

Examples:

Preferred food types (e.g. halal)
Credit card rewards preferences (e.g. AMEX cashback)
Accessibility requirements
Navigation habits
Airport Integrations

Our airport MVP integrates with:

Indoor airport schematics
Flight tracking systems
Terminal store information
Loyalty and rewards programs
Inbound and outbound call systems
Tech Stack
Technology	Purpose
LangGraph	AI agent orchestration
Moss.dev	Hydration layer + low-latency integrations
AgentPhone	Telephone-based agent access
Supermemory	Persistent cross-context memory
Custom Mapping Software	Indoor navigation and routing
Custom Flight Tracking	Real-time airport updates
Example User Experience

A traveler lands at a new airport and calls DeepIndoors.

The agent already remembers:

They prefer halal food
They use AMEX cards
They prefer accessible walking routes

DeepIndoors then:

Navigates them to their next gate
Finds nearby halal food options
Recommends locations offering AMEX cashback
Alerts them if their gate changes
Adjusts directions dynamically in real time
Architecture
User
  ↓
AgentPhone (Voice Access)
  ↓
LangGraph Agent
  ↓
Moss.dev Hydration Layer
  ↓
Integrated Indoor Systems
    • Airport Maps
    • Flight Tracking
    • Rewards Systems
    • Store Information
    • Accessibility Data
  ↓
Supermemory
(Persistent Cross-Context Personalization)
Vision

DeepIndoors transforms indoor navigation from a static and confusing experience into a personalized, dynamic, and universally accessible system.

Our goal is to ensure that anyone — regardless of language, accommodations, age, or familiarity with a space — can confidently navigate any indoor environment.

Future Expansion
Multi-airport deployments
Stadium event navigation
University campus integrations
Multilingual conversational support
Smart accessibility routing
Retail and loyalty personalization
Real-time crowd and congestion awareness

Status

MVP currently focused on airport navigation and integrations.

Demo video and additional materials coming soon.
