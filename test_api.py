#!/usr/bin/env python
"""Quick test of the API endpoints."""

import httpx
import json
import asyncio

BASE_URL = "http://localhost:8000"


async def test_health():
    async with httpx.AsyncClient() as client:
        print("\n1. Testing /health endpoint...")
        try:
            response = await client.get(f"{BASE_URL}/health", timeout=2)
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")


async def test_webhook():
    """Test webhook with a mock AgentPhone payload."""
    async with httpx.AsyncClient() as client:
        print("\n2. Testing /webhook endpoint...")
        payload = {
            "event": "agent.message",
            "channel": "voice",
            "data": {
                "callId": "call_test_123",
                "from": "+15559876543",
                "to": "+15551234567",
                "transcript": "I need help finding the bathroom",
                "confidence": 0.95,
            }
        }
        try:
            response = await client.post(
                f"{BASE_URL}/webhook",
                json=payload,
                timeout=10
            )
            print(f"   Status: {response.status_code}")
            result = response.json()
            print(f"   Response: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"   Error: {e}")


async def test_gate_change():
    """Test gate-change endpoint."""
    async with httpx.AsyncClient() as client:
        print("\n3. Testing /gate-change endpoint...")
        payload = {
            "phone": "+15551234567",
            "flight": "UA123",
            "old_gate": "B4",
            "new_gate": "C7",
        }
        try:
            response = await client.post(
                f"{BASE_URL}/gate-change",
                json=payload,
                timeout=5
            )
            print(f"   Status: {response.status_code}")
            result = response.json()
            print(f"   Response: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"   Error: {e}")


async def main():
    print("=" * 60)
    print("API Tests - Lincoln Airport Agent")
    print("=" * 60)
    print("\nMake sure the server is running: uv run uvicorn server:app")
    print("\nRunning tests...")

    await test_health()
    await test_webhook()
    await test_gate_change()

    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
