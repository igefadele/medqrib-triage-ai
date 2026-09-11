"""
MediPulse AI: Client Integration & Test Script
Author: Ige Fadele (https://igefadele.savadub.com)

Demonstrates client-side consumption of Server-Sent Events (SSE)
and validates physiological emergency routing.
"""

import json
import requests

SERVER_URL = "http://127.0.0.1:8000"

def test_health():
    print("--- [1] Testing Health Endpoint ---")
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        print("Health Status:", resp.status_code, resp.json())
    except Exception as e:
        print(f"Error reaching server at {SERVER_URL}: {e}")

def test_triage_stream(scenario_name: str, payload: dict):
    print(f"\n--- [2] Running Scenario: {scenario_name} ---")
    try:
        response = requests.post(
            f"{SERVER_URL}/triage/stream",
            json=payload,
            stream=True,
            timeout=30
        )
        print(f"HTTP Status: {response.status_code}")
        print("Receiving Streamed Tokens:\n" + "="*50)
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    token = decoded.replace("data: ", "")
                    print(token, end="", flush=True)
        print("\n" + "="*50 + "\nStream Complete.\n")
    except Exception as e:
        print(f"Error during stream: {e}")

if __name__ == "__main__":
    test_health()

    # Scenario A: Emergency Red-Flag (Severe Hypoxemia SpO2 84%)
    emergency_patient = {
        "patient_id": "PAT-EMERGENCY-01",
        "age_years": 58.0,
        "temperature_celsius": 38.2,
        "heart_rate_bpm": 128,
        "spo2_percent": 84,  # Triggers Tier 1 Red Flag (<90%)
        "systolic_bp": 140,
        "is_pregnant": False,
        "reported_symptoms": "Severe shortness of breath, gasping, cannot finish sentences"
    }
    test_triage_stream("Scenario A: Severe Hypoxemia Emergency", emergency_patient)

    # Scenario B: Urgent Pediatric Intake (Colloquial Dialect)
    pediatric_patient = {
        "patient_id": "PAT-PEDIATRIC-44",
        "age_years": 3.5,
        "temperature_celsius": 39.4,
        "heart_rate_bpm": 142,
        "spo2_percent": 96,
        "is_pregnant": False,
        "reported_symptoms": "Body dey hot like fire, vomiting anything wey e drink, breathing very fast"
    }
    test_triage_stream("Scenario B: Urgent Pediatric Intake", pediatric_patient)
