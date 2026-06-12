"""Test script for PhishRAG pipeline"""
import requests
import json

FLASK_URL = 'http://localhost:5000'

phishing_email = """Subject: Urgent: Your PayPal account has been suspended

Dear user,

We detected suspicious activity on your account. Click here immediately to verify your identity and restore access: http://paypa1-secure-verify.net/login

Failure to respond within 24 hours will result in permanent account termination.

PayPal Security Team"""

print("=== Test 1: Full pipeline analyze (hybrid mode) ===")
resp = requests.post(f'{FLASK_URL}/api/v1/analyze',
    json={
        'email_content': phishing_email,
        'user_query': 'Analyze the ATT&CK techniques used in this phishing email and provide mitigation recommendations.',
        'rag_mode': 'hybrid'
    },
    timeout=300)

data = resp.json()
print(f"HTTP Status: {resp.status_code}")
print(f"Pipeline Status: {data.get('status')}")
if data.get('status') == 'error':
    print(f"Error: {data.get('message')}")
else:
    l2 = data.get('layer2_risk', {})
    l3 = data.get('layer3_classification', {})
    print(f"\n--- Layer 2 Risk ---")
    print(f"  Is Phishing: {l2.get('is_phishing')}")
    print(f"  Risk Score: {l2.get('risk_score')}")
    print(f"  Risk Level: {l2.get('risk_level')}")
    print(f"  Severity: {l2.get('severity')}")
    print(f"\n--- Layer 3 Classification ---")
    print(f"  Phishing Type: {l3.get('phishing_type')}")
    print(f"  Confidence: {l3.get('type_confidence')}")
    print(f"\n--- Layer 4 RAG Report (first 600 chars) ---")
    rag = data.get('layer4_rag_report', 'N/A')
    print(rag[:600])
    print("\n[...truncated...]" if len(rag) > 600 else "")

    print("\n=== Test 2: Multi-turn chat (local mode) ===")
    chat_resp = requests.post(f'{FLASK_URL}/api/v1/chat',
        json={
            'email_content': phishing_email,
            'current_query': 'What social engineering techniques did the attacker use?',
            'rag_mode': 'local',
            'chat_history': [
                {'role': 'user', 'content': 'Analyze ATT&CK techniques'},
                {'role': 'assistant', 'content': rag[:200]}
            ]
        },
        timeout=300)
    chat_data = chat_resp.json()
    print(f"Chat Status: {chat_data.get('status')}")
    print(f"Chat Reply (first 400 chars):\n{chat_data.get('reply', 'N/A')[:400]}")

print("\n=== Test 3: Legitimate email (bypass mode) ===")
legit_email = """Subject: Q3 Roadmap Review Meeting

Hi team,

Please find attached the draft Q3 roadmap for your review. Feedback by Friday would be appreciated.

Best regards,
Project Manager"""

resp3 = requests.post(f'{FLASK_URL}/api/v1/analyze',
    json={
        'email_content': legit_email,
        'user_query': 'Is this email safe?',
        'rag_mode': 'bypass'
    },
    timeout=300)
data3 = resp3.json()
l2_3 = data3.get('layer2_risk', {})
l3_3 = data3.get('layer3_classification', {})
print(f"Status: {data3.get('status')}")
print(f"Is Phishing: {l2_3.get('is_phishing')} (expected: 0)")
print(f"Phishing Type: {l3_3.get('phishing_type')} (expected: legitimate)")
print(f"Risk Level: {l2_3.get('risk_level')}")

print("\n=== All tests completed ===")
