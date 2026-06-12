"""Quick ML-only test"""
import requests

FLASK_URL = 'http://localhost:5000'

# Simple phishing email test
phishing_email = "Your account has been suspended. Verify immediately at http://paypa1-fake.net/login. Failure to respond will terminate your account."

resp = requests.post(f'{FLASK_URL}/api/v1/analyze',
    json={
        'email_content': phishing_email,
        'user_query': 'bypass test',
        'rag_mode': 'bypass'
    },
    timeout=60)

print(f"HTTP: {resp.status_code}")
data = resp.json()
print(f"Status: {data.get('status')}")
if data.get('status') == 'error':
    print(f"Error: {data.get('message')}")
else:
    print(f"Layer2: {data.get('layer2_risk')}")
    print(f"Layer3: {data.get('layer3_classification')}")
    print(f"RAG (bypass): {str(data.get('layer4_rag_report',''))[:200]}")
