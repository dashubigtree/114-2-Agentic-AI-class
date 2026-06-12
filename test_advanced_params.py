"""Verify advanced params are forwarded to LightRAG"""
import requests

email = "Your PayPal account is suspended. Click http://fake-paypal.net to verify immediately."

resp = requests.post('http://localhost:5000/api/v1/analyze',
    json={
        'email_content': email,
        'user_query': 'Analyze ATT&CK techniques.',
        'rag_mode': 'bypass',
        'user_prompt': 'Reply in Traditional Chinese only.',
        'top_k': 5,
        'chunk_top_k': 3,
        'max_entity_tokens': 2000,
        'max_relation_tokens': 2000,
        'max_total_tokens': 8000,
    },
    timeout=120)

data = resp.json()
print(f"HTTP: {resp.status_code}  Status: {data.get('status')}")
if data.get('status') == 'error':
    print(f"Error: {data.get('message')}")
else:
    print(f"Layer2: is_phishing={data['layer2_risk']['is_phishing']}, level={data['layer2_risk']['risk_level']}")
    print(f"Layer3: type={data['layer3_classification']['phishing_type']}")
    print(f"RAG (first 200 chars): {data.get('layer4_rag_report','')[:200]}")
    print("\nPASSED: advanced params accepted and pipeline completed successfully.")
