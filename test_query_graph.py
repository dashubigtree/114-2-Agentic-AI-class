"""Test /api/v1/query_graph endpoint"""
import requests, json

email = "Your PayPal account is suspended. Click http://fake-paypal.net to verify immediately."

resp = requests.post('http://localhost:5000/api/v1/query_graph', json={
    'email_content': email,
    'user_query': '請分析此郵件的知識圖譜',
    'rag_mode': 'local',
    'layer2_risk': {'is_phishing': 1, 'risk_score': 0.9997, 'risk_level': 'High', 'severity': 'high'},
    'layer3_classification': {'phishing_type': 'credential_harvesting', 'type_confidence': 0.99},
}, timeout=90)

data = resp.json()
print(f"HTTP: {resp.status_code}  Status: {data.get('status')}")
if data.get('status') == 'error':
    print(f"Error: {data.get('message')}")
else:
    ents = data.get('entities', [])
    rels = data.get('relationships', [])
    chks = data.get('chunks', [])
    print(f"Entities: {len(ents)}")
    print(f"Relationships: {len(rels)}")
    print(f"Chunks: {len(chks)}")
    if ents:
        print(f"\nFirst entity: {json.dumps(ents[0], ensure_ascii=False, indent=2)[:300]}")
    if rels:
        print(f"\nFirst relation: src={rels[0].get('src_id')} → tgt={rels[0].get('tgt_id')}, weight={rels[0].get('weight')}")
    meta = data.get('metadata', {})
    proc = meta.get('processing_info', {})
    if proc:
        print(f"\nMetadata: {json.dumps(proc, indent=2)}")
    print("\nPASSED: query_graph endpoint working correctly.")
