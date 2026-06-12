"""
app.py — PhishRAG Flask 後端
四層流水線：特徵擷取 → 風險評分 → 釣魚分類 → LightRAG 知識圖譜檢索
"""

import re
import os
import logging
import numpy as np
import pandas as pd
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ─── 路徑設定 ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR   = os.path.join(BASE_DIR, 'PhishRAG_MLPipeline')
_LIGHTRAG_BASE     = os.getenv('LIGHTRAG_URL', 'http://localhost:9621')
LIGHTRAG_QUERY_URL = f'{_LIGHTRAG_BASE}/query'
LIGHTRAG_DATA_URL  = f'{_LIGHTRAG_BASE}/query/data'

# ─── 載入 ML 模型與 Pipeline 元件 ──────────────────────────────────────────
logger.info("正在從 %s 載入 ML 元件...", ML_DIR)

def _load_xgb(path):
    """Load XGBoost model and patch missing attributes from older serialized format."""
    m = joblib.load(path)
    _missing_defaults = {
        'callbacks': None,
        'early_stopping_rounds': None,
        'eval_metric': None,
        'feature_types': None,
        'grow_policy': None,
        'max_bin': None,
        'max_cat_threshold': None,
        'max_cat_to_onehot': None,
        'max_leaves': None,
        'sampling_method': None,
    }
    for attr, default in _missing_defaults.items():
        if not hasattr(m, attr):
            setattr(m, attr, default)
    return m

model_a      = _load_xgb(os.path.join(ML_DIR, 'model_a_binary.pkl'))
model_b      = _load_xgb(os.path.join(ML_DIR, 'model_b_type.pkl'))
model_c      = _load_xgb(os.path.join(ML_DIR, 'model_c_severity.pkl'))
le_type      = joblib.load(os.path.join(ML_DIR, 'label_encoder_type.pkl'))
le_sev       = joblib.load(os.path.join(ML_DIR, 'label_encoder_severity.pkl'))
FEATURE_COLS = joblib.load(os.path.join(ML_DIR, 'feature_cols.pkl'))
vocab        = joblib.load(os.path.join(ML_DIR, 'tfidf_vocabulary.pkl'))

# 以固定詞彙表重建 TF-IDF 向量化器
# 注意：訓練時使用整個語料庫計算 IDF；此處以 dummy doc fit，IDF=1（最佳近似值）
_dummy_doc = ' '.join(vocab.keys())
tfidf_inf  = TfidfVectorizer(
    vocabulary=vocab,
    ngram_range=(1, 2),
    sublinear_tf=True,
    token_pattern=r'\b[a-zA-Z]{2,}\b',
    stop_words='english',
)
tfidf_inf.fit([_dummy_doc])
TFIDF_VOCAB_COLS = [f'tfidf_{t}' for t in tfidf_inf.get_feature_names_out()]

logger.info(
    "ML 元件載入完成 | FEATURE_COLS=%d | TF-IDF vocab=%d",
    len(FEATURE_COLS), len(vocab)
)

# ─── 詞彙字典（與訓練腳本完全一致）────────────────────────────────────────
LEXICONS = {
    'urgency_score': [
        'urgent', 'immediately', 'expire', 'expires', 'act now', 'limited time',
        'last chance', 'hurry', "don't wait", 'time sensitive', 'deadline',
        'today only', 'right away', 'instantly', 'asap',
    ],
    'financial_score': [
        'refund', 'winner', 'prize', 'reward', 'gift', 'lottery', 'million',
        'credit card', 'ssn', 'wire transfer', 'payment', 'invoice', 'cash',
        'bonus', 'offer', '$', '£', '€',
    ],
    'authority_score': [
        'official', 'government', 'irs', 'fbi', 'mcafee', 'apple', 'microsoft',
        'paypal', 'amazon', 'netflix', 'google', 'facebook', 'compliance',
        'fraud prevention', 'security team', 'telegram',
    ],
    'credential_score': [
        'verify', 'verification', 'username', 'password', 'pin', 'sign in',
        'log in', 'login', 'account locked', 'restore access', 'confirm',
        'secure link', 'click here', 'update your',
    ],
    'threat_score': [
        'suspend', 'suspended', 'suspension', 'penalty', 'violation',
        'failure to respond', 'legal action', 'terminated', 'blocked',
        'unauthorized', 'breach',
    ],
    'pii_request_score': [
        'provide your', 'enter your', 'submit your', 'send us your',
        'social security', 'date of birth', "mother's maiden",
        'credit card number', 'bank account',
    ],
    'brand_impersonation_score': [
        'apple', 'paypal', 'amazon', 'netflix', 'microsoft', 'google',
        'facebook', 'instagram', 'twitter', 'linkedin', 'bank', 'chase',
        'citibank', 'wells fargo',
    ],
    'social_engineering_score': [
        'i miss you', 'i love you', 'special connection', 'lonely',
        'someone special', 'lonely heart', 'romantic', 'dating',
        'meet you', "let's meet",
    ],
}

# 合成特徵（URL、寄件者）在推論時無法從純文字取得，使用釣魚信件類別平均值估計
SYNTH_MEAN = {
    'has_url':                  0.75,
    'url_count':                2.0,
    'has_suspicious_domain':    0.55,
    'url_text_mismatch':        0.50,
    'has_ip_url':               0.20,
    'sender_domain_suspicious': 0.60,
    'display_name_mismatch':    0.45,
    'reply_to_mismatch':        0.35,
    'has_attachment':           0.30,
}

# ─── Layer 1：特徵擷取函式（與 phishrag_pipeline.py 邏輯完全一致）─────────

def split_body_keywords(text: str) -> tuple:
    """將郵件拆分為 (body_text, keywords_string)。"""
    if not isinstance(text, str):
        return '', ''
    parts = re.split(r'\nKeywords:', text, maxsplit=1)
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else '')


def count_keywords(text: str, kw_list: list) -> int:
    """計算關鍵字出現次數（不分大小寫）。"""
    if not isinstance(text, str):
        return 0
    t = text.lower()
    return sum(1 for kw in kw_list if kw in t)


def extract_hand_features(raw_text: str) -> dict:
    """從原始郵件文字擷取 28 個手工特徵（排除 data leakage 欄位）。"""
    body, _ = split_body_keywords(raw_text)
    words   = re.findall(r'\b\w+\b', body)
    sents   = [s for s in re.split(r'[.!?]+', body) if s.strip()]
    uc      = re.findall(r'\b[A-Z]{2,}\b', body)

    feat = {
        'char_count':           len(body),
        'word_count':           len(words),
        'sentence_count':       len(sents),
        'avg_word_length':      len(body) / len(words) if words else 0.0,
        'avg_sentence_len':     len(words) / len(sents) if sents else 0.0,
        'exclamation_count':    body.count('!'),
        'question_count':       body.count('?'),
        'uppercase_word_count': len(uc),
        'uppercase_ratio':      len(uc) / len(words) if words else 0.0,
        'special_char_count':   sum(1 for c in body if c in '@#$%^&*+=<>[]{}|\\~`'),
    }
    for fname, kw_list in LEXICONS.items():
        feat[fname] = count_keywords(body, kw_list)
    feat['total_suspicious_score'] = sum(feat[k] for k in LEXICONS)
    feat.update(SYNTH_MEAN)  # 合成特徵使用平均值
    return feat


def build_feature_vector(raw_text: str) -> tuple:
    """
    建立 76 維特徵向量 (DataFrame) 與 Layer 1 輸出 dict。
    特徵排列順序以 feature_cols.pkl 為準，確保與訓練時一致。
    """
    body, _    = split_body_keywords(raw_text)
    hand_feat  = extract_hand_features(raw_text)
    tfidf_vec  = tfidf_inf.transform([body]).toarray()[0]
    tfidf_dict = dict(zip(TFIDF_VOCAB_COLS, tfidf_vec.tolist()))

    combined = {**hand_feat, **tfidf_dict}

    X = pd.DataFrame(
        [[combined.get(c, 0.0) for c in FEATURE_COLS]],
        columns=FEATURE_COLS,
        dtype=float,
    )
    layer1 = {c: round(float(combined.get(c, 0.0)), 6) for c in FEATURE_COLS}
    return X, layer1


# ─── Layer 2 & 3：ML 推論流水線 ─────────────────────────────────────────────

def run_ml_pipeline(email_text: str) -> dict:
    """
    執行三層 XGBoost 推論：
      Layer 2 - Model A：是否為釣魚（binary）
      Layer 3 - Model B：釣魚類型（11 類）
      Layer 3 - Model C：嚴重程度（low/medium/high，輸入加入 Model B 的類型機率）
    """
    X, layer1 = build_feature_vector(email_text)

    # Model A：二元分類
    is_phishing = int(model_a.predict(X)[0])
    risk_score  = float(model_a.predict_proba(X)[0, 1])

    if is_phishing == 0:
        return {
            'layer1_features': layer1,
            'layer2_risk': {
                'is_phishing': 0,
                'risk_score':  round(risk_score, 4),
                'risk_level':  'Low',
                'severity':    'low',
            },
            'layer3_classification': {
                'phishing_type':   'legitimate',
                'type_confidence': 0.0,
            },
        }

    # Model B：多類型分類（11 類）
    type_proba    = model_b.predict_proba(X)          # shape: (1, 11)
    type_idx      = int(model_b.predict(X)[0])
    phishing_type = le_type.classes_[type_idx]
    type_conf     = float(type_proba[0, type_idx])

    # Model C：嚴重程度（輸入：76 維特徵 + 11 維類型機率 = 87 維）
    X_c      = np.hstack([X.values, type_proba])
    sev_idx  = int(model_c.predict(X_c)[0])
    severity = le_sev.classes_[sev_idx]

    risk_level_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}

    return {
        'layer1_features': layer1,
        'layer2_risk': {
            'is_phishing': 1,
            'risk_score':  round(risk_score, 4),
            'risk_level':  risk_level_map.get(severity, 'Unknown'),
            'severity':    severity,
        },
        'layer3_classification': {
            'phishing_type':   phishing_type,
            'type_confidence': round(type_conf, 4),
        },
    }


# ─── Layer 4：LightRAG 知識圖譜查詢 ──────────────────────────────────────────

def query_lightrag(
    query: str,
    mode: str = 'hybrid',
    user_prompt: str | None = None,
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    max_entity_tokens: int | None = None,
    max_relation_tokens: int | None = None,
    max_total_tokens: int | None = None,
) -> str:
    """
    POST 至 LightRAG /query 端點，依 mode 動態切換檢索策略。
    支援 hybrid / local / global / mix / naive / bypass。
    進階參數傳 None 時不放入 payload，讓伺服器沿用 .env 預設值。
    """
    payload: dict = {'query': query, 'mode': mode}
    if user_prompt:
        payload['user_prompt'] = user_prompt
    if top_k is not None:
        payload['top_k'] = top_k
    if chunk_top_k is not None:
        payload['chunk_top_k'] = chunk_top_k
    if max_entity_tokens is not None:
        payload['max_entity_tokens'] = max_entity_tokens
    if max_relation_tokens is not None:
        payload['max_relation_tokens'] = max_relation_tokens
    if max_total_tokens is not None:
        payload['max_total_tokens'] = max_total_tokens
    try:
        resp = requests.post(LIGHTRAG_QUERY_URL, json=payload, timeout=360)
        resp.raise_for_status()
        data = resp.json()
        return data.get('response', str(data))
    except requests.exceptions.ConnectionError:
        return (
            '⚠️ [LightRAG 連線失敗]\n'
            '請確認 LightRAG 伺服器正在 http://localhost:9621 執行，'
            '並已完成知識庫索引。'
        )
    except requests.exceptions.Timeout:
        return '⚠️ [LightRAG 請求逾時] 伺服器回應超過 360 秒，請確認 LLM 後端正常運作。'
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else 'N/A'
        body = exc.response.text[:300] if exc.response is not None else ''
        return f'⚠️ [LightRAG HTTP 錯誤 {code}] {body}'
    except Exception as exc:
        return f'⚠️ [LightRAG 未知錯誤] {exc}'


def _build_analysis_query(email_content: str, ml_result: dict, user_query: str) -> str:
    """組合初始分析查詢：郵件內容 + ML 摘要 + 使用者指令。"""
    l2 = ml_result['layer2_risk']
    l3 = ml_result['layer3_classification']
    phish_label = '釣魚郵件 (Phishing)' if l2['is_phishing'] else '正常郵件 (Legitimate)'
    conf_str = f"{l3['type_confidence']:.4f}" if l3['type_confidence'] > 0 else 'N/A'

    return (
        "【PhishRAG 郵件威脅情資分析任務】\n\n"
        "【待分析郵件內容】\n"
        f"{email_content}\n\n"
        "【ML 流水線分析摘要（Layer 1–3）】\n"
        f"- 釣魚判定：{phish_label}\n"
        f"- 風險分數：{l2['risk_score']:.4f}\n"
        f"- 風險等級：{l2['risk_level']} ({l2.get('severity', 'low')})\n"
        f"- 釣魚類型：{l3['phishing_type']}\n"
        f"- 分類信心度：{conf_str}\n\n"
        "【分析師指令】\n"
        f"{user_query}"
    )


def _build_chat_query(email_content: str, chat_history: list, current_query: str) -> str:
    """組合多輪追問查詢：郵件上下文 + 對話歷史 + 當前問題。"""
    history_lines = []
    for msg in chat_history:
        role = '使用者' if msg.get('role') == 'user' else 'AI 助手'
        history_lines.append(f"{role}：{msg.get('content', '').strip()}")
    history_text = '\n'.join(history_lines) if history_lines else '（無歷史記錄）'

    return (
        "【PhishRAG 持續威脅追問對話】\n\n"
        "【郵件上下文（已鎖定）】\n"
        f"{email_content}\n\n"
        "【歷史對話記錄】\n"
        f"{history_text}\n\n"
        "【當前分析師提問】\n"
        f"{current_query}"
    )


# ─── Flask 應用程式 ──────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'feature_cols': len(FEATURE_COLS), 'vocab_size': len(vocab)})


@app.route('/api/v1/analyze', methods=['POST'])
def analyze():
    """
    初始分析端點：執行完整四層流水線。
    輸入：{"email_content": str, "user_query": str, "rag_mode": str}
    輸出：{"status", "layer1_features", "layer2_risk", "layer3_classification", "layer4_rag_report"}
    """
    data = request.get_json(force=True, silent=True) or {}
    email_content       = str(data.get('email_content', '')).strip()
    user_query          = str(data.get('user_query', '')).strip() or '請對此郵件進行完整的 PhishRAG 流水線威脅分析。'
    rag_mode            = str(data.get('rag_mode', 'hybrid')).strip()
    user_prompt         = data.get('user_prompt') or None
    top_k               = data.get('top_k')
    chunk_top_k         = data.get('chunk_top_k')
    max_entity_tokens   = data.get('max_entity_tokens')
    max_relation_tokens = data.get('max_relation_tokens')
    max_total_tokens    = data.get('max_total_tokens')

    if not email_content:
        return jsonify({'status': 'error', 'message': '郵件內容不得為空'}), 400

    # Layer 1-3：ML 推論
    try:
        ml_result = run_ml_pipeline(email_content)
    except Exception as exc:
        logger.exception("ML 流水線錯誤")
        return jsonify({'status': 'error', 'message': f'ML 分析錯誤：{exc}'}), 500

    # Layer 4：LightRAG 知識圖譜檢索
    query_text = _build_analysis_query(email_content, ml_result, user_query)
    rag_report = query_lightrag(
        query_text, rag_mode,
        user_prompt=user_prompt,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        max_entity_tokens=max_entity_tokens,
        max_relation_tokens=max_relation_tokens,
        max_total_tokens=max_total_tokens,
    )

    return jsonify({
        'status':                'success',
        'layer1_features':       ml_result['layer1_features'],
        'layer2_risk':           ml_result['layer2_risk'],
        'layer3_classification': ml_result['layer3_classification'],
        'layer4_rag_report':     rag_report,
    })


@app.route('/api/v1/chat', methods=['POST'])
def chat():
    """
    多輪追問端點：跳過 Layer 1-3，直接以動態模式查詢 LightRAG。
    輸入：{"email_content": str, "current_query": str, "rag_mode": str, "chat_history": list}
    輸出：{"status": "success", "reply": str}
    """
    data                = request.get_json(force=True, silent=True) or {}
    email_content       = str(data.get('email_content', '')).strip()
    current_query       = str(data.get('current_query', '')).strip()
    rag_mode            = str(data.get('rag_mode', 'hybrid')).strip()
    chat_history        = data.get('chat_history', [])
    user_prompt         = data.get('user_prompt') or None
    top_k               = data.get('top_k')
    chunk_top_k         = data.get('chunk_top_k')
    max_entity_tokens   = data.get('max_entity_tokens')
    max_relation_tokens = data.get('max_relation_tokens')
    max_total_tokens    = data.get('max_total_tokens')

    if not current_query:
        return jsonify({'status': 'error', 'message': '問題內容不得為空'}), 400

    query_text = _build_chat_query(email_content, chat_history, current_query)
    reply      = query_lightrag(
        query_text, rag_mode,
        user_prompt=user_prompt,
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        max_entity_tokens=max_entity_tokens,
        max_relation_tokens=max_relation_tokens,
        max_total_tokens=max_total_tokens,
    )

    return jsonify({'status': 'success', 'reply': reply})


@app.route('/api/v1/query_graph', methods=['POST'])
def query_graph():
    """
    呼叫 LightRAG /query/data，取得知識圖譜原始資料（不呼叫 LLM）。
    輸入：{"email_content", "user_query", "rag_mode", "layer2_risk", "layer3_classification"}
    輸出：{"status", "entities", "relationships", "chunks", "metadata"}
    """
    data          = request.get_json(force=True, silent=True) or {}
    email_content = str(data.get('email_content', '')).strip()
    user_query    = str(data.get('user_query', '請分析此郵件的知識圖譜')).strip()
    rag_mode      = str(data.get('rag_mode', 'hybrid')).strip()
    ml_result     = {
        'layer2_risk':           data.get('layer2_risk', {}),
        'layer3_classification': data.get('layer3_classification', {}),
    }

    query_text = _build_analysis_query(email_content, ml_result, user_query)
    payload    = {'query': query_text, 'mode': rag_mode}

    try:
        resp      = requests.post(LIGHTRAG_DATA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        body      = resp.json()
        graph_raw = body.get('data', {})
        return jsonify({
            'status':        'success',
            'entities':      graph_raw.get('entities',      []),
            'relationships': graph_raw.get('relationships', []),
            'chunks':        graph_raw.get('chunks',        []),
            'metadata':      body.get('metadata', {}),
        })
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else 'N/A'
        body_text = exc.response.text[:300] if exc.response is not None else ''
        return jsonify({'status': 'error', 'message': f'LightRAG HTTP {code}: {body_text}'}), 502
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
