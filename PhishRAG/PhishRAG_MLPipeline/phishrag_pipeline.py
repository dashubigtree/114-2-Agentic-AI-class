"""
phishrag_pipeline.py
====================
PhishRAG 釣魚郵件多標籤分類系統
課程：LLM Applications in Cybersecurity

功能：
    - 特徵工程：從原始郵件文字擷取 76 個特徵
    - 模型訓練：三層階層式 XGBoost 分類器
        - Model A：是否為釣魚（binary）
        - Model B：釣魚類型（11 類）
        - Model C：嚴重程度（low / medium / high）
    - 推論函式：analyze_email()

使用方式：
    python phishrag_pipeline.py

輸入檔案：
    phishing_legit_dataset_KD_10000.csv

輸出檔案：
    features.csv             — 工程化特徵矩陣
    tfidf_vocabulary.pkl     — 固定 TF-IDF 詞彙表
    model_a_binary.pkl       — Model A 模型
    model_b_type.pkl         — Model B 模型
    model_c_severity.pkl     — Model C 模型
    label_encoder_type.pkl   — 釣魚類型 label encoder
    label_encoder_severity.pkl — 嚴重程度 label encoder
    feature_cols.pkl         — 特徵欄位名稱清單
    hand_feature_cols.pkl    — 手工特徵欄位名稱清單
    confusion_matrices.png   — 三個模型的混淆矩陣
    shap_beeswarm_modelA.png — Model A SHAP Beeswarm 圖
    shap_per_type.png        — 各釣魚類型 SHAP 特徵重要性圖
"""

# ─────────────────────────────────────────────
# 匯入套件
# ─────────────────────────────────────────────
import re
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay,
    roc_auc_score, accuracy_score, f1_score
)
from xgboost import XGBClassifier

print('套件匯入完成')


# ─────────────────────────────────────────────
# 區塊 A：特徵工程
# ─────────────────────────────────────────────

# A1. 讀取原始資料
# ─────────────────────────────────────────────
INPUT_PATH = 'phishing_legit_dataset_KD_10000.csv'

df = pd.read_csv(INPUT_PATH)
print(f'載入 {len(df):,} 筆資料 | 欄位：{df.columns.tolist()}')
print()
print('Label 分佈：')
print(df['label'].value_counts())
print()
print('釣魚類型分佈：')
print(df['phishing_type'].value_counts())
print()
print('嚴重程度分佈：')
print(df['severity'].value_counts())


# A2. 輔助函式定義
# ─────────────────────────────────────────────

def split_body_keywords(text: str):
    """
    釣魚信件樣本包含 'Keywords: ...' 區塊。
    此函式將郵件拆分為 (body_text, keywords_string)。
    正常信件沒有關鍵字區塊，回傳空字串。
    """
    if not isinstance(text, str):
        return '', ''
    parts = re.split(r'\nKeywords:', text, maxsplit=1)
    body = parts[0].strip()
    kws  = parts[1].strip() if len(parts) > 1 else ''
    return body, kws


# 釣魚信件常見詞彙字典，用於計算各類型的詞彙分數
LEXICONS = {
    # 緊迫感 / 時間壓力詞彙
    'urgency_score': [
        'urgent', 'immediately', 'expire', 'expires', 'act now', 'limited time',
        'last chance', 'hurry', "don't wait", 'time sensitive', 'deadline',
        'today only', 'right away', 'instantly', 'asap',
    ],
    # 財務 / 獎品誘餌詞彙
    'financial_score': [
        'refund', 'winner', 'prize', 'reward', 'gift', 'lottery', 'million',
        'credit card', 'ssn', 'wire transfer', 'payment', 'invoice', 'cash',
        'bonus', 'offer', '$', '£', '€',
    ],
    # 權威機構 / 品牌假冒詞彙
    'authority_score': [
        'official', 'government', 'irs', 'fbi', 'mcafee', 'apple', 'microsoft',
        'paypal', 'amazon', 'netflix', 'google', 'facebook', 'compliance',
        'fraud prevention', 'security team', 'telegram',
    ],
    # 憑證竊取詞彙
    'credential_score': [
        'verify', 'verification', 'username', 'password', 'pin', 'sign in',
        'log in', 'login', 'account locked', 'restore access', 'confirm',
        'secure link', 'click here', 'update your',
    ],
    # 威脅 / 恐嚇詞彙
    'threat_score': [
        'suspend', 'suspended', 'suspension', 'penalty', 'violation',
        'failure to respond', 'legal action', 'terminated', 'blocked',
        'unauthorized', 'breach',
    ],
    # 個人資料索取詞彙
    'pii_request_score': [
        'provide your', 'enter your', 'submit your', 'send us your',
        'social security', 'date of birth', "mother's maiden",
        'credit card number', 'bank account',
    ],
    # 品牌假冒詞彙
    'brand_impersonation_score': [
        'apple', 'paypal', 'amazon', 'netflix', 'microsoft', 'google',
        'facebook', 'instagram', 'twitter', 'linkedin', 'bank', 'chase',
        'citibank', 'wells fargo',
    ],
    # 社交工程 / 交友詐騙詞彙
    'social_engineering_score': [
        'i miss you', 'i love you', 'special connection', 'lonely',
        'someone special', 'lonely heart', 'romantic', 'dating',
        'meet you', "let's meet",
    ],
}


def count_keywords(text: str, kw_list: list) -> int:
    """計算文字中出現關鍵字的數量（不分大小寫）。"""
    if not isinstance(text, str):
        return 0
    t = text.lower()
    return sum(1 for kw in kw_list if kw in t)


print('輔助函式定義完成')


# A3. 擷取特徵
# ─────────────────────────────────────────────
t0 = time.time()
print('開始擷取特徵...')

# 拆分郵件 body 與 keywords 區塊
df['body']     = df['text'].apply(lambda t: split_body_keywords(t)[0])
df['keywords'] = df['text'].apply(lambda t: split_body_keywords(t)[1])

# --- 文字長度特徵 ---
df['char_count']       = df['body'].str.len()
df['word_count']       = df['body'].apply(lambda t: len(re.findall(r'\b\w+\b', t)))
df['sentence_count']   = df['body'].apply(lambda t: len(re.split(r'[.!?]+', t)))
df['avg_word_length']  = df.apply(
    lambda r: r['char_count'] / r['word_count'] if r['word_count'] > 0 else 0, axis=1)
df['avg_sentence_len'] = df.apply(
    lambda r: r['word_count'] / r['sentence_count'] if r['sentence_count'] > 0 else 0, axis=1)

# --- 結構性標記特徵 ---
df['has_subject_line']   = df['body'].str.contains(r'^Subject:', flags=re.IGNORECASE).astype(int)
df['has_greeting']       = df['body'].str.contains(
    r'^(Hello|Hi|Dear|Attention|Notice|Greetings)', flags=re.IGNORECASE).astype(int)
df['has_signature']      = df['body'].str.contains(
    r'(Sincerely|Best regards|Best,|Regards,|Support Desk|Security Team)',
    flags=re.IGNORECASE).astype(int)
df['has_keywords_block'] = (df['keywords'] != '').astype(int)
df['keyword_count']      = df['keywords'].apply(
    lambda k: len(re.findall(r'\b\w+\b', k)) if k else 0)

# --- 標點符號 / 書寫風格特徵 ---
df['exclamation_count']    = df['body'].str.count(r'!')
df['question_count']       = df['body'].str.count(r'\?')
df['uppercase_word_count'] = df['body'].apply(
    lambda t: sum(1 for w in re.findall(r'\b[A-Z]{2,}\b', t)))
df['uppercase_ratio']      = df.apply(
    lambda r: r['uppercase_word_count'] / r['word_count'] if r['word_count'] > 0 else 0, axis=1)
df['special_char_count']   = df['body'].apply(
    lambda t: sum(1 for c in t if c in '@#$%^&*+=<>[]{}|\\~`'))

# --- 詞彙分數特徵（從 LEXICONS 字典計算）---
for fname, kw_list in LEXICONS.items():
    df[fname] = df['body'].apply(lambda t: count_keywords(t, kw_list))
# 所有詞彙分數加總，作為綜合可疑程度指標
df['total_suspicious_score'] = df[list(LEXICONS.keys())].sum(axis=1)

# --- 合成 URL 特徵（seed=42 確保可重現性）---
# 真實資料集沒有 URL 欄位，以隨機抽樣方式模擬釣魚信件和正常信件的 URL 分佈差異
rng = np.random.default_rng(seed=42)
phish_mask = df['label'] == 1

# 是否含有 URL（釣魚信件機率 75%，正常信件機率 30%）
df['has_url'] = 0
df.loc[phish_mask,  'has_url'] = rng.binomial(1, 0.75, phish_mask.sum())
df.loc[~phish_mask, 'has_url'] = rng.binomial(1, 0.30, (~phish_mask).sum())


def gen_url_count(has_url, is_phish):
    """依是否為釣魚信件產生 URL 數量。"""
    if has_url == 0:
        return 0
    return int(rng.integers(1, 5)) if is_phish else int(rng.integers(1, 3))


df['url_count'] = df.apply(lambda r: gen_url_count(r['has_url'], r['label'] == 1), axis=1)

# 是否含可疑 domain（bit.ly、.ru、.xyz 等）
df['has_suspicious_domain'] = 0
sus_mask = (df['has_url'] == 1) & phish_mask
df.loc[sus_mask, 'has_suspicious_domain'] = rng.binomial(1, 0.55, sus_mask.sum())

# URL 顯示文字與實際連結不符
df['url_text_mismatch'] = 0
mm_mask = (df['has_url'] == 1) & phish_mask
df.loc[mm_mask, 'url_text_mismatch'] = rng.binomial(1, 0.50, mm_mask.sum())

# 使用 IP 位址作為 URL host
df['has_ip_url'] = 0
ip_mask = (df['has_url'] == 1) & phish_mask
df.loc[ip_mask, 'has_ip_url'] = rng.binomial(1, 0.20, ip_mask.sum())

# --- 合成寄件者 / 標頭特徵 ---
# 寄件者 domain 可疑（釣魚：60%，正常：5%）
df['sender_domain_suspicious'] = 0
df.loc[phish_mask,  'sender_domain_suspicious'] = rng.binomial(1, 0.60, phish_mask.sum())
df.loc[~phish_mask, 'sender_domain_suspicious'] = rng.binomial(1, 0.05, (~phish_mask).sum())

# 顯示名稱與實際 domain 不符（釣魚：45%，正常：3%）
df['display_name_mismatch'] = 0
df.loc[phish_mask,  'display_name_mismatch'] = rng.binomial(1, 0.45, phish_mask.sum())
df.loc[~phish_mask, 'display_name_mismatch'] = rng.binomial(1, 0.03, (~phish_mask).sum())

# Reply-To 與 From 不一致（釣魚：35%，正常：2%）
df['reply_to_mismatch'] = 0
df.loc[phish_mask,  'reply_to_mismatch'] = rng.binomial(1, 0.35, phish_mask.sum())
df.loc[~phish_mask, 'reply_to_mismatch'] = rng.binomial(1, 0.02, (~phish_mask).sum())

# 含附件（釣魚：30%，正常：15%）
df['has_attachment'] = 0
df.loc[phish_mask,  'has_attachment'] = rng.binomial(1, 0.30, phish_mask.sum())
df.loc[~phish_mask, 'has_attachment'] = rng.binomial(1, 0.15, (~phish_mask).sum())

# --- 元資料編碼 ---
# 將 severity 文字轉為數值（low=0, medium=1, high=2）
severity_map = {'low': 0, 'medium': 1, 'high': 2}
df['severity_encoded'] = df['severity'].map(severity_map)

# 將 phishing_type 做 one-hot 編碼（排除 legitimate 作為基準類別）
type_dummies = pd.get_dummies(df['phishing_type'], prefix='type', drop_first=False)
type_dummies = type_dummies.drop(columns=['type_legitimate'], errors='ignore')

# --- 風險分數計算（規則型，不依賴 label）---
def compute_risk_score(row) -> float:
    """
    依各種信號計算 0~100 的風險分數。
    結合詞彙信號、結構信號、標頭信號和書寫風格信號。
    """
    score = 0.0
    # 詞彙信號（最高貢獻約 40 分）
    score += min(row['urgency_score']            * 4, 12)
    score += min(row['credential_score']         * 4, 10)
    score += min(row['financial_score']          * 3, 10)
    score += min(row['threat_score']             * 4,  8)
    score += min(row['authority_score']          * 2,  6)
    score += min(row['pii_request_score']        * 5, 10)
    score += min(row['social_engineering_score'] * 3,  6)
    # 結構信號（最高貢獻約 30 分）
    score += row['has_keywords_block']           * 20
    score += row['has_url']                      *  5
    score += row['has_suspicious_domain']        *  5
    score += row['url_text_mismatch']            *  4
    score += row['has_ip_url']                   *  3
    score += row['has_attachment']               *  2
    # 標頭 / 寄件者信號（最高貢獻約 15 分）
    score += row['sender_domain_suspicious']     *  6
    score += row['display_name_mismatch']        *  5
    score += row['reply_to_mismatch']            *  4
    # 書寫風格信號（最高貢獻約 10 分）
    score += min(row['exclamation_count']        *  1,  4)
    score += min(row['uppercase_ratio']          * 20,  4)
    score += min(row['url_count']                *  1,  3)
    # 依嚴重程度調整倍率
    severity_mult = {0: 0.8, 1: 1.0, 2: 1.2}.get(int(row['severity_encoded']), 1.0)
    score *= severity_mult
    return round(min(score, 100), 2)


df['risk_score']      = df.apply(compute_risk_score, axis=1)
df['risk_prediction'] = (df['risk_score'] >= 35).astype(int)

# --- TF-IDF 詞彙特徵 ---
# 從郵件 body 擷取 top-50 bigram 詞彙的 TF-IDF 值
tfidf = TfidfVectorizer(
    max_features=50,
    ngram_range=(1, 2),
    sublinear_tf=True,
    token_pattern=r'\b[a-zA-Z]{2,}\b',
    stop_words='english'
)
tfidf_matrix = tfidf.fit_transform(df['body'].fillna(''))
tfidf_cols   = [f'tfidf_{t}' for t in tfidf.get_feature_names_out()]
tfidf_df     = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index)

# 儲存固定詞彙表，確保推論時每次使用相同的詞彙集合
joblib.dump(tfidf.vocabulary_, 'tfidf_vocabulary.pkl')

# --- 組裝最終特徵 DataFrame ---
HAND_CRAFTED_COLS = [
    'char_count', 'word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_len',
    'has_subject_line', 'has_greeting', 'has_signature', 'has_keywords_block', 'keyword_count',
    'exclamation_count', 'question_count', 'uppercase_word_count', 'uppercase_ratio', 'special_char_count',
    'urgency_score', 'financial_score', 'authority_score', 'credential_score', 'threat_score',
    'pii_request_score', 'brand_impersonation_score', 'social_engineering_score', 'total_suspicious_score',
    'has_url', 'url_count', 'has_suspicious_domain', 'url_text_mismatch', 'has_ip_url',
    'sender_domain_suspicious', 'display_name_mismatch', 'reply_to_mismatch', 'has_attachment',
    'severity_encoded', 'confidence', 'risk_score', 'risk_prediction',
]

feature_df = pd.concat([df[HAND_CRAFTED_COLS], type_dummies, tfidf_df, df['label']], axis=1)
feature_df.to_csv('features.csv', index=False)

elapsed = time.time() - t0
print(f'特徵工程完成，耗時 {elapsed:.1f}s')
print(f'features.csv shape: {feature_df.shape}')
print(f'tfidf_vocabulary.pkl 已儲存（{len(tfidf.vocabulary_)} 個詞彙）')


# A4. 準備特徵矩陣（排除 Data Leakage 欄位）
# ─────────────────────────────────────────────
df_feat = pd.read_csv('features.csv')

# 第一輪排除：直接由 label 衍生的欄位
leakage_cols = ['risk_prediction', 'risk_score', 'confidence', 'severity_encoded']
type_cols    = [c for c in df_feat.columns if c.startswith('type_')]

# 第二輪排除：資料集生成規則造成的結構性洩漏（相關係數 >= 0.92）
structure_leakage = [
    'has_keywords_block',  # 與 label 相關係數 +1.000
    'has_subject_line',    # 與 label 相關係數 -1.000
    'has_signature',       # 與 label 相關係數 +0.969
    'tfidf_best',          # 只出現在正常信件
    'tfidf_subject',       # 只出現在正常信件
    'keyword_count',       # 與 label 相關係數 +0.924
    'has_greeting',        # 與 label 相關係數 +0.919
]

exclude_cols      = leakage_cols + type_cols + structure_leakage + ['label']
FEATURE_COLS      = [c for c in df_feat.columns if c not in exclude_cols]
HAND_FEATURE_COLS = [c for c in FEATURE_COLS if not c.startswith('tfidf_')]
TFIDF_COLS        = [c for c in FEATURE_COLS if c.startswith('tfidf_')]

X_full = df_feat[FEATURE_COLS].copy()

print(f'使用特徵總數 : {len(FEATURE_COLS)}')
print(f'  手工特徵   : {len(HAND_FEATURE_COLS)}')
print(f'  TF-IDF     : {len(TFIDF_COLS)}')
print(f'排除欄位數   : {len(exclude_cols) - 1}')


# A5. 準備標籤與訓練/測試分割
# ─────────────────────────────────────────────
# Model A 標籤：二元（是否釣魚）
y_binary = df['label'].values

# Model B 標籤：釣魚類型（字串 → 數字）
le_type = LabelEncoder()
y_type  = le_type.fit_transform(df['phishing_type'])

# Model C 標籤：嚴重程度（字串 → 數字）
le_sev  = LabelEncoder()
y_sev   = le_sev.fit_transform(df['severity'])

print('Model A 類別：', {0: 'legitimate', 1: 'phishing'})
print('Model B 類別：', dict(enumerate(le_type.classes_)))
print('Model C 類別：', dict(enumerate(le_sev.classes_)))

# 依 label 做 stratified split，確保訓練集和測試集的類別比例一致
idx_train, idx_test = train_test_split(
    np.arange(len(X_full)), test_size=0.2, random_state=42, stratify=y_binary
)
X_train = X_full.iloc[idx_train].reset_index(drop=True)
X_test  = X_full.iloc[idx_test].reset_index(drop=True)

yA_train, yA_test = y_binary[idx_train], y_binary[idx_test]
yB_train, yB_test = y_type[idx_train],   y_type[idx_test]
yC_train, yC_test = y_sev[idx_train],    y_sev[idx_test]

print(f'訓練集：{len(X_train)} 筆 | 測試集：{len(X_test)} 筆')


# ─────────────────────────────────────────────
# 區塊 B：模型訓練與評估
# ─────────────────────────────────────────────

# B1. Model A — 是否為釣魚（Binary Classification）
# ─────────────────────────────────────────────
# 輸入：76 維特徵向量
# 輸出：0（正常）或 1（釣魚）
print('\n訓練 Model A...')
t0 = time.time()

model_a = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    random_state=42, eval_metric='logloss', verbosity=0
)
model_a.fit(X_train, yA_train)

train_time_a = time.time() - t0
print(f'Model A 訓練時間：{train_time_a:.2f}s')

yA_pred = model_a.predict(X_test)
yA_prob = model_a.predict_proba(X_test)[:, 1]

print()
print('=== Model A: Binary Classification ===')
print(classification_report(yA_test, yA_pred, target_names=['Legitimate', 'Phishing']))
print(f'Accuracy : {accuracy_score(yA_test, yA_pred):.4f}')
print(f'F1 macro : {f1_score(yA_test, yA_pred, average="macro"):.4f}')
print(f'ROC-AUC  : {roc_auc_score(yA_test, yA_prob):.4f}')


# B2. Model B — 釣魚類型（Multi-class，11 類）
# ─────────────────────────────────────────────
# 輸入：76 維特徵向量
# 輸出：11 種類別之一（含 legitimate）
print('\n訓練 Model B...')
t0 = time.time()

model_b = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    random_state=42, eval_metric='mlogloss', verbosity=0
)
model_b.fit(X_train, yB_train)

train_time_b = time.time() - t0
print(f'Model B 訓練時間：{train_time_b:.2f}s')

yB_pred = model_b.predict(X_test)
yB_prob = model_b.predict_proba(X_test)

print()
print('=== Model B: Phishing Type Classification ===')
print(classification_report(yB_test, yB_pred, target_names=le_type.classes_))
print(f'Accuracy : {accuracy_score(yB_test, yB_pred):.4f}')
print(f'F1 macro : {f1_score(yB_test, yB_pred, average="macro"):.4f}')

# 使用 cross_val_predict 在訓練集上產生 Model B 的預測機率
# 作為 Model C 的額外輸入特徵，避免 Data Leakage
print('\n執行 cross_val_predict（供 Model C 使用）...')
t0 = time.time()
yB_train_cv = cross_val_predict(
    model_b, X_train, yB_train, cv=5, method='predict_proba'
)
print(f'cross_val_predict 完成，耗時 {time.time()-t0:.2f}s | shape: {yB_train_cv.shape}')


# B3. Model C — 嚴重程度（Multi-class：low / medium / high）
# ─────────────────────────────────────────────
# 輸入：76 維特徵 + 11 維 Model B 類型機率 = 87 維
# 輸出：low / medium / high
# 加入 Model B 輸出讓 Model C 能學到「釣魚類型與嚴重程度的對應關係」
X_train_c = np.hstack([X_train.values, yB_train_cv])
X_test_c  = np.hstack([X_test.values,  yB_prob])
print(f'\nModel C 輸入維度：{X_train_c.shape[1]}（{X_train.shape[1]} 特徵 + {yB_train_cv.shape[1]} 類型機率）')

print('訓練 Model C...')
t0 = time.time()

model_c = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    random_state=42, eval_metric='mlogloss', verbosity=0
)
model_c.fit(X_train_c, yC_train)

train_time_c = time.time() - t0
print(f'Model C 訓練時間：{train_time_c:.2f}s')

yC_pred = model_c.predict(X_test_c)

print()
print('=== Model C: Severity Classification ===')
print(classification_report(yC_test, yC_pred, target_names=le_sev.classes_))
print(f'Accuracy : {accuracy_score(yC_test, yC_pred):.4f}')
print(f'F1 macro : {f1_score(yC_test, yC_pred, average="macro"):.4f}')


# B4. 訓練時間與效能摘要
# ─────────────────────────────────────────────
print('\n=== 訓練時間摘要 ===')
print(f'Model A（binary）          : {train_time_a:.2f}s')
print(f'Model B（type 11-class）   : {train_time_b:.2f}s')
print(f'Model C（severity 3-class）: {train_time_c:.2f}s')
print(f'總計                        : {train_time_a+train_time_b+train_time_c:.2f}s')

summary = pd.DataFrame({
    'Model': ['Model A (binary)', 'Model B (type 11-class)', 'Model C (severity 3-class)'],
    'Task':  ['Is phishing?', 'Phishing type', 'Severity'],
    'Accuracy':       [accuracy_score(yA_test, yA_pred),
                       accuracy_score(yB_test, yB_pred),
                       accuracy_score(yC_test, yC_pred)],
    'F1 (macro)':     [f1_score(yA_test, yA_pred, average='macro'),
                       f1_score(yB_test, yB_pred, average='macro'),
                       f1_score(yC_test, yC_pred, average='macro')],
    'Train Time (s)': [round(train_time_a, 2), round(train_time_b, 2), round(train_time_c, 2)]
}).set_index('Model').round(4)
print(summary.to_string())


# B5. 混淆矩陣
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ConfusionMatrixDisplay.from_predictions(
    yA_test, yA_pred, display_labels=['Legitimate', 'Phishing'],
    cmap='Blues', ax=axes[0])
axes[0].set_title('Model A - Is Phishing?')

ConfusionMatrixDisplay.from_predictions(
    yB_test, yB_pred, display_labels=le_type.classes_,
    cmap='Blues', ax=axes[1], xticks_rotation=45)
axes[1].set_title('Model B - Phishing Type')

ConfusionMatrixDisplay.from_predictions(
    yC_test, yC_pred, display_labels=le_sev.classes_,
    cmap='Blues', ax=axes[2])
axes[2].set_title('Model C - Severity')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print('confusion_matrices.png 已儲存')


# B6. SHAP 分析 — Model A 特徵重要性
# ─────────────────────────────────────────────
# SHAP Beeswarm Plot：每個點代表一筆資料
# 橫軸正值表示該特徵讓模型傾向判定為釣魚，負值則相反
explainer_a   = shap.TreeExplainer(model_a)
shap_values_a = explainer_a.shap_values(X_test[:500])

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values_a, X_test[:500],
                  feature_names=FEATURE_COLS, max_display=20, show=False)
plt.title('SHAP Beeswarm - Model A (Phishing Risk)')
plt.tight_layout()
plt.savefig('shap_beeswarm_modelA.png', dpi=150, bbox_inches='tight')
plt.show()
print('shap_beeswarm_modelA.png 已儲存')


# B7. SHAP 分析 — Model B 各釣魚類型特徵重要性
# ─────────────────────────────────────────────
# 對每種釣魚類型各畫一個 Top-10 特徵長條圖
# 可以看到不同類型的釣魚信件各自依賴哪些特徵
explainer_b   = shap.TreeExplainer(model_b)
shap_values_b = np.array(explainer_b.shap_values(X_test[:300])).transpose(1, 2, 0)
# shape: (n_samples, n_features, n_classes)

# 排除 legitimate，只畫釣魚類型
phishing_cls_idx   = [i for i, c in enumerate(le_type.classes_) if c != 'legitimate']
phishing_cls_names = [le_type.classes_[i] for i in phishing_cls_idx]

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()
for ax_i, (ci, cn) in enumerate(zip(phishing_cls_idx, phishing_cls_names)):
    sv       = shap_values_b[:, :, ci]
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:10]
    axes[ax_i].barh([FEATURE_COLS[i] for i in top_idx[::-1]],
                    mean_abs[top_idx[::-1]], color='#DD8452')
    axes[ax_i].set_title(cn.replace('_', ' ').title(), fontsize=9)
    axes[ax_i].set_xlabel('Mean |SHAP|', fontsize=8)
    axes[ax_i].tick_params(labelsize=7)

plt.suptitle('SHAP Feature Importance per Phishing Type', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('shap_per_type.png', dpi=150, bbox_inches='tight')
plt.show()
print('shap_per_type.png 已儲存')


# B8. 儲存所有模型與 Pipeline 元件
# ─────────────────────────────────────────────
joblib.dump(model_a,           'model_a_binary.pkl')
joblib.dump(model_b,           'model_b_type.pkl')
joblib.dump(model_c,           'model_c_severity.pkl')
joblib.dump(le_type,           'label_encoder_type.pkl')
joblib.dump(le_sev,            'label_encoder_severity.pkl')
joblib.dump(FEATURE_COLS,      'feature_cols.pkl')
joblib.dump(HAND_FEATURE_COLS, 'hand_feature_cols.pkl')

print('\n已儲存的檔案：')
for f in ['model_a_binary.pkl', 'model_b_type.pkl', 'model_c_severity.pkl',
          'label_encoder_type.pkl', 'label_encoder_severity.pkl',
          'feature_cols.pkl', 'hand_feature_cols.pkl', 'tfidf_vocabulary.pkl']:
    if os.path.exists(f):
        print(f'  {f}: {os.path.getsize(f)/1024:.1f} KB')


# ─────────────────────────────────────────────
# 區塊 C：推論函式
# ─────────────────────────────────────────────

# C1. 載入固定 TF-IDF 詞彙表
# ─────────────────────────────────────────────
# 使用特徵工程時儲存的固定詞彙表，確保推論結果與訓練一致
vocab = joblib.load('tfidf_vocabulary.pkl')
tfidf_inference = TfidfVectorizer(
    vocabulary=vocab,
    ngram_range=(1, 2),
    sublinear_tf=True,
    token_pattern=r'\b[a-zA-Z]{2,}\b',
    stop_words='english',
)
# 用訓練集 body 文字 fit（固定詞彙下 fit 只是建立 idf，不改變詞彙集合）
tfidf_inference.fit(df['body'].iloc[idx_train])
TFIDF_VOCAB_COLS = [f'tfidf_{t}' for t in tfidf_inference.get_feature_names_out()]

# 合成特徵在推論時無法從真實郵件取得，使用釣魚信件類別的平均值作為估計
SYNTH_MEAN = {
    'has_url': 0.75,
    'url_count': 2.0,
    'has_suspicious_domain': 0.55,
    'url_text_mismatch': 0.50,
    'has_ip_url': 0.20,
    'sender_domain_suspicious': 0.60,
    'display_name_mismatch': 0.45,
    'reply_to_mismatch': 0.35,
    'has_attachment': 0.30,
}

print(f'TF-IDF 詞彙表載入完成：{len(vocab)} 個詞彙')


# C2. 推論用特徵擷取函式
# ─────────────────────────────────────────────
def extract_features_for_inference(raw_text: str) -> dict:
    """
    從單封原始郵件文字擷取特徵，供推論使用。
    邏輯與訓練階段的特徵工程相同。
    合成特徵（URL、寄件者）使用釣魚信件類別的平均值估計。
    """
    body, _ = split_body_keywords(raw_text)
    words   = re.findall(r'\b\w+\b', body)
    sents   = [s for s in re.split(r'[.!?]+', body) if s.strip()]
    uc      = re.findall(r'\b[A-Z]{2,}\b', body)

    feat = {
        'char_count':           len(body),
        'word_count':           len(words),
        'sentence_count':       len(sents),
        'avg_word_length':      len(body) / len(words) if words else 0,
        'avg_sentence_len':     len(words) / len(sents) if sents else 0,
        'exclamation_count':    body.count('!'),
        'question_count':       body.count('?'),
        'uppercase_word_count': len(uc),
        'uppercase_ratio':      len(uc) / len(words) if words else 0,
        'special_char_count':   sum(1 for c in body if c in '@#$%^&*+=<>[]{}|\\~`'),
    }
    # 計算各類詞彙分數
    for fname, kw_list in LEXICONS.items():
        feat[fname] = count_keywords(body, kw_list)
    feat['total_suspicious_score'] = sum(feat[k] for k in LEXICONS)
    # 合成特徵使用平均值
    feat.update(SYNTH_MEAN)
    return feat


# C3. 核心推論函式
# ─────────────────────────────────────────────
def analyze_email(raw_email_text: str, show_shap: bool = False) -> dict:
    """
    PhishRAG 完整推論流程。

    參數：
        raw_email_text : 原始郵件文字（字串）
        show_shap      : 是否計算 SHAP 特徵說明（預設 False）

    回傳：
        dict，包含以下欄位：
            is_phishing     : int   (0 = 正常，1 = 釣魚)
            risk_score      : float (Model A 的釣魚機率，0.0 ~ 1.0)
            phishing_type   : str   (釣魚類型，如 'credential_harvesting')
            type_confidence : float (Model B 的預測信心度)
            severity        : str   ('low', 'medium', 'high')
            top_features    : list  (只有 show_shap=True 時才有，
                                     格式：[(特徵名, 特徵值, SHAP貢獻), ...])
    """
    # Step 1：特徵工程
    body, _    = split_body_keywords(raw_email_text)
    hand_feat  = extract_features_for_inference(raw_email_text)
    tfidf_vec  = tfidf_inference.transform([body]).toarray()[0]
    tfidf_dict = dict(zip(TFIDF_VOCAB_COLS, tfidf_vec))
    combined   = {**hand_feat, **tfidf_dict}
    # 確保特徵排列順序與訓練時一致
    X_single   = pd.DataFrame(
        [[combined.get(c, 0.0) for c in FEATURE_COLS]],
        columns=FEATURE_COLS
    )

    # Step 2：Model A — 是否為釣魚？
    is_phishing = int(model_a.predict(X_single)[0])
    risk_score  = float(model_a.predict_proba(X_single)[0, 1])

    # 若判定為正常信件，直接回傳，不執行後續模型
    if is_phishing == 0:
        return {
            'is_phishing':   0,
            'risk_score':    round(risk_score, 4),
            'phishing_type': 'legitimate',
            'severity':      'low'
        }

    # Step 3：Model B — 釣魚類型
    type_proba    = model_b.predict_proba(X_single)
    type_idx      = int(model_b.predict(X_single)[0])
    phishing_type = le_type.classes_[type_idx]
    type_conf     = float(type_proba[0, type_idx])

    # Step 4：Model C — 嚴重程度
    # 輸入包含原始特徵 + Model B 的類型機率
    X_single_c = np.hstack([X_single.values, type_proba])
    sev_idx    = int(model_c.predict(X_single_c)[0])
    severity   = le_sev.classes_[sev_idx]

    result = {
        'is_phishing':     1,
        'risk_score':      round(risk_score, 4),
        'phishing_type':   phishing_type,
        'type_confidence': round(type_conf, 4),
        'severity':        severity
    }

    # Step 5：SHAP 特徵說明（只有 show_shap=True 才計算）
    if show_shap:
        sv = explainer_a.shap_values(X_single)[0]
        # 篩選正向貢獻（讓釣魚判定信心上升）且特徵值非零的特徵
        top_features = sorted(
            [(FEATURE_COLS[i], float(X_single.iloc[0, i]), float(sv[i]))
             for i in range(len(FEATURE_COLS))
             if sv[i] > 0.005 and X_single.iloc[0, i] != 0],
            key=lambda x: x[2], reverse=True
        )[:5]
        result['top_features'] = top_features

    return result


print('analyze_email() 函式定義完成')


# C4. 示範：分析樣本郵件
# ─────────────────────────────────────────────
test_df = df.iloc[idx_test].reset_index(drop=True)

phishing_sample = test_df[test_df['label'] == 1].iloc[0]['text']
legit_sample    = test_df[test_df['label'] == 0].iloc[0]['text']

print('\n' + '=' * 50)
print('SAMPLE 1：釣魚信件')
print('=' * 50)
print('內容（前 200 字元）：', phishing_sample[:200], '...')

# 傳入 show_shap=True 以顯示 SHAP 特徵說明
r1 = analyze_email(phishing_sample, show_shap=True)
print()
for k, v in r1.items():
    if k == 'top_features':
        print(f'  {"top_features":<18}:')
        for name, val, shap_v in v:
            print(f'    {name}: {val:.3f}  (SHAP +{shap_v:.3f})')
    else:
        print(f'  {k:<18}: {v}')

print()
print('=' * 50)
print('SAMPLE 2：正常信件')
print('=' * 50)
print('內容（前 200 字元）：', legit_sample[:200], '...')
r2 = analyze_email(legit_sample)
print()
for k, v in r2.items():
    print(f'  {k:<18}: {v}')


# C5. 批次推論驗證
# ─────────────────────────────────────────────
# 對測試集前 20 筆釣魚信件執行推論，並與真實標籤比對
phishing_test = test_df[test_df['label'] == 1].head(20)

batch_results = []
for _, row in phishing_test.iterrows():
    result = analyze_email(row['text'])
    result['true_type']     = row['phishing_type']
    result['true_severity'] = row['severity']
    batch_results.append(result)

batch_df = pd.DataFrame(batch_results)
print('\n批次推論結果（20 筆釣魚信件）：')
print(batch_df[['is_phishing', 'risk_score', 'phishing_type', 'true_type',
                'severity', 'true_severity']].to_string(index=False))

type_correct = (batch_df['phishing_type'] == batch_df['true_type']).mean()
sev_correct  = (batch_df['severity'] == batch_df['true_severity']).mean()
print(f'\n類型準確率（樣本）  : {type_correct:.2%}')
print(f'嚴重程度準確率（樣本）: {sev_correct:.2%}')
