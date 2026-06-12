"""
dashboard.py — PhishRAG Streamlit 前端儀表板
雙對話框佈局：左側郵件輸入（分析後鎖定）+ 右側 RAG 問答（支援多輪追問）
"""

import os
import streamlit as st
import requests

BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000')

st.set_page_config(
    page_title='PhishRAG — 釣魚郵件威脅情資分析儀表板',
    page_icon='🎣',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ─── 全局 CSS 樣式 ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-label { font-size: 0.85rem; color: #888; }
    .stChatMessage { border-radius: 8px; }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTextArea"]) { height: 100%; }
</style>
""", unsafe_allow_html=True)

# ─── Session State 初始化 ──────────────────────────────────────────────────
_defaults = {
    'analysis_done':         False,
    'email_content':         '',
    'layer1_features':       {},
    'layer2_risk':           {},
    'layer3_classification': {},
    'chat_history':          [],   # list[{role: str, content: str}]
    'user_prompt':           '',
    'top_k':                 None,
    'chunk_top_k':           None,
    'max_entity_tokens':     None,
    'max_relation_tokens':   None,
    'max_total_tokens':      None,
    'graph_data':            None,   # None = 未載入；dict = 圖譜資料
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── API 輔助函式 ─────────────────────────────────────────────────────────

def _rag_extra_params() -> dict:
    """從 session state 收集進階查詢參數，None 或空值不附加。"""
    raw = {
        'user_prompt':         st.session_state.get('user_prompt_input', ''),
        'top_k':               st.session_state.get('top_k_input'),
        'chunk_top_k':         st.session_state.get('chunk_top_k_input'),
        'max_entity_tokens':   st.session_state.get('max_entity_tokens_input'),
        'max_relation_tokens': st.session_state.get('max_relation_tokens_input'),
        'max_total_tokens':    st.session_state.get('max_total_tokens_input'),
    }
    return {k: v for k, v in raw.items() if v}


def call_analyze(email_content: str, user_query: str, rag_mode: str) -> dict:
    """呼叫後端 /api/v1/analyze，執行完整四層流水線。"""
    try:
        resp = requests.post(
            f'{BACKEND_URL}/api/v1/analyze',
            json={'email_content': email_content, 'user_query': user_query, 'rag_mode': rag_mode,
                  **_rag_extra_params()},
            timeout=200,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {'status': 'error', 'message': '無法連接後端服務。請確認 Flask 伺服器正在 http://localhost:5000 執行。'}
    except requests.exceptions.Timeout:
        return {'status': 'error', 'message': '後端服務回應超時（>200s）。請稍後再試。'}
    except Exception as exc:
        return {'status': 'error', 'message': f'API 錯誤：{exc}'}


def call_chat(email_content: str, current_query: str, rag_mode: str, chat_history: list) -> str:
    """呼叫後端 /api/v1/chat，執行多輪追問（跳過 ML 流水線）。"""
    try:
        resp = requests.post(
            f'{BACKEND_URL}/api/v1/chat',
            json={
                'email_content': email_content,
                'current_query': current_query,
                'rag_mode':      rag_mode,
                'chat_history':  chat_history,
                **_rag_extra_params(),
            },
            timeout=200,
        )
        resp.raise_for_status()
        return resp.json().get('reply', '（空回應）')
    except requests.exceptions.ConnectionError:
        return '⚠️ 無法連接後端服務，請確認 Flask 伺服器正在運行。'
    except requests.exceptions.Timeout:
        return '⚠️ 後端服務回應超時，請稍後再試。'
    except Exception as exc:
        return f'⚠️ 聊天 API 錯誤：{exc}'


def call_query_graph(
    email_content: str,
    user_query: str,
    rag_mode: str,
    layer2_risk: dict,
    layer3_classification: dict,
) -> dict:
    """呼叫後端 /api/v1/query_graph，取得知識圖譜原始資料（不觸發 LLM）。"""
    try:
        resp = requests.post(
            f'{BACKEND_URL}/api/v1/query_graph',
            json={
                'email_content':         email_content,
                'user_query':            user_query,
                'rag_mode':              rag_mode,
                'layer2_risk':           layer2_risk,
                'layer3_classification': layer3_classification,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {'status': 'error', 'message': '⚠️ 無法連接後端服務'}
    except requests.exceptions.Timeout:
        return {'status': 'error', 'message': '⚠️ 知識圖譜查詢超時（>60s）'}
    except Exception as exc:
        return {'status': 'error', 'message': f'⚠️ 圖譜查詢錯誤：{exc}'}


def _generate_vis_html(entities: list, relationships: list) -> str:
    """生成自包含的 vis.js 網路圖 HTML 字串。"""
    import json as _json

    TYPE_COLORS = {
        'TECHNIQUE':    '#e74c3c',
        'ATTACK':       '#e74c3c',
        'TACTIC':       '#c0392b',
        'ORGANIZATION': '#3498db',
        'ACTOR':        '#9b59b6',
        'MALWARE':      '#e67e22',
        'TOOL':         '#f39c12',
        'VULNERABILITY':'#c0392b',
        'MITIGATION':   '#27ae60',
        'INDICATOR':    '#1abc9c',
        'CONCEPT':      '#2980b9',
        'EVENT':        '#8e44ad',
    }
    DEFAULT_COLOR = '#95a5a6'

    nodes = []
    for i, e in enumerate(entities[:80]):
        etype = (e.get('entity_type') or 'default').upper()
        color = TYPE_COLORS.get(etype, DEFAULT_COLOR)
        nodes.append({
            'id':    i,
            'label': (e.get('entity_name') or '?')[:30],
            'title': (e.get('description') or '')[:150],
            'color': {'background': color, 'border': color},
            'font':  {'color': '#ffffff', 'size': 13},
            'shape': 'dot',
            'size':  18,
        })

    name_to_id = {(e.get('entity_name') or ''): i for i, e in enumerate(entities[:80])}
    edges = []
    for j, r in enumerate(relationships[:120]):
        src = name_to_id.get(r.get('src_id') or '')
        tgt = name_to_id.get(r.get('tgt_id') or '')
        if src is not None and tgt is not None:
            weight = float(r.get('weight') or 0.5)
            edges.append({
                'id':     j,
                'from':   src,
                'to':     tgt,
                'label':  (r.get('keywords') or '')[:25],
                'title':  (r.get('description') or '')[:150],
                'width':  max(1, round(weight * 5)),
                'color':  {'color': '#7f8c8d', 'highlight': '#2c3e50'},
                'arrows': 'to',
                'smooth': {'type': 'curvedCW', 'roundness': 0.2},
            })

    nodes_json = _json.dumps(nodes)
    edges_json = _json.dumps(edges)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin:0; background:#1a1a2e; }}
  #graph {{ width:100%; height:480px; }}
  #legend {{ position:absolute; top:8px; right:8px; background:rgba(0,0,0,0.6);
             padding:8px 12px; border-radius:6px; font:12px sans-serif; color:#eee; }}
  #legend div {{ margin:3px 0; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
</style>
</head>
<body>
<div id="graph"></div>
<div id="legend">
  <b>Entity Types</b>
  <div><span class="dot" style="background:#e74c3c"></span>Technique/Attack/Tactic</div>
  <div><span class="dot" style="background:#3498db"></span>Organization</div>
  <div><span class="dot" style="background:#9b59b6"></span>Actor</div>
  <div><span class="dot" style="background:#e67e22"></span>Malware</div>
  <div><span class="dot" style="background:#f39c12"></span>Tool</div>
  <div><span class="dot" style="background:#27ae60"></span>Mitigation</div>
  <div><span class="dot" style="background:#1abc9c"></span>Indicator</div>
  <div><span class="dot" style="background:#95a5a6"></span>Other</div>
</div>
<script>
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});
  var options = {{
    nodes: {{ borderWidth: 2, shadow: true }},
    edges: {{ font: {{ size: 10, color: '#bdc3c7', align: 'middle' }}, shadow: true }},
    physics: {{
      stabilization: {{ iterations: 150 }},
      barnesHut: {{ gravitationalConstant: -8000, springLength: 120 }},
    }},
    interaction: {{ hover: true, tooltipDelay: 100, navigationButtons: true }},
    background: {{ color: '#1a1a2e' }},
  }};
  new vis.Network(document.getElementById('graph'), {{nodes, edges}}, options);
</script>
</body>
</html>"""


def _render_knowledge_graph(graph_data: dict) -> None:
    """在 Streamlit 渲染知識圖譜（vis.js 網路圖 + 結構化表格）。"""
    import streamlit.components.v1 as components
    import pandas as pd

    if graph_data.get('status') == 'bypass':
        st.info('ℹ️ bypass 模式不進行圖譜檢索，請切換至其他模式後重新查詢。')
        return
    if graph_data.get('status') == 'error':
        st.error(f"❌ {graph_data.get('message', '未知錯誤')}")
        return

    entities      = graph_data.get('entities',      [])
    relationships = graph_data.get('relationships', [])
    chunks        = graph_data.get('chunks',        [])
    metadata      = graph_data.get('metadata',      {})

    if not entities and not relationships:
        st.warning('⚠️ 本次查詢未從知識圖譜中檢索到任何實體或關係。請嘗試其他 RAG 模式。')
        return

    with st.expander(
        f'🕸️ 知識圖譜（{len(entities)} 實體 · {len(relationships)} 關係 · {len(chunks)} 文字區塊）',
        expanded=True,
    ):
        # ── vis.js 網路圖 ──
        vis_html = _generate_vis_html(entities, relationships)
        components.html(vis_html, height=490, scrolling=False)

        # ── 統計摘要 ──
        proc = metadata.get('processing_info', {})
        if proc:
            c1, c2, c3 = st.columns(3)
            c1.metric('檢索實體數', proc.get('total_entities_found', len(entities)))
            c2.metric('檢索關係數', proc.get('total_relations_found', len(relationships)))
            c3.metric('最終區塊數', proc.get('final_chunks_count', len(chunks)))

        # ── 結構化資料 tab ──
        tab_e, tab_r, tab_c = st.tabs([
            f'📌 Entities ({len(entities)})',
            f'🔗 Relationships ({len(relationships)})',
            f'📄 Chunks ({len(chunks)})',
        ])

        with tab_e:
            if entities:
                df_e = pd.DataFrame([{
                    'Entity':      e.get('entity_name', ''),
                    'Type':        e.get('entity_type', ''),
                    'Description': (e.get('description') or '')[:200],
                } for e in entities])
                st.dataframe(df_e, use_container_width=True, hide_index=True)
            else:
                st.caption('無實體資料')

        with tab_r:
            if relationships:
                df_r = pd.DataFrame([{
                    'Source':      r.get('src_id', ''),
                    'Target':      r.get('tgt_id', ''),
                    'Keywords':    r.get('keywords', ''),
                    'Weight':      round(float(r.get('weight') or 0), 3),
                    'Description': (r.get('description') or '')[:150],
                } for r in relationships])
                st.dataframe(df_r, use_container_width=True, hide_index=True)
            else:
                st.caption('無關係資料')

        with tab_c:
            if chunks:
                for i, c in enumerate(chunks[:10]):
                    with st.expander(f"Chunk {i+1} — {c.get('file_path', '')}"):
                        st.text(c.get('content', '')[:500])
            else:
                st.caption('無文字區塊資料')


# ─── 頁面標題 ──────────────────────────────────────────────────────────────
st.title('🎣 PhishRAG — 釣魚郵件威脅情資分析儀表板')
st.caption('XGBoost 三層分類器 × LightRAG 知識圖譜 × MITRE ATT&CK 技術映射')

# ─── 分析完成後：頂部指標與 Layer 1 特徵展示 ──────────────────────────────
if st.session_state.analysis_done:
    l2 = st.session_state.layer2_risk
    l3 = st.session_state.layer3_classification

    # 4 個核心指標
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        is_p = l2.get('is_phishing', 0)
        st.metric(
            label='🔍 釣魚判定',
            value='釣魚 ⚠️' if is_p else '正常 ✅',
        )
    with m2:
        st.metric(
            label='📊 風險分數 (Model A)',
            value=f"{l2.get('risk_score', 0.0):.4f}",
        )
    with m3:
        level = l2.get('risk_level', 'N/A')
        level_icon = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(level, '⚪')
        st.metric(label='⚡ 風險等級 (Model C)', value=f'{level_icon} {level}')
    with m4:
        ptype = l3.get('phishing_type', 'N/A').replace('_', ' ').title()
        st.markdown(
            f'<div style="font-size:0.875rem;color:#a3a8b8;margin-bottom:4px;">🎯 釣魚類型 (Model B)</div>'
            f'<div style="font-size:1.5rem;font-weight:700;line-height:1.3;word-break:break-word;white-space:normal;">{ptype}</div>',
            unsafe_allow_html=True,
        )

    # 分類信心度進度條
    conf = l3.get('type_confidence', 0.0)
    st.progress(float(conf), text=f"Model B 分類信心度：{conf:.2%}")

    # Layer 1 技術特徵展示（76 維）
    with st.expander('🔍 檢視 Layer 1 技術特徵指標（76 維特徵向量）'):
        feat = st.session_state.layer1_features
        if feat:
            # 分組顯示：手工特徵 vs TF-IDF 特徵
            hand_feat = {k: v for k, v in feat.items() if not k.startswith('tfidf_')}
            tfidf_feat = {k: v for k, v in feat.items() if k.startswith('tfidf_')}
            tab1, tab2 = st.tabs([f'手工特徵（{len(hand_feat)} 個）', f'TF-IDF 特徵（{len(tfidf_feat)} 個）'])
            with tab1:
                st.json(hand_feat)
            with tab2:
                st.json(tfidf_feat)

    # 知識圖譜查看按鈕
    kg_btn_col, kg_info_col = st.columns([1, 4])
    with kg_btn_col:
        if st.button('🕸️ 查看知識圖譜', use_container_width=True):
            rag_mode_now = st.session_state.get('rag_mode_select', 'hybrid')
            if rag_mode_now == 'bypass':
                st.session_state.graph_data = {'status': 'bypass'}
            else:
                with st.spinner('正在從知識圖譜檢索實體與關係資料（不呼叫 LLM）...'):
                    result = call_query_graph(
                        email_content         = st.session_state.email_content,
                        user_query            = '請分析此郵件的知識圖譜',
                        rag_mode              = rag_mode_now,
                        layer2_risk           = st.session_state.layer2_risk,
                        layer3_classification = st.session_state.layer3_classification,
                    )
                st.session_state.graph_data = result
            st.rerun()
    with kg_info_col:
        if st.session_state.graph_data:
            mode_used = st.session_state.get('rag_mode_select', 'hybrid')
            n_ent = len(st.session_state.graph_data.get('entities', []))
            n_rel = len(st.session_state.graph_data.get('relationships', []))
            st.caption(f'圖譜已載入 · 模式：{mode_used} · {n_ent} 實體 · {n_rel} 關係')

    # 知識圖譜渲染
    if st.session_state.graph_data:
        _render_knowledge_graph(st.session_state.graph_data)

    st.divider()

# ═══ 上半區：郵件上下文 ══════════════════════════════════════════════════
st.subheader('✉️ 原始郵件內容 (Context)')

if not st.session_state.analysis_done:
    # ── 分析前：可編輯狀態 ──
    email_input = st.text_area(
        '請在此貼入待分析的原始郵件文字',
        height=250,
        key='email_input_widget',
        placeholder=(
            'Subject: Your account requires immediate verification\n\n'
            'Dear user,\n\n'
            'We have detected suspicious activity on your account. '
            'Please click the link below to verify your identity immediately...'
        ),
    )
    query_col, mode_col = st.columns([3, 1], gap='small')
    with query_col:
        initial_query = st.text_area(
            '初始分析指令（可留空，留空則使用預設指令）',
            height=68,
            key='initial_query_widget',
            placeholder='請對此郵件進行完整的 PhishRAG 流水線威脅分析。',
        )
    with mode_col:
        rag_mode = st.selectbox(
            '🔍 LightRAG 模式',
            options=['hybrid', 'local', 'global', 'mix', 'naive', 'bypass'],
            index=0,
            key='rag_mode_select',
            help=(
                'hybrid：本地圖遍歷 + 全局圖摘要混合（預設，推薦）\n'
                'local：以實體關係進行局部圖遍歷\n'
                'global：以社群摘要進行全局圖推理\n'
                'mix：向量語義搜索 + 知識圖譜混合\n'
                'naive：純向量相似度搜索\n'
                'bypass：跳過圖譜，直接交由底層 LLM 回答'
            ),
        )
    with st.expander('⚙️ 進階查詢參數（選填，留空使用伺服器預設值）', expanded=False):
        st.text_area(
            'Additional Output Prompt',
            placeholder='例：請以繁體中文回答，並附上 MITRE ATT&CK 技術編號。',
            key='user_prompt_input',
            height=80,
        )
        tk_col, ctk_col = st.columns(2)
        with tk_col:
            st.number_input('KG TOP K', min_value=1, max_value=200, step=1,
                            value=None, placeholder='預設', key='top_k_input')
        with ctk_col:
            st.number_input('Chunk TOP K', min_value=1, max_value=50, step=1,
                            value=None, placeholder='預設', key='chunk_top_k_input')
        et_col, rt_col, tt_col = st.columns(3)
        with et_col:
            st.number_input('Max Entity Tokens', min_value=100, max_value=20000, step=100,
                            value=None, placeholder='預設', key='max_entity_tokens_input')
        with rt_col:
            st.number_input('Max Relation Tokens', min_value=100, max_value=20000, step=100,
                            value=None, placeholder='預設', key='max_relation_tokens_input')
        with tt_col:
            st.number_input('Max Total Tokens', min_value=100, max_value=60000, step=100,
                            value=None, placeholder='預設', key='max_total_tokens_input')

    st.caption('💡 執行後，郵件內容將鎖定為本次會話上下文。')

    if st.button('🚀 執行分析', type='primary', use_container_width=True):
        email_val = email_input.strip()
        if not email_val:
            st.warning('⚠️ 請先貼入郵件內容。')
        else:
            rag_mode_val    = st.session_state.get('rag_mode_select', 'hybrid')
            user_query_text = initial_query.strip() or '請對此郵件進行完整的 PhishRAG 流水線威脅分析。'

            with st.spinner('⚙️ 正在執行四層流水線分析，請稍候...'):
                result = call_analyze(email_val, user_query_text, rag_mode_val)

            if result.get('status') == 'success':
                st.session_state.analysis_done         = True
                st.session_state.email_content         = email_val
                st.session_state.layer1_features       = result['layer1_features']
                st.session_state.layer2_risk           = result['layer2_risk']
                st.session_state.layer3_classification = result['layer3_classification']
                st.session_state.chat_history          = [
                    {'role': 'user',      'content': user_query_text},
                    {'role': 'assistant', 'content': result['layer4_rag_report']},
                ]
                st.rerun()
            else:
                st.error(f"❌ {result.get('message', '未知錯誤')}")
else:
    # ── 分析後：唯讀鎖定狀態 ──
    lock_col, btn_col = st.columns([4, 1], gap='small')
    with lock_col:
        st.text_area(
            '郵件內容（已鎖定為本次會話上下文）',
            value=st.session_state.email_content,
            height=150,
            disabled=True,
            key='email_locked_widget',
        )
    with btn_col:
        st.caption('🔒 郵件已鎖定')
        if st.button('🔄 重置 / 分析新郵件', use_container_width=True):
            st.session_state.analysis_done         = False
            st.session_state.email_content         = ''
            st.session_state.layer1_features       = {}
            st.session_state.layer2_risk           = {}
            st.session_state.layer3_classification = {}
            st.session_state.chat_history          = []
            st.session_state.graph_data            = None
            st.rerun()

st.divider()

# ═══ 下半區：資安 AI 助手 ════════════════════════════════════════════════
st.subheader('💬 資安 AI 助手 (User Query)')

if not st.session_state.analysis_done:
    # ── 分析前：等待狀態 ──
    st.info('👆 請在上方貼入郵件內容，選擇檢索模式後，點擊「執行分析」開始分析。')
    st.markdown("""
    **流水線說明：**
    1. **Layer 1** — 擷取 76 維技術特徵（文字統計、詞彙分數、TF-IDF）
    2. **Layer 2** — XGBoost 二元分類（是否為釣魚？風險分數）
    3. **Layer 3** — XGBoost 多類型分類（10 種釣魚類型 × 嚴重程度）
    4. **Layer 4** — LightRAG 知識圖譜檢索（ATT&CK 技術映射 × 威脅情資報告）
    """)
else:
    # ── 分析後：互動式聊天介面 ──
    rag_mode = st.selectbox(
        '🔍 LightRAG 檢索模式',
        options=['hybrid', 'local', 'global', 'mix', 'naive', 'bypass'],
        index=0,
        key='rag_mode_select',
        help=(
            'hybrid：本地圖遍歷 + 全局圖摘要混合（預設，推薦）\n'
            'local：以實體關係進行局部圖遍歷\n'
            'global：以社群摘要進行全局圖推理\n'
            'mix：向量語義搜索 + 知識圖譜混合\n'
            'naive：純向量相似度搜索\n'
            'bypass：跳過圖譜，直接交由底層 LLM 回答'
        ),
    )

    with st.expander('⚙️ 進階查詢參數（選填，留空使用伺服器預設值）', expanded=False):
        st.text_area(
            'Additional Output Prompt',
            placeholder='例：請以繁體中文回答，並附上 MITRE ATT&CK 技術編號。',
            key='user_prompt_input',
            height=80,
        )
        tk_col, ctk_col = st.columns(2)
        with tk_col:
            st.number_input('KG TOP K', min_value=1, max_value=200, step=1,
                            value=None, placeholder='預設', key='top_k_input')
        with ctk_col:
            st.number_input('Chunk TOP K', min_value=1, max_value=50, step=1,
                            value=None, placeholder='預設', key='chunk_top_k_input')
        et_col, rt_col, tt_col = st.columns(3)
        with et_col:
            st.number_input('Max Entity Tokens', min_value=100, max_value=20000, step=100,
                            value=None, placeholder='預設', key='max_entity_tokens_input')
        with rt_col:
            st.number_input('Max Relation Tokens', min_value=100, max_value=20000, step=100,
                            value=None, placeholder='預設', key='max_relation_tokens_input')
        with tt_col:
            st.number_input('Max Total Tokens', min_value=100, max_value=60000, step=100,
                            value=None, placeholder='預設', key='max_total_tokens_input')

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

    # 聊天輸入框
    user_input = st.chat_input('輸入追問問題，可隨時切換上方檢索模式...', key='chat_input')
    if user_input:
        current_mode = rag_mode
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})

        with st.spinner(f'🔍 正在以 [{current_mode}] 模式查詢知識圖譜...'):
            reply = call_chat(
                email_content=st.session_state.email_content,
                current_query=user_input,
                rag_mode=current_mode,
                chat_history=st.session_state.chat_history[:-1],
            )

        st.session_state.chat_history.append({'role': 'assistant', 'content': reply})
        st.rerun()
