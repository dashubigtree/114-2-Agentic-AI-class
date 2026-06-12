"""
dashboard_en.py — PhishRAG Streamlit Frontend Dashboard (English Version)
Dual-panel layout: left email input (locked after analysis) + right RAG Q&A (multi-turn)
"""

import os
import streamlit as st
import requests

BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000')

st.set_page_config(
    page_title='PhishRAG — Phishing Email Threat Intelligence Dashboard',
    page_icon='🎣',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ─── Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-label { font-size: 0.85rem; color: #888; }
    .stChatMessage { border-radius: 8px; }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTextArea"]) { height: 100%; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ─────────────────────────────────────────
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
    'graph_data':            None,   # None = not loaded; dict = graph data
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── API Helper Functions ──────────────────────────────────────────────────

def _rag_extra_params() -> dict:
    """Collect advanced query parameters from session state; skip None or empty values."""
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
    """Call backend /api/v1/analyze to run the full four-layer pipeline."""
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
        return {'status': 'error', 'message': 'Cannot connect to backend. Please ensure the Flask server is running at http://localhost:5000.'}
    except requests.exceptions.Timeout:
        return {'status': 'error', 'message': 'Backend request timed out (>200s). Please try again later.'}
    except Exception as exc:
        return {'status': 'error', 'message': f'API error: {exc}'}


def call_chat(email_content: str, current_query: str, rag_mode: str, chat_history: list) -> str:
    """Call backend /api/v1/chat for multi-turn follow-up (skips ML pipeline)."""
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
        return resp.json().get('reply', '(empty response)')
    except requests.exceptions.ConnectionError:
        return '⚠️ Cannot connect to backend. Please ensure the Flask server is running.'
    except requests.exceptions.Timeout:
        return '⚠️ Backend request timed out. Please try again later.'
    except Exception as exc:
        return f'⚠️ Chat API error: {exc}'


def call_query_graph(
    email_content: str,
    user_query: str,
    rag_mode: str,
    layer2_risk: dict,
    layer3_classification: dict,
) -> dict:
    """Call backend /api/v1/query_graph to fetch raw knowledge graph data (no LLM call)."""
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
        return {'status': 'error', 'message': '⚠️ Cannot connect to backend'}
    except requests.exceptions.Timeout:
        return {'status': 'error', 'message': '⚠️ Knowledge graph query timed out (>60s)'}
    except Exception as exc:
        return {'status': 'error', 'message': f'⚠️ Graph query error: {exc}'}


def _generate_vis_html(entities: list, relationships: list) -> str:
    """Generate a self-contained vis.js network graph HTML string."""
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
    """Render the knowledge graph in Streamlit (vis.js network + structured tables)."""
    import streamlit.components.v1 as components
    import pandas as pd

    if graph_data.get('status') == 'bypass':
        st.info('ℹ️ Bypass mode skips graph retrieval. Switch to another mode and re-query.')
        return
    if graph_data.get('status') == 'error':
        st.error(f"❌ {graph_data.get('message', 'Unknown error')}")
        return

    entities      = graph_data.get('entities',      [])
    relationships = graph_data.get('relationships', [])
    chunks        = graph_data.get('chunks',        [])
    metadata      = graph_data.get('metadata',      {})

    if not entities and not relationships:
        st.warning('⚠️ No entities or relationships retrieved from the knowledge graph. Try a different RAG mode.')
        return

    with st.expander(
        f'🕸️ Knowledge Graph ({len(entities)} entities · {len(relationships)} relationships · {len(chunks)} text chunks)',
        expanded=True,
    ):
        # ── vis.js network graph ──
        vis_html = _generate_vis_html(entities, relationships)
        components.html(vis_html, height=490, scrolling=False)

        # ── Statistics summary ──
        proc = metadata.get('processing_info', {})
        if proc:
            c1, c2, c3 = st.columns(3)
            c1.metric('Entities Retrieved', proc.get('total_entities_found', len(entities)))
            c2.metric('Relationships Retrieved', proc.get('total_relations_found', len(relationships)))
            c3.metric('Final Chunk Count', proc.get('final_chunks_count', len(chunks)))

        # ── Structured data tabs ──
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
                st.caption('No entity data')

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
                st.caption('No relationship data')

        with tab_c:
            if chunks:
                for i, c in enumerate(chunks[:10]):
                    with st.expander(f"Chunk {i+1} — {c.get('file_path', '')}"):
                        st.text(c.get('content', '')[:500])
            else:
                st.caption('No chunk data')


# ─── Page Title ────────────────────────────────────────────────────────────
st.title('🎣 PhishRAG — Phishing Email Threat Intelligence Dashboard')
st.caption('XGBoost Three-Layer Classifier × LightRAG Knowledge Graph × MITRE ATT&CK Technique Mapping')

# ─── Post-analysis: Top metrics and Layer 1 feature display ───────────────
if st.session_state.analysis_done:
    l2 = st.session_state.layer2_risk
    l3 = st.session_state.layer3_classification

    # 4 core metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        is_p = l2.get('is_phishing', 0)
        st.metric(
            label='🔍 Phishing Verdict',
            value='Phishing ⚠️' if is_p else 'Legitimate ✅',
        )
    with m2:
        st.metric(
            label='📊 Risk Score (Model A)',
            value=f"{l2.get('risk_score', 0.0):.4f}",
        )
    with m3:
        level = l2.get('risk_level', 'N/A')
        level_icon = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}.get(level, '⚪')
        st.metric(label='⚡ Risk Level (Model C)', value=f'{level_icon} {level}')
    with m4:
        ptype = l3.get('phishing_type', 'N/A').replace('_', ' ').title()
        st.markdown(
            f'<div style="font-size:0.875rem;color:#a3a8b8;margin-bottom:4px;">🎯 Phishing Type (Model B)</div>'
            f'<div style="font-size:1.5rem;font-weight:700;line-height:1.3;word-break:break-word;white-space:normal;">{ptype}</div>',
            unsafe_allow_html=True,
        )

    # Classification confidence progress bar
    conf = l3.get('type_confidence', 0.0)
    st.progress(float(conf), text=f"Model B Classification Confidence: {conf:.2%}")

    # Layer 1 technical feature display (76 dimensions)
    with st.expander('🔍 View Layer 1 Technical Feature Indicators (76-dimensional feature vector)'):
        feat = st.session_state.layer1_features
        if feat:
            hand_feat  = {k: v for k, v in feat.items() if not k.startswith('tfidf_')}
            tfidf_feat = {k: v for k, v in feat.items() if k.startswith('tfidf_')}
            tab1, tab2 = st.tabs([f'Handcrafted Features ({len(hand_feat)})', f'TF-IDF Features ({len(tfidf_feat)})'])
            with tab1:
                st.json(hand_feat)
            with tab2:
                st.json(tfidf_feat)

    # Knowledge graph button
    kg_btn_col, kg_info_col = st.columns([1, 4])
    with kg_btn_col:
        if st.button('🕸️ View Knowledge Graph', use_container_width=True):
            rag_mode_now = st.session_state.get('rag_mode_select', 'hybrid')
            if rag_mode_now == 'bypass':
                st.session_state.graph_data = {'status': 'bypass'}
            else:
                with st.spinner('Retrieving entities and relationships from the knowledge graph (no LLM call)...'):
                    result = call_query_graph(
                        email_content         = st.session_state.email_content,
                        user_query            = 'Analyze the knowledge graph for this email',
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
            st.caption(f'Graph loaded · Mode: {mode_used} · {n_ent} entities · {n_rel} relationships')

    # Knowledge graph rendering
    if st.session_state.graph_data:
        _render_knowledge_graph(st.session_state.graph_data)

    st.divider()

# ═══ Upper Section: Email Context ════════════════════════════════════════
st.subheader('✉️ Raw Email Content (Context)')

if not st.session_state.analysis_done:
    # ── Pre-analysis: editable state ──
    email_input = st.text_area(
        'Paste the raw email text to analyze here',
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
            'Initial analysis instruction (optional — leave blank to use the default)',
            height=68,
            key='initial_query_widget',
            placeholder='Please perform a full PhishRAG pipeline threat analysis on this email.',
        )
    with mode_col:
        rag_mode = st.selectbox(
            '🔍 LightRAG Mode',
            options=['hybrid', 'local', 'global', 'mix', 'naive', 'bypass'],
            index=0,
            key='rag_mode_select',
            help=(
                'hybrid: Local graph traversal + global graph summary mixed (default, recommended)\n'
                'local: Local graph traversal via entity relationships\n'
                'global: Global graph reasoning via community summaries\n'
                'mix: Vector semantic search + knowledge graph combined\n'
                'naive: Pure vector similarity search\n'
                'bypass: Skip graph, answer directly via the underlying LLM'
            ),
        )
    with st.expander('⚙️ Advanced Query Parameters (optional — leave blank to use server defaults)', expanded=False):
        st.text_area(
            'Additional Output Prompt',
            placeholder='e.g. Please respond in English and include MITRE ATT&CK technique IDs.',
            key='user_prompt_input',
            height=80,
        )
        tk_col, ctk_col = st.columns(2)
        with tk_col:
            st.number_input('KG TOP K', min_value=1, max_value=200, step=1,
                            value=None, placeholder='Default', key='top_k_input')
        with ctk_col:
            st.number_input('Chunk TOP K', min_value=1, max_value=50, step=1,
                            value=None, placeholder='Default', key='chunk_top_k_input')
        et_col, rt_col, tt_col = st.columns(3)
        with et_col:
            st.number_input('Max Entity Tokens', min_value=100, max_value=20000, step=100,
                            value=None, placeholder='Default', key='max_entity_tokens_input')
        with rt_col:
            st.number_input('Max Relation Tokens', min_value=100, max_value=20000, step=100,
                            value=None, placeholder='Default', key='max_relation_tokens_input')
        with tt_col:
            st.number_input('Max Total Tokens', min_value=100, max_value=60000, step=100,
                            value=None, placeholder='Default', key='max_total_tokens_input')

    st.caption('💡 After running, the email content will be locked as the context for this session.')

    if st.button('🚀 Run Analysis', type='primary', use_container_width=True):
        email_val = email_input.strip()
        if not email_val:
            st.warning('⚠️ Please paste email content before running.')
        else:
            rag_mode_val    = st.session_state.get('rag_mode_select', 'hybrid')
            user_query_text = initial_query.strip() or 'Please perform a full PhishRAG pipeline threat analysis on this email.'

            with st.spinner('⚙️ Running four-layer pipeline analysis, please wait...'):
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
                st.error(f"❌ {result.get('message', 'Unknown error')}")
else:
    # ── Post-analysis: read-only locked state ──
    lock_col, btn_col = st.columns([4, 1], gap='small')
    with lock_col:
        st.text_area(
            'Email Content (locked as context for this session)',
            value=st.session_state.email_content,
            height=150,
            disabled=True,
            key='email_locked_widget',
        )
    with btn_col:
        st.caption('🔒 Email locked')
        if st.button('🔄 Reset / Analyze New Email', use_container_width=True):
            st.session_state.analysis_done         = False
            st.session_state.email_content         = ''
            st.session_state.layer1_features       = {}
            st.session_state.layer2_risk           = {}
            st.session_state.layer3_classification = {}
            st.session_state.chat_history          = []
            st.session_state.graph_data            = None
            st.rerun()

st.divider()

# ═══ Lower Section: Security AI Assistant ════════════════════════════════
st.subheader('💬 Security AI Assistant (User Query)')

if not st.session_state.analysis_done:
    # ── Pre-analysis: waiting state ──
    st.info('👆 Paste email content above, select a retrieval mode, then click "Run Analysis" to begin.')
    st.markdown("""
    **Pipeline Overview:**
    1. **Layer 1** — Extract 76-dimensional technical features (text statistics, lexical scores, TF-IDF)
    2. **Layer 2** — XGBoost binary classification (phishing or not? risk score)
    3. **Layer 3** — XGBoost multi-class classification (10 phishing types × severity)
    4. **Layer 4** — LightRAG knowledge graph retrieval (ATT&CK technique mapping × threat intelligence report)
    """)
else:
    # ── Post-analysis: interactive chat interface ──
    rag_mode = st.selectbox(
        '🔍 LightRAG Retrieval Mode',
        options=['hybrid', 'local', 'global', 'mix', 'naive', 'bypass'],
        index=0,
        key='rag_mode_select',
        help=(
            'hybrid: Local graph traversal + global graph summary mixed (default, recommended)\n'
            'local: Local graph traversal via entity relationships\n'
            'global: Global graph reasoning via community summaries\n'
            'mix: Vector semantic search + knowledge graph combined\n'
            'naive: Pure vector similarity search\n'
            'bypass: Skip graph, answer directly via the underlying LLM'
        ),
    )

    with st.expander('⚙️ Advanced Query Parameters (optional — leave blank to use server defaults)', expanded=False):
        st.text_area(
            'Additional Output Prompt',
            placeholder='e.g. Please respond in English and include MITRE ATT&CK technique IDs.',
            key='user_prompt_input',
            height=80,
        )
        tk_col, ctk_col = st.columns(2)
        with tk_col:
            st.number_input('KG TOP K', min_value=1, max_value=200, step=1,
                            value=None, placeholder='Default', key='top_k_input')
        with ctk_col:
            st.number_input('Chunk TOP K', min_value=1, max_value=50, step=1,
                            value=None, placeholder='Default', key='chunk_top_k_input')
        et_col, rt_col, tt_col = st.columns(3)
        with et_col:
            st.number_input('Max Entity Tokens', min_value=100, max_value=20000, step=100,
                            value=None, placeholder='Default', key='max_entity_tokens_input')
        with rt_col:
            st.number_input('Max Relation Tokens', min_value=100, max_value=20000, step=100,
                            value=None, placeholder='Default', key='max_relation_tokens_input')
        with tt_col:
            st.number_input('Max Total Tokens', min_value=100, max_value=60000, step=100,
                            value=None, placeholder='Default', key='max_total_tokens_input')

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

    # Chat input box
    user_input = st.chat_input('Type a follow-up question — you can switch the retrieval mode above at any time...', key='chat_input')
    if user_input:
        current_mode = rag_mode
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})

        with st.spinner(f'🔍 Querying knowledge graph in [{current_mode}] mode...'):
            reply = call_chat(
                email_content=st.session_state.email_content,
                current_query=user_input,
                rag_mode=current_mode,
                chat_history=st.session_state.chat_history[:-1],
            )

        st.session_state.chat_history.append({'role': 'assistant', 'content': reply})
        st.rerun()
