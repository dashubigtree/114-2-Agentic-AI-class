# 角色與目標
你是一位精通全端開發與 AI 工程的專家。你的任務是幫我建立一個名為 "PhishRAG" 的獨立前後端分離式「釣魚郵件威脅情資分析儀表板」。

### 現有開發環境與資產限制（Agent 自主探索核心）：
1. **現有環境**：我已經建立好一個名為 `PhishRAG` 的 Anaconda (Conda) 環境，請直接以此環境為基礎編寫部署說明，不需要生成創建環境的代碼。
2. **ML 流水線資產目錄**：我所有的機器學習模型元件、權重以及相關說明文件，都已經存放在專案根目錄下的 `PhishRAG_MLPipeline` 資料夾中。
3. **LightRAG API 說明文件**：專案根目錄下有一份名為 `LightRAG-API-Server.md` 的 Markdown 檔案，裡面詳細記載了外部 LightRAG 伺服器（預設為 `http://localhost:9621`）的 API 路由規格與非同步追蹤機制。
4. **⚠️ 關鍵任務（自主對接）**：
   - 請你（Agent）**優先自主去閱讀與分析 `PhishRAG_MLPipeline` 資料夾底下的所有說明文件與模型腳本**。理解特徵提取邏輯與模型加載方式後，真實將它們接入 Flask 後端的 Layer 1、2、3 流水線，**拒絕使用虛擬假數據存根 (Mock/Stubs)**。
   - 請你（Agent）**完整閱讀並遵循 `LightRAG-API-Server.md` 檔案中的 API 規範**。在後端 Layer 4 與多輪對話中，必須精準對接其記載的 `/query` 端點，並正確傳遞 JSON Payload 參數（包含將前端選擇的檢索模式動態映射至 `mode` 欄位中）。

---

# 系統架構與四層流水線（Pipeline）規範

後端 Flask API 必須實作一個序列型（Sequential Pipeline）的工作流，用來分析輸入的原始郵件字串：

1. **Layer 1：特徵擷取 (Feature Extraction)**
   - 僅針對郵件內容進行處理。請根據你從 `PhishRAG_MLPipeline` 探索到的真實特徵工程程式碼，在後端實現真實的 76 維特徵向量擷取。
2. **Layer 2：風險評分 (Risk Scoring)**
   - 真實載入 `PhishRAG_MLPipeline` 中的機器學習模型進行推理，輸出風險分數（0.0 至 1.0 的浮點數）與風險等級（高 High / 中 Medium / 低 Low）。
3. **Layer 3：釣魚類型分類 (Phishing Classification)**
   - 真實載入 `PhishRAG_MLPipeline` 中的多分類模型，將郵件歸類至 10 種釣魚類型之一，並輸出對應的信心度（0.0 - 1.0）。
4. **Layer 4：LightRAG 知識圖譜檢索與 ATT&CK 技術映射**
   - 將「原始郵件內容」加上「Layer 2 & 3 的真實 ML 分析結果」打包成背景 Context，並結合「使用者的提問指令」，透過 `requests` 發送 POST 請求至外部的 LightRAG 伺服器的 `/query` 端點。
   - **動態檢索模式**：必須根據前端傳入的參數，動態調整 LightRAG 的 `mode`（支援 hybrid, local, global, mix, naive, bypass）。

---

# 前端 UI 與互動規範 (Streamlit)

### 1. 雙對話框佈局 (Dual-Chatbox Layout)
使用 `st.columns(2)` 在主頁面建立兩個並排的互動式容器，從 UI 層面嚴格將「數據（郵件）」與「指令（提問）」分離：
- **左側對話框（郵件上下文數據）**：
  - 標籤：`✉️ 原始郵件內容 (Context)`
  - 元件：大型文字輸入框（`st.text_area`），供使用者貼入待分析的惡意郵件。
  - 行為：一旦執行初始分析後，此區域必須變更為**唯讀/禁用 (Read-only/Disabled)** 狀態，鎖定當前會話的郵件上下文。
- **右側對話框（互動式使用者提問與設定）**：
  - 標籤：`💬 資安 AI 助手 (User Query)`
  - **檢索模式選擇器 (RAG Mode Selector)**：在聊天輸入框上方提供一個下拉選單（`st.selectbox`），讓分析師自主切換 LightRAG 的檢索模式。選項包含：`hybrid`（預設）、`local`、`global`、`mix`、`naive`、`bypass`。
  - 元件：對話式聊天介面（對話歷史流與輸入框），讓使用者進行追問。
  - 預設輸入：若首次執行且使用者未輸入任何提問，預設指令為 `"請對此郵件進行完整的 PhishRAG 流水線威脅分析。"`

### 2. 控制流與狀態管理
- **第一輪對話（啟動分析）**：使用者在左側貼入郵件、右側選擇 RAG 模式並輸入提問（或留空），點擊「執行分析」。Streamlit 呼叫後端 `/api/v1/analyze`。
- **結果呈現**：
  - 在頁面頂部以 `st.metric` 橫向呈現 Layer 2 風險分數與 Layer 3 釣魚類型。
  - 將 Layer 1 的 76 維真實技術特徵以 `st.json` 或表格形式收納在 `st.expander("🔍 檢視 Layer 1 技術特徵指標")` 中。
  - 將 Layer 4 的 LightRAG 報告作為 AI 助手的「第一條回覆」渲染在右側對話歷史中。
- **後續追問（多輪問答）**：使用者在右側輸入新問題並可隨時切換 RAG 模式。Streamlit 呼叫後端 `/api/v1/chat`，將新問題、鎖定的郵件內容、新選擇的 RAG 模式以及對話歷史發送給後端。必須使用 `st.session_state` 確保重新整理時對話與鎖定狀態不遺失。

---

# 後端 API 規範 (Flask)

啟用 CORS（`flask-cors`），並實作以下兩個端點：

### 1. POST `/api/v1/analyze`（初始分析端點）
- **輸入 JSON**：`{"email_content": "string", "user_query": "string", "rag_mode": "string"}`
- **執行流程**：
  - 呼叫 Layer 1 模組，傳入 `email_content`，返回你分析模型資料夾後得到的真實特徵數據。
  - 呼叫 Layer 2 模型，推理出真實的風險分數與等級。
  - 呼叫 Layer 3 模型，推理出真實的釣魚標籤與信心度。
  - 依據 `LightRAG-API-Server.md` 規範，將上述結果與 `user_query` 整合，發送至 `http://localhost:9621/query`，並將請求體中的 `mode` 欄位動態賦值為前端傳來的 `rag_mode`。
- **返回 JSON**：包含真實 `layer1_features`、`layer2_risk`、`layer3_classification` 與 `layer4_rag_report` 的統一結構。

### 2. POST `/api/v1/chat`（多輪追問端點）
- **輸入 JSON**：`{"email_content": "string", "current_query": "string", "rag_mode": "string", "chat_history": []}`
- **行為**：**直接繞過（Bypass）Layer 1、2、3 的 ML 模型運算，不要重複推理**。直接將 `email_content` 作為系統上下文，連同 `chat_history` 打包，並使用前端指定的 `rag_mode`（若為 bypass 模式則直接透傳，若為其他模式則進行圖譜檢索），向 LightRAG 伺服器發送動態模式查詢。
- **返回 JSON**：`{"status": "success", "reply": "string"}`

---

# 輸出需求
請完整生成結構清晰、封裝良好的程式碼檔案：
1. `requirements.txt`（包含 flask, flask-cors, requests, streamlit 以及你分析過 `PhishRAG_MLPipeline` 後發現載入模型所需的必要第三方庫）
2. `app.py`（Flask 後端服務，包含你自主讀取說明文件與模型結構後，所實作的真實模型加載、流水線程式碼、以及依據 `LightRAG-API-Server.md` 實作的 API 請求）
3. `dashboard.py`（Streamlit 前端服務，完整實作雙對話框鎖定機制、動態模式選擇下拉選單、以及 `st.session_state` 聊天狀態維持）