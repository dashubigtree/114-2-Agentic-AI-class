# PhishRAG

> 結合 XGBoost 風險評分與 LightRAG／MITRE ATT&CK 檢索的智慧釣魚郵件分析系統。

PhishRAG 是國立臺灣科技大學 114-2 Agentic AI 課程專題。系統將機器學習推論與圖譜感知檢索結合，把一封郵件轉換為具有可解釋性、可追溯脈絡的威脅情報報告。

本專題回應釣魚郵件流程中的實務落差：二元過濾器雖能標示可疑郵件，分析人員仍需要理解可能的攻擊技術、判斷依據、相關 ATT&CK 知識，以及可採取的緩解或偵測方向。

## 系統功能

使用者輸入郵件內容與選填的分析問題後，PhishRAG 會：

1. 擷取詞彙、結構、寄件者、網址與可疑語句等特徵；
2. 以二元 XGBoost 模型估計釣魚風險；
3. 透過後續 XGBoost 模型辨識釣魚類型與嚴重程度；
4. 以 LightRAG 檢索相關的 MITRE ATT&CK 知識；以及
5. 產生具備依據的威脅情報報告，並支援後續追問。

## 系統架構

應用程式由四個 Docker Compose 服務組成。

```text
Streamlit 前端介面（:8501）
        |
        | HTTP REST
        v
Flask 後端（:5000）
  - 機器學習推論
  - LightRAG 請求代理
        |
        | POST /query
        v
LightRAG API（:9621） ----------------> 相容於 Ollama 的 LLM 與嵌入模型端點
        |
        v
PostgreSQL（:5432）
  - pgvector 向量檢索
  - Apache AGE 圖譜儲存
  - 文件與鍵值儲存
```

Streamlit 介面接收郵件內容、選填的分析指令、檢索模式與進階檢索參數；Flask 後端則負責執行 ML 分析階段、組合檢索問題，並回傳模型結果與 LightRAG 回應。

## 分析流程

### 第 1 層：特徵工程

系統從郵件結構、可疑語言型態、寄件者與網址訊號，以及 TF-IDF 詞彙建立特徵；同時明確檢查可能洩漏標籤、或僅反映合成資料產製痕跡的特徵。

### 第 2 層：釣魚風險偵測

**模型 A** 為 XGBoost 二元分類器，用以判斷郵件較可能為釣魚或正常郵件，並輸出風險分數。

### 第 3 層：釣魚類型與嚴重程度

**模型 B** 預測釣魚類型；**模型 C** 預測低、中、高三種嚴重程度，並將模型 B 的類別機率作為額外輸入特徵。

### 第 4 層：LightRAG 威脅情報檢索

LightRAG 會從 MITRE ATT&CK 知識圖譜中檢索相關技術、偵測指引、分析 ID、關係與緩解措施。支援的檢索模式包括 `hybrid`、`local`、`global`、`mix`、`naive` 與 `bypass`。

## 知識庫

ATT&CK 知識庫採以釣魚技術為核心的星狀圖譜，串連緩解措施、偵測來源、偵測分析、威脅行為者、惡意軟體、攻擊活動、工具、偵察、資源開發與執行階段。

圖譜包含 ATT&CK 技術、戰術、威脅行為者、工具與緩解措施等實體；重要關係包含 `paired_with`、`uses`、`enables` 與 `mitigates`。這樣的結構可支援需要跨多個節點與文件脈絡的多跳問題。

## 為什麼使用圖譜感知檢索

課堂評估比較了純向量式檢索（naive）與圖譜加向量的混合式檢索（hybrid）。

| 問題類型 | 觀察結果 |
| --- | --- |
| 事實查詢 | 兩種方法皆可回答正確事實；混合式檢索的回答通常較精簡。 |
| 結構化檢索 | 混合式檢索較能在跨文件內容中一致保留 ATT&CK 識別碼與圖譜關聯脈絡。 |
| 情境分析 | 混合式檢索可根據圖譜節點，回傳偵測 ID、分析內容與可調整參數。 |
| 範圍外問題 | 兩者皆可保守拒答；混合式檢索可補充圖譜中確實存在的相鄰事實。 |

知識圖譜不會讓模型本身「更聰明」。它的價值在於將名稱、識別碼與關係以結構化資料儲存，使多跳查詢與需精確辨識 ID 的檢索更可靠，並降低模型以生成文字補足缺失結構化事實的風險。

## 可解釋性

系統以 SHAP 分析說明單一預測中各特徵的貢獻。在課堂實驗中，整體可疑語句分數、帳號相關 TF-IDF 詞彙與連結相關 TF-IDF 詞彙是較強的釣魚訊號；常見於正常商務郵件的特徵則會使預測傾向正常類別。

## 評估說明

課堂簡報中，模型在提供的**合成**郵件資料集上呈現以下結果：

| 階段 | 任務 | 簡報結果 |
| --- | --- | --- |
| 模型 A | 二元釣魚郵件偵測 | Accuracy 1.00、ROC-AUC 1.00 |
| 模型 B | 釣魚類型分類 | Accuracy 1.00、Macro F1 1.00 |
| 模型 C | 嚴重程度分類 | Accuracy 0.87、Macro F1 0.730 |

以上數值僅描述以合成資料完成的課堂實驗，並不代表正式環境中的效能。若應用於真實郵件場景，仍須使用具代表性且獨立保留的資料進行驗證，且任何資安處置都應保留分析人員覆核。

## 資料集與參考資料

- 合成郵件資料集：共 10,000 封郵件，包含 4,000 封正常郵件與 6,000 封、涵蓋 10 種釣魚類型的釣魚郵件。來源：[Phishing and Legitimate Emails Dataset](https://www.kaggle.com/datasets/kuladeep19/phishing-and-legitimate-emails-dataset)
- MITRE ATT&CK 知識庫：[attack-stix-data](https://github.com/mitre-attack/attack-stix-data) 與 [MITRE ATT&CK framework](https://attack.mitre.org/)
- Guo et al. (2024), [LightRAG](https://arxiv.org/abs/2410.05779)
- Lundberg and Lee (2017), [SHAP](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- Es et al. (2023), [RAGAS](https://arxiv.org/abs/2309.15217)
- Edge et al. (2024), [GraphRAG](https://arxiv.org/abs/2404.16130)

## 本機執行

### 前置需求

- 已安裝 Docker Desktop 與 Docker Compose
- 可連線的 Ollama 相容 LLM 與嵌入模型端點，並已在 `LightRAG/.env` 設定
- 已準備 LightRAG 資料庫或知識庫初始化資料

完整的 LightRAG SQL 備份因檔案較大而未納入此儲存庫；若要取得圖譜檢索結果，請先準備或還原知識庫。

### 啟動服務

```bash
git clone https://github.com/dashubigtree/114-2-Agentic-AI-class.git
cd 114-2-Agentic-AI-class
docker compose up --build -d
docker compose ps
```

待所有服務皆正常啟動後，開啟 [http://localhost:8501](http://localhost:8501)。

### 停止服務

```bash
docker compose down
```

只有在確定要刪除本機 PostgreSQL Volume 及其知識庫資料時，才使用 `docker compose down -v`。

## 儲存庫導覽

| 路徑 | 用途 |
| --- | --- |
| `PhishRAG/` | Flask API、Streamlit 儀表板、ML 模型與推論流程 |
| `LightRAG/` | LightRAG 服務設定與儲存目錄 |
| `lightrag-package/` | 資料庫初始化資源 |
| `docker-compose.yml` | 四服務的本機部署設定 |
| [`使用說明書.md`](使用說明書.md) | 詳細使用說明 |
| [`技術文件.md`](技術文件.md) | 詳細技術文件 |

## 使用範圍與責任聲明

PhishRAG 為課程與研究展示用途而建立。系統分析使用者提供的郵件文字，並檢索公開的 ATT&CK 知識；請勿將風險分數、分類結果或生成報告作為唯一的資安決策依據，也不要將機密郵件內容傳送至未經資料安全核准的端點。
