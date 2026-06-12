# start_phishrag.ps1 — PhishRAG 全系統啟動腳本
# 啟動順序：PostgreSQL (Docker) → LightRAG Server → Flask 後端 → Streamlit 前端

$LIGHTRAG_PKG  = "C:\Users\games\OneDrive\Desktop\PhishRAG資料夾\lightrag-package"
$PHISHRAG_DIR  = "C:\Users\games\OneDrive\Desktop\PhishRAG資料夾\PhishRAG"
$PYTHON_EXE    = "C:\Users\games\miniconda3\envs\PhishRAG\python.exe"
$STREAMLIT_EXE = "C:\Users\games\miniconda3\envs\PhishRAG\Scripts\streamlit.exe"

Write-Host "=== PhishRAG 全系統啟動 ===" -ForegroundColor Cyan

# ── 步驟 1：啟動 PostgreSQL Docker 容器 ─────────────────────────────────────
Write-Host "`n[1/4] 啟動 PostgreSQL 知識庫容器..." -ForegroundColor Yellow
Set-Location $LIGHTRAG_PKG
docker compose up -d
if ($LASTEXITCODE -ne 0) { Write-Host "Docker 啟動失敗，請確認 Docker Desktop 正在執行。" -ForegroundColor Red; exit 1 }

# 等待 PostgreSQL 健康檢查通過
Write-Host "  等待 PostgreSQL 就緒（最多 90 秒）..." -ForegroundColor Gray
$timeout = 90; $elapsed = 0
while ($elapsed -lt $timeout) {
    $status = docker inspect lightrag-postgres --format "{{.State.Health.Status}}" 2>$null
    if ($status -eq "healthy") { Write-Host "  PostgreSQL 已就緒！" -ForegroundColor Green; break }
    Start-Sleep -Seconds 5; $elapsed += 5
    Write-Host "  等待中... ($elapsed / $timeout s)" -ForegroundColor Gray
}
if ($status -ne "healthy") {
    Write-Host "  警告：PostgreSQL 健康檢查未通過，繼續嘗試..." -ForegroundColor Yellow
}

# ── 步驟 2：啟動 LightRAG Server ────────────────────────────────────────────
Write-Host "`n[2/4] 啟動 LightRAG Server（Port 9621）..." -ForegroundColor Yellow
Set-Location $LIGHTRAG_PKG  # lightrag-server 從此目錄讀取 .env
Start-Process -FilePath "lightrag-server" -WorkingDirectory $LIGHTRAG_PKG `
    -RedirectStandardOutput "$LIGHTRAG_PKG\lightrag_out.txt" `
    -RedirectStandardError  "$LIGHTRAG_PKG\lightrag_err.txt" `
    -WindowStyle Hidden

# 等待 LightRAG API 就緒
Write-Host "  等待 LightRAG API 就緒（最多 60 秒）..." -ForegroundColor Gray
$timeout = 60; $elapsed = 0
while ($elapsed -lt $timeout) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:9621/health" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) { Write-Host "  LightRAG 已就緒！" -ForegroundColor Green; break }
    } catch {}
    Start-Sleep -Seconds 5; $elapsed += 5
    Write-Host "  等待中... ($elapsed / $timeout s)" -ForegroundColor Gray
}

# ── 步驟 3：啟動 Flask 後端 ──────────────────────────────────────────────────
Write-Host "`n[3/4] 啟動 Flask 後端（Port 5000）..." -ForegroundColor Yellow
Set-Location $PHISHRAG_DIR
Start-Process -FilePath $PYTHON_EXE -ArgumentList "app.py" -WorkingDirectory $PHISHRAG_DIR `
    -RedirectStandardOutput "$PHISHRAG_DIR\flask_out.txt" `
    -RedirectStandardError  "$PHISHRAG_DIR\flask_err.txt" `
    -WindowStyle Hidden

Start-Sleep -Seconds 6
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    Write-Host "  Flask 後端已就緒！(feature_cols=$(($resp.Content | ConvertFrom-Json).feature_cols))" -ForegroundColor Green
} catch {
    Write-Host "  Flask 啟動中，請稍候..." -ForegroundColor Yellow
}

# ── 步驟 4：啟動 Streamlit 前端 ─────────────────────────────────────────────
Write-Host "`n[4/4] 啟動 Streamlit 前端（Port 8501）..." -ForegroundColor Yellow
Start-Process -FilePath $STREAMLIT_EXE `
    -ArgumentList "run dashboard.py --server.port 8501 --server.headless true" `
    -WorkingDirectory $PHISHRAG_DIR `
    -RedirectStandardOutput "$PHISHRAG_DIR\streamlit_out.txt" `
    -RedirectStandardError  "$PHISHRAG_DIR\streamlit_err.txt" `
    -WindowStyle Hidden

Start-Sleep -Seconds 5

Write-Host "`n=== 所有服務已啟動 ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "  服務               URL" -ForegroundColor White
Write-Host "  ─────────────────────────────────────────" -ForegroundColor Gray
Write-Host "  PostgreSQL (DB)  → localhost:5432" -ForegroundColor Gray
Write-Host "  LightRAG Server  → http://localhost:9621" -ForegroundColor Gray
Write-Host "  LightRAG WebUI   → http://localhost:9621/webui" -ForegroundColor Gray
Write-Host "  Flask 後端       → http://localhost:5000" -ForegroundColor Gray
Write-Host "  Streamlit 儀表板 → http://localhost:8501  ★ 主入口" -ForegroundColor Green
Write-Host ""
Write-Host "  日誌檔案位置：" -ForegroundColor Gray
Write-Host "    LightRAG → $LIGHTRAG_PKG\lightrag_err.txt" -ForegroundColor Gray
Write-Host "    Flask    → $PHISHRAG_DIR\flask_err.txt" -ForegroundColor Gray
Write-Host ""

# 自動開啟瀏覽器
Start-Process "http://localhost:8501"
