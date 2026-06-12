# lightrag-postgres 部署說明

## 快速啟動

```bash
docker compose up -d
```

啟動後資料會自動還原，首次啟動約需 1-2 分鐘。

## 連線資訊

請參考 docker-compose.yml 中的 environment 設定。

## 注意事項

- `init/` 資料夾內的 SQL 只在**第一次**建立 volume 時執行
- 若要重新匯入，請先刪除 volume：`docker compose down -v` 再重新 up
