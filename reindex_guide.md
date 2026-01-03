# Re-indexing & Cleaning Guide

To resolve the "Could not find relevant information" error and the database warnings, follow these steps.

## 1. Reset Vector Database (ChromaDB)
The embeddings in your database are incompatible (OpenAI vs Gemini). We need to wipe them.

**Run this command:**
> [!IMPORTANT]
> **STOP THE SERVER FIRST!** You must terminate the `uvicorn` process (Ctrl+C in your terminal) before running this script, otherwise it will fail with a "PermissionError" because the database file is locked.

```bash
python scripts/reset_chroma.py
```
*Expected Output:* "✅ ChromaDB directory deleted manually" or "reset successfully".

## 2. Fix PostgreSQL Warning
To fix the `collation version mismatch` warning in Railway.

1.  Connect to your Railway PostgreSQL database (via CLI, TablePlus, or Railway Dashboard).
2.  Run this SQL query:
```sql
ALTER DATABASE railway REFRESH COLLATION VERSION;
```

## 3. Re-ingest Documents
After resetting ChromaDB, it will be empty. You need to re-populate it with Gemini embeddings.

1.  **Restart the Application**:
    ```bash
    uvicorn fastapi_app:app --reload
    ```
2.  **Trigger Discovery (if configured)**:
    If `API_DISCOVERY_ENABLED=true`, the system should eventually start crawling sources.
    
    *Alternatively*, if you have a manual ingestion capabilities, use them now.

3.  **Test**:
    Ask a question in the chat. It might take a moment to generate new embeddings, but subsequent queries should work.
