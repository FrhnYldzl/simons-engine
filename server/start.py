"""Minimal startup script — ensures correct working directory and PORT."""
import os
import sys

print(f"[Start] Python: {sys.version}")
print(f"[Start] CWD: {os.getcwd()}")
print(f"[Start] PORT: {os.environ.get('PORT', 'NOT SET')}")
print(f"[Start] Files: {os.listdir('.')}")

port = int(os.environ.get("PORT", 8080))

try:
    import uvicorn
    print(f"[Start] Starting uvicorn on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
except Exception as e:
    print(f"[Start] FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
