"""One-off script: build LandDev-Pro-AI-deploy.zip for server transfer (no secrets, no venv)."""
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "_deploy_zip_staging"
ZIP_PATH = ROOT / "LandDev-Pro-AI-deploy.zip"

PLACEHOLDER_SECRETS = 'ANTHROPIC_API_KEY = "YOUR_KEY_HERE"\n'

if STAGE.exists():
    shutil.rmtree(STAGE)
(STAGE / ".streamlit").mkdir(parents=True)
(STAGE / "data" / "index").mkdir(parents=True)
(STAGE / "data" / "inbox").mkdir(parents=True)
(STAGE / "assets").mkdir(parents=True)

for name in ("app.py", "Dockerfile", "docker-compose.yml", "requirements.txt", ".dockerignore"):
    src = ROOT / name
    if src.exists():
        shutil.copy2(src, STAGE / name)

(STAGE / ".streamlit" / "secrets.toml").write_text(PLACEHOLDER_SECRETS, encoding="utf-8")

idx = ROOT / "data" / "index"
if idx.exists():
    for p in idx.iterdir():
        if p.is_file():
            shutil.copy2(p, STAGE / "data" / "index" / p.name)

inbox = ROOT / "data" / "inbox"
if inbox.exists():
    skip_inbox = {"desktop.ini", "thumbs.db"}
    for p in inbox.iterdir():
        if p.is_file() and p.name.lower() not in skip_inbox:
            shutil.copy2(p, STAGE / "data" / "inbox" / p.name)

assets = ROOT / "assets"
if assets.exists():
    for p in assets.iterdir():
        if p.is_file():
            shutil.copy2(p, STAGE / "assets" / p.name)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder, _, files in os.walk(STAGE):
        for f in files:
            fp = Path(folder) / f
            arc = fp.relative_to(STAGE).as_posix()
            zf.write(fp, arcname=arc)

shutil.rmtree(STAGE)
print(f"Wrote {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")
