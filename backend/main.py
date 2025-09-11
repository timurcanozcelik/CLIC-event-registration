# backend/main.py
import os
import uuid
import sys
import shutil
import csv
import logging
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ttct_backend")

# ---------------- Paths / config ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT = os.path.join(HERE, "submissions")
os.makedirs(SAVE_ROOT, exist_ok=True)

# AuDrA (unchanged)
AUDRA_BASE_DIR = os.path.join(HERE, "AuDrA_files", "AuDrA")
AUDRA_ENV = "audra_cpu"

app = FastAPI(title="TTCT Figural Registration Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Helpers ----------------
def compute_ink_density(pil_img: Image.Image) -> float:
    arr = np.array(pil_img.convert("RGB"))
    white_mask = (arr > 250).all(axis=2)
    ink_pixels = (~white_mask).sum()
    total = arr.shape[0] * arr.shape[1]
    return float(ink_pixels) / float(total)

def find_conda_env_python(env_name: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        for prefix in data.get("envs", []):
            if os.path.basename(prefix) == env_name or prefix.rstrip(os.path.sep).endswith(os.path.sep + env_name):
                python_path = os.path.join(prefix, "bin", "python")
                if os.path.exists(python_path):
                    return python_path
        return None
    except Exception as e:
        logger.warning("Could not introspect conda environments: %s", e)
        return None

def run_audra_on_file(src_path: str, audra_base_dir: str, env_name: Optional[str] = None) -> float:
    filename = os.path.basename(src_path)

    user_images = os.path.join(audra_base_dir, f"user_images_{uuid.uuid4().hex[:8]}")
    os.makedirs(user_images, exist_ok=True)
    dest_path = os.path.join(user_images, filename)
    shutil.copy(src_path, dest_path)

    abs_user_images = os.path.abspath(user_images)
    abs_output = os.path.abspath(os.path.join(audra_base_dir, f"audra_out_{uuid.uuid4().hex[:8]}.csv"))
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    tried_commands = []
    result = None

    if env_name:
        cmd = ["conda", "run", "-n", env_name, "python", "AuDrA_run.py",
               "--input-dir", abs_user_images, "--output-filename", abs_output]
        tried_commands.append(("conda_run", cmd))
        result = subprocess.run(cmd, cwd=audra_base_dir, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("`conda run` failed for env '%s': %s", env_name, result.stderr.strip())
            env_python = find_conda_env_python(env_name)
            if env_python:
                cmd2 = [env_python, "AuDrA_run.py",
                        "--input-dir", abs_user_images, "--output-filename", abs_output]
                tried_commands.append(("env_python_direct", cmd2))
                result = subprocess.run(cmd2, cwd=audra_base_dir, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning("Direct env python invocation failed: %s", result.stderr.strip())
            else:
                logger.warning("Could not locate python in env '%s'; falling back to current interpreter.", env_name)
                cmd3 = [sys.executable, "AuDrA_run.py",
                        "--input-dir", abs_user_images, "--output-filename", abs_output]
                tried_commands.append(("current_python_fallback", cmd3))
                result = subprocess.run(cmd3, cwd=audra_base_dir, capture_output=True, text=True)
    else:
        cmd = [sys.executable, "AuDrA_run.py",
               "--input-dir", abs_user_images, "--output-filename", abs_output]
        tried_commands.append(("current_python", cmd))
        result = subprocess.run(cmd, cwd=audra_base_dir, capture_output=True, text=True)

    if result is None:
        raise RuntimeError("No command executed for AuDrA invocation.")

    if result.returncode != 0:
        logger.error(
            "AuDrA subprocess failed. Tried: %s\nstdout:\n%s\nstderr:\n%s",
            [(name, " ".join(cmd)) for name, cmd in tried_commands],
            result.stdout, result.stderr,
        )
        raise RuntimeError(f"AuDrA execution failed: {result.stderr.strip()}")

    if not os.path.exists(abs_output):
        raise RuntimeError(f"Expected output file not created: {abs_output}")

    creativity_score = None
    with open(abs_output, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("filenames") == filename:
                creativity_score = float(row.get("predictions", 0.0))
                break
    if creativity_score is None:
        raise RuntimeError(f"AuDrA did not return a score for {filename}")
    return creativity_score

def read_meta_file(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out

def list_submissions() -> List[Dict[str, Any]]:
    """Scan SAVE_ROOT and return a list of dicts describing each PNG + meta."""
    items: List[Dict[str, Any]] = []
    for participant in os.listdir(SAVE_ROOT):
        part_dir = os.path.join(SAVE_ROOT, participant)
        if not os.path.isdir(part_dir):
            continue
        for fname in os.listdir(part_dir):
            if not fname.lower().endswith(".png"):
                continue
            meta = read_meta_file(os.path.join(part_dir, fname.replace(".png", ".txt")))
            uploaded_at = meta.get("uploaded_at") or ""
            try:
                score = float(meta.get("creativity_score", "nan"))
            except Exception:
                score = None
            used_fallback = (meta.get("used_fallback", "").lower() == "true")
            items.append({
                "participant_id": participant,
                "filename": fname,
                "uploaded_at": uploaded_at,
                "creativity_score": score,
                "used_fallback": used_fallback,
                "image_url": f"/submission/{participant}/{fname}",
            })
    # newest first by uploaded_at, fallback to filename
    items.sort(key=lambda x: (x.get("uploaded_at") or x["filename"]), reverse=True)
    return items

# ---------------- API ----------------
@app.post("/upload/")
async def upload(participant_id: str = Form(...), file: UploadFile = File(...)):
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = str(uuid.uuid4())[:8]
    base_name = f"{timestamp}_{participant_id}_{uid}"
    participant_dir = os.path.join(SAVE_ROOT, participant_id)
    os.makedirs(participant_dir, exist_ok=True)

    filename = f"{base_name}.png"
    file_path = os.path.join(participant_dir, filename)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    pil_img = Image.open(file_path)
    ink_ratio = compute_ink_density(pil_img)

    creativity_score = None
    used_fallback = False
    try:
        creativity_score = run_audra_on_file(file_path, AUDRA_BASE_DIR, env_name=AUDRA_ENV)
    except Exception as e:
        logger.warning("AuDrA inference failed for %s: %s", filename, e)
        creativity_score = ink_ratio  # fallback
        used_fallback = True

    creativity_score_rounded = round(float(creativity_score), 4)

    meta_path = os.path.join(participant_dir, f"{base_name}.txt")
    with open(meta_path, "w") as f:
        f.write(f"participant_id: {participant_id}\n")
        f.write(f"uploaded_at: {timestamp}\n")
        f.write(f"ink_density: {ink_ratio:.4f}\n")
        f.write(f"creativity_score: {creativity_score_rounded}\n")
        f.write(f"used_fallback: {used_fallback}\n")
        f.write(f"raw_filename: {filename}\n")

    return {
        "participant_id": participant_id,
        "proxy_creativity_score": round(ink_ratio, 4),
        "creativity_score": creativity_score_rounded,
        "drawing_file": filename,
        "used_fallback": used_fallback,
        "note": "AuDrA score" if not used_fallback else "Fallback to ink density",
        "image_url": f"/submission/{participant_id}/{filename}",
        "uploaded_at": timestamp,
    }

@app.get("/submission/{participant_id}/{filename}")
def serve_submission(participant_id: str, filename: str):
    path = os.path.join(SAVE_ROOT, participant_id, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/api/submissions")
def api_submissions():
    items = list_submissions()
    return {"items": items, "count": len(items)}

@app.get("/api/submissions.csv", response_class=PlainTextResponse)
def api_submissions_csv():
    items = list_submissions()
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["participant_id", "filename", "uploaded_at", "creativity_score", "used_fallback", "image_url"])
    for it in items:
        w.writerow([
            it["participant_id"], it["filename"], it["uploaded_at"],
            "" if it["creativity_score"] is None else it["creativity_score"],
            it["used_fallback"], it["image_url"]
        ])
    return buf.getvalue()

# ---------------- Slideshow page (self-contained) ----------------
@app.get("/slideshow", response_class=HTMLResponse)
def slideshow():
    # Uses same-origin /api/submissions, so images load without CORS issues.
    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CLIC Submissions – Slideshow</title>
<style>
  :root {{ --bg:#0b0b0e; --fg:#f5f7fa; --muted:#aab4c6; --panel:#111622; --btn:#1d2332; --btn2:#222a3d; --border:#2a3245; --accent:#4ea1ff; --err:#ff7d7d; }}
  html,body{{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
  .bar{{position:fixed;inset:0 0 auto 0;display:flex;gap:12px;align-items:center;padding:10px 14px;background:var(--panel);box-shadow:0 2px 8px rgba(0,0,0,.35);z-index:10}}
  .bar h1{{margin:0;font-size:16px;font-weight:600;opacity:.95}}
  .spacer{{flex:1}}
  .bar a,.bar button{{color:var(--fg);background:var(--btn);border:1px solid var(--border);padding:8px 12px;border-radius:8px;cursor:pointer;text-decoration:none}}
  .bar a:hover,.bar button:hover{{background:var(--btn2)}}
  .wrap{{position:fixed;inset:56px 0 64px 0;display:flex;align-items:center;justify-content:center}}
  .stage{{text-align:center;max-width:92vw;max-height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center}}
  /* Key bit: invert the uploaded (white-on-black) images so they show black-on-white */
  .stage img{{max-width:92vw;max-height:72vh;object-fit:contain;background:#fff;border-radius:10px;box-shadow:0 10px 30px rgba(0,0,0,.45);filter: invert(1);}}
  .caption{{margin-top:10px;font-size:20px;color:#fff;font-weight:600}}
  .controls{{position:fixed;inset:auto 0 12px 0;display:flex;justify-content:center;gap:10px}}
  .controls button{{background:var(--btn);color:var(--fg);border:1px solid var(--border);padding:10px 14px;border-radius:10px;font-size:15px;cursor:pointer}}
  .controls button:hover{{background:var(--btn2)}}
  .error{{color:var(--err)}}
  .debug{{position:fixed;left:12px;bottom:12px;color:#9aa3b7;font-size:12px;opacity:.8}}
</style>
</head>
<body>
  <div class="bar">
    <h1>CLIC Submissions – Slideshow</h1>
    <div class="spacer"></div>
    <button id="fsBtn" title="Fullscreen (F)">Fullscreen</button>
    <button id="refreshBtn">Refresh</button>
    <a href="/api/submissions.csv" target="_blank" rel="noopener">Download CSV</a>
  </div>

  <div class="wrap">
    <div class="stage">
      <img id="slide" alt="submission">
      <!-- Caption now shows ONLY participant + score -->
      <div id="caption" class="caption">Loading…</div>
    </div>
  </div>

  <div class="controls">
    <button id="prevBtn">◀︎ Previous</button>
    <button id="playBtn">⏸ Pause</button>
    <button id="nextBtn">Next ▶︎</button>
  </div>

  <div id="debug" class="debug"></div>

<script>
  const FEED = "/api/submissions";
  const STEP_MS = 5000;

  let items = [];
  let idx = 0;
  let playing = true;
  let timer = null;

  const imgEl  = document.getElementById("slide");
  const capEl  = document.getElementById("caption");
  const dbgEl  = document.getElementById("debug");
  const fsBtn  = document.getElementById("fsBtn");
  const playBtn= document.getElementById("playBtn");

  function dbg(m){ console.log("[slideshow]", m); dbgEl.textContent = String(m); }

  // --- Fullscreen helpers (must be called from a user gesture) ---
  async function goFullscreen(){
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
      }
    } catch(e){ console.warn(e); }
  }
  fsBtn.addEventListener("click", goFullscreen);
  // also try when Play is pressed
  function tryFullscreenOnPlay(){ if (!document.fullscreenElement) goFullscreen(); }
  document.addEventListener("keydown",(e)=>{
    if (e.key.toLowerCase()==='f') goFullscreen();
  });

  imgEl.addEventListener("error", () => {
    const it = items[idx];
    if (!it) return;
    if (imgEl.dataset.tried === "1") {
      capEl.innerHTML += ' <span class="error">(image not found)</span>';
      return;
    }
    imgEl.dataset.tried = "1";
    imgEl.src = it.image_url + (it.image_url.includes('?') ? '&' : '?') + 't=' + Date.now();
  });

  async function load() {
    stop();
    capEl.textContent = "Loading…";
    try {
      const r = await fetch(FEED + "?t=" + Date.now(), { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const data = await r.json();
      const list = Array.isArray(data) ? data : (data.items || []);
      if (!list.length) {
        capEl.textContent = "No submissions yet.";
        imgEl.removeAttribute("src");
        return;
      }
      // newest first
      items = list.slice().sort((a,b)=> String(b.uploaded_at||b.filename).localeCompare(String(a.uploaded_at||a.filename)));
      idx = 0;
      show(idx);
      playing ? start() : stop();
      dbg("Loaded " + items.length + " submissions.");
    } catch (e) {
      console.error(e);
      capEl.innerHTML = '<span class="error">Failed to load submissions.</span>';
      dbg(e.message);
    }
  }

  function show(i) {
    if (!items.length) return;
    idx = (i + items.length) % items.length;
    const it = items[idx];
    imgEl.dataset.tried = "0";
    // cache-buster so we always get the latest file
    imgEl.src = it.image_url + (it.image_url.includes('?') ? '&' : '?') + 't=' + Date.now();

    const score = (it.creativity_score == null || isNaN(it.creativity_score)) ? "–" : Number(it.creativity_score).toFixed(4);
    // Caption: ONLY participant + score (no filename or links)
    capEl.textContent = `Participant: ${it.participant_id ?? "—"}   •   Creativity score: ${score}`;
    // prefetch next
    const pre = new Image();
    pre.src = items[(idx+1)%items.length].image_url;
  }

  function next() { show(idx+1); }
  function prev() { show(idx-1); }
  function start() { stop(); playing = true; playBtn.textContent = "⏸ Pause"; timer = setInterval(next, STEP_MS); tryFullscreenOnPlay(); }
  function stop()  { playing = false; playBtn.textContent = "▶︎ Play"; clearInterval(timer); timer = null; }
  function toggle(){ playing ? stop() : start(); }

  document.getElementById("refreshBtn").addEventListener("click", load);
  document.getElementById("nextBtn").addEventListener("click", () => { stop(); next(); });
  document.getElementById("prevBtn").addEventListener("click", () => { stop(); prev(); });
  playBtn.addEventListener("click", toggle);

  // auto-refresh list every 5s (adds new submissions while presenting)
  setInterval(async () => {
    try {
      const r = await fetch(FEED + "?t=" + Date.now(), { cache:"no-store" });
      if (!r.ok) return;
      const data = await r.json();
      const list = Array.isArray(data) ? data : (data.items || []);
      if (list.length !== items.length) {
        const wasPlaying = playing;
        await load();
        if (wasPlaying) start();
      }
    } catch(_e) {}
  }, 5000);

  load();
</script>
</body></html>"""
    return HTMLResponse(html)
