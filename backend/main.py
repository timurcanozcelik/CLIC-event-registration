import os
import uuid
import sys
import shutil
import csv
import logging
import subprocess
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ttct_backend")

SAVE_ROOT = "submissions"
os.makedirs(SAVE_ROOT, exist_ok=True)

# Path to the AuDrA directory (adjust if moved)
AUDRA_BASE_DIR = os.path.join("AuDrA_files", "AuDrA")  # relative to this script
# Name of the AuDrA conda environment
AUDRA_ENV = "audra_cpu"

app = FastAPI(title="TTCT Figural Registration Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def compute_ink_density(pil_img: Image.Image) -> float:
    arr = np.array(pil_img.convert("RGB"))
    white_mask = (arr > 250).all(axis=2)
    ink_pixels = (~white_mask).sum()
    total = arr.shape[0] * arr.shape[1]
    return float(ink_pixels) / float(total)


def find_conda_env_python(env_name: str) -> str | None:
    """
    Try to locate the python binary inside a conda environment by name using `conda env list --json`.
    """
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


def run_audra_on_file(src_path: str, audra_base_dir: str, env_name: str | None = None) -> float:
    filename = os.path.basename(src_path)

    # Prepare a unique user_images input folder and copy the drawing there
    user_images = os.path.join(audra_base_dir, f"user_images_{uuid.uuid4().hex[:8]}")
    os.makedirs(user_images, exist_ok=True)
    dest_path = os.path.join(user_images, filename)
    shutil.copy(src_path, dest_path)

    abs_user_images = os.path.abspath(user_images)
    abs_output = os.path.abspath(os.path.join(audra_base_dir, f"audra_out_{uuid.uuid4().hex[:8]}.csv"))
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    tried_commands = []
    result = None

    # Primary attempt: conda run if env_name provided
    if env_name:
        cmd = [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            "AuDrA_run.py",
            "--input-dir",
            abs_user_images,
            "--output-filename",
            abs_output,
        ]
        tried_commands.append(("conda_run", cmd))
        result = subprocess.run(cmd, cwd=audra_base_dir, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("`conda run` failed for env '%s': %s", env_name, result.stderr.strip())

            # Try directly invoking the python inside that env if we can discover it
            env_python = find_conda_env_python(env_name)
            if env_python:
                cmd2 = [
                    env_python,
                    "AuDrA_run.py",
                    "--input-dir",
                    abs_user_images,
                    "--output-filename",
                    abs_output,
                ]
                tried_commands.append(("env_python_direct", cmd2))
                result = subprocess.run(cmd2, cwd=audra_base_dir, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning("Direct env python invocation also failed: %s", result.stderr.strip())
            else:
                logger.warning("Could not locate python in conda env '%s'; falling back to current interpreter.", env_name)
                cmd3 = [
                    sys.executable,
                    "AuDrA_run.py",
                    "--input-dir",
                    abs_user_images,
                    "--output-filename",
                    abs_output,
                ]
                tried_commands.append(("current_python_fallback", cmd3))
                result = subprocess.run(cmd3, cwd=audra_base_dir, capture_output=True, text=True)
    else:
        # No env specified: just use current interpreter
        cmd = [
            sys.executable,
            "AuDrA_run.py",
            "--input-dir",
            abs_user_images,
            "--output-filename",
            abs_output,
        ]
        tried_commands.append(("current_python", cmd))
        result = subprocess.run(cmd, cwd=audra_base_dir, capture_output=True, text=True)

    if result is None:
        raise RuntimeError("No command was executed for AuDrA invocation.")

    if result.returncode != 0:
        logger.error(
            "AuDrA subprocess failed. Tried commands: %s\nLast stdout:\n%s\nLast stderr:\n%s",
            [(name, " ".join(cmd)) for name, cmd in tried_commands],
            result.stdout,
            result.stderr,
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

    creativity_score_rounded = round(creativity_score, 4)

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
    }


@app.get("/admin/", response_class=HTMLResponse)
def admin_view():
    html = ["<html><head><title>Submissions</title></head><body><h1>Recent Drawings</h1><ul>"]
    for participant in sorted(os.listdir(SAVE_ROOT)):
        part_dir = os.path.join(SAVE_ROOT, participant)
        if not os.path.isdir(part_dir):
            continue
        html.append(f"<li><strong>{participant}</strong><ul>")
        for fname in sorted(os.listdir(part_dir), reverse=True):
            if fname.endswith(".png"):
                thumb_url = f"/submission/{participant}/{fname}"
                meta_file = fname.replace(".png", ".txt")
                meta_text = ""
                meta_path = os.path.join(part_dir, meta_file)
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as m:
                        meta_text = "<br/>".join(m.read().splitlines())
                html.append(
                    f"<li>"
                    f"<div style='margin-bottom:4px;'>"
                    f"<a href='{thumb_url}' target='_blank'>{fname}</a>"
                    f"</div>"
                    f"<div style='font-size:0.8em; background:#f0f0f0; padding:4px; border-radius:4px;'>{meta_text}</div>"
                    f"</li>"
                )
        html.append("</ul></li>")
    html.append("</ul></body></html>")
    return "\n".join(html)


@app.get("/submission/{participant_id}/{filename}")
def serve_submission(participant_id: str, filename: str):
    path = os.path.join(SAVE_ROOT, participant_id, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "Not found"}
