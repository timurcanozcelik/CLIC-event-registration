// Configuration
const BACKEND_URL = "https://827224a51b6e.ngrok-free.app/upload/"; // ← update this

// Setup canvas
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
let drawing = false;
let last = { x: 0, y: 0 };

// Resize & load template
function resizeCanvas() {
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  drawTemplate();
}
window.addEventListener("resize", resizeCanvas);
const template = new Image();
template.src = "FigureTemplate.png"; // put your template file alongside
template.onload = () => resizeCanvas();

function drawTemplate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // draw background template to cover entire canvas
  ctx.globalAlpha = 1;
  ctx.drawImage(template, 0, 0, canvas.width, canvas.height);
  // set up drawing style
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
}

// Pointer event handlers
canvas.addEventListener("pointerdown", e => {
  drawing = true;
  last = { x: e.offsetX, y: e.offsetY };
});
canvas.addEventListener("pointermove", e => {
  if (!drawing) return;
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  last = { x: e.offsetX, y: e.offsetY };
});
window.addEventListener("pointerup", () => drawing = false);

// Clear button
document.getElementById("clearBtn").onclick = () => {
  drawTemplate();
  setStatus("");
};

// Status helper
function setStatus(msg, isError = false) {
  const s = document.getElementById("status");
  s.textContent = msg;
  s.style.color = isError ? "crimson" : "#333";
}

// Submit
document.getElementById("submitBtn").onclick = async () => {
  const pid = document.getElementById("participantId").value.trim();
  if (!pid) {
    setStatus("Enter a Participant ID first!", true);
    return;
  }
  setStatus("Preparing image…");

  // Merge template+drawing into a single blob
  canvas.toBlob(async blob => {
    if (!blob) return setStatus("Failed to capture drawing", true);

    // build multipart form
    const form = new FormData();
    form.append("participant_id", pid);
    form.append("file", blob, "drawing.png");

    setStatus("Uploading…");
    try {
      const resp = await fetch(BACKEND_URL, {
        method: "POST",
        body: form
      });
      if (!resp.ok) throw new Error(resp.statusText);
      const data = await resp.json();
      setStatus(`Success! Score: ${data.creativity_score}`);
    } catch (err) {
      console.error(err);
      setStatus("Upload failed: " + err.message, true);
    }
  }, "image/png");
};
