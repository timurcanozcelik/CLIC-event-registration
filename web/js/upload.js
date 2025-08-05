// js/upload.js
document.addEventListener("DOMContentLoaded", () => {
    const submitBtn = document.getElementById("submit-btn");
    const nameInput = document.getElementById("participant-id");
    const templateImg = document.getElementById("template-img");
    const drawCanvas  = document.getElementById("draw-canvas");
  
    submitBtn.addEventListener("click", async () => {
      const participantId = nameInput.value.trim();
      if (!participantId) {
        alert("Please enter your name first.");
        return;
      }
  
      // Composite background + drawing
      const w = drawCanvas.width, h = drawCanvas.height;
      const off = document.createElement("canvas");
      off.width = w; off.height = h;
      const ctx = off.getContext("2d");
  
      // draw template
      ctx.drawImage(templateImg, 0, 0, w, h);
      // draw strokes
      ctx.drawImage(drawCanvas, 0, 0, w, h);
  
      // turn into PNG blob
      off.toBlob(async blob => {
        if (!blob) return alert("Export failed.");
        const fd = new FormData();
        fd.append("participant_id", participantId);
        fd.append("file", blob, "drawing.png");
  
        try {
          const res = await fetch("https://4221420e4a6b.ngrok-free.app/upload/", {
            method: "POST",
            body: fd
          });
          if (!res.ok) throw new Error(res.status);
          const json = await res.json();
          alert(`✅ Score: ${json.creativity_score}`);
        } catch (err) {
          console.error(err);
          alert("Upload failed: " + err);
        }
      }, "image/png");
    });
  });
  