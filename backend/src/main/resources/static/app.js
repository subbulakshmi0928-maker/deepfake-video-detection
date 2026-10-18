/* Deepfake Video Detection — front-end glue for the Spring Boot API. */
(() => {
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), fileInput = $("file"), preview = $("preview"), analyzeBtn = $("analyze");
  const fileName = $("file-name"), progress = $("progress"), bar = progress.querySelector(".bar");
  const progressText = progress.querySelector(".progress-text"), errorBox = $("error"), result = $("result");
  const mlStatus = $("ml-status");
  let selected = null;

  const fmtBytes = (n) => n < 1048576 ? (n / 1024).toFixed(0) + " KB" : (n / 1048576).toFixed(1) + " MB";
  const fmtPct = (x) => x == null ? "–" : (x * 100).toFixed(1) + "%";
  const fmtWhen = (iso) => new Date(iso).toLocaleString();
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

  function showError(msg) { errorBox.textContent = msg; errorBox.hidden = !msg; }

  function setFile(f) {
    if (!f) return;
    const ok = /\.(mp4|mov|avi|mkv|webm)$/i.test(f.name);
    if (!ok) { showError("Unsupported file type. Use MP4, MOV, AVI, MKV or WEBM."); return; }
    showError("");
    selected = f;
    fileName.textContent = `${f.name} · ${fmtBytes(f.size)}`;
    preview.src = URL.createObjectURL(f); preview.hidden = false;
    analyzeBtn.disabled = false;
  }

  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  function renderResult(a) {
    const v = (a.verdict || "ERROR").toLowerCase();
    const conf = a.confidence ?? 0;
    const untrained = a.modelName && /untrained/i.test(a.modelName);
    result.className = "result";
    result.innerHTML = `
      <div class="verdict">
        <span class="badge ${v}">${esc(a.verdict)}</span>
        <div><div style="font-weight:700">${esc(a.fileName)}</div><div class="muted" style="font-size:13px">${fmtBytes(a.sizeBytes)} · ${fmtWhen(a.createdAt)}</div></div>
      </div>
      ${v === "error" ? `<p class="error">${esc(a.errorMessage || "The ML service could not analyze this video.")}</p>` : `
      <div>
        <div class="row" style="margin:0 0 6px"><span class="muted">Confidence</span><strong>${fmtPct(conf)}</strong></div>
        <div class="meter ${v}"><div style="width:${(conf * 100).toFixed(1)}%"></div></div>
      </div>
      <dl class="kv">
        <div><dt>Fake probability</dt><dd>${fmtPct(a.fakeProbability)}</dd></div>
        <div><dt>Frames analyzed</dt><dd>${a.framesAnalyzed ?? "–"}</dd></div>
        <div><dt>Faces detected</dt><dd>${a.facesDetected ?? "–"}</dd></div>
        <div><dt>Processing time</dt><dd>${a.processingMs != null ? (a.processingMs / 1000).toFixed(2) + " s" : "–"}</dd></div>
        <div style="grid-column:1/-1"><dt>Model</dt><dd>${esc(a.modelName || "–")}</dd></div>
      </dl>
      ${untrained ? `<p class="note">The ML service is running without trained weights, so this verdict only demonstrates the pipeline.</p>` : ""}`}`;
  }

  async function analyze() {
    if (!selected) return;
    analyzeBtn.disabled = true; showError("");
    progress.hidden = false; bar.classList.remove("indeterminate"); bar.style.width = "0%"; progressText.textContent = "Uploading… 0%";
    const form = new FormData(); form.append("file", selected, selected.name);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/analyses");
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      const pct = Math.round((e.loaded / e.total) * 100);
      bar.style.width = pct + "%"; progressText.textContent = `Uploading… ${pct}%`;
      if (pct >= 100) { bar.classList.add("indeterminate"); progressText.textContent = "Analyzing frames with the ML model…"; }
    };
    xhr.onload = () => {
      progress.hidden = true; analyzeBtn.disabled = false;
      let body = {}; try { body = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status >= 200 && xhr.status < 300) { renderResult(body); loadHistory(); loadStats(); }
      else showError(body.message || `Request failed (${xhr.status})`);
      if (xhr.status === 502) { loadHistory(); loadStats(); }
    };
    xhr.onerror = () => { progress.hidden = true; analyzeBtn.disabled = false; showError("Network error while uploading."); };
    xhr.send(form);
  }
  analyzeBtn.addEventListener("click", analyze);

  async function loadHistory() {
    const r = await fetch("/api/analyses?page=0&size=25");
    if (!r.ok) return;
    const { items } = await r.json();
    const tbody = $("history").querySelector("tbody");
    tbody.innerHTML = items.map((a) => {
      const v = (a.verdict || "ERROR").toLowerCase();
      return `<tr>
        <td>${fmtWhen(a.createdAt)}</td>
        <td class="file" title="${esc(a.fileName)}">${esc(a.fileName)}</td>
        <td><span class="tag ${v}">${esc(a.verdict)}</span></td>
        <td>${fmtPct(a.confidence)}</td>
        <td>${a.framesAnalyzed ?? "–"}</td>
        <td>${a.facesDetected ?? "–"}</td>
        <td>${a.processingMs != null ? (a.processingMs / 1000).toFixed(1) + " s" : "–"}</td>
        <td><button class="icon-btn" data-id="${a.id}" title="Delete">✕</button></td>
      </tr>`;
    }).join("");
    $("history-empty").hidden = items.length > 0;
    tbody.querySelectorAll("button[data-id]").forEach((b) => b.addEventListener("click", async () => {
      await fetch(`/api/analyses/${b.dataset.id}`, { method: "DELETE" }); loadHistory(); loadStats();
    }));
  }

  async function loadStats() {
    const r = await fetch("/api/analyses/stats");
    if (!r.ok) return;
    const s = await r.json();
    $("s-total").textContent = s.total; $("s-fake").textContent = s.fake; $("s-real").textContent = s.real;
    $("s-avg").textContent = fmtPct(s.avgConfidence);
  }

  async function loadMlStatus() {
    try {
      const r = await fetch("/api/health"); const h = await r.json(); const ml = h.mlService || {};
      if (ml.status === "ok") {
        mlStatus.className = "pill " + (ml.trained ? "pill-ok" : "pill-warn");
        mlStatus.textContent = ml.trained ? `ML ready · ${ml.weights} · ${ml.device}` : `ML ready · untrained demo weights · ${ml.device}`;
      } else { mlStatus.className = "pill pill-down"; mlStatus.textContent = "ML service unreachable"; }
    } catch (_) { mlStatus.className = "pill pill-down"; mlStatus.textContent = "Backend unreachable"; }
  }

  async function showFromUrl() {
    // Deep link: /?analysis=<id> renders that stored analysis in the result card.
    const id = new URLSearchParams(location.search).get("analysis");
    if (!id) return;
    const r = await fetch(`/api/analyses/${encodeURIComponent(id)}`);
    if (r.ok) renderResult(await r.json());
  }

  $("refresh").addEventListener("click", () => { loadHistory(); loadStats(); loadMlStatus(); });
  loadMlStatus(); loadHistory(); loadStats(); showFromUrl();
})();
