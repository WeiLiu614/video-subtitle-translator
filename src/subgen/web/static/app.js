const state = {
  inputMode: "upload",
  selectedFile: null,
  currentJobId: null,
  segments: [],
  mediaAvailable: false,
  eventSource: null,
  saveTimer: null,
  pendingEdits: new Map(),
};

const els = {
  envBadges: document.getElementById("env-badges"),
  tabs: document.querySelectorAll(".tab"),
  panelUpload: document.getElementById("panel-upload"),
  panelLocal: document.getElementById("panel-local"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  selectedFile: document.getElementById("selected-file"),
  localPath: document.getElementById("local-path"),
  form: document.getElementById("job-form"),
  startBtn: document.getElementById("start-btn"),
  formError: document.getElementById("form-error"),
  sourceLang: document.getElementById("source-lang"),
  whisperModel: document.getElementById("whisper-model"),
  asrBackend: document.getElementById("asr-backend"),
  translator: document.getElementById("translator"),
  translationModel: document.getElementById("translation-model"),
  apiKey: document.getElementById("api-key"),
  apiBaseUrl: document.getElementById("api-base-url"),
  batchSize: document.getElementById("batch-size"),
  contextPrompt: document.getElementById("context-prompt"),
  bilingual: document.getElementById("bilingual"),
  saveSource: document.getElementById("save-source"),
  progressCard: document.getElementById("progress-card"),
  jobTitle: document.getElementById("job-title"),
  progressMsg: document.getElementById("progress-msg"),
  progressPct: document.getElementById("progress-pct"),
  progressBar: document.getElementById("progress-bar"),
  subProgressWrap: document.getElementById("sub-progress-wrap"),
  subProgressLabel: document.getElementById("sub-progress-label"),
  subProgressPct: document.getElementById("sub-progress-pct"),
  subProgressBar: document.getElementById("sub-progress-bar"),
  stagePills: document.querySelectorAll(".pill"),
  workspaceCard: document.getElementById("workspace-card"),
  workspaceGrid: document.querySelector(".workspace-grid"),
  workspaceMeta: document.getElementById("workspace-meta"),
  playerPane: document.getElementById("player-pane"),
  videoPlayer: document.getElementById("video-player"),
  trackMode: document.getElementById("track-mode"),
  saveIndicator: document.getElementById("save-indicator"),
  cueSearch: document.getElementById("cue-search"),
  cueCount: document.getElementById("cue-count"),
  cueList: document.getElementById("cue-list"),
  btnDlZh: document.getElementById("btn-dl-zh"),
  btnDlBilingual: document.getElementById("btn-dl-bilingual"),
  btnDlSource: document.getElementById("btn-dl-source"),
};

function fmtTime(sec) {
  const msTotal = Math.max(0, Math.round(sec * 1000));
  const h = Math.floor(msTotal / 3600000);
  const m = Math.floor((msTotal % 3600000) / 60000);
  const s = Math.floor((msTotal % 60000) / 1000);
  const ms = msTotal % 1000;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const badges = [];
    const hw = data.hardware;
    if (hw) {
      if (hw.primary_accelerator === "apple_silicon") {
        badges.push(`<span class="badge ok">macOS · Apple Silicon</span>`);
      } else if (hw.primary_accelerator === "nvidia_cuda") {
        const gpuLabel = hw.primary_gpu_name || "NVIDIA CUDA";
        badges.push(`<span class="badge ok">${hw.os_name} · ${gpuLabel}</span>`);
      } else if (hw.primary_accelerator === "intel_igpu") {
        const gpuLabel = hw.primary_gpu_name || "Intel GPU";
        badges.push(`<span class="badge ok">${hw.os_name} · ${gpuLabel}</span>`);
      } else {
        badges.push(`<span class="badge">${hw.os_name} · CPU</span>`);
      }

      const autoOpt = els.whisperModel.querySelector('option[value="auto"]');
      if (autoOpt && hw.recommended_whisper_model) {
        autoOpt.textContent = `自动选择（推荐 ${hw.recommended_whisper_model} · ${hw.recommended_backend}）`;
      }
      const autoAsrOpt = els.asrBackend.querySelector('option[value="auto"]');
      if (autoAsrOpt && hw.recommended_backend) {
        autoAsrOpt.textContent = `自动选择（推荐 ${hw.recommended_backend} / ${hw.recommended_device}）`;
      }
    } else if (data.apple_silicon) {
      badges.push(`<span class="badge ok">Apple Silicon</span>`);
    }

    const installedAsr = [];
    if (data.asr_backends.mlx) installedAsr.push("MLX Whisper ✓");
    if (data.asr_backends.faster_whisper) installedAsr.push("faster-whisper ✓");
    if (data.asr_backends.openvino) installedAsr.push("OpenVINO GPU ✓");

    if (installedAsr.length > 0) {
      for (const label of installedAsr) {
        badges.push(`<span class="badge ok">${label}</span>`);
      }
    } else {
      badges.push(`<span class="badge warn">未安装本地 Whisper</span>`);
    }

    badges.push(
      data.ffmpeg_available
        ? `<span class="badge ok">FFmpeg ✓</span>`
        : `<span class="badge warn">缺少 FFmpeg</span>`
    );
    if (data.api_keys_configured.gemini) {
      badges.push(`<span class="badge ok">Gemini API ✓</span>`);
    }
    if (data.api_keys_configured.openai) {
      badges.push(`<span class="badge ok">OpenAI API ✓</span>`);
    }
    if (!data.api_keys_configured.gemini && !data.api_keys_configured.openai) {
      badges.push(`<span class="badge warn">请在 .env 或高级选项填写 API Key</span>`);
    }
    els.envBadges.innerHTML = badges.join("");

    if (data.defaults) {
      els.asrBackend.value = data.defaults.asr_backend || "auto";
      if (data.defaults.whisper_model) {
        els.whisperModel.value = data.defaults.whisper_model;
      }
      els.translator.value = data.defaults.translator || "openai";
      if (data.defaults.openai_base_url) {
        els.apiBaseUrl.value = data.defaults.openai_base_url;
      }
    }
  } catch (err) {
    els.envBadges.innerHTML = `<span class="badge warn">无法连接本地服务</span>`;
  }
}

// Mode tabs
els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    state.inputMode = tab.dataset.mode;
    els.tabs.forEach((t) => t.classList.toggle("active", t === tab));
    els.panelUpload.classList.toggle("active", state.inputMode === "upload");
    els.panelLocal.classList.toggle("active", state.inputMode === "local");
  });
});

// Drag and drop
function selectFile(file) {
  if (!file) return;
  state.selectedFile = file;
  const mb = (file.size / (1024 * 1024)).toFixed(1);
  els.selectedFile.textContent = `已选择: ${file.name} (${mb} MB)`;
  els.selectedFile.classList.remove("hidden");
  els.formError.textContent = "";
}

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (e) => selectFile(e.target.files[0]));
els.dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  els.dropzone.classList.add("dragover");
});
els.dropzone.addEventListener("dragleave", () => els.dropzone.classList.remove("dragover"));
els.dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  els.dropzone.classList.remove("dragover");
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    selectFile(e.dataTransfer.files[0]);
  }
});

// Progress pill helper
const STAGE_ORDER = ["extracting", "transcribing", "translating", "writing"];

function updateProgressUI(stage, pct, message, detail = {}) {
  els.progressPct.textContent = `${Math.round(pct)}%`;
  els.progressBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  els.progressMsg.textContent = message;

  const stagePct = typeof detail?.stage_progress === "number" ? Math.max(0, Math.min(100, detail.stage_progress)) : null;
  if (stagePct !== null && (stage === "transcribing" || stage === "translating")) {
    els.subProgressWrap.classList.remove("hidden");
    els.subProgressLabel.textContent = stage === "transcribing" ? "Whisper 阶段进度" : "LLM 翻译阶段进度";
    els.subProgressPct.textContent = `${Math.round(stagePct)}%`;
    els.subProgressBar.style.width = `${stagePct}%`;
  } else {
    els.subProgressWrap.classList.add("hidden");
  }

  const idx = STAGE_ORDER.indexOf(stage);
  els.stagePills.forEach((pill) => {
    const pIdx = STAGE_ORDER.indexOf(pill.dataset.stage);
    const baseLabel = pill.dataset.label || pill.textContent;
    pill.classList.remove("active", "done");
    if (stage === "completed") {
      pill.classList.add("done");
      pill.textContent = `${baseLabel} ✓`;
    } else if (idx !== -1) {
      if (pIdx < idx) {
        pill.classList.add("done");
        pill.textContent = `${baseLabel} ✓`;
      } else if (pIdx === idx) {
        pill.classList.add("active");
        pill.textContent = stagePct !== null ? `${baseLabel} · ${Math.round(stagePct)}%` : baseLabel;
      } else {
        pill.textContent = baseLabel;
      }
    } else {
      pill.textContent = baseLabel;
    }
  });
}

// Form submission
els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  els.formError.textContent = "";

  const formData = new FormData();
  if (state.inputMode === "upload") {
    if (!state.selectedFile) {
      els.formError.textContent = "请先拖拽或选择一个视频/音频/SRT 文件。";
      return;
    }
    formData.append("file", state.selectedFile);
  } else {
    const lp = els.localPath.value.trim();
    if (!lp) {
      els.formError.textContent = "请输入本地文件的绝对路径。";
      return;
    }
    formData.append("local_path", lp);
  }

  formData.append("source_lang", els.sourceLang.value);
  formData.append("asr_backend", els.asrBackend.value);
  formData.append("whisper_model", els.whisperModel.value);
  formData.append("translator", els.translator.value);
  formData.append("translation_model", els.translationModel.value.trim());
  formData.append("api_key", els.apiKey.value.trim());
  formData.append("api_base_url", els.apiBaseUrl.value.trim());
  formData.append("batch_size", els.batchSize.value || "30");
  formData.append("context_prompt", els.contextPrompt.value.trim());
  formData.append("bilingual", els.bilingual.checked ? "true" : "false");
  formData.append("save_source_srt", els.saveSource.checked ? "true" : "false");

  els.startBtn.disabled = true;
  els.workspaceCard.classList.add("hidden");
  els.progressCard.classList.remove("hidden");
  els.jobTitle.textContent = "正在提交任务…";
  updateProgressUI("preparing", 2, "正在准备文件…");

  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }

  try {
    const res = await fetch("/api/jobs", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "任务创建失败");
    }

    state.currentJobId = data.job_id;
    els.jobTitle.textContent = `正在处理：${data.filename}`;
    subscribeToJob(data.job_id);
  } catch (err) {
    els.startBtn.disabled = false;
    els.formError.textContent = err.message;
    els.progressCard.classList.add("hidden");
  }
});

function subscribeToJob(jobId) {
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource = es;

  es.onmessage = (evt) => {
    const payload = JSON.parse(evt.data);
    updateProgressUI(payload.stage, payload.progress, payload.message, payload.detail || {});
  };

  es.addEventListener("done", async (evt) => {
    es.close();
    state.eventSource = null;
    els.startBtn.disabled = false;

    const info = JSON.parse(evt.data);
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();

    if (info.status === "completed") {
      renderWorkspace(job);
    } else {
      els.formError.textContent = job.error || "生成失败，请检查配置或 API Key。";
    }
  });

  es.onerror = () => {
    es.close();
    els.startBtn.disabled = false;
  };
}

function reloadVideoTrack() {
  if (!state.mediaAvailable || !state.currentJobId) return;
  const video = els.videoPlayer;
  const mode = els.trackMode.value;

  // Remove existing tracks
  video.querySelectorAll("track").forEach((t) => t.remove());
  if (mode === "off") return;

  const track = document.createElement("track");
  track.kind = "subtitles";
  track.label = "SubGen";
  track.srclang = "zh";
  track.default = true;
  track.src = `/api/jobs/${state.currentJobId}/subtitles.vtt?mode=${mode}&t=${Date.now()}`;
  video.appendChild(track);
  if (video.textTracks && video.textTracks[0]) {
    video.textTracks[0].mode = "showing";
  }
}

function renderWorkspace(job) {
  state.segments = job.segments || [];
  state.mediaAvailable = Boolean(job.media_available);

  els.workspaceCard.classList.remove("hidden");
  els.workspaceMeta.textContent = `检测语言：${job.detected_language || "auto"} · 共 ${state.segments.length} 条字幕 · 输出路径：${job.output_srt_path || ""}`;

  if (state.mediaAvailable) {
    els.playerPane.classList.remove("hidden");
    els.workspaceGrid.classList.remove("no-media");
    els.videoPlayer.src = `/api/jobs/${job.job_id}/media`;
    reloadVideoTrack();
  } else {
    els.playerPane.classList.add("hidden");
    els.workspaceGrid.classList.add("no-media");
  }

  renderCueList();
}

function renderCueList(filter = "") {
  const q = filter.trim().toLowerCase();
  const filtered = q
    ? state.segments.filter(
        (s) =>
          s.source_text.toLowerCase().includes(q) ||
          s.translated_text.toLowerCase().includes(q)
      )
    : state.segments;

  els.cueCount.textContent = `${filtered.length} / ${state.segments.length} 条字幕`;
  els.cueList.innerHTML = "";

  const frag = document.createDocumentFragment();
  for (const seg of filtered) {
    const card = document.createElement("div");
    card.className = "cue-item";
    card.dataset.index = String(seg.index);

    const meta = document.createElement("div");
    meta.className = "cue-meta";

    const idxSpan = document.createElement("span");
    idxSpan.className = "muted";
    idxSpan.textContent = `#${seg.index}`;

    const timeSpan = document.createElement("span");
    timeSpan.className = "cue-time";
    timeSpan.textContent = `${fmtTime(seg.start)} → ${fmtTime(seg.end)}`;
    timeSpan.title = "点击跳转到视频该时间点";
    timeSpan.addEventListener("click", () => {
      if (state.mediaAvailable) {
        els.videoPlayer.currentTime = seg.start;
        els.videoPlayer.play().catch(() => {});
      }
    });

    meta.append(idxSpan, timeSpan);

    const srcDiv = document.createElement("div");
    srcDiv.className = "cue-source";
    srcDiv.textContent = seg.source_text;

    const textarea = document.createElement("textarea");
    textarea.className = "cue-input";
    textarea.rows = 1;
    textarea.value = seg.translated_text;
    textarea.addEventListener("input", () => {
      seg.translated_text = textarea.value;
      scheduleSave(seg.index, textarea.value);
    });

    card.append(meta, srcDiv, textarea);
    frag.appendChild(card);
  }
  els.cueList.appendChild(frag);
}

function scheduleSave(index, text) {
  state.pendingEdits.set(index, text);
  els.saveIndicator.textContent = "保存中…";
  els.saveIndicator.classList.remove("saved");

  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(async () => {
    if (!state.currentJobId || state.pendingEdits.size === 0) return;
    const payload = {
      segments: Array.from(state.pendingEdits.entries()).map(([idx, translated_text]) => ({
        index: idx,
        translated_text,
      })),
    };
    state.pendingEdits.clear();

    try {
      const res = await fetch(`/api/jobs/${state.currentJobId}/segments`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        els.saveIndicator.textContent = "✓ 已同步到字幕预览与 .srt";
        els.saveIndicator.classList.add("saved");
        reloadVideoTrack();
      }
    } catch {
      els.saveIndicator.textContent = "保存失败";
    }
  }, 450);
}

// Highlight active cue while video plays
els.videoPlayer.addEventListener("timeupdate", () => {
  const t = els.videoPlayer.currentTime;
  const active = state.segments.find((s) => t >= s.start && t <= s.end);
  const activeIdx = active ? String(active.index) : null;
  els.cueList.querySelectorAll(".cue-item").forEach((el) => {
    const isPlaying = el.dataset.index === activeIdx;
    el.classList.toggle("playing", isPlaying);
  });
});

els.trackMode.addEventListener("change", reloadVideoTrack);
els.cueSearch.addEventListener("input", (e) => renderCueList(e.target.value));

function triggerDownload(mode) {
  if (!state.currentJobId) return;
  window.location.href = `/api/jobs/${state.currentJobId}/download?mode=${mode}`;
}

els.btnDlZh.addEventListener("click", () => triggerDownload("translated"));
els.btnDlBilingual.addEventListener("click", () => triggerDownload("bilingual"));
els.btnDlSource.addEventListener("click", () => triggerDownload("source"));

loadStatus();
