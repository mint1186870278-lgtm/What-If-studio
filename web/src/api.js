function getViteEnvValue(key) {
  const env = import.meta?.env;
  if (!env || typeof env !== "object") return "";
  return String(env[key] || "").trim();
}

export function resolveApiBase() {
  const envBase = getViteEnvValue("VITE_API_BASE");
  if (envBase) return envBase.replace(/\/+$/, "");
  if (typeof window === "undefined") return "http://localhost:8000";
  // Prefer same-origin /api in browser so dev proxy and production reverse-proxy both work.
  return "";
}

const API_BASE = resolveApiBase();

function joinApiUrl(path) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (!API_BASE) return normalizedPath;
  return `${API_BASE}${normalizedPath}`;
}

function safeParseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

export function parseSseDataLine(rawEvent) {
  const line = String(rawEvent || "")
    .split("\n")
    .find((item) => item.startsWith("data: "));
  if (!line) return null;
  return safeParseJson(line.slice(6));
}

async function parseApiError(res, fallbackMessage) {
  try {
    const data = await res.json();
    if (data?.error?.message) return data.error.message;
    if (data?.error && typeof data.error === "string") return data.error;
    if (data?.message) return data.message;
  } catch {
    // Ignore non-JSON error responses and fall back to generic text.
  }
  return fallbackMessage;
}

export async function streamProjectDiscussion(projectId, onTurn, { userId } = {}) {
  const params = new URLSearchParams();
  if (userId) params.set("user_id", userId);
  const query = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(joinApiUrl(`/api/projects/${projectId}/script/stream${query}`), {
    method: "POST",
    headers: { "Content-Type": "application/json" }
  });
  if (!res.ok || !res.body) {
    throw new Error(await parseApiError(res, "讨论流打开失败"));
  }

  const decoder = new TextDecoder();
  const reader = res.body.getReader();
  let buffer = "";

  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const raw of events) {
      const payload = parseSseDataLine(raw);
      if (!payload) continue;
      onTurn(payload);
    }
  }
}

export async function createProjectVideoJob(projectId) {
  const res = await fetch(joinApiUrl(`/api/projects/${projectId}/video-jobs`), {
    method: "POST",
    headers: { "Content-Type": "application/json" }
  });
  if (!res.ok) throw new Error(await parseApiError(res, "视频任务创建失败"));
  const data = await res.json();
  return data?.job ? data : { job: { ...data, jobId: data.id } };
}

export async function uploadVideoSource(projectId, file) {
  if (!(file instanceof Blob)) throw new Error("无效文件");
  const fileName = String(file.name || "upload.mp4").trim() || "upload.mp4";
  const formData = new FormData();
  formData.append("file", file, fileName);
  formData.append("asset_type", "video");
  const res = await fetch(joinApiUrl(`/api/projects/${projectId}/assets`), {
    method: "POST",
    body: formData
  });
  if (!res.ok) throw new Error(await parseApiError(res, "素材上传失败"));
  const asset = await res.json();
  return { upload: { ...asset, uploadId: asset.id, originalName: asset.file_name } };
}

export async function fetchAgents() {
  const res = await fetch(joinApiUrl("/api/agents"));
  if (!res.ok) throw new Error(await parseApiError(res, "角色列表加载失败"));
  const data = await res.json();
  return Array.isArray(data?.agents) ? data.agents : [];
}

export function watchVideoJob(jobId, onEvent, options = {}) {
  const params = new URLSearchParams();
  if (options.videoUrl) params.set("video_url", options.videoUrl);
  if (options.refImageUrls?.length) {
    options.refImageUrls.forEach((url) => params.append("ref_image_url", url));
  }
  const query = params.toString() ? `?${params.toString()}` : "";
  const source = new EventSource(joinApiUrl(`/api/video-jobs/${jobId}/events${query}`));
  source.onmessage = (event) => {
    const payload = safeParseJson(event.data);
    if (!payload) return;
    const mapped = payload.type ? { ...payload, event: payload.type } : payload;
    if (mapped.event === "complete") {
      mapped.result = {
        type: "video-mp4",
        publicUrl: joinApiUrl(`/api/video-jobs/${jobId}/output`),
        outputPath: mapped.output_path || "",
      };
    }
    onEvent(mapped);
    if (mapped.event === "complete" || mapped.event === "error") source.close();
  };
  source.onerror = () => {
    if (typeof options.onDisconnect === "function") options.onDisconnect();
    onEvent({ event: "error", message: "任务事件流中断，请稍后重试。" });
    source.close();
  };
  return () => source.close();
}

export async function getProject(projectId) {
  const res = await fetch(joinApiUrl(`/api/projects/${projectId}`));
  if (!res.ok) throw new Error(await parseApiError(res, "工程加载失败"));
  return res.json();
}

export async function createProject(payload) {
  const res = await fetch(joinApiUrl("/api/projects"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseApiError(res, "工程创建失败"));
  return res.json();
}

export async function updateProject(projectId, payload) {
  const res = await fetch(joinApiUrl(`/api/projects/${projectId}`), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseApiError(res, "工程更新失败"));
  return res.json();
}

// -------------------------------------------------------------------------
// WebSocket for real-time intervention
// -------------------------------------------------------------------------

let wsIntervention = null;
let wsInterventionCallbacks = {};

export function connectInterventionWebSocket(sessionId, callbacks = {}) {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host || "localhost:8010";
  const url = `${proto}//${host}/ws/${sessionId}`;
  wsInterventionCallbacks = callbacks;
  try {
    wsIntervention = new WebSocket(url);
  } catch (e) {
    console.warn("WebSocket creation failed:", e);
    return null;
  }
  wsIntervention.onopen = () => callbacks.onOpen?.();
  wsIntervention.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      callbacks.onMessage?.(msg);
    } catch {}
  };
  wsIntervention.onerror = (e) => callbacks.onError?.(e);
  wsIntervention.onclose = (e) => callbacks.onClose?.(e);
  return wsIntervention;
}

export function sendIntervene(sessionId, text) {
  if (wsIntervention && wsIntervention.readyState === WebSocket.OPEN) {
    wsIntervention.send(JSON.stringify({ action: "intervene", text }));
    return true;
  }
  // REST fallback
  fetch(joinApiUrl(`/api/sessions/${sessionId}/intervene`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }).catch(() => {});
  return false;
}

export function sendPause(sessionId) {
  if (wsIntervention && wsIntervention.readyState === WebSocket.OPEN) {
    wsIntervention.send(JSON.stringify({ action: "pause_now" }));
  }
}

export function sendResume(sessionId) {
  if (wsIntervention && wsIntervention.readyState === WebSocket.OPEN) {
    wsIntervention.send(JSON.stringify({ action: "resume_now" }));
  }
}

export function closeInterventionWebSocket() {
  if (wsIntervention) {
    try { wsIntervention.close(); } catch {}
    wsIntervention = null;
  }
}

// -------------------------------------------------------------------------
// Output format selection
// -------------------------------------------------------------------------

export async function selectOutputFormat(projectId, outputType) {
  const res = await fetch(joinApiUrl(`/api/projects/${projectId}/output/select`), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_type: outputType }),
  });
  if (!res.ok) throw new Error(await parseApiError(res, "输出格式设置失败"));
  return res.json();
}

// -------------------------------------------------------------------------
// Storyboard
// -------------------------------------------------------------------------

export async function generateStoryboard(projectId) {
  const res = await fetch(joinApiUrl(`/api/projects/${projectId}/storyboard/generate`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) throw new Error(await parseApiError(res, "分镜生成失败"));
  return res.json();
}

export async function confirmStoryboard(projectId, confirmed, feedback) {
  const res = await fetch(joinApiUrl(`/api/projects/${projectId}/storyboard/confirm`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmed, feedback: feedback || null }),
  });
  if (!res.ok) throw new Error(await parseApiError(res, "分镜确认失败"));
  return res.json();
}

// -------------------------------------------------------------------------
// Feedback
// -------------------------------------------------------------------------

export async function submitFeedback(payload) {
  const res = await fetch(joinApiUrl("/api/feedback"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseApiError(res, "反馈提交失败"));
  return res.json();
}

// -------------------------------------------------------------------------
// Script export
// -------------------------------------------------------------------------

export function getScriptExportUrl(projectId, format) {
  return joinApiUrl(`/api/projects/${projectId}/script/export?format=${encodeURIComponent(format || "markdown")}`);
}
