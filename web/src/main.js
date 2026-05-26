import "./style.css";
import { parseCreativeSession } from "./sessionSchema";
import { createNetwork } from "./network";
import {
  createProject,
  createProjectVideoJob,
  fetchAgents,
  getProject,
  streamProjectDiscussion,
  uploadVideoSource,
  updateProject,
  watchVideoJob,
  connectInterventionWebSocket,
  closeInterventionWebSocket,
  sendIntervene,
  sendPause,
  sendResume,
  selectOutputFormat,
  generateStoryboard,
  confirmStoryboard,
  getScriptExportUrl,
} from "./api";
import { getDisplayName } from "./displayNames";
import { createIdleDialogueEngine } from "./idleDialogueEngine";
import { createEmptyRoleBuckets, inferTagsFromSession, pickRoleBuckets } from "./roleBuckets";

const IDLE_SPEECH_DURATION_MS = 3200;
const IDLE_REPLY_GAP_MS = 3000;
const DISCUSSION_PLAYBACK_MIN_DELAY_MS = 520;
const DISCUSSION_PLAYBACK_MAX_DELAY_MS = 1300;

const app = document.querySelector("#app");

app.innerHTML = `
  <div class="app-shell app-shell--stage-only">
    <div id="network-stage" class="network-stage">
      <svg class="network-svg"></svg>

      <form class="input-overlay" id="input-overlay">
        <div class="input-overlay-head">
          <div class="card-kicker">YINANPING STUDIO</div>
          <button type="button" id="overlay-toggle-btn" class="ghost-btn overlay-toggle-btn">收起</button>
        </div>
        <h1>意难平剧组</h1>
        <p class="card-summary">无限画布浏览模式已开启，可先逛房间再开机。</p>
        <div id="memory-badge" class="memory-badge is-hidden">
          <span class="memory-badge-icon">🧠</span>
          <span id="memory-badge-text">记忆已加载</span>
        </div>
        <label class="field-label" for="work-title">作品名称</label>
        <input id="work-title" name="workTitle" placeholder="例如：哈利波特与凤凰社" required />

        <label class="field-label" for="ending-direction">你想要的结局</label>
        <textarea id="ending-direction" name="endingDirection" placeholder="例如：小天狼星被救下，哈利和他拥抱收尾" required></textarea>

        <label class="field-label" for="style-preference">导演风格偏好</label>
        <select id="style-preference" name="stylePreference">
          <option value="auto">AI 自动决定</option>
          <option value="darkEpic">偏黑暗史诗</option>
          <option value="warmHealing">偏温情治愈</option>
          <option value="realism">偏文艺写实</option>
          <option value="fantasyGrand">偏奇幻宏大</option>
        </select>

        <label class="field-label" for="source-video-file">上传素材视频 <strong style="color:#c9a84c">*</strong></label>
        <input id="source-video-file" name="sourceVideoFile" type="file" accept="video/*" required />
        <label class="field-label" for="reference-images">参考图片（可选，最多5张）</label>
        <input id="reference-images" name="referenceImages" type="file" accept="image/*" multiple />
        <p id="idle-dialogue-warning" class="card-summary card-summary-soft form-warning is-hidden">
          闲聊语料加载失败，已降级为基础闲聊模式。
        </p>

        <div class="quick-actions">
          <button type="button" id="load-demo-btn" class="ghost-btn">加载示例</button>
          <button type="submit">开机 →</button>
        </div>
      </form>

      <div id="browse-controls" class="browse-controls">
        <button type="button" id="zoom-directors-btn" class="ghost-btn">导演室特写</button>
        <button type="button" id="zoom-reset-btn" class="ghost-btn">返回全景</button>
      </div>

      <div id="idle-banner" class="idle-banner">
        <div class="idle-pulse"></div>
        <span>浏览模式：你可以随时缩放逛剧组</span>
        <span class="idle-sub">点击 DIRECTORS 可窥视导演室闲聊，准备好再开机</span>
      </div>

      <div id="queue-indicator" class="queue-indicator">
        <span class="queue-dot"></span>
        <span id="queue-text">STANDBY</span>
      </div>


      <div id="live-overlay" class="live-overlay is-hidden">
        <div class="live-overlay-header">
          <span class="live-dot"></span>
          <span class="live-title" id="live-title">PRODUCTION IN PROGRESS</span>
          <button type="button" id="retry-stream-btn" class="ghost-btn is-hidden">重试流连接</button>
        </div>
        <div class="phase-timeline" id="phase-timeline">
          <span data-phase="collect">collect</span>
          <span data-phase="analyze">analyze</span>
          <span data-phase="discuss">discuss</span>
          <span data-phase="edit">edit</span>
          <span data-phase="render">render</span>
          <span data-phase="deliver">deliver</span>
        </div>
        <div class="live-feed" id="live-feed"></div>
        <div id="intervene-bar" class="intervene-bar is-hidden">
          <input type="text" id="intervene-input" class="intervene-input" placeholder="随时输入你的想法注入讨论..." />
          <button type="button" id="intervene-send-btn" class="intervene-send-btn">发送</button>
          <button type="button" id="intervene-pause-btn" class="intervene-pause-btn">暂停</button>
        </div>
      </div>

      <div id="spotlight-card" class="spotlight-card">
        <div class="spotlight-kicker">CREW SPOTLIGHT</div>
        <div class="spotlight-name" id="spotlight-name"></div>
        <div class="spotlight-subname" id="spotlight-subname"></div>
        <div class="spotlight-role" id="spotlight-role"></div>
        <div class="spotlight-summary" id="spotlight-summary"></div>
      </div>

      <div id="agent-detail-overlay" class="agent-detail-overlay is-hidden">
        <button class="agent-detail-close" id="agent-detail-close" type="button">&times;</button>
        <div class="agent-detail-name" id="detail-name"></div>
        <div class="agent-detail-subname" id="detail-subname"></div>
        <div class="agent-detail-role" id="detail-role"></div>
        <div class="agent-detail-stage" id="detail-stage">当前状态：等待调度</div>
        <div class="agent-detail-summary" id="detail-summary"></div>
      </div>

      <section id="result-overlay" class="result-overlay is-hidden">
        <article class="result-panel">
          <header class="result-head">
            <div class="result-head-left">
              <span class="result-status-dot" aria-hidden="true"></span>
              <span id="result-title" class="result-title">平行结局已完成</span>
            </div>
            <div class="result-head-right">
              <span id="result-work-name" class="result-work-name">意难平剧组</span>
              <button type="button" id="result-close-btn" class="result-close-btn" aria-label="关闭结果弹窗">&times;</button>
            </div>
          </header>
          <div id="result-media" class="result-media is-hidden"></div>
          <section class="result-notes">
            <div class="result-notes-head">CREW NOTES</div>
            <div id="result-notes-list" class="result-notes-list"></div>
          </section>
          <footer class="result-actions">
            <div class="result-actions-left">
              <button type="button" id="download-share-btn" class="ghost-btn result-ghost-btn">下载 / 分享</button>
            </div>
            <button type="button" id="restart-btn" class="result-primary-btn">重新制作</button>
          </footer>
        </article>
      </section>

      <section id="script-review-overlay" class="result-overlay is-hidden">
        <article class="result-panel">
          <header class="result-head">
            <div class="result-head-left">
              <span class="result-status-dot" aria-hidden="true"></span>
              <span class="result-title">导演讨论剧本</span>
            </div>
            <div class="result-head-right">
              <button type="button" id="script-review-close-btn" class="result-close-btn" aria-label="关闭弹窗">&times;</button>
            </div>
          </header>
          <div class="result-notes">
            <div class="result-notes-head">SCRIPT</div>
            <pre id="script-review-content" class="result-note-text" style="white-space: pre-wrap; max-height: 65vh; overflow:auto;"></pre>
          </div>
          <div class="result-notes">
            <div class="result-notes-head">PROGRESS</div>
            <div id="script-review-progress" class="result-note-text">等待生成</div>
          </div>
          <div id="script-review-video" class="result-media is-hidden"></div>
          <div class="result-notes" id="output-format-section">
            <div class="result-notes-head">OUTPUT MODE</div>
            <div class="output-format-options">
              <label class="output-format-option">
                <input type="radio" name="output-type" value="script_only" checked />
                <span>仅剧本</span>
              </label>
              <label class="output-format-option">
                <input type="radio" name="output-type" value="script_and_storyboard" />
                <span>剧本 + 分镜预览</span>
              </label>
              <label class="output-format-option">
                <input type="radio" name="output-type" value="script_and_video" />
                <span>剧本 + 视频</span>
              </label>
            </div>
            <div id="storyboard-preview" class="storyboard-preview is-hidden"></div>
            <div class="storyboard-actions is-hidden" id="storyboard-actions">
              <button type="button" id="storyboard-confirm-btn" class="result-primary-btn">确认分镜，生成视频</button>
              <input type="text" id="storyboard-feedback" class="intervene-input" placeholder="调整意见（可选）" style="margin-top:8px;" />
            </div>
          </div>
          <footer class="result-actions">
            <div class="result-actions-left">
              <a id="script-export-link" class="ghost-btn" href="#" style="display:none;">导出剧本</a>
            </div>
            <button type="button" id="script-review-generate-btn" class="result-primary-btn">确认并继续</button>
          </footer>
        </article>
      </section>
    </div>
  </div>
`;

const stage = document.querySelector(".network-svg");
const inputOverlay = document.querySelector("#input-overlay");
const idleBanner = document.querySelector("#idle-banner");
const liveOverlay = document.querySelector("#live-overlay");
const liveFeed = document.querySelector("#live-feed");
const liveTitle = document.querySelector("#live-title");
const resultOverlay = document.querySelector("#result-overlay");
const resultTitle = document.querySelector("#result-title");
const resultWorkName = document.querySelector("#result-work-name");
const resultCloseBtn = document.querySelector("#result-close-btn");
const resultMedia = document.querySelector("#result-media");
const resultNotesList = document.querySelector("#result-notes-list");
const downloadShareBtn = document.querySelector("#download-share-btn");
const restartBtn = document.querySelector("#restart-btn");
const spotlightCard = document.querySelector("#spotlight-card");
const spotlightName = document.querySelector("#spotlight-name");
const spotlightSubname = document.querySelector("#spotlight-subname");
const spotlightRole = document.querySelector("#spotlight-role");
const spotlightSummary = document.querySelector("#spotlight-summary");
const detailOverlay = document.querySelector("#agent-detail-overlay");
const detailName = document.querySelector("#detail-name");
const detailSubname = document.querySelector("#detail-subname");
const detailRole = document.querySelector("#detail-role");
const detailStage = document.querySelector("#detail-stage");
const detailSummary = document.querySelector("#detail-summary");
const detailClose = document.querySelector("#agent-detail-close");
const queueText = document.querySelector("#queue-text");
const phaseTimeline = document.querySelector("#phase-timeline");
const loadDemoBtn = document.querySelector("#load-demo-btn");
const overlayToggleBtn = document.querySelector("#overlay-toggle-btn");
const zoomDirectorsBtn = document.querySelector("#zoom-directors-btn");
const zoomResetBtn = document.querySelector("#zoom-reset-btn");
const sourceVideoFileInput = document.querySelector("#source-video-file");
const idleDialogueWarning = document.querySelector("#idle-dialogue-warning");
const retryStreamBtn = document.querySelector("#retry-stream-btn");
const scriptReviewOverlay = document.querySelector("#script-review-overlay");
const scriptReviewCloseBtn = document.querySelector("#script-review-close-btn");
const scriptReviewGenerateBtn = document.querySelector("#script-review-generate-btn");
const scriptReviewContent = document.querySelector("#script-review-content");
const scriptReviewProgress = document.querySelector("#script-review-progress");
const scriptReviewVideo = document.querySelector("#script-review-video");
const interveneBar = document.querySelector("#intervene-bar");
const interveneInput = document.querySelector("#intervene-input");
const interveneSendBtn = document.querySelector("#intervene-send-btn");
const intervenePauseBtn = document.querySelector("#intervene-pause-btn");
const storyboardPreview = document.querySelector("#storyboard-preview");
const storyboardActions = document.querySelector("#storyboard-actions");
const storyboardConfirmBtn = document.querySelector("#storyboard-confirm-btn");
const storyboardFeedback = document.querySelector("#storyboard-feedback");
const scriptExportLink = document.querySelector("#script-export-link");
const outputFormatRadios = document.querySelectorAll("input[name=\"output-type\"]");
const memoryBadge = document.querySelector("#memory-badge");
const memoryBadgeText = document.querySelector("#memory-badge-text");

// --- User identity for cross-session memory ---
function getUserId() {
  const stored = window.localStorage.getItem("whatif_user_id");
  if (stored) return stored;
  const id = "user_" + Math.random().toString(36).slice(2, 10);
  window.localStorage.setItem("whatif_user_id", id);
  return id;
}

function getUserMemorySummary() {
  try { return JSON.parse(window.localStorage.getItem("whatif_memory_summary") || "{}"); } catch { return {}; }
}
function setUserMemorySummary(summary) {
  window.localStorage.setItem("whatif_memory_summary", JSON.stringify(summary));
}

let networkHandle;
let wsSessionId = null;
let isPaused = false;
let outputTypeSelected = "script_only";
let unsubscribeJob = null;
let spotlightTimer = null;
let spotlightPinned = false;
let allAgents = [];
const activeAgentIds = new Set();
let lastPhaseSection = "";
let idleChatterTimer = null;
let idleFollowupTimer = null;
let idleDialogueEngine = null;
let lastIdleStatsSnapshot = null;
let overlayCollapsed = false;
let sessionRunning = false;
let activeChatChannel = "idle";
let streamRetryAction = null;
let currentRoleBuckets = createEmptyRoleBuckets();
let idleSpeechVisibleByZoom = false;
let latestResultPayload = null;
let latestSessionTitle = "";
let latestDiscussionTranscript = [];
let activeProjectId = null;
let latestGeneratedScript = "";
let scriptReviewOpenedAtMs = 0;
let discussionPlaybackQueue = [];
let discussionPlaybackDone = false;
const discussionChunkBySpeaker = new Map();
let renderedTurnKeys = null;

function setProjectQuery(projectId) {
  const url = new URL(window.location.href);
  if (projectId) url.searchParams.set("project", projectId);
  else url.searchParams.delete("project");
  window.history.replaceState({}, "", url.toString());
}

async function loadProjectFromQuery() {
  const url = new URL(window.location.href);
  const projectId = String(url.searchParams.get("project") || "").trim();
  if (!projectId) return;
  try {
    const project = await getProject(projectId);
    activeProjectId = String(project.id);
    const workTitleInput = document.querySelector("#work-title");
    const endingDirectionInput = document.querySelector("#ending-direction");
    const stylePreferenceInput = document.querySelector("#style-preference");
    if (workTitleInput && project.name) workTitleInput.value = project.name;
    if (endingDirectionInput && project.prompt) endingDirectionInput.value = project.prompt;
    if (stylePreferenceInput && project.style_preference) stylePreferenceInput.value = project.style_preference;
    if (project.script) latestGeneratedScript = String(project.script);
  } catch {
    activeProjectId = null;
    setProjectQuery(null);
  }
}

function openScriptReview(scriptText) {
  scriptReviewOpenedAtMs = Date.now();
  scriptReviewContent.textContent = scriptText || "暂无脚本";
  scriptReviewProgress.textContent = "讨论完成，请选择输出模式";
  scriptReviewVideo.classList.add("is-hidden");
  scriptReviewVideo.innerHTML = "";
  scriptReviewGenerateBtn.classList.remove("is-hidden");
  scriptReviewGenerateBtn.disabled = false;
  scriptReviewGenerateBtn.textContent = "确认并继续";
  scriptReviewOverlay.classList.remove("is-hidden");
  // Reset output format UI
  storyboardPreview.classList.add("is-hidden");
  storyboardPreview.innerHTML = "";
  storyboardActions.classList.add("is-hidden");
  scriptExportLink.style.display = "none";
  const radio = document.querySelector("input[name=\"output-type\"][value=\"script_only\"]");
  if (radio) radio.checked = true;
  outputTypeSelected = "script_only";
}

function closeScriptReview() {
  scriptReviewOverlay.classList.add("is-hidden");
}

function waitForGenerateConfirm() {
  return new Promise((resolve, reject) => {
    const onConfirm = () => {
      // Read selected output type
      const checked = document.querySelector("input[name=\"output-type\"]:checked");
      outputTypeSelected = checked ? checked.value : "script_only";
      cleanup();
      resolve(outputTypeSelected);  // resolve with the selected output type
    };
    const onClose = () => {
      cleanup();
      closeScriptReview();
      reject(new Error("已取消生成"));
    };
    // Also listen to radio changes
    const onRadioChange = () => {
      const checked = document.querySelector("input[name=\"output-type\"]:checked");
      outputTypeSelected = checked ? checked.value : "script_only";
      if (outputTypeSelected === "script_only") {
        scriptReviewGenerateBtn.textContent = "仅导出剧本";
      } else if (outputTypeSelected === "script_and_storyboard") {
        scriptReviewGenerateBtn.textContent = "生成分镜预览";
      } else {
        scriptReviewGenerateBtn.textContent = "确认并生成视频";
      }
    };
    for (const r of outputFormatRadios) {
      r.addEventListener("change", onRadioChange);
    }
    function cleanup() {
      scriptReviewGenerateBtn.removeEventListener("click", onConfirm);
      scriptReviewCloseBtn.removeEventListener("click", onClose);
      for (const r of outputFormatRadios) {
        r.removeEventListener("change", onRadioChange);
      }
    }
    scriptReviewGenerateBtn.addEventListener("click", onConfirm);
    scriptReviewCloseBtn.addEventListener("click", onClose);
  });
}

function clearStreamRetryAction() {
  streamRetryAction = null;
  retryStreamBtn.classList.add("is-hidden");
  retryStreamBtn.textContent = "重试流连接";
}

function setStreamRetryAction(label, action) {
  streamRetryAction = typeof action === "function" ? action : null;
  retryStreamBtn.textContent = label || "重试流连接";
  retryStreamBtn.classList.toggle("is-hidden", !streamRetryAction);
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function buildAvatarCandidates(url) {
  const raw = String(url || "").trim();
  if (!raw) return [];
  const queryIdx = raw.indexOf("?");
  const hashIdx = raw.indexOf("#");
  const tailIdx = [queryIdx, hashIdx].filter((idx) => idx >= 0).sort((a, b) => a - b)[0] ?? raw.length;
  const path = raw.slice(0, tailIdx);
  const suffix = raw.slice(tailIdx);
  const dot = path.lastIndexOf(".");
  if (dot < 0) return [raw];
  const base = path.slice(0, dot);
  const currentExt = path.slice(dot + 1).toLowerCase();
  const extPriority = ["png", "jpg", "jpeg", "webp", "svg"];
  const order = [currentExt, ...extPriority.filter((ext) => ext !== currentExt)];
  const seen = new Set();
  return order
    .map((ext) => `${base}.${ext}${suffix}`)
    .filter((candidate) => {
      if (seen.has(candidate)) return false;
      seen.add(candidate);
      return true;
    });
}

function buildAvatarNameCandidates(agent) {
  const display = getDisplayName(agent || {});
  const names = new Set();
  const pushName = (value) => {
    const text = String(value || "").trim();
    if (text) names.add(text);
  };

  pushName(display.zh);
  pushName(display.en);
  pushName(agent?.name);

  if (display.zh && display.en) {
    names.add(`${display.zh}, ${display.en}`);
    names.add(`${display.zh},${display.en}`);
    names.add(`${display.en}, ${display.zh}`);
  }

  const exts = ["png", "jpg", "jpeg", "webp", "svg"];
  const urls = [];
  for (const name of names) {
    const encoded = encodeURIComponent(name);
    for (const ext of exts) {
      urls.push(`/mock/avatars/${encoded}.${ext}`);
    }
  }
  return urls;
}

function probeImage(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(true);
    img.onerror = () => resolve(false);
    img.src = url;
  });
}

const STRICT_AVATAR_BY_AGENT_ID = {
  "agent-burton": "/mock/avatars/director-burton.png",
  "agent-columbus": "/mock/avatars/director-columbus.png",
  "agent-cuaron": "/mock/avatars/director-cuaron.png",
  "agent-curtis": "/mock/avatars/director-curtis.png",
  "agent-jackson": "/mock/avatars/director-jackson.png",
  "agent-newell": "/mock/avatars/director-newell.png",
  "agent-spielberg": "/mock/avatars/director-spielberg.png",
  "agent-yates": "/mock/avatars/director-yates.png",
  "agent-rowling": "/mock/avatars/guardian-rowling.png",
  "agent-tolkien": "/mock/avatars/guardian-tolkien.png"
};

async function resolveAvatarUrl(agent) {
  const strictAvatar = STRICT_AVATAR_BY_AGENT_ID[String(agent?.agentId || "")];
  if (strictAvatar) return strictAvatar;
  const fromConfig = buildAvatarCandidates(agent?.avatarUrl);
  const fromName = buildAvatarNameCandidates(agent);
  const candidates = [...new Set([...fromConfig, ...fromName])];
  for (const candidate of candidates) {
    // Prefer real portrait files (png/jpg/webp) when available.
    // Falls back to the original configured file.
    if (await probeImage(candidate)) return candidate;
  }
  return "";
}

async function normalizeAgentAvatars(agents) {
  return Promise.all(
    agents.map(async (agent) => ({
      ...agent,
      avatarUrl: await resolveAvatarUrl(agent)
    }))
  );
}

function renderSystemLine(text) {
  const el = document.createElement("div");
  el.className = "live-summary";
  el.innerHTML = `<span class="live-summary-label">系统</span><span>${escapeHtml(text)}</span>`;
  liveFeed.appendChild(el);
  liveFeed.scrollTop = liveFeed.scrollHeight;
}

function renderSectionLine(title) {
  const el = document.createElement("div");
  el.className = "live-section";
  el.innerHTML = `<span>${escapeHtml(title)}</span>`;
  liveFeed.appendChild(el);
  liveFeed.scrollTop = liveFeed.scrollHeight;
}

function renderSummaryLine(text) {
  const el = document.createElement("div");
  el.className = "live-summary live-summary-final";
  el.innerHTML = `<span class="live-summary-label">总结</span><span>${escapeHtml(text)}</span>`;
  liveFeed.appendChild(el);
  liveFeed.scrollTop = liveFeed.scrollHeight;
}

function discussionSectionLabel(stage) {
  if (stage === "briefing") return "briefing";
  if (stage === "topic-1") return "topic 1 · 原著底线";
  if (stage === "topic-2") return "topic 2 · 情绪曲线";
  if (stage === "topic-3") return "topic 3 · 镜头执行";
  if (stage === "finalize") return "finalize";
  return "discussion";
}

function ensureDiscussionSection(stage) {
  const section = discussionSectionLabel(stage);
  if (lastPhaseSection === section) return;
  renderSectionLine(section);
  lastPhaseSection = section;
}

function resolveSpeakerDisplayName(speaker) {
  const raw = String(speaker || "").trim();
  if (!raw) return "未知";
  for (const agent of allAgents) {
    const display = getDisplayName(agent);
    if (agent.name === raw || display.en === raw || display.zh === raw) {
      return display.zh || raw;
    }
  }
  return raw;
}

function findAgentBySpeaker(speaker) {
  const raw = String(speaker || "").trim();
  if (!raw) return null;
  for (const agent of allAgents) {
    const display = getDisplayName(agent);
    if (agent.agentId === raw || agent.name === raw || display.en === raw || display.zh === raw) {
      return agent;
    }
  }
  return null;
}

function renderMessageLine(speaker, content) {
  const agent = findAgentBySpeaker(speaker);
  const name = agent ? getDisplayName(agent).zh : resolveSpeakerDisplayName(speaker);
  const avatarHtml = agent?.avatarUrl
    ? `<img class="live-msg-avatar live-msg-avatar-image" src="${escapeHtml(agent.avatarUrl)}" alt="${escapeHtml(name)}">`
    : `<div class="live-msg-avatar live-msg-avatar-placeholder">${escapeHtml((name || "?").slice(0, 1))}</div>`;
  const bubble = document.createElement("div");
  bubble.className = "live-msg";
  bubble.innerHTML = `
    ${avatarHtml}
    <div class="live-msg-body">
      <span class="live-msg-name">${escapeHtml(name)}</span>
      <span class="live-msg-text">${escapeHtml(content)}</span>
    </div>
  `;
  liveFeed.appendChild(bubble);
  requestAnimationFrame(() => bubble.classList.add("is-visible"));
  liveFeed.scrollTop = liveFeed.scrollHeight;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function cleanContent(text) {
  // Strip <think>...</think> blocks (including nested)
  let cleaned = String(text || "").replace(/<think>[\s\S]*?<\/think>/gi, "");
  // Strip JOIN: / SKIP: prefix and any leading whitespace/punctuation
  cleaned = cleaned.replace(/^\s*(JOIN|SKIP)\s*:\s*/i, "");
  // Strip leading instruction-like boilerplate (e.g. "JOIN: 我认为" → "我认为")
  cleaned = cleaned.replace(/^\s*(?:JOIN|SKIP)\s*[：:]\s*/, "");
  // Strip FINAL_JSON and everything after (structured output, not discussion)
  cleaned = cleaned.replace(/\s*FINAL_JSON\s*[\s\S]*/gi, "");
  // Strip markdown headers
  cleaned = cleaned.replace(/^#{1,6}\s+/gm, "");
  return cleaned.trim();
}

function calcPlaybackDelay(text, fallback = 760) {
  const length = String(text || "").trim().length;
  if (!length) return fallback;
  return Math.max(
    DISCUSSION_PLAYBACK_MIN_DELAY_MS,
    Math.min(DISCUSSION_PLAYBACK_MAX_DELAY_MS, Math.floor(380 + length * 18))
  );
}

async function replayDiscussionEvent(turn) {
  if (!sessionRunning || activeChatChannel !== "live") return;
  if (turn.event === "turn_chunk" || turn.type === "turn_chunk") {
    const speaker = String(turn.speaker || "").trim();
    const chunk = String(turn.content || "");
    if (!speaker || !chunk) return;
    const speakerAgent = findAgentBySpeaker(speaker);
    const merged = `${discussionChunkBySpeaker.get(speaker) || ""}${chunk}`;
    discussionChunkBySpeaker.set(speaker, merged);
    ensureDiscussionSection(turn.stage);
    if (speakerAgent) {
      updateActiveAgentsFromNames([speakerAgent.name], speakerAgent.name);
      networkHandle.showAgentSpeech(speakerAgent.agentId, cleanContent(merged), {
        duration: 30000,
        force: true,
        append: false
      });
    }
    return;
  }
  if (turn.event === "turn" || turn.type === "turn") {
    const speakerAgent = findAgentBySpeaker(turn.speaker);
    const speaker = String(turn.speaker || "").trim();
    const chunkMerged = speaker ? discussionChunkBySpeaker.get(speaker) || "" : "";
    const finalContent = cleanContent(chunkMerged || String(turn.content || ""));
    if (speaker) discussionChunkBySpeaker.delete(speaker);
    // Dedup: skip if same speaker+content already rendered anywhere in this discussion (AG2 sometimes emits duplicate messages)
    const turnKey = `${speaker}|${finalContent}`;
    if (renderedTurnKeys && renderedTurnKeys.has(turnKey)) return;
    if (renderedTurnKeys) renderedTurnKeys.add(turnKey);
    if (!finalContent) return;
    ensureDiscussionSection(turn.stage);
    renderMessageLine(turn.speaker, finalContent);
    latestDiscussionTranscript.push(`${resolveSpeakerDisplayName(turn.speaker)}：${finalContent}`);
      if (speakerAgent) {
        updateActiveAgentsFromNames([speakerAgent.name], speakerAgent.name);
        networkHandle.showAgentSpeech(speakerAgent.agentId, finalContent, {
          duration: 30000,
          force: true,
          append: false
        });
      }
    await sleep(calcPlaybackDelay(finalContent, 900));
    return;
  }
  if (turn.event === "topic" || turn.type === "topic") {
    ensureDiscussionSection(turn.stage);
    renderSystemLine(`${turn.title}｜${turn.goal}`);
    latestDiscussionTranscript.push(`系统｜${turn.title}：${turn.goal}`);
    await sleep(calcPlaybackDelay(`${turn.title}${turn.goal}`, 780));
    return;
  }
  if (turn.event === "system" || turn.type === "system") {
    ensureDiscussionSection(turn.stage);
    renderSystemLine(turn.content);
    latestDiscussionTranscript.push(`系统：${turn.content}`);
    await sleep(calcPlaybackDelay(turn.content, 700));
    return;
  }
  if (turn.event === "summary" || turn.type === "summary") {
    ensureDiscussionSection(turn.stage);
    renderSummaryLine(turn.content);
    latestDiscussionTranscript.push(`总结：${turn.content}`);
    await sleep(calcPlaybackDelay(turn.content, 820));
    return;
  }
  if (turn.event === "done" || turn.type === "done") {
    renderSummaryLine("讨论结论已达成，剧组进入制作管线。");
    latestDiscussionTranscript.push("系统：讨论结论已达成，剧组进入制作管线。");
    renderSectionLine("pipeline");
    await sleep(640);
  }
}

async function replayDiscussionEvents() {
  while (!discussionPlaybackDone || discussionPlaybackQueue.length > 0) {
    if (!sessionRunning || activeChatChannel !== "live") return;
    if (!discussionPlaybackQueue.length) {
      await sleep(50);
      continue;
    }
    const turn = discussionPlaybackQueue.shift();
    if (!turn) continue;
    await replayDiscussionEvent(turn);
  }
}

async function replayDiscussionEvents_DEPRECATED(events) {
  for (const turn of events) {
    if (!sessionRunning || activeChatChannel !== "live") return;
    if (turn.event === "turn" || turn.type === "turn") {
      const speakerAgent = allAgents.find((agent) => agent.name === turn.speaker);
      ensureDiscussionSection(turn.stage);
      renderMessageLine(turn.speaker, turn.content);
      latestDiscussionTranscript.push(`${resolveSpeakerDisplayName(turn.speaker)}：${turn.content}`);
      if (speakerAgent) {
        updateActiveAgentsFromNames([turn.speaker], turn.speaker);
        networkHandle.showAgentSpeech(speakerAgent.agentId, turn.content, {
          duration: 30000,
          force: true,
          append: true
        });
      }
      await sleep(calcPlaybackDelay(turn.content, 900));
      continue;
    }
    if (turn.event === "topic" || turn.type === "topic") {
      ensureDiscussionSection(turn.stage);
      renderSystemLine(`${turn.title}｜${turn.goal}`);
      latestDiscussionTranscript.push(`系统｜${turn.title}：${turn.goal}`);
      await sleep(calcPlaybackDelay(`${turn.title}${turn.goal}`, 780));
      continue;
    }
    if (turn.event === "system" || turn.type === "system") {
      ensureDiscussionSection(turn.stage);
      renderSystemLine(turn.content);
      latestDiscussionTranscript.push(`系统：${turn.content}`);
      await sleep(calcPlaybackDelay(turn.content, 700));
      continue;
    }
    if (turn.event === "summary" || turn.type === "summary") {
      ensureDiscussionSection(turn.stage);
      renderSummaryLine(turn.content);
      latestDiscussionTranscript.push(`总结：${turn.content}`);
      await sleep(calcPlaybackDelay(turn.content, 820));
      continue;
    }
    if (turn.event === "done" || turn.type === "done") {
      renderSummaryLine("讨论结论已达成，剧组进入制作管线。");
      latestDiscussionTranscript.push("系统：讨论结论已达成，剧组进入制作管线。");
      renderSectionLine("pipeline");
      await sleep(640);
    }
  }
}

function openLive(question) {
  sessionRunning = true;
  activeChatChannel = "live";
  idleSpeechVisibleByZoom = false;
  latestDiscussionTranscript = [];
  liveTitle.textContent = `片场直播 | ${question}`;
  liveFeed.innerHTML = "";
  liveFeed.insertAdjacentHTML(
    "beforeend",
    `<div class="live-topic-card"><div class="live-topic-q">${escapeHtml(question)}</div></div>`
  );
  liveOverlay.classList.remove("is-fading");
  liveOverlay.classList.remove("is-hidden");
  idleBanner.classList.add("is-hidden");
  stopIdleChatter();
  networkHandle?.setIdleSpeechEnabled(false);
  queueText.textContent = "LIVE";
  clearStreamRetryAction();
  discussionChunkBySpeaker.clear();
  renderedTurnKeys = new Set();
  lastPhaseSection = "";
  ensureDiscussionSection("briefing");
  renderSystemLine("导演组就位，讨论系统启动。");
  // Show intervention bar
  interveneBar.classList.remove("is-hidden");
  interveneInput.disabled = false;
  intervenePauseBtn.textContent = "暂停";
  intervenePauseBtn.style.background = "";
  isPaused = false;
}

function closeLive() {
  sessionRunning = false;
  activeChatChannel = "idle";
  idleSpeechVisibleByZoom = false;
  liveOverlay.classList.remove("is-fading");
  liveOverlay.classList.add("is-hidden");
  queueText.textContent = "STANDBY";
  clearStreamRetryAction();
  networkHandle?.setIdleSpeechEnabled(false);
  // Hide intervention bar
  interveneBar.classList.add("is-hidden");
  closeInterventionWebSocket();
  wsSessionId = null;
  isPaused = false;
}

function hideResultOverlay() {
  resultOverlay.classList.add("is-hidden");
}

function clearResultOverlay() {
  hideResultOverlay();
  resultMedia.classList.add("is-hidden");
  resultMedia.innerHTML = "";
  resultNotesList.innerHTML = "";
  latestResultPayload = null;
}

function resetSessionUiToIdle() {
  if (unsubscribeJob) unsubscribeJob();
  clearStreamRetryAction();
  closeLive();
  toggleOverlayCollapsed(false);
  idleBanner.classList.remove("is-hidden");
  spotlightCard.classList.remove("is-hidden");
  hideAgentDetail();
  queueText.textContent = "STANDBY";
  activeAgentIds.clear();
  setRoleBuckets(createEmptyRoleBuckets());
  if (networkHandle) {
    networkHandle.setActiveAgents([]);
    networkHandle.setPhase("idle");
  }
  liveFeed.innerHTML = "";
  startIdleChatter();
}

function keepErrorState(message) {
  sessionRunning = false;
  activeChatChannel = "idle";
  queueText.textContent = "ERROR";
  renderSystemLine(`任务失败：${message}`);
  setStreamRetryAction("重试讨论流（重新开机）", async () => {
    inputOverlay.requestSubmit();
  });
}

function buildCrewNotes() {
  const participantIds = Array.from(currentRoleBuckets.participants || []).slice(0, 3);
  const picked = participantIds
    .map((id) => allAgents.find((agent) => agent.agentId === id))
    .filter(Boolean);
  if (picked.length) return picked;
  return allAgents.slice(0, 3);
}

function renderCrewNotes() {
  const notes = buildCrewNotes();
  if (!notes.length) {
    resultNotesList.innerHTML = `<div class="result-note-card"><div class="result-note-text">暂无导演组点评。</div></div>`;
    return;
  }
  resultNotesList.innerHTML = notes
    .map((agent) => {
      const display = getDisplayName(agent);
      const initial = escapeHtml((display.zh || display.en || "?").slice(0, 1));
      const noteText = escapeHtml(agent.stance || "本轮结局已生成，可继续迭代。");
      const avatarHtml = agent?.avatarUrl
        ? `<img class="result-note-avatar-image" src="${escapeHtml(agent.avatarUrl)}" alt="${escapeHtml(
            display.zh || display.en || "匿名成员"
          )}">`
        : `<span class="result-note-avatar-placeholder" aria-hidden="true">${initial}</span>`;
      return `
        <article class="result-note-card">
          <div class="result-note-avatar">${avatarHtml}</div>
          <div class="result-note-body">
            <div class="result-note-name">${escapeHtml(display.zh || display.en || "匿名成员")}</div>
            <p class="result-note-text">${noteText}</p>
          </div>
        </article>
      `;
    })
    .join("");
}

function roleLabel(type) {
  if (type === "guardian") return "原著守门人";
  if (type === "crew") return "制作组";
  return "执行导演";
}

function toggleOverlayCollapsed(force) {
  overlayCollapsed = typeof force === "boolean" ? force : !overlayCollapsed;
  inputOverlay.classList.toggle("is-collapsed", overlayCollapsed);
  overlayToggleBtn.textContent = overlayCollapsed ? "展开" : "收起";
}

function setRoleBuckets(bucketSets) {
  currentRoleBuckets = bucketSets;
  if (!networkHandle) return;
  networkHandle.setRoleBuckets({
    participants: [...bucketSets.participants],
    lateJoiners: [],
    observers: [...bucketSets.observers],
    absent: [...bucketSets.absent]
  });
}

function ensureActiveFromBuckets() {
  activeAgentIds.clear();
  for (const id of currentRoleBuckets.participants) activeAgentIds.add(id);
}

function countStageAuthorsAndDirectors(bucketSets) {
  let total = 0;
  for (const id of bucketSets.participants) {
    const agent = allAgents.find((item) => item.agentId === id);
    if (agent?.type === "guardian" || agent?.type === "director") total += 1;
  }
  return total;
}

function inferIdleTagsFromForm() {
  const workTitleInput = document.querySelector("#work-title");
  const endingDirectionInput = document.querySelector("#ending-direction");
  const stylePreferenceInput = document.querySelector("#style-preference");
  const pseudoSession = {
    workTitle: String(workTitleInput?.value || "").trim(),
    endingDirection: String(endingDirectionInput?.value || "").trim(),
    stylePreference: String(stylePreferenceInput?.value || "auto")
  };
  return [...inferTagsFromSession(pseudoSession)];
}

function reportIdleStats() {
  if (!idleDialogueEngine) return;
  const stats = idleDialogueEngine.getStats();
  if (!stats) return;
  const changed =
    !lastIdleStatsSnapshot ||
    stats.rounds !== lastIdleStatsSnapshot.rounds ||
    stats.repetitionRate !== lastIdleStatsSnapshot.repetitionRate;
  if (!changed || stats.rounds === 0 || stats.rounds % 5 !== 0) return;
  lastIdleStatsSnapshot = stats;
  console.log(
    `[idle-dialogue] rounds=${stats.rounds} repeatRate=${stats.repetitionRate} duplicateAvoided=${stats.duplicateAvoidedCount} fallback=${stats.fallbackCount}`
  );
}

function renderIdleDialogueTurn() {
  if (
    sessionRunning ||
    activeChatChannel !== "idle" ||
    !idleSpeechVisibleByZoom ||
    !allAgents.length ||
    !networkHandle ||
    !idleDialogueEngine
  ) {
    return null;
  }
  const stylePreferenceInput = document.querySelector("#style-preference");
  const round = idleDialogueEngine.nextRound({
    stylePreference: String(stylePreferenceInput?.value || "auto"),
    tags: inferIdleTagsFromForm()
  });
  if (!round) return null;
  networkHandle.showAgentSpeech(round.speakerAId, round.lineA, { duration: IDLE_SPEECH_DURATION_MS });
  if (idleFollowupTimer) window.clearTimeout(idleFollowupTimer);
  const followupDelayMs = IDLE_SPEECH_DURATION_MS + IDLE_REPLY_GAP_MS;
  idleFollowupTimer = window.setTimeout(() => {
    if (sessionRunning || !networkHandle) return;
    networkHandle.showAgentSpeech(round.speakerBId, round.lineB, { duration: IDLE_SPEECH_DURATION_MS });
  }, followupDelayMs);
  reportIdleStats();
  return round.nextDelayMs;
}

function runIdleChatterCycle() {
  if (sessionRunning || activeChatChannel !== "idle" || !idleSpeechVisibleByZoom) return;
  const nextDelayMs = renderIdleDialogueTurn();
  const delay = Number.isFinite(nextDelayMs) ? nextDelayMs : 9000;
  idleChatterTimer = window.setTimeout(runIdleChatterCycle, delay);
}

function startIdleChatter() {
  if (sessionRunning || activeChatChannel !== "idle" || !idleSpeechVisibleByZoom) return;
  if (idleChatterTimer) window.clearTimeout(idleChatterTimer);
  runIdleChatterCycle();
}

function stopIdleChatter() {
  if (idleChatterTimer) window.clearTimeout(idleChatterTimer);
  if (idleFollowupTimer) window.clearTimeout(idleFollowupTimer);
  idleChatterTimer = null;
  idleFollowupTimer = null;
  if (networkHandle) networkHandle.clearAllSpeech();
}

function updateIdleSpeechByZoom(zoneKey) {
  const nextVisible = !sessionRunning && zoneKey === "directors";
  if (nextVisible === idleSpeechVisibleByZoom) return;
  idleSpeechVisibleByZoom = nextVisible;
  networkHandle?.setIdleSpeechEnabled(nextVisible);
  if (nextVisible) startIdleChatter();
  else stopIdleChatter();
}

function updateSpotlight(agent) {
  if (!agent || spotlightPinned) return;
  const name = getDisplayName(agent);
  spotlightName.textContent = name.zh;
  spotlightSubname.textContent = name.en;
  spotlightRole.textContent = roleLabel(agent.type);
  spotlightSummary.textContent = agent.stance || "等待分配任务。";
}

function startSpotlight() {
  if (spotlightTimer) window.clearInterval(spotlightTimer);
  if (!allAgents.length) return;
  updateSpotlight(allAgents[0]);
  spotlightTimer = window.setInterval(() => {
    if (spotlightPinned) return;
    const idx = Math.floor(Math.random() * allAgents.length);
    updateSpotlight(allAgents[idx]);
  }, 9000);
}

function showAgentDetail(agent) {
  spotlightPinned = true;
  spotlightCard.classList.add("is-hidden");
  const name = getDisplayName(agent);
  detailName.textContent = name.zh;
  detailSubname.textContent = name.en;
  detailRole.textContent = roleLabel(agent.type);
  detailStage.textContent = activeAgentIds.has(agent.agentId) ? "当前状态：参与本轮任务" : "当前状态：空闲";
  detailSummary.textContent = agent.stance || "暂无描述";
  detailOverlay.classList.remove("is-hidden");
  if (networkHandle) networkHandle.setSelected(agent.agentId);
}

function hideAgentDetail() {
  spotlightPinned = false;
  detailOverlay.classList.add("is-hidden");
  spotlightCard.classList.remove("is-hidden");
  if (networkHandle) networkHandle.setSelected(null);
}

function mapPhaseByProgressText(text) {
  if (text.includes("collect") || text.includes("素材")) return "collect";
  if (text.includes("analyze") || text.includes("分析")) return "analyze";
  if (text.includes("discuss") || text.includes("讨论")) return "discuss";
  if (text.includes("edit") || text.includes("剪辑")) return "edit";
  if (text.includes("render") || text.includes("导出")) return "render";
  return "forming";
}

retryStreamBtn.addEventListener("click", async () => {
  if (!streamRetryAction) return;
  const action = streamRetryAction;
  clearStreamRetryAction();
  try {
    await action();
  } catch (error) {
    renderSystemLine(`重试失败：${error.message}`);
  }
});

// --- Intervention bar events ---
interveneSendBtn.addEventListener("click", async () => {
  const text = interveneInput.value.trim();
  if (!text || !wsSessionId) return;
  interveneInput.value = "";
  const sent = sendIntervene(wsSessionId, text);
  if (sent) {
    renderMessageLine("user", text);
    latestDiscussionTranscript.push(`用户：${text}`);
  } else {
    renderSystemLine("（已通过 HTTP 发送干预意见，将在下一轮讨论中生效）");
  }
});

interveneInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    interveneSendBtn.click();
  }
});

intervenePauseBtn.addEventListener("click", () => {
  if (!wsSessionId) return;
  if (isPaused) {
    sendResume(wsSessionId);
    intervenePauseBtn.textContent = "暂停";
    intervenePauseBtn.style.background = "";
    queueText.textContent = "LIVE";
    renderSystemLine("讨论已恢复。");
    isPaused = false;
  } else {
    sendPause(wsSessionId);
    intervenePauseBtn.textContent = "继续";
    intervenePauseBtn.style.background = "#c9a84c";
    queueText.textContent = "PAUSED";
    renderSystemLine("讨论已暂停，点击「继续」或发送消息恢复。");
    isPaused = true;
  }
});

function phaseLabel(phase) {
  if (phase === "collect") return "collect";
  if (phase === "analyze") return "analyze";
  if (phase === "discuss") return "discuss";
  if (phase === "edit") return "edit";
  if (phase === "render") return "render";
  if (phase === "deliver") return "deliver";
  return "runtime";
}

function markTimelinePhase(phase) {
  const labels = phaseTimeline.querySelectorAll("span");
  labels.forEach((label) => {
    label.classList.toggle("is-active", label.dataset.phase === phase);
  });
}

function updateActiveAgentsFromNames(names, leadName = null) {
  if (!allAgents.length || !networkHandle) return;
  for (const name of names || []) {
    const hit = allAgents.find((agent) => agent.name === name);
    if (!hit) continue;
    if (!currentRoleBuckets.participants.has(hit.agentId)) {
      continue;
    }
    activeAgentIds.add(hit.agentId);
  }
  const ids = Array.from(activeAgentIds);
  const leadCandidateId = allAgents.find((agent) => agent.name === leadName)?.agentId || null;
  const leadId = leadCandidateId && currentRoleBuckets.participants.has(leadCandidateId) ? leadCandidateId : null;
  networkHandle.setActiveAgents(ids, leadId);
}

function isParticipantAgentId(agentId) {
  if (!agentId) return false;
  return currentRoleBuckets.participants.has(agentId);
}

function showResult(result) {
  latestResultPayload = result || null;
  resultTitle.textContent = "平行结局已完成";
  resultWorkName.textContent = latestSessionTitle || "意难平剧组";
  renderCrewNotes();

  const publicUrl = String(result?.publicUrl || "");
  const hasPlayableMp4 =
    publicUrl &&
    (result?.type === "video-mp4" ||
      result?.type === "video-mp4-fallback" ||
      publicUrl.toLowerCase().endsWith(".mp4"));
  if (hasPlayableMp4) {
    resultMedia.classList.remove("is-hidden");
    resultMedia.innerHTML = `
      <video controls preload="metadata" src="${escapeHtml(publicUrl)}"></video>
      <a href="${escapeHtml(publicUrl)}" target="_blank" rel="noreferrer">在新窗口打开视频</a>
    `;
  } else {
    resultMedia.classList.remove("is-hidden");
    resultMedia.innerHTML = `
      <div class="result-video-placeholder" role="status" aria-live="polite">
        <span>VIDEO PLAYER</span>
        <small>当前暂无可播放视频，请重新制作或稍后重试。</small>
      </div>
    `;
  }
  resultOverlay.classList.remove("is-hidden");
}

async function loadIdleDialogueLibrary() {
  try {
    const res = await fetch("/mock/idle-directors-dialogues.json");
    if (!res.ok) throw new Error(`status ${res.status}`);
    const data = await res.json();
    return { dialogues: Array.isArray(data) ? data : [], degraded: false };
  } catch (error) {
    console.warn("idle dialogue library load failed, fallback to empty pool", error);
    return { dialogues: [], degraded: true };
  }
}

async function bootstrap() {
  const [fetchedAgents, idleDialogueResult] = await Promise.all([
    fetchAgents().catch(async () => {
      const fallback = await fetch("/mock/agents.json").then((res) => res.json());
      return Array.isArray(fallback) ? fallback : [];
    }),
    loadIdleDialogueLibrary()
  ]);
  const agents = await normalizeAgentAvatars(fetchedAgents);
  allAgents = agents;
  if (idleDialogueResult.degraded) {
    idleDialogueWarning?.classList.remove("is-hidden");
  }
  idleDialogueEngine = createIdleDialogueEngine({
    agents,
    dialogues: idleDialogueResult.dialogues,
    pairCooldown: 8,
    dialogueCooldown: 16,
    signatureCooldown: 20
  });
  networkHandle = createNetwork(stage, agents, {
    onSelectAgent: (agent) => showAgentDetail(agent),
    onClearSelection: () => hideAgentDetail(),
    onZoneFocusChange: (zoneKey, zoomScale) => {
      if (!zoneKey) {
        detailStage.textContent = activeAgentIds.size ? "当前状态：参与本轮任务" : "当前状态：空闲";
        updateIdleSpeechByZoom(null, zoomScale);
        return;
      }
      detailStage.textContent = `当前视角：${zoneKey.toUpperCase()} (${zoomScale.toFixed(2)}x)`;
      updateIdleSpeechByZoom(zoneKey, zoomScale);
    }
  });
  networkHandle.setIdleSpeechEnabled(false);
  startSpotlight();
  setRoleBuckets(createEmptyRoleBuckets());
  // Show memory badge if we have stored preferences
  const memSummary = getUserMemorySummary();
  if (memSummary.preferences_count > 0) {
    memoryBadgeText.textContent = `${memSummary.preferences_count} 条偏好已记住`;
    memoryBadge.classList.remove("is-hidden");
  }
  await loadProjectFromQuery();
}

inputOverlay.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(inputOverlay);

  const candidate = {
    workTitle: String(formData.get("workTitle") || "").trim(),
    endingDirection: String(formData.get("endingDirection") || "").trim(),
    stylePreference: String(formData.get("stylePreference") || "auto")
  };
  const rawSourceVideoFile = formData.get("sourceVideoFile");
  const sourceVideoFile =
    rawSourceVideoFile instanceof File && rawSourceVideoFile.size > 0 ? rawSourceVideoFile : null;
  if (!sourceVideoFile) {
    alert("请上传一段素材视频，HappyHorse 将对其进行 AI 编辑");
    return;
  }

  const rawRefImages = formData.getAll("referenceImages");
  const refImageFiles = rawRefImages
    .filter((f) => f instanceof File && f.size > 0)
    .slice(0, 5);

  let payload;
  try {
    payload = parseCreativeSession(candidate);
  } catch (error) {
    alert(`输入校验失败：${error.message}`);
    return;
  }

  toggleOverlayCollapsed(true);
  latestSessionTitle = payload.workTitle || "未命名作品";
  spotlightCard.classList.add("is-hidden");
  setRoleBuckets(pickRoleBuckets(allAgents, payload));
  ensureActiveFromBuckets();
  networkHandle.gather();
  networkHandle.setPhase("forming");
  networkHandle.setActiveAgents([...activeAgentIds]);
  openLive(payload.workTitle);
  markTimelinePhase("collect");
  renderSystemLine("会话创建中…");
  renderSystemLine(`主舞台核心成员 ${countStageAuthorsAndDirectors(currentRoleBuckets)} 位。`);

  try {
    if (!activeProjectId) {
      const project = await createProject({
        name: payload.workTitle,
        description: "",
        prompt: payload.endingDirection,
        style_preference: payload.stylePreference
      });
      activeProjectId = String(project.id);
      setProjectQuery(activeProjectId);
    } else {
      await updateProject(activeProjectId, {
        name: payload.workTitle,
        prompt: payload.endingDirection,
        style_preference: payload.stylePreference
      });
    }

    renderSystemLine(`工程已就绪：${activeProjectId}`);
    renderSystemLine("导演与原著守门人讨论中…");
    networkHandle.pulseZone("mainStage");
    networkHandle.setPhase("discuss");
    markTimelinePhase("discuss");

    try {
      let playbackError = null;
      discussionPlaybackQueue = [];
      discussionPlaybackDone = false;
      const playbackTask = replayDiscussionEvents();
      // Open WebSocket for intervention
      wsSessionId = activeProjectId;
      connectInterventionWebSocket(activeProjectId, {
        onOpen: () => console.log("[ws] intervention channel open"),
        onMessage: (msg) => {
          if (msg.type === "paused") renderSystemLine("（讨论已暂停）");
          if (msg.type === "resumed") renderSystemLine("（讨论已恢复）");
          if (msg.type === "question") {
            renderSystemLine(`🟡 ${msg.question || "导演组需要你的意见"}`);
            interveneInput.focus();
          }
          if (msg.type === "ack" && msg.action === "intervene") {
            renderSystemLine("（你的意见已注入讨论）");
          }
        },
        onClose: () => console.log("[ws] intervention channel closed"),
      });
      const currentUserId = getUserId();
      await streamProjectDiscussion(activeProjectId, (turn) => {
        if (turn.type === "error" || turn.event === "error") {
          playbackError = new Error(String(turn.message || turn.error || "讨论失败"));
          return;
        }
        if (turn.type === "script") {
          latestGeneratedScript = String(turn.script || "");
          return;
        }
        // Handle new SSE event types from backend
        if (turn.type === "awaiting_input") {
          // Question rendered via WS "question" message instead (no duplicate)
          return;
        }
        if (turn.type === "user_intervention") {
          renderMessageLine("user", turn.content || "");
          return;
        }
        if (turn.type === "memory_loaded") {
          if (turn.preferences_count > 0 || turn.similar_scripts > 0) {
            const msg = `🧠 记忆系统已加载：${turn.preferences_count} 条偏好，${turn.similar_scripts} 个相似历史剧本`;
            renderSystemLine(msg);
            setUserMemorySummary({
              preferences_count: turn.preferences_count,
              similar_scripts: turn.similar_scripts,
              lastLoaded: new Date().toISOString(),
              userId: currentUserId,
            });
          }
          return;
        }
        discussionPlaybackQueue.push(turn);
      }, { userId: currentUserId });
      discussionPlaybackDone = true;
      await playbackTask;
      if (playbackError) throw playbackError;
    } catch (discussionError) {
      discussionPlaybackDone = true;
      setStreamRetryAction("重试讨论流（重新开机）", async () => {
        inputOverlay.requestSubmit();
      });
      throw new Error(`讨论流中断：${discussionError.message}`);
    }

    if (!latestGeneratedScript && latestDiscussionTranscript.length) {
      latestGeneratedScript = latestDiscussionTranscript.join("\n");
    }
    if (!latestGeneratedScript.trim()) {
      throw new Error("讨论未产出可用剧本，请检查 OPENAI 配置或模型响应。");
    }
    openScriptReview(latestGeneratedScript);
    const chosenOutput = await waitForGenerateConfirm();
    scriptReviewGenerateBtn.disabled = true;

    // Save output format preference
    try { await selectOutputFormat(activeProjectId, chosenOutput); } catch {}

    // --- Handle "script_only" ---
    if (chosenOutput === "script_only") {
      scriptReviewProgress.textContent = "剧本已生成，点击下方链接下载";
      scriptExportLink.style.display = "inline-block";
      scriptExportLink.href = getScriptExportUrl(activeProjectId, "markdown");
      scriptExportLink.textContent = "下载 Markdown 剧本";
      scriptReviewGenerateBtn.classList.add("is-hidden");
      closeInterventionWebSocket();
      return;
    }

    // --- Handle "script_and_storyboard" ---
    if (chosenOutput === "script_and_storyboard") {
      scriptReviewProgress.textContent = "分镜预览生成中…";
      try {
        const sb = await generateStoryboard(activeProjectId);
        const frames = sb.frames || [];
        storyboardPreview.classList.remove("is-hidden");
        storyboardPreview.innerHTML = frames.map((f, i) => `
          <div class="storyboard-frame">
            <strong>第${i + 1}幕 (${f.timing || "?"})</strong>
            <p>${escapeHtml(f.description || "")}</p>
            ${f.visual_prompt ? `<small>视觉提示: ${escapeHtml(f.visual_prompt)}</small>` : ""}
          </div>
        `).join("");
        storyboardActions.classList.remove("is-hidden");
        scriptReviewGenerateBtn.classList.add("is-hidden");
        scriptReviewProgress.textContent = `分镜预览共 ${frames.length} 帧 — 确认后生成视频？`;
        // Wait for storyboard confirm → then ask "generate video?"
        await new Promise((resolveSb, rejectSb) => {
          storyboardConfirmBtn.addEventListener("click", async () => {
            storyboardConfirmBtn.disabled = true;
            const fb = storyboardFeedback.value.trim();
            try {
              await confirmStoryboard(activeProjectId, true, fb || undefined);
              resolveSb(true);
            } catch (e) { rejectSb(e); }
          }, { once: true });
        });
        // Ask: generate video now?
        storyboardActions.classList.add("is-hidden");
        scriptReviewGenerateBtn.classList.remove("is-hidden");
        scriptReviewGenerateBtn.textContent = "生成视频";
        scriptReviewGenerateBtn.disabled = false;
        scriptReviewProgress.textContent = "分镜已确认。是否生成视频？";
        const proceed = await waitForGenerateConfirm();
        if (proceed === "script_only") {
          // User closed instead of generating video
          scriptReviewGenerateBtn.classList.add("is-hidden");
          closeInterventionWebSocket();
          return;
        }
        // Fall through to video generation below
      } catch (e) {
        scriptReviewProgress.textContent = `分镜生成失败：${e.message}`;
        scriptReviewGenerateBtn.classList.remove("is-hidden");
        scriptReviewGenerateBtn.disabled = false;
        return;
      }
    }

    // --- Video generation (for "script_and_video" or "script_and_storyboard" after confirm) ---
    scriptReviewProgress.textContent = "视频任务创建中…";

    // Upload source video (required for HappyHorse)
    renderSystemLine(`上传素材视频中：${sourceVideoFile.name}`);
    const { upload: videoUpload } = await uploadVideoSource(activeProjectId, sourceVideoFile);
    renderSystemLine(`视频上传完成：${videoUpload.originalName}`);
    const videoAssetId = videoUpload.uploadId;

    // Upload reference images (optional)
    const refImgIds = [];
    for (const imgFile of refImageFiles) {
      renderSystemLine(`上传参考图片：${imgFile.name}`);
      const { upload } = await uploadVideoSource(activeProjectId, imgFile, "image");
      renderSystemLine(`图片上传完成：${upload.originalName}`);
      if (upload.uploadId) refImgIds.push(upload.uploadId);
    }

    const { job } = await createProjectVideoJob(activeProjectId);
    renderSystemLine(`视频任务已创建：${job.jobId}`);
    const origin = window.location.origin;
    const sourceVideoUrl = videoAssetId ? `${origin}/api/assets/${videoAssetId}/download` : "";
    const refImgUrls = refImgIds.map((id) => `${origin}/api/assets/${id}/download`);
    if (sourceVideoUrl) renderSystemLine(`素材视频已就绪，即将进行 HappyHorse 编辑`);
    networkHandle.setPhase("collect");
    markTimelinePhase("collect");

    const attachVideoStream = () => {
      if (unsubscribeJob) unsubscribeJob();
      unsubscribeJob = watchVideoJob(
        job.jobId,
        (evt) => {
          if (evt.event === "progress") {
            const phase = mapPhaseByProgressText(`${evt.phase}: ${evt.message}`);
            if (lastPhaseSection !== phase) {
              renderSectionLine(phaseLabel(phase));
              lastPhaseSection = phase;
            }
            renderSystemLine(`${evt.phase}: ${evt.message}`);
            scriptReviewProgress.textContent = `${evt.phase}: ${evt.message}`;
            networkHandle.setPhase(phase);
            markTimelinePhase(phase === "forming" ? "collect" : phase);
            return;
          }
          if (evt.event === "complete") {
            clearStreamRetryAction();
            renderSummaryLine("制作完成，正在回放平行结局。");
            markTimelinePhase("deliver");
            networkHandle.scatterAfterWrapup();
            networkHandle.setActiveAgents([]);
            window.setTimeout(() => networkHandle.release(), 7600);
            liveOverlay.classList.add("is-fading");
            scriptReviewProgress.textContent = "生成完成";
            const publicUrl = String(evt?.result?.publicUrl || "");
            if (publicUrl) {
              scriptReviewVideo.classList.remove("is-hidden");
              scriptReviewVideo.innerHTML = `
                <video controls preload="metadata" src="${escapeHtml(publicUrl)}"></video>
                <a href="${escapeHtml(publicUrl)}" target="_blank" rel="noreferrer">在新窗口打开视频</a>
              `;
            }
            scriptReviewGenerateBtn.classList.add("is-hidden");
            setTimeout(() => {
              closeLive();
              spotlightCard.classList.remove("is-hidden");
              showResult(evt.result);
            }, 1400);
            return;
          }
          if (evt.event === "error") {
            renderSystemLine(evt.message || "任务事件流中断。");
            scriptReviewProgress.textContent = evt.message || "任务事件流中断";
            scriptReviewGenerateBtn.disabled = false;
            setStreamRetryAction("重连任务流", async () => {
              renderSystemLine("正在重连任务流…");
              attachVideoStream();
            });
          }
        },
        {
          videoUrl: sourceVideoUrl,
          refImageUrls: refImgUrls,
          onDisconnect: () => {
            setStreamRetryAction("重连任务流", async () => {
              renderSystemLine("正在重连任务流…");
              attachVideoStream();
            });
          }
        }
      );
    };
    attachVideoStream();
  } catch (error) {
    if (String(error?.message || "") === "已取消生成") {
      renderSystemLine("已取消本次生成。");
      resetSessionUiToIdle();
      return;
    }
    const errMsg = String(error?.message || "未知错误");
    scriptReviewProgress.textContent = `失败：${errMsg}`;
    scriptReviewGenerateBtn.disabled = false;
    networkHandle.release();
    keepErrorState(errMsg);
  }
});

restartBtn.addEventListener("click", () => {
  clearResultOverlay();
  resetSessionUiToIdle();
});

resultCloseBtn.addEventListener("click", () => {
  hideResultOverlay();
});

downloadShareBtn.addEventListener("click", async () => {
  const publicUrl = String(latestResultPayload?.publicUrl || "");
  if (!publicUrl) {
    alert("当前暂无可下载或分享的视频链接。");
    return;
  }
  const safeUrl = publicUrl.trim();
  const anchor = document.createElement("a");
  anchor.href = safeUrl;
  anchor.download = `${latestSessionTitle || "parallel-ending"}.mp4`;
  anchor.target = "_blank";
  anchor.rel = "noreferrer";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);

  let copied = false;
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(safeUrl);
      copied = true;
    } catch {
      copied = false;
    }
  }
  if (copied) renderSystemLine("视频链接已复制到剪贴板，可直接分享。");
});

resultOverlay.addEventListener("click", (event) => {
  if (event.target === resultOverlay) hideResultOverlay();
});
scriptReviewOverlay.addEventListener("click", (event) => {
  if (Date.now() - scriptReviewOpenedAtMs < 300) return;
  if (event.target === scriptReviewOverlay) closeScriptReview();
});

window.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!scriptReviewOverlay.classList.contains("is-hidden")) {
    closeScriptReview();
    return;
  }
  if (resultOverlay.classList.contains("is-hidden")) return;
  hideResultOverlay();
});

detailClose.addEventListener("click", hideAgentDetail);
overlayToggleBtn.addEventListener("click", () => toggleOverlayCollapsed());
zoomDirectorsBtn.addEventListener("click", () => networkHandle?.zoomToZone("directors"));
zoomResetBtn.addEventListener("click", () => networkHandle?.resetCamera());
loadDemoBtn.addEventListener("click", () => {
  const workTitleInput = document.querySelector("#work-title");
  const endingDirectionInput = document.querySelector("#ending-direction");
  const stylePreferenceInput = document.querySelector("#style-preference");
  workTitleInput.value = "哈利波特与凤凰社";
  endingDirectionInput.value = "小天狼星在神秘事务司被救下，最终和哈利拥抱和解。";
  stylePreferenceInput.value = "warmHealing";
  sourceVideoFileInput.value = "";
});

bootstrap();
