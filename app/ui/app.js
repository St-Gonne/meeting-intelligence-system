"use strict";
const $ = (id) => document.getElementById(id);
const tokenFromURL = new URLSearchParams(location.hash.slice(1)).get("token");
if (tokenFromURL) { sessionStorage.setItem("mi-session", tokenFromURL); history.replaceState(null, "", "/"); }
const sessionToken = sessionStorage.getItem("mi-session") || "";
let items = [], selectedId = null, showOlder = false, activeView = "inbox", confirmation = null, lastOperation = null, refreshing = false;

const errors = {
  invalid_session: "This browser session has expired. Reopen the local inbox with mi ui.",
  operation_busy: "Another operation is still active. Wait for its outcome, then refresh.",
  confirmation_expired_or_used: "That confirmation expired or was already used. Review the recording and try again.",
  action_no_longer_available: "The recording changed or was already processed. Refresh to see its current state.",
  source_unavailable: "This recording is no longer available. Refresh the inbox.",
  artifact_unavailable: "The saved artifact is unavailable. Refresh to check its current state.",
  local_evidence_unavailable: "Local evidence could not be read. No completion can be confirmed right now.",
  operation_unavailable: "The action could not start. Refresh and inspect the recording before retrying.",
  worker_start_failed: "The processing worker could not start. No new processing was confirmed."
};
function node(tag, className, text) { const element = document.createElement(tag); if (className) element.className = className; if (text !== undefined) element.textContent = text; return element; }
function button(label, className, action) { const element = node("button", "button " + className, label); element.addEventListener("click", action); return element; }
function human(value) { if (value === null || value === undefined || value === "") return "Not yet known"; if (typeof value === "object") return human(value.label || value.status || value.state); return String(value).replace(/_/g, " "); }
function dateLabel(value) { const date = new Date(value); return Number.isNaN(date.getTime()) ? "Time unavailable" : date.toLocaleString("en-IN", {day:"numeric", month:"short", hour:"2-digit", minute:"2-digit"}); }
function today(value) { const date = new Date(value), now = new Date(); return date.toDateString() === now.toDateString(); }
function issue(item) { return Boolean(item.warning) || /interrupt|incomplete|source.loss|missing.conversation/i.test(JSON.stringify(item.recording_state || "")); }
function showError(error) { $("connection").textContent = errors[error.message] || "The inbox cannot confirm current state. Reconnect or refresh before taking another action."; $("connection").hidden = false; $("connection").className = "notice error"; }
async function api(path, body) { const options = {headers:{"X-MI-Token":sessionToken}, cache:"no-store"}; if (body !== undefined) { options.method="POST"; options.headers["Content-Type"]="application/json"; options.body=JSON.stringify(body); } const response = await fetch(path, options); const result = await response.json(); if (!response.ok) throw new Error(result.error || "request_failed"); return result; }

function renderList() {
  const list = $("meeting-list"); list.replaceChildren(); $("meeting-count").textContent = items.length;
  if (!items.length) { list.append(node("p", "empty", "No local recordings found. Record in Terminal or use mi sync to download phone recordings.")); return; }
  const urgent = items.filter(issue), current = items.filter(x => !issue(x) && today(x.created_at)), older = items.filter(x => !issue(x) && !today(x.created_at));
  const group = (title, rows) => { if (!rows.length) return; list.append(node("h3", "group-heading", title)); for (const item of rows) {
    const row = node("button", "meeting-item" + (item.id === selectedId ? " selected" : "")); row.setAttribute("aria-pressed", String(item.id === selectedId));
    const top = node("span", "meeting-topline"); top.append(node("span", "", human(item.source_kind)), node("span", "", dateLabel(item.created_at)));
    row.append(top, node("span", "meeting-title", item.title || "Recording"), node("span", "badge" + (issue(item) ? " problem" : item.category === "needs_decision" ? " warning" : ""), human(item.status)));
    row.addEventListener("click", () => select(item.id)); list.append(row);
  }};
  group("Needs attention", urgent); group("Today", current);
  if (older.length) { const toggle = button((showOlder ? "Hide" : "Show") + " older recordings (" + older.length + ")", "ghost older-toggle", () => { showOlder = !showOlder; renderList(); }); toggle.setAttribute("aria-expanded", String(showOlder)); list.append(toggle); if (showOlder) group("Earlier", older); }
}

function renderDetail(item) {
  const panel = $("meeting-detail"); panel.replaceChildren();
  panel.append(node("p", "detail-meta", human(item.source_kind) + " · " + dateLabel(item.created_at)), node("h2", "detail-title", item.title || "Recording"));
  if (Number.isFinite(item.duration_seconds)) { const seconds = Math.round(item.duration_seconds); panel.append(node("p", "subtle", (item.duration_is_final ? "Saved audio: " : "Last observed audio: ") + Math.floor(seconds / 60) + " min " + (seconds % 60) + " sec")); }
  if (item.warning) panel.append(node("div", "notice error", human(item.warning)));
  const grid = node("div", "state-grid");
  for (const [label, key] of [["Recording", "recording_state"], ["Transcript", "transcript_state"], ["Report", "report_state"], ["Daily brief", "brief_state"]]) {
    const cell = node("div", "state-cell"); cell.append(node("span", "state-label", label), node("span", "state-value", human(item[key]))); grid.append(cell);
  } panel.append(grid);
  const next = node("section", "next-step"); next.append(node("h3", "", "Next step"), node("p", "", item.next_action || "There is no processing action to take for this recording."));
  const actions = node("div", "action-row");
  if (item.can_process) actions.append(button(/fail|stop|retry/i.test(item.status || "") ? "Review processing retry" : "Process this recording", "primary", () => prepare("process", item.id)));
  if (item.can_repair_brief) actions.append(button("Repair daily brief", "primary", () => prepare("repair_brief", item.id)));
  if (item.handoff) actions.append(button("Continue in Terminal", "secondary", () => handoff(item.handoff, item.next_action || "Continue the guided workflow in Terminal.", item.id)));
  if (actions.childElementCount) next.append(actions); panel.append(next);
  const artifacts = (item.artifacts || []).filter(x => ["transcript", "report", "brief"].includes(x.kind));
  panel.append(node("h3", "artifacts-title", "Saved results"));
  if (!artifacts.length) panel.append(node("p", "subtle", "No saved results are available for this recording yet."));
  else { const row = node("div", "action-row"); for (const artifact of artifacts) row.append(button("Open " + artifact.kind, "secondary", () => openArtifact(item.id, artifact.kind))); panel.append(row); }
  panel.append(node("p", "panel-footer", "A transcript covers the saved audio. It does not establish that the whole meeting was captured."));
}
async function select(id) { selectedId = id; renderList(); try { const detail = await api("/api/detail?id=" + encodeURIComponent(id)); if (selectedId === id) renderDetail(detail); } catch(error) { showError(error); } }

function renderOperation(operation) {
  lastOperation = operation; const target = $("operation"); target.hidden = !operation;
  if (!operation) return;
  const labels = {deferred:"Local GPU work was deferred. Your recording is saved; retry when the other local AI work has finished or the service check is resolved.", running:"Processing is active. You can refresh or leave this browser tab; the local worker continues.", starting:"Starting the selected operation…", finished:"The operation finished. The states below reflect the currently saved artifacts.", failed:"The operation stopped: " + human(operation.outcome) + ". Inspect the saved stages before retrying.", interrupted:"Processing stopped without a confirmed outcome. Saved artifacts have been preserved; inspect the recording before retrying.", unknown:"Worker state is uncertain. The inbox cannot claim completion or safely start another operation. Check Terminal diagnostics."};
  target.textContent = labels[operation.status] || "Operation state is unavailable.";
}
async function refresh() {
  if (refreshing) return; refreshing = true;
  try { const result = await api("/api/snapshot"); items = result.items || []; $("connection").hidden = true; $("warnings").replaceChildren(); for (const warning of result.warnings || []) $("warnings").append(node("div", "notice", human(warning))); renderList(); renderOperation(result.operation);
    if (selectedId && items.some(x => x.id === selectedId)) { const requestedId = selectedId; const detail = await api("/api/detail?id=" + encodeURIComponent(requestedId)); if (selectedId === requestedId) renderDetail(detail); }
    else if (items.length) { selectedId = (items.find(issue) || items.find(x => today(x.created_at)) || items[0]).id; if (!today(items.find(x => x.id === selectedId).created_at) && !issue(items.find(x => x.id === selectedId))) showOlder = true; await select(selectedId); }
    else { selectedId = null; $("meeting-detail").replaceChildren(node("p", "empty", "Select a recording when one becomes available.")); }
    if (activeView === "learning") await loadLearning();
  } catch(error) { showError(error); $("operation").hidden = false; $("operation").textContent = "Connection lost: displayed recording and processing states may be stale. No completion is confirmed."; } finally { refreshing = false; }
}
async function prepare(action, sourceId) {
  try { confirmation = await api("/api/prepare", {action, source_id:sourceId}); const repair = action === "repair_brief"; $("confirm-title").textContent = repair ? "Repair the daily brief?" : "Process this recording?"; $("confirm-source").textContent = (confirmation.source.title || "Recording") + " · " + dateLabel(confirmation.source.created_at); $("confirm-explanation").textContent = repair ? "Rebuild the brief from already saved records. This does not rerun transcription or analysis." : "Only this selected recording will be processed. A retry may repeat transcription. Original recordings are retained."; $("confirm-submit").textContent = repair ? "Repair daily brief" : "Process this recording"; $("confirm-dialog").showModal(); } catch(error) { showError(error); }
}
async function cancelConfirmation() { const previous = confirmation; confirmation = null; $("confirm-dialog").close(); if (previous) { try { await api("/api/cancel", {confirmation_id:previous.confirmation_id}); } catch(error) { showError(error); } } }
$("confirm-cancel").addEventListener("click", cancelConfirmation);
$("confirm-dialog").addEventListener("cancel", event => { event.preventDefault(); cancelConfirmation(); });
$("confirm-submit").addEventListener("click", async () => { if (!confirmation) return; const previous = confirmation; confirmation = null; $("confirm-submit").disabled = true; try { renderOperation(await api("/api/action", {confirmation_id:previous.confirmation_id})); $("confirm-dialog").close(); await refresh(); } catch(error) { $("confirm-dialog").close(); showError(error); } finally { $("confirm-submit").disabled = false; } });

function handoff(command, explanation, sourceId = null) { $("handoff-command").textContent = typeof command === "string" ? command : (command.command || "mi inbox"); $("handoff-explanation").textContent = explanation; $("handoff-copy").textContent = "Copy command"; $("handoff-dialog").showModal(); if (sourceId) api("/api/handoff", {source_id:sourceId}).catch(showError); }
$("handoff-close").addEventListener("click", () => $("handoff-dialog").close());
$("handoff-copy").addEventListener("click", async () => { try { await navigator.clipboard.writeText($("handoff-command").textContent); $("handoff-copy").textContent = "Copied"; } catch(error) { $("handoff-copy").textContent = "Select and copy the command"; } });
$("record-button").addEventListener("click", () => { api("/api/terminal", {action:"record"}).catch(showError); handoff("mi record", "Record using the existing guarded Terminal recorder. Keep the lid open. Press Control-C once to stop, then wait for the saved recording message."); });
async function openArtifact(sourceId, kind) { try { const artifact = await api("/api/artifact?id=" + encodeURIComponent(sourceId) + "&kind=" + encodeURIComponent(kind)); $("artifact-title").textContent = kind.charAt(0).toUpperCase() + kind.slice(1); $("artifact-text").textContent = artifact.text; $("artifact-dialog").showModal(); } catch(error) { showError(error); } }
$("artifact-close").addEventListener("click", () => { $("artifact-dialog").close(); $("artifact-text").textContent = ""; });

async function loadLearning() {
  const data = await api("/api/learning"), panel = $("learning-view"); panel.replaceChildren();
  const grid = node("div", "learning-grid");
  for (const [label, value, explanation] of [["Measurement health", human(data.status), "Unavailable collection is not counted as zero usage."], ["Observed events", data.event_count == null ? "—" : String(data.event_count), "CLI and GUI events share the same local history."], ["Review window due", data.next_review_due ? dateLabel(data.next_review_due) : "Collection not started", "This date does not confirm a schedule is installed. A sparse window may have insufficient evidence."]]) { const card = node("section", "metric-card"); card.append(node("h3", "state-label", label), node("span", "metric", value), node("p", "", explanation)); grid.append(card); } panel.append(grid);
  const overview = node("section", "learning-card"); overview.append(node("h2", "", "A measured improvement loop"), node("p", "", "Local events describe actions, failures and saved outcomes. Reviews produce structured findings and candidate changes. They do not modify software or rearrange this interface automatically."), button("Run a review in Terminal", "secondary", () => { api("/api/terminal", {action:"learning_review"}).catch(showError); handoff("mi learning review", "Run a local review of recorded usage. This does not read meeting content or change processing behavior."); })); panel.append(overview);
  const changes = node("section", "learning-card"); changes.append(node("h2", "", "Findings & proposed improvements"));
  const findings = data.changes || []; if (!findings.length) changes.append(node("p", "", "No proposed changes are available yet. Collect usage first; a quiet period is a valid outcome."));
  for (const item of findings) { const evidence = item.evaluation || {}, row = node("article", "change-item"); row.append(node("h3", "", human(item.title || item.finding_type || "Candidate improvement")), node("p", "", human(item.proposed_change || evidence.proposed_action || item.hypothesis || item.description || "Review the structured evidence before making a change."))); if (Number.isFinite(evidence.observed_count) && Number.isFinite(evidence.eligible_count)) row.append(node("p", "", "Evidence: " + evidence.observed_count + " matching observations; " + evidence.eligible_count + " eligible observations.")); row.append(node("span", "badge", human(item.state || item.status || "proposed"))); if (evidence.outcome) row.append(node("p", "", "Evaluation: " + human(evidence.outcome))); changes.append(row); } panel.append(changes);
}
function switchView(view) { activeView = view; $("inbox-view").hidden = view !== "inbox"; $("learning-view").hidden = view !== "learning"; $("inbox-tab").classList.toggle("active", view === "inbox"); $("learning-tab").classList.toggle("active", view === "learning"); $("inbox-tab").setAttribute("aria-current", view === "inbox" ? "page" : "false"); $("learning-tab").setAttribute("aria-current", view === "learning" ? "page" : "false"); $("page-title").textContent = view === "inbox" ? "Every meeting. One clear next step." : "Learn from what actually happened."; $("page-subtitle").textContent = view === "inbox" ? "See what was saved, what is ready, and what needs attention." : "Content-free local evidence, structured findings, and deliberate improvements."; if (view === "learning") loadLearning().catch(showError); }
$("inbox-tab").addEventListener("click", () => switchView("inbox")); $("learning-tab").addEventListener("click", () => switchView("learning")); $("refresh-button").addEventListener("click", refresh);
refresh(); setInterval(() => { if (!document.hidden && !$("confirm-dialog").open) refresh(); }, 5000);
window.addEventListener("focus", () => { if (!document.hidden) refresh(); });
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
