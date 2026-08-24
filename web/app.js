"use strict";

const state = {
  bootstrap: null,
  watchlist: [],
  selectedCandidate: null,
  lastAssessment: null,
  lastLlmReview: null,
  lastDispatch: null,
  governance: null,
};

const pageMeta = {
  overview: ["监管总览", "全路网风险筛查、复核与处置态势"],
  monitor: ["风险监测", "候选路段筛选与优先复核队列"],
  assessment: ["监督研判", "基于过去时空证据的二次风险确认"],
  llm: ["大模型复核", "政策约束下的结构化证据与处置建议"],
  dispatch: ["资源调度", "容量、距离与成本情景约束下的处置组合"],
  governance: ["模型治理", "性能证据、验证结果与人工安全门"],
  audit: ["处置闭环", "人工确认、驳回与交接记录"],
};

const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmtNumber(value) {
  return new Intl.NumberFormat("zh-CN").format(Number(value || 0));
}

function fmtPercent(value, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

function riskClass(score) {
  if (Number(score) >= 0.8) return "high";
  if (Number(score) >= 0.6) return "mid";
  return "low";
}

function riskColor(score) {
  if (score >= 0.8) return "#d7484e";
  if (score >= 0.6) return "#ee941f";
  if (score >= 0.35) return "#e3bd36";
  return "#42b883";
}

const candidateSourceLabels = {
  future_event: "事件关联",
  random_negative: "随机对照",
  hard_negative: "高风险对照",
};

const actionCatalog = {
  VMS_WARNING: { icon: "▣", title: "发布上游可变情报板预警", note: "发布受影响方向、公里桩范围和减速提示。" },
  VERIFY: { icon: "◎", title: "调取视频或派出巡查核验", note: "由值班人员确认现场状态后再升级响应。" },
  HEAVY_RESCUE: { icon: "✚", title: "准备重型救援与起吊力量", note: "预备救援、起吊和货物泄漏检查资源。" },
  LANE_ISOLATION: { icon: "⇆", title: "临时隔离相邻车道", note: "评估车辆稳定性和散落物后决定是否执行。" },
  LOW_SPEED_WARNING: { icon: "◴", title: "布设低速队尾预警", note: "关注低速冲击波并保护上游排队尾部。" },
};

const topologyPlayback = {
  source: "assets/risk_topology_2025-09-01.gif",
  poster: "assets/risk_topology_poster.png",
  date: "2025-09-01",
  frameCount: 288,
  frameStepMinutes: 5,
  cropTop: 52,
  frameIndex: 0,
  intervalMs: 5000,
  playing: true,
  decoding: false,
  decoder: null,
  timer: null,
};

async function api(path, options = {}) {
  const request = { ...options, headers: { ...(options.headers || {}) } };
  if (request.body && typeof request.body !== "string") {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(request.body);
  }
  const response = await fetch(path, request);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `请求失败（${response.status}）`);
  return payload;
}

function toast(message, type = "info") {
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  $("#toastContainer").append(node);
  setTimeout(() => node.remove(), 3600);
}

function setLoading(button, loading, text = "处理中…") {
  if (!button) return;
  if (loading) {
    button.dataset.originalText = button.textContent;
    button.textContent = text;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
  }
}

function navigate(page) {
  if (!pageMeta[page]) return;
  $$(".page").forEach((node) => node.classList.toggle("active", node.id === `page-${page}`));
  $$(".nav-item").forEach((node) => node.classList.toggle("active", node.dataset.page === page));
  $("#pageTitle").textContent = pageMeta[page][0];
  $("#pageSubtitle").textContent = pageMeta[page][1];
  $("#sidebar").classList.remove("open");
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (page === "governance") loadGovernance();
  if (page === "audit") loadAudit();
}

function renderKpis() {
  const metrics = state.bootstrap.metrics;
  const cards = [
    { tone: "red", icon: "⌁", label: "独立测试事件召回", value: fmtPercent(metrics.event_recall), unit: "", foot: `${metrics.detected_events} / ${metrics.test_events} 起正式事故` },
    { tone: "blue", icon: "↓", label: "候选误报削减", value: fmtPercent(metrics.false_positive_reduction), unit: "", foot: "相对第一阶段直接阈值法" },
    { tone: "green", icon: "◴", label: "中位最大提前量", value: metrics.median_lead_time_min.toFixed(0), unit: "分钟", foot: "未来 30 分钟标签窗口" },
    { tone: "amber", icon: "▦", label: "大模型复核工作量", value: fmtNumber(metrics.test_watchlist_sites), unit: "站点窗", foot: `较测试期原始流压缩 ${fmtPercent(metrics.workload_reduction, 2)}` },
  ];
  $("#overviewKpis").innerHTML = cards.map((card) => `
    <article class="kpi-card ${card.tone}">
      <div class="kpi-top"><span>${card.label}</span><span class="kpi-icon">${card.icon}</span></div>
      <div class="kpi-value">${card.value}<small>${card.unit}</small></div>
      <div class="kpi-foot">${card.foot}</div>
    </article>`).join("");
}

function candidateTypeHypothesis(row) {
  const selectedCase = state.bootstrap.selected_case;
  if (row.target_event_id && row.target_event_id === selectedCase.event_id) return `${selectedCase.type_hypotheses[0].type}（假设）`;
  return "待复核";
}

function topologyTimeText(frameIndex) {
  const totalMinutes = frameIndex * topologyPlayback.frameStepMinutes;
  const hours = String(Math.floor(totalMinutes / 60) % 24).padStart(2, "0");
  const minutes = String(totalMinutes % 60).padStart(2, "0");
  return `${topologyPlayback.date} ${hours}:${minutes}`;
}

function updateTopologyControls() {
  const time = $("#topologyTime");
  const slider = $("#topologyTimeline");
  const button = $("#topologyPlayToggle");
  if (time) time.textContent = topologyTimeText(topologyPlayback.frameIndex);
  if (slider) slider.value = topologyPlayback.frameIndex;
  if (button) button.textContent = topologyPlayback.playing ? "暂停" : "播放";
}

async function drawTopologyFrame(frameIndex) {
  if (!topologyPlayback.decoder || topologyPlayback.decoding) return;
  topologyPlayback.decoding = true;
  try {
    const result = await topologyPlayback.decoder.decode({ frameIndex, completeFramesOnly: true });
    const canvas = $("#riskTopologyCanvas");
    if (!canvas) return;
    const image = result.image;
    const sourceWidth = image.displayWidth || image.codedWidth;
    const sourceHeight = image.displayHeight || image.codedHeight;
    const cropTop = Math.min(topologyPlayback.cropTop, sourceHeight - 1);
    canvas.width = sourceWidth;
    canvas.height = sourceHeight - cropTop;
    const context = canvas.getContext("2d", { alpha: false });
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, cropTop, sourceWidth, sourceHeight - cropTop, 0, 0, canvas.width, canvas.height);
    image.close();
  } finally {
    topologyPlayback.decoding = false;
  }
}

function scheduleTopologyFrame() {
  clearTimeout(topologyPlayback.timer);
  if (!topologyPlayback.playing || !topologyPlayback.decoder) return;
  topologyPlayback.timer = setTimeout(async () => {
    topologyPlayback.frameIndex = (topologyPlayback.frameIndex + 1) % topologyPlayback.frameCount;
    await drawTopologyFrame(topologyPlayback.frameIndex);
    updateTopologyControls();
    scheduleTopologyFrame();
  }, topologyPlayback.intervalMs);
}

async function initializeTopologyPlayback() {
  clearTimeout(topologyPlayback.timer);
  topologyPlayback.decoder?.close?.();
  topologyPlayback.decoder = null;
  topologyPlayback.frameIndex = 0;
  topologyPlayback.playing = true;
  updateTopologyControls();
  const fallback = $("#riskTopologyFallback");
  const canvas = $("#riskTopologyCanvas");
  const mode = $("#topologyMode");
  try {
    if (!("ImageDecoder" in window)) throw new Error("当前浏览器不支持逐帧解码");
    const response = await fetch(topologyPlayback.source);
    if (!response.ok) throw new Error(`动图载入失败（${response.status}）`);
    const decoder = new ImageDecoder({ data: await response.arrayBuffer(), type: "image/gif" });
    await decoder.tracks.ready;
    topologyPlayback.decoder = decoder;
    topologyPlayback.frameCount = decoder.tracks.selectedTrack?.frameCount || 288;
    $("#topologyTimeline").max = topologyPlayback.frameCount - 1;
    fallback.hidden = true;
    canvas.hidden = false;
    mode.textContent = "逐帧交互播放";
    await drawTopologyFrame(0);
    scheduleTopologyFrame();
  } catch (error) {
    canvas.hidden = true;
    fallback.hidden = false;
    $("#topologyTimeline").disabled = true;
    $("#topologyPlayToggle").disabled = true;
    mode.textContent = "兼容模式 · 首帧静态预览";
  }
}

function renderSpatialRiskMap() {
  const routes = state.bootstrap.route_series || [];
  const watchlist = state.bootstrap.watchlist || [];
  const totalPoints = routes.reduce((sum, route) => sum + route.points.length, 0);
  const highRiskCount = watchlist.filter((row) => row.score >= 0.8).length;
  $("#networkSpatialMap").innerHTML = `
    <div class="topology-time-heading" id="topologyTime">${topologyTimeText(0)}</div>
    <div class="topology-stage">
      <img id="riskTopologyFallback" class="risk-topology-fallback" src="${topologyPlayback.poster}" alt="2025年9月1日全天路网各路段风险拓扑首帧">
      <canvas id="riskTopologyCanvas" class="risk-topology-canvas" hidden aria-label="全天路网风险逐帧动态拓扑"></canvas>
      <div class="topology-mode" id="topologyMode">正在初始化逐帧播放…</div>
    </div>
    <div class="topology-controls">
      <button class="secondary-button" id="topologyPlayToggle" type="button">暂停</button>
      <input id="topologyTimeline" type="range" min="0" max="287" value="0" step="1" aria-label="全天时间轴">
      <label>播放速度<select id="topologySpeed"><option value="8000">慢速</option><option value="5000" selected>标准 · 5秒</option><option value="2500">快速</option></select></label>
      <button class="text-button" id="topologyRestart" type="button">回到 00:00</button>
    </div>`;
  $("#spatialMapSummary").textContent = `${totalPoints} 个空间路段点 · 288 个全天时刻 · ${watchlist.length} 条复核候选 · ${highRiskCount} 条高风险热点`;
  $("#topologyPlayToggle").addEventListener("click", () => {
    topologyPlayback.playing = !topologyPlayback.playing;
    updateTopologyControls();
    scheduleTopologyFrame();
  });
  $("#topologyTimeline").addEventListener("input", async (event) => {
    topologyPlayback.frameIndex = Number(event.target.value);
    await drawTopologyFrame(topologyPlayback.frameIndex);
    updateTopologyControls();
  });
  $("#topologySpeed").addEventListener("change", (event) => {
    topologyPlayback.intervalMs = Number(event.target.value);
    scheduleTopologyFrame();
  });
  $("#topologyRestart").addEventListener("click", async () => {
    topologyPlayback.frameIndex = 0;
    await drawTopologyFrame(0);
    updateTopologyControls();
    scheduleTopologyFrame();
  });
  initializeTopologyPlayback();
}

function renderNetworkCandidateTable() {
  const rows = state.bootstrap.watchlist.slice(0, 5);
  $("#networkCandidateTable").innerHTML = rows.map((row, index) => {
    const active = state.selectedCandidate?.segment_id === row.segment_id ? " active" : "";
    const windowText = row.lead_time_min > 0 ? `${row.lead_time_min.toFixed(0)} 分钟内` : "待复核";
    return `<tr class="network-candidate-row${active}" data-network-row="${escapeHtml(row.segment_id)}"><td><span class="network-rank rank-${index + 1}">${index + 1}</span></td><td><strong>${escapeHtml(row.road_code)} · K${row.stake_km.toFixed(3)}</strong><small>方向 ${row.direction_code} · ${escapeHtml(candidateSourceLabels[row.candidate_source] || row.candidate_source)}</small></td><td><b class="score-number ${riskClass(row.score)}">${row.score.toFixed(3)}</b></td><td><span class="status-pill ${row.risk_level === "高" ? "red" : row.risk_level === "中" ? "amber" : "green"}">${row.risk_level}风险</span></td><td>${escapeHtml(candidateTypeHypothesis(row))}</td><td>${windowText}</td></tr>`;
  }).join("");
  $$("[data-network-row]", $("#networkCandidateTable")).forEach((row) => row.addEventListener("click", () => selectNetworkCandidate(row.dataset.networkRow)));
}

function renderNetworkPriorityDetail() {
  const row = state.selectedCandidate || state.bootstrap.watchlist[0];
  if (!row) return;
  const selectedCase = state.bootstrap.selected_case;
  const matchesCase = row.target_event_id && row.target_event_id === selectedCase.event_id;
  const delta = row.score - row.stage1_score;
  const spatial = selectedCase.memory.directed_spatial_context;
  const trafficText = matchesCase ? `${fmtNumber(selectedCase.traffic_volume)} 辆 · 中位速度 ${selectedCase.median_speed_km_h.toFixed(1)} km/h` : `第一阶段风险 ${row.stage1_score.toFixed(3)}，待补充交通状态`;
  const spatialText = matchesCase ? `上游 ${spatial.upstream_mean_300m.toFixed(3)} / 下游 ${spatial.downstream_mean_300m.toFixed(3)}，邻域峰值 ${spatial.neighborhood_max_500m.toFixed(3)}` : "进入监督研判后读取同路同向上下游证据";
  const patternText = matchesCase ? `${selectedCase.type_hypotheses[0].type}为首位类型假设，严重程度仍须人工确认` : "尚未形成事故类型假设，不得将风险分数视为事故结论";
  $("#networkPriorityDetail").innerHTML = `<div class="priority-segment-identity"><div><span>路段</span><strong>${escapeHtml(row.road_code)} · K${row.stake_km.toFixed(3)}</strong><small>方向 ${row.direction_code} · ${escapeHtml(row.segment_id)}</small></div><div><span>监督分数</span><b>${row.score.toFixed(3)}</b><em>${row.risk_level}风险</em></div></div><div class="priority-evidence-list"><div><i>车</i><p><b>交通状态</b><span>${trafficText}</span></p></div><div><i>↗</i><p><b>局部风险趋势</b><span>监督分数较第一阶段${delta >= 0 ? "上升" : "下降"} ${Math.abs(delta).toFixed(3)}，当前为${escapeHtml(row.review_status)}</span></p></div><div><i>⇅</i><p><b>上下游空间证据</b><span>${spatialText}</span></p></div><div><i>!</i><p><b>可能事件模式</b><span>${patternText}</span></p></div></div><button class="secondary-button priority-detail-button" id="openNetworkCandidate">查看完整证据</button>`;
  $("#openNetworkCandidate").addEventListener("click", () => openCandidate(row.segment_id));
  let actionCodes = ["VMS_WARNING", "VERIFY"];
  if (matchesCase && selectedCase.management_actions) actionCodes = selectedCase.management_actions.split(",").map((item) => item.trim()).filter(Boolean);
  else if (row.score >= 0.8) actionCodes.push("LOW_SPEED_WARNING");
  $("#networkAuthorizedActions").innerHTML = actionCodes.filter((code) => actionCatalog[code]).slice(0, 4).map((code) => {
    const action = actionCatalog[code];
    return `<div class="authorized-action-item"><i>${action.icon}</i><div><b>${action.title}</b><span>${action.note}</span></div><em>待确认</em></div>`;
  }).join("");
}

function selectNetworkCandidate(segmentId) {
  const row = state.bootstrap.watchlist.find((item) => item.segment_id === segmentId);
  if (!row) return;
  state.selectedCandidate = row;
  renderNetworkCandidateTable();
  renderNetworkPriorityDetail();
}

function renderNetworkOverview() {
  renderSpatialRiskMap();
  renderNetworkCandidateTable();
  renderNetworkPriorityDetail();
}

function renderEvidencePacket() {
  const item = state.bootstrap.selected_case;
  const decision = item.decision;
  const spatial = item.memory.directed_spatial_context;
  $("#evidencePacket").innerHTML = `
    <div class="case-location"><strong>${escapeHtml(item.location.road_code)} · K${(item.location.stake_m / 1000).toFixed(3)}</strong><span>${escapeHtml(decision.decision_time)}</span></div>
    <div class="evidence-grid">
      <div class="evidence-item"><span>监督分数</span><b>${decision.supervisory_score.toFixed(3)} / 阈值 ${decision.threshold.toFixed(2)}</b></div>
      <div class="evidence-item"><span>交通状态</span><b>${fmtNumber(item.traffic_volume)} 辆 · ${item.median_speed_km_h.toFixed(1)} km/h</b></div>
      <div class="evidence-item"><span>上下游 300 米</span><b>${spatial.upstream_mean_300m.toFixed(3)} / ${spatial.downstream_mean_300m.toFixed(3)}</b></div>
      <div class="evidence-item"><span>类型假设</span><b>${escapeHtml(item.type_hypotheses[0].type)} · 需人工确认</b></div>
    </div>`;
}

function renderPipeline() {
  const stages = state.bootstrap.pipeline;
  const maxLog = Math.log10(Math.max(...stages.map((item) => item.records)) + 1);
  const labels = {
    raw_5min_100m_rows: "原始 5 分钟路段流",
    causal_distilled_candidates: "因果候选集",
    "10min_100m_llm_watchlist_sites": "大模型复核站点窗",
  };
  const preferred = stages.filter((item) => item.scope === "independent_test");
  $("#pipelineChart").innerHTML = `<div class="pipeline-chart">${preferred.map((item) => {
    const width = Math.max(1, Math.log10(item.records + 1) / maxLog * 100);
    return `<div class="pipeline-stage"><span>${labels[item.stage] || item.stage}</span><div class="pipeline-bar"><i style="width:${width}%"></i></div><b>${fmtNumber(item.records)}</b></div>`;
  }).join("")}<div class="compression-note">测试期从 ${fmtNumber(preferred[0]?.records)} 条原始路段状态压缩到 ${fmtNumber(preferred.at(-1)?.records)} 个大模型复核站点窗，保留比例约 ${(preferred.at(-1)?.retained_fraction_vs_raw * 100).toFixed(3)}%。</div></div>`;
}

async function loadWatchlist() {
  const params = new URLSearchParams({
    road: $("#roadFilter").value,
    direction: $("#directionFilter").value,
    level: $("#levelFilter").value,
    q: $("#monitorSearch").value.trim(),
  });
  try {
    const payload = await api(`/api/watchlist?${params}`);
    state.watchlist = payload.rows;
    renderMonitorTable();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderMonitorTable() {
  $("#monitorCount").textContent = `${state.watchlist.length} 条`;
  $("#monitorTable").innerHTML = state.watchlist.length ? state.watchlist.map((row, index) => `
    <tr>
      <td><strong>${String(index + 1).padStart(2, "0")}</strong></td>
      <td><strong>${escapeHtml(row.road_code)}</strong> / 方向 ${row.direction_code}</td>
      <td>K${row.stake_km.toFixed(3)}</td>
      <td>${row.stage1_score.toFixed(3)}</td>
      <td><span class="score-number ${riskClass(row.score)}">${row.score.toFixed(3)}</span></td>
      <td><span class="status-pill ${row.risk_level === "高" ? "red" : row.risk_level === "中" ? "amber" : "green"}">${row.risk_level}风险</span></td>
      <td>${fmtPercent(row.calibrated_probability, 2)}</td>
      <td>${escapeHtml(row.review_status)}</td>
      <td><button class="table-action" data-open-candidate="${escapeHtml(row.segment_id)}">查看证据</button></td>
    </tr>`).join("") : '<tr><td colspan="9"><div class="empty-state" style="height:180px">没有符合筛选条件的候选路段</div></td></tr>';
  $$('[data-open-candidate]').forEach((button) => button.addEventListener("click", () => openCandidate(button.dataset.openCandidate)));
}

function openCandidate(segmentId) {
  const rows = [...(state.bootstrap?.watchlist || []), ...state.watchlist];
  const row = rows.find((item) => item.segment_id === segmentId);
  if (!row) return;
  state.selectedCandidate = row;
  $("#drawerTitle").textContent = `${row.road_code} · K${row.stake_km.toFixed(3)}`;
  $("#drawerContent").innerHTML = `
    <div class="drawer-score"><div><span>蒸馏学生监督分数</span><b>${row.score.toFixed(3)}</b></div><span class="status-pill ${row.risk_level === "高" ? "red" : "amber"}">${row.risk_level}风险</span></div>
    <div class="drawer-section"><h3>定位与时窗</h3><div class="drawer-grid">
      <div class="drawer-item"><span>路段编号</span><b>${escapeHtml(row.segment_id)}</b></div><div class="drawer-item"><span>决策时刻</span><b>${escapeHtml(row.time)}</b></div>
      <div class="drawer-item"><span>道路方向</span><b>${escapeHtml(row.road_code)} / 方向 ${row.direction_code}</b></div><div class="drawer-item"><span>标签窗口</span><b>未来 30 分钟</b></div>
    </div></div>
    <div class="drawer-section"><h3>模型对照</h3><div class="drawer-grid">
      <div class="drawer-item"><span>第一阶段分数</span><b>${row.stage1_score.toFixed(3)}</b></div><div class="drawer-item"><span>监督分数</span><b>${row.score.toFixed(3)}</b></div>
      <div class="drawer-item"><span>校准概率诊断</span><b>${fmtPercent(row.calibrated_probability, 2)}</b></div><div class="drawer-item"><span>候选来源</span><b>${escapeHtml(row.candidate_source)}</b></div>
    </div></div>
    <div class="boundary-card" style="margin:18px 0 0"><b>注意</b><p>该条目来自历史回放候选集。监督分数达到阈值只触发人工复核，不说明事故已经发生。</p></div>
    <div class="drawer-actions"><button class="secondary-button" id="drawerAssess">进入监督研判</button><button class="primary-button" id="drawerAudit">记录人工结论</button></div>`;
  $("#detailDrawer").classList.add("open");
  $("#drawerBackdrop").classList.add("open");
  $("#detailDrawer").setAttribute("aria-hidden", "false");
  $("#drawerAssess").addEventListener("click", () => { closeDrawer(); navigate("assessment"); });
  $("#drawerAudit").addEventListener("click", () => openAuditModal(row.segment_id, "风险监测"));
}

function closeDrawer() {
  $("#detailDrawer").classList.remove("open");
  $("#drawerBackdrop").classList.remove("open");
  $("#detailDrawer").setAttribute("aria-hidden", "true");
}

function fillAssessmentDefaults() {
  const values = state.bootstrap.default_features;
  Object.entries(values).forEach(([name, value]) => {
    const input = $(`[name="${name}"]`, $("#assessmentForm"));
    if (input) input.value = Number(value).toFixed(4);
  });
  $("#assessRoad").value = values.road_is_s5 ? "S5" : "G9411";
  renderGauge(null);
  $("#assessmentEvidence").className = "evidence-bars empty-state";
  $("#assessmentEvidence").textContent = "运行模型后显示时空证据";
  $("#assessmentStatus").className = "status-pill";
  $("#assessmentStatus").textContent = "等待计算";
  $("#calibratedProbability").textContent = "—";
  $("#scoreSemantics").textContent = "待运行";
}

function assessmentPayload() {
  const payload = { ...state.bootstrap.default_features };
  new FormData($("#assessmentForm")).forEach((value, key) => { payload[key] = Number(value); });
  payload.road_is_s5 = $("#assessRoad").value === "S5" ? 1 : 0;
  return payload;
}

function renderGauge(result) {
  const score = result ? Number(result.score) : 0;
  const length = 283;
  $("#gaugeFill").style.strokeDashoffset = String(length * (1 - score));
  $("#gaugeFill").style.stroke = result ? riskColor(score) : "#1f6df2";
  $("#gaugeScore").textContent = result ? score.toFixed(3) : "—";
  const angle = Math.PI * (1 - 0.6);
  const cx = 120 + 90 * Math.cos(angle);
  const cy = 125 - 90 * Math.sin(angle);
  const innerX = 120 + 77 * Math.cos(angle);
  const innerY = 125 - 77 * Math.sin(angle);
  Object.entries({ x1: cx, y1: cy, x2: innerX, y2: innerY }).forEach(([key, value]) => $("#thresholdMark").setAttribute(key, value));
}

function renderAssessment(result) {
  state.lastAssessment = result;
  renderGauge(result);
  const status = $("#assessmentStatus");
  status.className = `status-pill ${result.alert ? result.risk_level === "高" ? "red" : "amber" : "green"}`;
  status.textContent = result.alert ? "进入人工复核" : "保留观察";
  $("#calibratedProbability").textContent = result.calibrated_probability == null ? "未提供" : fmtPercent(result.calibrated_probability, 2);
  $("#scoreSemantics").textContent = result.semantics;
  $("#assessmentEvidence").className = "evidence-bars";
  $("#assessmentEvidence").innerHTML = result.evidence.map((item) => `
    <div class="evidence-bar" title="${escapeHtml(item.note)}"><span>${escapeHtml(item.key)}</span><div class="evidence-bar-track"><i style="width:${Math.min(100, Math.abs(item.value) * 100)}%"></i></div><b>${Number(item.value).toFixed(3)}</b></div>`).join("") +
    (result.degradation_flags.length ? `<div class="compression-note" style="color:#985800;background:#fff0cf">降级提示：${result.degradation_flags.map(escapeHtml).join("；")}</div>` : "");
}

function renderLlmPacket() {
  const item = state.bootstrap.selected_case;
  const spatial = item.memory.directed_spatial_context;
  $("#llmSegment").textContent = `${item.location.road_code} · K${(item.location.stake_m / 1000).toFixed(3)}`;
  $("#llmPacket").innerHTML = [
    ["监督分数", item.decision.supervisory_score.toFixed(3)],
    ["30 分钟风险均值", item.memory.past_30min.mean.toFixed(3)],
    ["上游 300 米", spatial.upstream_mean_300m.toFixed(3)],
    ["下游 300 米", spatial.downstream_mean_300m.toFixed(3)],
    ["邻域峰值", spatial.neighborhood_max_500m.toFixed(3)],
    ["人工复核", "必须"],
  ].map(([label, value]) => `<div class="packet-item"><span>${label}</span><b>${value}</b></div>`).join("");
  $("#typeHypothesis").value = item.type_hypotheses[0].type;
  $("#llmSpeed").value = item.median_speed_km_h.toFixed(1);
  $("#llmTraffic").value = `${item.upstream_traffic} / ${item.downstream_traffic}`;
}

async function runLlmReview() {
  const button = $("#runLlmReview");
  const parts = $("#llmTraffic").value.split("/").map((item) => Number(item.trim()));
  const packet = {
    segment_id: state.bootstrap.selected_case.location.segment_id,
    score: state.lastAssessment?.score ?? state.bootstrap.selected_case.decision.supervisory_score,
    type_hypothesis: $("#typeHypothesis").value,
    weather: $("#llmWeather").value,
    median_speed_km_h: Number($("#llmSpeed").value),
    upstream_traffic: Number.isFinite(parts[0]) ? parts[0] : 0,
    downstream_traffic: Number.isFinite(parts[1]) ? parts[1] : 0,
    evidence: state.lastAssessment?.evidence || [],
  };
  try {
    setLoading(button, true, "正在校验政策与生成建议…");
    const result = await api("/api/llm/review", { method: "POST", body: packet });
    state.lastLlmReview = result;
    renderLlmReview(result);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setLoading(button, false);
  }
}

function renderLlmReview(result) {
  $("#llmEngine").textContent = result.engine;
  $("#llmOutput").className = "llm-output";
  $("#llmOutput").innerHTML = `
    <div class="llm-state"><b>${escapeHtml(result.operational_state)}</b><span>政策版本 ${escapeHtml(result.policy_version)}</span></div>
    <div class="reason-box">${escapeHtml(result.reasoning)}</div>
    <div class="action-list">${result.actions.map((item) => `<div class="action-item"><span class="action-check">✓</span><div><b>${escapeHtml(item.label)}</b><span>${escapeHtml(item.code)} · 授权动作</span></div><span class="status-pill green">待确认</span></div>`).join("")}</div>
    <div class="boundary-card" style="margin:13px 0 0"><b>安全门</b><p>事故类型为假设，严重程度为“${escapeHtml(result.severity)}”。${escapeHtml(result.engine_note)}</p></div>
    <div class="llm-footer"><button class="secondary-button" id="rejectLlm">驳回建议</button><button class="primary-button" id="confirmLlm">确认并留痕</button></div>`;
  $("#rejectLlm").addEventListener("click", () => openAuditModal(state.bootstrap.selected_case.event_id, "大模型复核", "驳回"));
  $("#confirmLlm").addEventListener("click", () => openAuditModal(state.bootstrap.selected_case.event_id, "大模型复核", "确认"));
}

async function runDispatch(event) {
  event.preventDefault();
  const button = $("#dispatchForm button[type=submit]");
  const payload = {};
  new FormData(event.currentTarget).forEach((value, key) => payload[key] = key === "scenario" ? value : Number(value));
  try {
    setLoading(button, true, "正在求解容量约束…");
    const result = await api("/api/dispatch/optimize", { method: "POST", body: payload });
    state.lastDispatch = result;
    renderDispatch(result);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setLoading(button, false);
  }
}

function renderDispatch(result) {
  $("#dispatchSummary").innerHTML = [
    ["候选路段", result.counts.candidate], ["预警动作", result.counts.warning], ["巡查动作", result.counts.patrol], ["应急准备", result.counts.emergency],
  ].map(([label, value]) => `<div class="summary-box"><span>${label}</span><b>${value}</b></div>`).join("");
  $("#dispatchActions").className = "dispatch-action-list";
  $("#dispatchActions").innerHTML = result.actions.length ? result.actions.map((item) => `
    <div class="dispatch-row"><span class="dispatch-kind ${item.kind}">${item.kind === "warning" ? "讯" : item.kind === "patrol" ? "巡" : "应"}</span><strong>${escapeHtml(item.label)}</strong><span>${escapeHtml(item.road_code)} · K${Number(item.stake_km).toFixed(3)} · 方向 ${item.direction_code}</span><span class="score-number ${riskClass(item.score)}">${Number(item.score).toFixed(3)}</span><span class="status-pill blue">${escapeHtml(item.status)}</span></div>`).join("") : '<div class="empty-state tall">当前约束下不建议配置动作</div>';
  $("#sendToAudit").disabled = !result.actions.length;
  $("#sendToAudit").textContent = `提交 ${result.actions.length} 项人工确认`;
}

async function loadGovernance() {
  try {
    if (!state.governance) state.governance = await api("/api/governance");
    renderGovernance();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderGovernance() {
  const data = state.governance;
  const metrics = state.bootstrap.metrics;
  const packet = data.model.memory_packet;
  const cards = [
    ["蒸馏学生模型", `${metrics.student_model_mb.toFixed(3)} MB`, "22 特征，本机实时筛查"],
    ["独立测试 ROC-AUC", metrics.roc_auc.toFixed(3), "2025 年 11—12 月时间外测试"],
    ["类型假设准确率", fmtPercent(metrics.type_accuracy), "六分类，仅作为人工复核线索"],
    ["紧凑证据包", `${packet.mean_compact_memory_packet_bytes.toFixed(0)} B`, `较完整包缩减 ${fmtPercent(packet.mean_packet_byte_reduction)}`],
  ];
  $("#governanceKpis").innerHTML = cards.map(([label, value, foot], index) => `<article class="kpi-card ${["blue","green","amber","red"][index]}"><div class="kpi-top"><span>${label}</span><span class="kpi-icon">${["◇","✓","!","▦"][index]}</span></div><div class="kpi-value">${value}</div><div class="kpi-foot">${foot}</div></article>`).join("");

  const models = data.benchmarks.models.sort((a, b) => b.composite_score - a.composite_score);
  $("#benchmarkChart").innerHTML = models.map((model) => `<div class="benchmark-row"><span title="${escapeHtml(model.model)}">${escapeHtml(model.model)}</span><div class="benchmark-track"><i style="width:${model.composite_score * 100}%"></i></div><b>${model.composite_score.toFixed(3)}</b><small>${model.seconds_per_case.toFixed(2)} s/例</small></div>`).join("") + '<div class="compression-note">Qwen3 8B 在固定结构化契约下与 32B 模型非劣；该结论不外推到通用推理任务。</div>';

  const checks = data.validation.checks || [];
  $("#validationChecks").innerHTML = checks.length ? checks.map((check) => `<div class="check-row"><span class="check-icon">✓</span><span>${escapeHtml(check.name || check.check || "验证项")}</span></div>`).join("") : '<div class="empty-state tall">验证报告未提供逐项结构</div>';
  $("#governanceBoundaries").innerHTML = data.boundaries.map((text) => `<div class="boundary-item"><i>!</i><span>${escapeHtml(text)}</span></div>`).join("");
}

async function loadAudit() {
  try {
    const payload = await api("/api/audit");
    renderAudit(payload.rows);
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderAudit(rows) {
  $("#auditTable").innerHTML = rows.length ? rows.map((row) => `<tr><td>${escapeHtml(row.time.replace("T", " "))}</td><td><strong>${escapeHtml(row.operator)}</strong></td><td><span class="status-pill ${row.action === "确认" ? "green" : row.action === "驳回" ? "red" : "blue"}">${escapeHtml(row.action)}</span></td><td>${escapeHtml(row.target)}</td><td title="${escapeHtml(row.note)}">${escapeHtml(row.note || "—")}</td><td>${escapeHtml(row.source)}</td></tr>`).join("") : '<tr><td colspan="6"><div class="empty-state" style="height:220px">尚无人工处置记录</div></td></tr>';
}

function openAuditModal(target = "", source = "平台操作", action = "确认") {
  const form = $("#auditForm");
  form.reset();
  form.elements.target.value = target;
  form.elements.source.value = source;
  form.elements.action.value = action;
  form.elements.operator.value = "值班员";
  $("#modalBackdrop").classList.add("open");
}

function closeModal() { $("#modalBackdrop").classList.remove("open"); }

async function submitAudit(event) {
  event.preventDefault();
  const payload = Object.fromEntries(new FormData(event.currentTarget));
  const button = event.currentTarget.querySelector("button[type=submit]");
  try {
    setLoading(button, true, "正在写入…");
    await api("/api/audit", { method: "POST", body: payload });
    closeModal();
    closeDrawer();
    toast("人工处置记录已写入审计日志");
    if ($("#page-audit").classList.contains("active")) loadAudit();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setLoading(button, false);
  }
}

function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.page)));
  $$('[data-goto]').forEach((button) => button.addEventListener("click", () => navigate(button.dataset.goto)));
  $("#menuToggle").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#dismissNotice").addEventListener("click", () => $("#boundaryNotice").classList.add("hidden"));
  $("#refreshButton").addEventListener("click", bootstrap);
  ["#roadFilter", "#directionFilter", "#levelFilter"].forEach((selector) => $(selector).addEventListener("change", loadWatchlist));
  let searchTimer;
  $("#monitorSearch").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadWatchlist, 240); });
  $("#monitorToAssess").addEventListener("click", () => navigate("assessment"));
  $("#assessmentForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.currentTarget.querySelector("button[type=submit]");
    try {
      setLoading(button, true, "模型计算中…");
      renderAssessment(await api("/api/risk/evaluate", { method: "POST", body: assessmentPayload() }));
    } catch (error) { toast(error.message, "error"); }
    finally { setLoading(button, false); }
  });
  $("#resetAssessment").addEventListener("click", fillAssessmentDefaults);
  $("#runLlmReview").addEventListener("click", runLlmReview);
  $("#dispatchForm").addEventListener("submit", runDispatch);
  $("#sendToAudit").addEventListener("click", () => openAuditModal(`调度组合：${state.lastDispatch?.actions.length || 0} 项`, "资源调度"));
  $("#addAuditNote").addEventListener("click", () => openAuditModal("值班交接", "处置闭环", "备注"));
  $("#closeDrawer").addEventListener("click", closeDrawer);
  $("#drawerBackdrop").addEventListener("click", closeDrawer);
  $("#closeModal").addEventListener("click", closeModal);
  $("#modalBackdrop").addEventListener("click", (event) => { if (event.target === event.currentTarget) closeModal(); });
  $("#auditForm").addEventListener("submit", submitAudit);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") { closeDrawer(); closeModal(); } });
}

async function bootstrap() {
  const refresh = $("#refreshButton");
  try {
    refresh.classList.add("spinning");
    const payload = await api("/api/bootstrap");
    state.bootstrap = payload;
    state.watchlist = payload.watchlist;
    state.selectedCandidate = payload.watchlist[0] || null;
    $("#snapshotClock").textContent = payload.snapshot.time;
    $("#navAlertCount").textContent = payload.watchlist.filter((row) => row.score >= .6).length;
    const modelReady = $("#modelReady");
    modelReady.classList.toggle("ready", payload.model_status.ready);
    modelReady.innerHTML = `<i></i>${payload.model_status.ready ? "真实模型已就绪" : "模型不可用"}`;
    renderKpis();
    renderNetworkOverview();
    renderEvidencePacket();
    renderPipeline();
    renderMonitorTable();
    fillAssessmentDefaults();
    renderLlmPacket();
    renderAudit(payload.audit || []);
    toast("平台数据已加载：历史因果回放模式");
  } catch (error) {
    toast(error.message, "error");
    console.error(error);
  } finally {
    refresh.classList.remove("spinning");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  bootstrap();
});
