"use strict";

const state = {
  products: [], activeProduct: null, report: null, findings: [], signals: [],
  showAllFindings: false,
};
const $ = (id) => document.getElementById(id);

async function requestJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* no-op */ }
    throw new Error(detail);
  }
  return response.json();
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function formatLabel(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "—";
}

function toast(message) {
  const target = $("toast");
  target.textContent = message;
  target.classList.add("visible");
  window.setTimeout(() => target.classList.remove("visible"), 2200);
}

function renderCatalog() {
  const grid = $("product-grid");
  grid.replaceChildren();
  state.products.forEach((product) => {
    const button = node("button", "product-card");
    button.type = "button";
    button.dataset.productId = product.product_id;
    const top = node("div", "card-top");
    top.append(node("span", `role-badge ${product.role === "benchmark" ? "benchmark" : "transfer"}`, product.role === "benchmark" ? "Benchmark source" : "Transfer case"));
    top.append(node("span", "os-label", product.operating_system));
    button.append(top);
    button.append(node("h3", "", product.display_name));
    button.append(node("p", "card-meta", `${product.reviews_analyzed} analyzed reviews · ${formatLabel(product.brand)}`));
    const action = node("span", "card-action", "Open intelligence report →");
    button.append(action);
    button.addEventListener("click", () => loadProduct(product.product_id));
    grid.append(button);
  });
}

function renderMetrics() {
  const sample = state.report.sample || {};
  const coverage = state.report.coverage || {};
  const mappedRate = coverage.mapped_review_coverage ?? coverage.mapped_finding_review_coverage;
  const metrics = [
    ["Reviews analyzed", sample.reviews_analyzed ?? "—", "Frozen cohort"],
    ["Evidence-backed findings", state.findings.length, "Ranked product patterns"],
    ["Finding coverage", formatPercent(mappedRate), "Reviews with mapped findings"],
    ["Other verified signals", state.signals.length, "Preserved without forced mapping"],
  ];
  const target = $("metrics");
  target.replaceChildren();
  metrics.forEach(([label, value, note]) => {
    const card = node("article", "metric-card");
    card.append(node("p", "metric-label", label));
    card.append(node("strong", "metric-value", String(value)));
    card.append(node("small", "metric-note", note));
    target.append(card);
  });
}

function renderFindings() {
  const target = $("findings");
  target.replaceChildren();
  const sorted = [...state.findings].sort((a, b) => b.supporting_review_count - a.supporting_review_count);
  if (!sorted.length) {
    target.append(node("p", "empty-state", "No taxonomy-backed findings were produced for this cohort."));
    return;
  }
  const visible = state.showAllFindings ? sorted : sorted.slice(0, 10);
  visible.forEach((finding) => {
    const button = node("button", "finding-row");
    button.type = "button";
    const sentiment = node("span", `sentiment ${finding.sentiment}`, finding.sentiment);
    const body = node("span", "finding-body");
    body.append(node("strong", "", formatLabel(finding.issue_group)));
    body.append(node("small", "", formatLabel(finding.category)));
    const support = node("span", "finding-support");
    support.append(node("strong", "", String(finding.supporting_review_count)));
    support.append(node("small", "", "reviews"));
    button.append(sentiment, body, support, node("span", "chevron", "›"));
    button.addEventListener("click", () => openEvidence(finding.finding_id));
    target.append(button);
  });
  if (sorted.length > 10) {
    const toggle = node(
      "button", "findings-toggle",
      state.showAllFindings ? "Show top 10 findings" : `Show all ${sorted.length} findings`,
    );
    toggle.type = "button";
    toggle.addEventListener("click", () => {
      state.showAllFindings = !state.showAllFindings;
      renderFindings();
    });
    target.append(toggle);
  }
}

function renderSignals() {
  const target = $("signals");
  target.replaceChildren();
  if (!state.signals.length) {
    target.append(node("p", "empty-state", "No additional verified signals in this cohort."));
    return;
  }
  state.signals.slice(0, 5).forEach((signal) => {
    const item = node("article", "signal-card");
    item.append(node("p", "signal-quote", `“${signal.quote}”`));
    item.append(node("small", "", `${signal.review_id || "Source review"} · intentionally unclassified`));
    target.append(item);
  });
  if (state.signals.length > 5) target.append(node("p", "more-note", `+ ${state.signals.length - 5} more verified signals`));
}

function reportSummary(report) {
  const sampleSize = report.sample?.reviews_analyzed;
  const covered = report.coverage?.reviews_with_product_findings;
  if (Number.isFinite(sampleSize) && Number.isFinite(covered)) {
    return `${covered} of ${sampleSize} reviews contributed at least one evidence-backed product finding. ReviewLens ranked ${state.findings.length} finding patterns below and preserved ${state.signals.length} additional verified signals without forcing a category.`;
  }
  return "Evidence-backed product findings from a frozen review cohort.";
}

async function loadProduct(productId, { scroll = true } = {}) {
  try {
    const [report, findings, signals] = await Promise.all([
      requestJson(`/api/products/${encodeURIComponent(productId)}/report`),
      requestJson(`/api/products/${encodeURIComponent(productId)}/findings`),
      requestJson(`/api/products/${encodeURIComponent(productId)}/signals`),
    ]);
    state.activeProduct = productId;
    state.report = report;
    state.findings = findings.findings || [];
    state.signals = signals.signals || [];
    state.showAllFindings = false;
    document.querySelectorAll(".product-card").forEach((card) => card.classList.toggle("active", card.dataset.productId === productId));
    $("product-name").textContent = report.product.display_name;
    $("product-summary").textContent = reportSummary(report);
    const badges = $("product-badges");
    badges.replaceChildren(
      node("span", "soft-badge", formatLabel(report.product.brand)),
      node("span", "soft-badge", report.product.operating_system),
      node("span", "soft-badge", report.product.role === "benchmark" ? "Benchmark source" : "Transfer demo")
    );
    renderMetrics(); renderFindings(); renderSignals();
    $("workspace").hidden = false;
    if (scroll) $("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) { toast(error.message); }
}

async function openEvidence(findingId) {
  try {
    const payload = await requestJson(`/api/products/${encodeURIComponent(state.activeProduct)}/findings/${encodeURIComponent(findingId)}/evidence`);
    $("drawer-title").textContent = formatLabel(payload.finding.issue_group);
    $("drawer-meta").textContent = `${formatLabel(payload.finding.category)} · ${payload.finding.supporting_review_count} supporting reviews`;
    const list = $("evidence-list");
    list.replaceChildren();
    payload.evidence.forEach((evidence) => {
      const item = node("article", "evidence-card");
      item.append(node("p", "evidence-quote", `“${evidence.quote}”`));
      item.append(node("p", "evidence-source", `${"★".repeat(Math.round(evidence.rating))}${"☆".repeat(5 - Math.round(evidence.rating))} · ${evidence.review_id}`));
      const verified = node("span", "verified", "✓ Exact source verified");
      item.append(verified);
      list.append(item);
    });
    $("drawer-backdrop").hidden = false;
    $("evidence-drawer").classList.add("open");
    $("evidence-drawer").setAttribute("aria-hidden", "false");
  } catch (error) { toast(error.message); }
}

function closeEvidence() {
  $("evidence-drawer").classList.remove("open");
  $("evidence-drawer").setAttribute("aria-hidden", "true");
  $("drawer-backdrop").hidden = true;
}

function populateCompareSelects() {
  [$("compare-left"), $("compare-right")].forEach((select) => {
    select.replaceChildren();
    state.products.forEach((product) => {
      const option = node("option", "", product.display_name);
      option.value = product.product_id;
      select.append(option);
    });
  });
  if (state.products.length > 1) $("compare-right").selectedIndex = 1;
}

async function compareProducts() {
  try {
    const comparison = await requestJson("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ left_product_id: $("compare-left").value, right_product_id: $("compare-right").value }),
    });
    const target = $("comparison-result");
    target.replaceChildren();
    comparison.products.forEach((entry) => {
      const column = node("article", "compare-column");
      column.append(node("h3", "", entry.product.display_name));
      const top = [...entry.findings].sort((a, b) => b.supporting_review_count - a.supporting_review_count).slice(0, 5);
      top.forEach((finding) => column.append(node("p", "compare-finding", `${formatLabel(finding.issue_group)} · ${finding.supporting_review_count}/${comparison.cohort_size_per_product}`)));
      target.append(column);
    });
    target.append(node("p", "comparison-disclaimer", comparison.disclaimer));
  } catch (error) { toast(error.message); }
}

async function copyBrief() {
  if (!state.report) return;
  const top = [...state.findings].sort((a, b) => b.supporting_review_count - a.supporting_review_count).slice(0, 5);
  const text = [state.report.product.display_name, reportSummary(state.report), "", "Priority findings:", ...top.map((item) => `• ${formatLabel(item.issue_group)} — ${item.supporting_review_count} reviews`), "", "Directional frozen-sample analysis; inspect evidence before action."].join("\n");
  try { await navigator.clipboard.writeText(text); toast("Decision brief copied"); }
  catch (_) { toast("Clipboard access is unavailable"); }
}

async function initialize() {
  try {
    const payload = await requestJson("/api/products");
    state.products = payload.products || [];
    renderCatalog(); populateCompareSelects();
    if (state.products.length) {
      await loadProduct(state.products[0].product_id, { scroll: false });
    }
  } catch (error) {
    $("product-grid").append(node("p", "error-state", `ReviewLens could not load the frozen bundle: ${error.message}`));
  }
}

$("close-drawer").addEventListener("click", closeEvidence);
$("drawer-backdrop").addEventListener("click", closeEvidence);
$("compare-button").addEventListener("click", compareProducts);
$("copy-brief").addEventListener("click", copyBrief);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeEvidence(); });
initialize();
