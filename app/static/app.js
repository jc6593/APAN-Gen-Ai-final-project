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
    top.append(node("span", `role-badge ${product.role === "benchmark" ? "benchmark" : "transfer"}`, product.role === "benchmark" ? "Reference cohort" : "Catalog case"));
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

function renderCatalogStats() {
  const target = $("catalog-stats");
  const productCount = state.products.length;
  const availableCount = state.products.reduce(
    (total, product) => total + (Number(product.available_review_count) || 0), 0,
  );
  const analyzedCount = state.products.reduce(
    (total, product) => total + (Number(product.reviews_analyzed) || 0), 0,
  );
  const brandCount = new Set(state.products.map((product) => product.brand)).size;
  const operatingSystemCount = new Set(
    state.products.map((product) => product.operating_system),
  ).size;
  const pairCount = productCount * (productCount - 1) / 2;
  const stats = [
    [productCount, "featured laptops"],
    [availableCount || "—", "qualifying reviews collected"],
    [analyzedCount, "frozen reviews analyzed"],
    [`${brandCount} · ${operatingSystemCount}`, "brands · operating systems"],
    [pairCount, "fair head-to-head pairs"],
  ];
  target.replaceChildren();
  stats.forEach(([value, label]) => {
    const item = node("div", "catalog-stat");
    item.append(node("strong", "", String(value)), node("span", "", label));
    target.append(item);
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

function formatSampleDate(value) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  const date = Number.isFinite(numeric)
    ? new Date(numeric > 1e12 ? numeric : numeric * 1000)
    : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleDateString("en-US", { year: "numeric", month: "short" });
}

function renderDecisionBrief() {
  const target = $("decision-brief");
  target.replaceChildren();
  const brief = state.report.decision_brief || {};
  const sample = state.report.sample || {};
  const sampleSize = sample.reviews_analyzed || "—";

  const addBriefCard = (label, className, findings, emptyText) => {
    const card = node("article", `brief-card ${className}`);
    card.append(node("p", "brief-label", label));
    if (findings?.length) {
      findings.forEach((finding, index) => {
        const button = node("button", "brief-finding");
        button.type = "button";
        button.append(
          node("strong", "", formatLabel(finding.issue_group)),
          node("span", "", `${finding.supporting_review_count} of ${sampleSize} sampled reviews${index ? " · tied" : ""}`),
          node("small", "", "View source evidence →"),
        );
        button.addEventListener("click", () => openEvidence(finding.finding_id));
        card.append(button);
      });
    } else {
      card.append(node("p", "brief-empty", emptyText));
    }
    target.append(card);
  };

  addBriefCard(
    "Key strength", "positive", brief.strengths,
    "No repeated positive pattern met the two-review threshold.",
  );
  addBriefCard(
    "Priority opportunity", "negative", brief.opportunities,
    "No repeated negative pattern met the two-review threshold.",
  );

  const context = node("article", "brief-card sample-context");
  context.append(node("p", "brief-label", "Sample context"));
  const distribution = sample.rating_distribution || {};
  const mix = [1, 2, 3, 4, 5].map((star) => `${star}★ ${distribution[String(star)] ?? 0}`).join(" · ");
  context.append(
    node("strong", "sample-rating", `${Number(sample.average_rating || 0).toFixed(1)}★ average`),
    node("span", "", mix),
    node("span", "", `${formatSampleDate(sample.date_range?.start)} – ${formatSampleDate(sample.date_range?.end)}`),
    node("small", "", brief.caveat || "Directional convenience sample; inspect evidence before action."),
  );
  target.append(context);
}

function renderFindings() {
  const target = $("findings");
  target.replaceChildren();
  const sorted = [...state.findings].sort((a, b) => b.supporting_review_count - a.supporting_review_count);
  if (!sorted.length) {
    target.append(node("p", "empty-state", "No taxonomy-backed findings were produced for this cohort."));
    return;
  }
  const groups = [
    ["Strengths", sorted.filter((item) => item.sentiment === "positive")],
    ["Pain points", sorted.filter((item) => item.sentiment === "negative")],
    ["Other mapped findings", sorted.filter((item) => !["positive", "negative"].includes(item.sentiment))],
  ];
  const renderFinding = (finding) => {
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
  };
  groups.forEach(([label, findings]) => {
    if (!findings.length) return;
    const visible = state.showAllFindings ? findings : findings.slice(0, 5);
    target.append(node("h4", "finding-group-title", label));
    visible.forEach(renderFinding);
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
      node("span", "soft-badge", report.product.role === "benchmark" ? "Reference cohort" : "Catalog case")
    );
    renderDecisionBrief(); renderMetrics(); renderFindings(); renderSignals();
    $("workspace").hidden = false;
    if (scroll) $("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) { toast(error.message); }
}

async function openEvidenceFor(productId, findingId, cohortSize = null) {
  try {
    const cohortQuery = cohortSize ? `?cohort_size=${encodeURIComponent(cohortSize)}` : "";
    const payload = await requestJson(`/api/products/${encodeURIComponent(productId)}/findings/${encodeURIComponent(findingId)}/evidence${cohortQuery}`);
    $("drawer-title").textContent = formatLabel(payload.finding.issue_group);
    const evidenceReviewCount = new Set(payload.evidence.map((item) => item.review_id)).size;
    const excerptSummary = payload.evidence.length
      ? `showing ${payload.evidence.length} stored excerpts from ${evidenceReviewCount} review${evidenceReviewCount === 1 ? "" : "s"}`
      : "no stored excerpt in this comparison prefix";
    $("drawer-meta").textContent = `${payload.product.display_name} · ${formatLabel(payload.finding.category)} · pattern in ${payload.finding.supporting_review_count} reviews · ${excerptSummary}`;
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
    if (!payload.evidence.length) {
      list.append(node("p", "empty-state", "This mapped comparison signal is verified, but its stored evidence excerpt falls outside the equal-size comparison prefix. Open the full product report to inspect the broader finding."));
    }
    $("drawer-backdrop").hidden = false;
    $("evidence-drawer").classList.add("open");
    $("evidence-drawer").setAttribute("aria-hidden", "false");
  } catch (error) { toast(error.message); }
}

async function openEvidence(findingId) {
  return openEvidenceFor(state.activeProduct, findingId);
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
    const [left, right] = comparison.products;
    const sameBrand = left.product.brand === right.product.brand;
    const sideLabel = (entry) => sameBrand ? entry.product.display_name : formatLabel(entry.product.brand);

    const overview = node("div", "compare-overview");
    comparison.products.forEach((entry) => {
      const column = node("article", "compare-column compare-overview-card");
      column.append(node("p", "compare-side-label", entry === left ? "PRODUCT ONE" : "PRODUCT TWO"));
      column.append(node("h3", "", entry.product.display_name));
      column.append(node(
        "p", "compare-sample-note",
        `${comparison.cohort_size_per_product} reviews · ${formatPercent(entry.coverage.mapped_review_coverage)} finding coverage`,
      ));
      const highlights = comparison.highlights?.[entry.product.product_id] || {};
      const highlightGrid = node("div", "compare-highlight-grid");
      [
        ["Key strength", "positive", highlights.strengths?.[0]],
        ["Top friction", "negative", highlights.pain_points?.[0]],
      ].forEach(([label, sentiment, finding]) => {
        const block = node("div", `compare-highlight ${sentiment}`);
        block.append(node("span", "compare-highlight-label", label));
        if (finding) {
          const button = node(
            "button", "compare-highlight-button",
            `${formatLabel(finding.issue_group)} · ${finding.supporting_review_count}/${comparison.cohort_size_per_product}`,
          );
          button.type = "button";
          button.addEventListener("click", () => openEvidenceFor(entry.product.product_id, finding.finding_id, comparison.cohort_size_per_product));
          block.append(button);
        } else {
          block.append(node("span", "compare-no-signal", "No mapped signal"));
        }
        highlightGrid.append(block);
      });
      column.append(highlightGrid);
      overview.append(column);
    });
    target.append(overview);

    const takeaways = comparison.takeaways || {};
    const takeawaySection = node("section", "compare-takeaways");
    const takeawayHeading = node("div", "compare-takeaway-heading");
    takeawayHeading.append(
      node("p", "eyebrow", "WHAT STANDS OUT"),
      node("p", "compare-threshold-note", `Only gaps of ${takeaways.minimum_gap_reviews || 2}+ reviews are promoted as differentiators.`),
    );
    takeawaySection.append(takeawayHeading);
    const takeawayGrid = node("div", "compare-takeaway-grid");

    const renderDifferentiator = (label, item, sentiment, emptyText) => {
      const card = node("article", `compare-takeaway-card ${sentiment}`);
      card.append(node("span", "compare-takeaway-label", label));
      if (!item) {
        card.append(node("strong", "", emptyText));
        return card;
      }
      const entry = item.edge === "left" ? left : right;
      const button = node("button", "compare-takeaway-button");
      button.type = "button";
      button.append(
        node("strong", "", `${sideLabel(entry)} · ${formatLabel(item.issue_group)}`),
        node("span", "", `${item.gap_reviews} more ${sentiment === "positive" ? "positive" : "friction"} reviews · ${item.left_supporting_review_count}/${comparison.cohort_size_per_product} vs ${item.right_supporting_review_count}/${comparison.cohort_size_per_product}`),
        node("small", "", "View source evidence →"),
      );
      button.addEventListener("click", () => openEvidenceFor(item.product_id, item.finding_id, comparison.cohort_size_per_product));
      card.append(button);
      return card;
    };

    takeawayGrid.append(
      renderDifferentiator(
        "Positive differentiator", takeaways.positive_differentiator, "positive",
        "No positive gap reached the threshold.",
      ),
      renderDifferentiator(
        "Reported friction gap", takeaways.friction_differentiator, "negative",
        "No friction gap reached the threshold.",
      ),
    );

    const shared = takeaways.strongest_shared_pattern;
    const sharedCard = node("article", "compare-takeaway-card shared");
    sharedCard.append(node("span", "compare-takeaway-label", "Strongest shared pattern"));
    if (shared) {
      sharedCard.append(
        node("strong", "", `${formatLabel(shared.issue_group)} · ${formatLabel(shared.sentiment)}`),
        node("span", "", `${shared.left_supporting_review_count}/${comparison.cohort_size_per_product} vs ${shared.right_supporting_review_count}/${comparison.cohort_size_per_product} reviews`),
      );
      const sharedLinks = node("div", "compare-shared-links");
      [[left, shared.left_finding_id], [right, shared.right_finding_id]].forEach(([entry, findingId]) => {
        const button = node("button", "compare-shared-button", `${sideLabel(entry)} evidence →`);
        button.type = "button";
        button.addEventListener("click", () => openEvidenceFor(entry.product.product_id, findingId, comparison.cohort_size_per_product));
        sharedLinks.append(button);
      });
      sharedCard.append(sharedLinks);
    } else {
      sharedCard.append(node("strong", "", "No repeated shared pattern met the threshold."));
    }
    takeawayGrid.append(sharedCard);
    takeawaySection.append(takeawayGrid);
    target.append(takeawaySection);

    const matrix = node("section", "compare-matrix");
    const matrixHeading = node("div", "compare-matrix-heading");
    matrixHeading.append(node("div", "", "ISSUE DIMENSION"));
    matrixHeading.append(node("div", "", sideLabel(left)));
    matrixHeading.append(node("div", "", sideLabel(right)));
    matrixHeading.append(node("div", "", "DIRECTIONAL READ"));
    matrix.append(matrixHeading);

    const renderSignal = (productId, finding, sentiment) => {
      if (!finding) return node("span", "compare-no-signal", "No mapped signal");
      const button = node("button", `compare-signal ${sentiment}`);
      button.type = "button";
      const top = node("span", "compare-signal-top");
      top.append(
        node("span", "compare-signal-sentiment", sentiment),
        node("strong", "", `${finding.supporting_review_count}/${comparison.cohort_size_per_product}`),
      );
      const track = node("span", "compare-bar-track");
      const fill = node("span", "compare-bar-fill");
      fill.style.width = `${Math.max(4, Math.round(finding.supporting_review_rate * 100))}%`;
      track.append(fill);
      button.append(top, track);
      button.addEventListener("click", () => openEvidenceFor(productId, finding.finding_id, comparison.cohort_size_per_product));
      return button;
    };

    const renderIssueCell = (productId, signals) => {
      const cell = node("div", "compare-issue-cell");
      if (signals.positive) cell.append(renderSignal(productId, signals.positive, "positive"));
      if (signals.negative) cell.append(renderSignal(productId, signals.negative, "negative"));
      if (!signals.positive && !signals.negative) cell.append(node("span", "compare-no-signal", "No mapped signal"));
      return cell;
    };

    const directionalRead = (row) => {
      const reads = [];
      const count = (side, sentiment) => row[side]?.[sentiment]?.supporting_review_count || 0;
      const positiveGap = Math.abs(count("left", "positive") - count("right", "positive"));
      const negativeGap = Math.abs(count("left", "negative") - count("right", "negative"));
      if (row.directional_read.positive_edge !== "even" && positiveGap) {
        const entry = row.directional_read.positive_edge === "left" ? left : right;
        reads.push(`${sideLabel(entry)} +${positiveGap} positive`);
      }
      if (row.directional_read.negative_edge !== "even" && negativeGap) {
        const entry = row.directional_read.negative_edge === "left" ? left : right;
        reads.push(`${sideLabel(entry)} +${negativeGap} friction`);
      }
      return reads.length ? reads.join(" · ") : "Similar mapped signal";
    };

    const visibleIssues = (comparison.aligned_issues || []).slice(0, 12);
    visibleIssues.forEach((row) => {
      const matrixRow = node("article", "compare-matrix-row");
      const dimension = node("div", "compare-dimension");
      dimension.append(
        node("strong", "", formatLabel(row.issue_group)),
        node("span", "", formatLabel(row.category)),
      );
      matrixRow.append(
        dimension,
        renderIssueCell(left.product.product_id, row.left),
        renderIssueCell(right.product.product_id, row.right),
        node("p", "compare-read", directionalRead(row)),
      );
      matrix.append(matrixRow);
    });
    if (!visibleIssues.length) {
      matrix.append(node("p", "empty-state", "No aligned positive or negative findings were available for this pair."));
    } else if (comparison.aligned_issues.length > visibleIssues.length) {
      matrix.append(node("p", "comparison-more", `Showing the 12 strongest aligned dimensions of ${comparison.aligned_issues.length}.`));
    }
    target.append(matrix);
    target.append(node("p", "comparison-disclaimer", comparison.disclaimer));
  } catch (error) { toast(error.message); }
}

async function copyBrief() {
  if (!state.report) return;
  const brief = state.report.decision_brief || {};
  const lines = (items, emptyText) => items?.length
    ? items.map((item) => `• ${formatLabel(item.issue_group)} — ${item.supporting_review_count} reviews`)
    : [`• ${emptyText}`];
  const text = [
    state.report.product.display_name,
    reportSummary(state.report),
    "",
    "Key strength:",
    ...lines(brief.strengths, "No repeated positive pattern met the threshold."),
    "",
    "Priority opportunity:",
    ...lines(brief.opportunities, "No repeated negative pattern met the threshold."),
    "",
    brief.caveat || "Directional frozen-sample analysis; inspect evidence before action.",
  ].join("\n");
  try { await navigator.clipboard.writeText(text); toast("Decision brief copied"); }
  catch (_) { toast("Clipboard access is unavailable"); }
}

async function initialize() {
  try {
    const payload = await requestJson("/api/products");
    state.products = payload.products || [];
    renderCatalog(); renderCatalogStats(); populateCompareSelects();
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
