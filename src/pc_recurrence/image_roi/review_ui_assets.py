from __future__ import annotations

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CT Series review</title>
  <link rel="stylesheet" href="/assets/app.css">
  <script src="/assets/app.js" defer></script>
</head>
<body>
  <a class="skip-link" href="#patient-list">Skip to patients</a>
  <header class="app-header">
    <div class="title-row">
      <div>
        <p class="eyebrow">Local montage reviewer</p>
        <h1>CT Series review</h1>
        <p id="selection-file" class="selection-file">Loading inventory…</p>
      </div>
      <div class="progress-panel" aria-live="polite">
        <strong id="selection-progress">— / — selected</strong>
        <span id="issue-progress">— blocked / — issues</span>
      </div>
    </div>
    <p class="workflow-note">
      Partial progress can be saved. Preprocessing still requires one ready choice for every
      targeted selectable patient and separately rejects blocked or stale inputs.
    </p>
    <div class="toolbar" aria-label="Review controls">
      <div class="search-control">
        <label for="patient-search">Search patients and Series</label>
        <div class="search-row">
          <input id="patient-search" type="search" autocomplete="off"
                 placeholder="Patient, UID, description, protocol, source…">
          <button id="clear-search" type="button">Clear</button>
        </div>
      </div>
      <div class="filter-control">
        <span id="filter-label">Show patients</span>
        <div class="segmented" role="group" aria-labelledby="filter-label">
          <button type="button" data-filter="all" aria-pressed="true">All</button>
          <button type="button" data-filter="unselected" aria-pressed="false">
            Unselected
          </button>
          <button type="button" data-filter="issues" aria-pressed="false">Issues</button>
        </div>
      </div>
      <button id="next-unselected" class="secondary-action" type="button">
        Next unselected
      </button>
      <div class="save-control">
        <span id="save-state" role="status" aria-live="polite">Loading…</span>
        <div class="save-actions">
          <button id="reload-inventory" type="button" hidden>Reload file</button>
          <button id="save-selections" class="primary-action" type="button" disabled>
            Save selections
          </button>
        </div>
      </div>
    </div>
    <div class="result-row">
      <span id="result-count" aria-live="polite"></span>
      <label class="compact-jump" for="patient-jump">Jump to patient</label>
      <select id="patient-jump" class="compact-jump"></select>
    </div>
  </header>

  <div class="review-layout">
    <aside class="patient-sidebar" aria-label="Patient jump list">
      <h2>Patients</h2>
      <nav id="patient-nav"></nav>
    </aside>
    <main id="patient-list" tabindex="-1">
      <div id="loading-panel" class="message-panel">Loading CT Series inventory…</div>
    </main>
  </div>

  <div id="lightbox" class="lightbox" role="dialog" aria-modal="true"
       aria-labelledby="lightbox-title" hidden>
    <div class="lightbox-panel">
      <div class="lightbox-toolbar">
        <h2 id="lightbox-title">Montage preview</h2>
        <div class="lightbox-actions">
          <button id="lightbox-fit" type="button" aria-pressed="true">Fit</button>
          <button id="lightbox-native" type="button" aria-pressed="false">100%</button>
          <button id="lightbox-close" type="button">Close</button>
        </div>
      </div>
      <div id="lightbox-viewport" class="lightbox-viewport">
        <img id="lightbox-image" alt="">
      </div>
    </div>
  </div>
</body>
</html>
"""


APP_CSS = """
:root {
  color-scheme: light;
  --ink: #17212b;
  --muted: #5a6875;
  --line: #ccd4db;
  --panel: #ffffff;
  --canvas: #eef2f4;
  --accent: #0b5f78;
  --accent-dark: #08465a;
  --ready: #146c43;
  --warning: #8a4b08;
  --danger: #a12828;
  --focus: #006fcc;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* {
  box-sizing: border-box;
}

html {
  min-width: 0;
  scroll-padding-top: 19rem;
}

body {
  min-width: 0;
  margin: 0;
  overflow-x: hidden;
  color: var(--ink);
  background: var(--canvas);
  line-height: 1.45;
}

button,
input,
select,
summary {
  font: inherit;
}

button,
select,
input[type="search"] {
  min-height: 44px;
}

button {
  border: 1px solid #84939f;
  border-radius: 0.4rem;
  padding: 0.55rem 0.8rem;
  color: var(--ink);
  background: #fff;
  cursor: pointer;
}

button:hover:not(:disabled) {
  border-color: var(--accent);
  background: #e8f2f5;
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

:focus-visible {
  outline: 3px solid var(--focus);
  outline-offset: 3px;
}

.skip-link {
  position: fixed;
  z-index: 100;
  top: 0.5rem;
  left: 0.5rem;
  padding: 0.75rem;
  color: #fff;
  background: #000;
  transform: translateY(-150%);
}

.skip-link:focus {
  transform: none;
}

.app-header {
  position: sticky;
  z-index: 20;
  top: 0;
  padding: 0.8rem clamp(0.75rem, 2vw, 1.5rem);
  border-bottom: 1px solid #aeb9c1;
  background: rgba(255, 255, 255, 0.98);
  box-shadow: 0 2px 8px rgba(20, 35, 45, 0.1);
}

.title-row,
.toolbar,
.result-row,
.patient-heading,
.card-heading,
.selection-row,
.save-actions,
.lightbox-toolbar,
.lightbox-actions {
  display: flex;
  align-items: center;
  gap: 0.65rem;
}

.title-row,
.patient-heading,
.lightbox-toolbar {
  justify-content: space-between;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1 {
  margin-bottom: 0.15rem;
  font-size: clamp(1.35rem, 3vw, 2rem);
}

.eyebrow {
  margin-bottom: 0.1rem;
  color: var(--accent);
  font-size: 0.75rem;
  font-weight: 750;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.selection-file,
.workflow-note,
.result-row,
.status-detail,
.field-note {
  color: var(--muted);
}

.selection-file {
  max-width: 70ch;
  margin-bottom: 0;
  overflow-wrap: anywhere;
}

.progress-panel {
  display: grid;
  flex: 0 0 auto;
  gap: 0.15rem;
  text-align: right;
}

.workflow-note {
  margin: 0.55rem 0;
  font-size: 0.9rem;
}

.toolbar {
  align-items: end;
  flex-wrap: wrap;
}

.search-control {
  flex: 2 1 22rem;
  min-width: 0;
}

.search-control label,
.filter-control > span {
  display: block;
  margin-bottom: 0.2rem;
  font-size: 0.85rem;
  font-weight: 700;
}

.search-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.4rem;
}

input[type="search"] {
  min-width: 0;
  width: 100%;
  border: 1px solid #84939f;
  border-radius: 0.4rem;
  padding: 0.5rem 0.65rem;
}

.segmented {
  display: flex;
}

.segmented button {
  border-radius: 0;
}

.segmented button:first-child {
  border-radius: 0.4rem 0 0 0.4rem;
}

.segmented button:last-child {
  border-radius: 0 0.4rem 0.4rem 0;
}

.segmented button + button {
  margin-left: -1px;
}

.segmented button[aria-pressed="true"] {
  z-index: 1;
  border-color: var(--accent);
  color: #fff;
  background: var(--accent);
}

.save-control {
  display: grid;
  flex: 1 1 16rem;
  justify-items: end;
  gap: 0.2rem;
  margin-left: auto;
}

.save-actions {
  flex-wrap: wrap;
  justify-content: end;
}

.primary-action {
  border-color: var(--accent-dark);
  color: #fff;
  background: var(--accent);
  font-weight: 750;
}

.primary-action:hover:not(:disabled) {
  color: #fff;
  background: var(--accent-dark);
}

.result-row {
  justify-content: space-between;
  min-height: 1.6rem;
  margin-top: 0.4rem;
}

.compact-jump {
  display: none;
}

.review-layout {
  display: grid;
  grid-template-columns: minmax(11rem, 15rem) minmax(0, 1fr);
  gap: 1rem;
  width: min(100%, 112rem);
  margin: 0 auto;
  padding: 1rem clamp(0.75rem, 2vw, 1.5rem) 4rem;
}

.patient-sidebar {
  position: sticky;
  top: 18.5rem;
  align-self: start;
  max-height: calc(100vh - 20rem);
  overflow-y: auto;
  border: 1px solid var(--line);
  border-radius: 0.55rem;
  padding: 0.8rem;
  background: var(--panel);
}

.patient-sidebar h2 {
  margin-bottom: 0.4rem;
  font-size: 1rem;
}

.patient-sidebar nav {
  display: grid;
  gap: 0.2rem;
}

.patient-sidebar a {
  display: block;
  min-height: 44px;
  padding: 0.65rem;
  border-radius: 0.3rem;
  color: var(--accent-dark);
  overflow-wrap: anywhere;
}

.patient-sidebar a:hover {
  background: #e8f2f5;
}

#patient-list {
  min-width: 0;
}

.patient-section {
  min-width: 0;
  margin-bottom: 1.25rem;
  border: 1px solid #b7c1c9;
  border-radius: 0.65rem;
  background: var(--panel);
  box-shadow: 0 1px 4px rgba(30, 45, 55, 0.08);
}

.patient-heading {
  align-items: flex-start;
  padding: 0.85rem 1rem;
  border-bottom: 1px solid var(--line);
  background: #f9fbfc;
}

.patient-heading h2 {
  margin-bottom: 0.1rem;
  font-size: 1.25rem;
  overflow-wrap: anywhere;
}

.patient-folder {
  margin-bottom: 0;
  color: var(--muted);
  overflow-wrap: anywhere;
}

.patient-actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  justify-content: end;
  gap: 0.5rem;
}

.state-badge,
.eligibility-badge {
  display: inline-flex;
  align-items: center;
  min-height: 1.8rem;
  border: 1px solid currentColor;
  border-radius: 999px;
  padding: 0.15rem 0.55rem;
  font-size: 0.82rem;
  font-weight: 750;
}

.state-selected,
.eligibility-ready {
  color: var(--ready);
  background: #edf8f2;
}

.state-unselected,
.eligibility-not-selectable {
  color: var(--warning);
  background: #fff6e9;
}

.state-blocked {
  color: var(--danger);
  background: #fff0f0;
}

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr));
  gap: 0.9rem;
  padding: 0.9rem;
}

.series-card {
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 0.5rem;
  background: #fff;
}

.series-card.is-selected {
  border: 2px solid var(--ready);
}

.card-heading {
  align-items: flex-start;
  justify-content: space-between;
  padding: 0.75rem;
}

.card-heading h3 {
  margin-bottom: 0.1rem;
  font-size: 1rem;
  overflow-wrap: anywhere;
}

.selection-row {
  min-height: 44px;
  margin: 0 0.75rem 0.6rem;
  align-items: flex-start;
}

.selection-label {
  display: flex;
  min-height: 44px;
  align-items: center;
  gap: 0.5rem;
  font-weight: 700;
  cursor: pointer;
}

.selection-label input {
  flex: 0 0 auto;
  width: 1.35rem;
  height: 1.35rem;
  margin: 0;
}

.selection-label:has(input:disabled) {
  cursor: not-allowed;
}

.disabled-reason {
  color: var(--danger);
  font-size: 0.85rem;
}

.montage-button {
  display: block;
  width: calc(100% - 1.5rem);
  min-height: 8rem;
  margin: 0 0.75rem 0.75rem;
  overflow: hidden;
  border-color: #768590;
  padding: 0;
  background: #12191e;
}

.montage-button img {
  display: block;
  width: 100%;
  height: auto;
  object-fit: contain;
}

.preview-placeholder {
  display: grid;
  min-height: 12rem;
  place-content: center;
  gap: 0.4rem;
  padding: 1rem;
  color: #f2f5f6;
  background: #26343e;
  text-align: center;
  overflow-wrap: anywhere;
}

.preview-placeholder strong {
  font-size: 1.05rem;
}

.metadata {
  display: grid;
  grid-template-columns: minmax(7.5rem, 0.75fr) minmax(0, 1.25fr);
  margin: 0;
  padding: 0 0.75rem 0.75rem;
  gap: 0.3rem 0.7rem;
}

.metadata dt {
  color: var(--muted);
  font-weight: 650;
}

.metadata dd {
  min-width: 0;
  margin: 0;
  overflow-wrap: anywhere;
}

.warning-value {
  color: var(--warning);
  font-weight: 700;
}

.audit-details {
  margin: 0 0.75rem 0.75rem;
  border-top: 1px solid var(--line);
  padding-top: 0.55rem;
}

.audit-details summary {
  min-height: 44px;
  padding: 0.55rem 0;
  color: var(--accent-dark);
  cursor: pointer;
  font-weight: 700;
}

.blocked-panel,
.message-panel,
.no-results {
  margin: 0.9rem;
  border: 1px solid #d6aeae;
  border-radius: 0.45rem;
  padding: 1rem;
  background: #fff5f5;
  overflow-wrap: anywhere;
}

.message-panel,
.no-results {
  border-color: var(--line);
  background: var(--panel);
}

.lightbox {
  position: fixed;
  z-index: 50;
  inset: 0;
  padding: max(0.5rem, env(safe-area-inset-top));
  background: rgba(5, 10, 14, 0.92);
}

.lightbox-panel {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  width: 100%;
  height: 100%;
  border-radius: 0.5rem;
  overflow: hidden;
  background: #17212b;
}

.lightbox-toolbar {
  min-width: 0;
  padding: 0.6rem;
  color: #fff;
  background: #26343e;
}

.lightbox-toolbar h2 {
  min-width: 0;
  margin: 0;
  font-size: 1rem;
  overflow-wrap: anywhere;
}

.lightbox-actions {
  flex: 0 0 auto;
}

.lightbox-actions button[aria-pressed="true"] {
  border-color: #8bd8f0;
  color: #fff;
  background: var(--accent);
}

.lightbox-viewport {
  min-width: 0;
  min-height: 0;
  overflow: auto;
  background: #05080a;
}

.lightbox-viewport.fit {
  position: relative;
  overflow: hidden;
}

.lightbox-image.fit {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.lightbox-image.native {
  position: static;
  display: block;
  width: auto;
  height: auto;
  max-width: none;
  max-height: none;
}

body.lightbox-open {
  overflow: hidden;
}

[hidden] {
  display: none !important;
}

@media (max-width: 1100px) {
  html {
    scroll-padding-top: 22rem;
  }

  .review-layout {
    display: block;
  }

  .patient-sidebar {
    display: none;
  }

  .compact-jump {
    display: inline-block;
  }

  #patient-jump {
    max-width: min(65vw, 24rem);
  }
}

@media (max-width: 620px) {
  html {
    scroll-padding-top: 2rem;
  }

  .app-header {
    position: static;
  }

  .title-row,
  .patient-heading,
  .lightbox-toolbar {
    align-items: stretch;
    flex-direction: column;
  }

  .progress-panel,
  .save-control {
    justify-items: start;
    text-align: left;
  }

  .toolbar > *,
  .secondary-action,
  .save-control,
  .save-actions,
  .save-actions button {
    width: 100%;
  }

  .segmented,
  .segmented button {
    flex: 1 1 0;
  }

  .patient-actions {
    justify-content: start;
  }

  .cards {
    grid-template-columns: minmax(0, 1fr);
    padding: 0.6rem;
  }

  .metadata {
    grid-template-columns: minmax(0, 1fr);
  }

  .metadata dd {
    margin-bottom: 0.35rem;
  }

  .lightbox-actions {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    width: 100%;
  }
}

@media (max-width: 360px) {
  .app-header,
  .review-layout {
    padding-right: 0.5rem;
    padding-left: 0.5rem;
  }

  .search-row {
    grid-template-columns: minmax(0, 1fr);
  }

  .segmented {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
  }

  .segmented button,
  .segmented button:first-child,
  .segmented button:last-child {
    margin: 0;
    border-radius: 0.4rem;
  }
}
"""


APP_JS = r"""
"use strict";

const dom = {
  selectionFile: document.querySelector("#selection-file"),
  selectionProgress: document.querySelector("#selection-progress"),
  issueProgress: document.querySelector("#issue-progress"),
  search: document.querySelector("#patient-search"),
  clearSearch: document.querySelector("#clear-search"),
  filterButtons: Array.from(document.querySelectorAll("[data-filter]")),
  next: document.querySelector("#next-unselected"),
  save: document.querySelector("#save-selections"),
  reload: document.querySelector("#reload-inventory"),
  saveState: document.querySelector("#save-state"),
  resultCount: document.querySelector("#result-count"),
  patientList: document.querySelector("#patient-list"),
  patientNav: document.querySelector("#patient-nav"),
  patientJump: document.querySelector("#patient-jump"),
  lightbox: document.querySelector("#lightbox"),
  lightboxTitle: document.querySelector("#lightbox-title"),
  lightboxViewport: document.querySelector("#lightbox-viewport"),
  lightboxImage: document.querySelector("#lightbox-image"),
  lightboxFit: document.querySelector("#lightbox-fit"),
  lightboxNative: document.querySelector("#lightbox-native"),
  lightboxClose: document.querySelector("#lightbox-close"),
};

let inventory = null;
let revision = null;
let selections = new Map();
let persistedSelections = new Map();
let activeFilter = "all";
let saving = false;
let conflict = false;
let navigationIndex = -1;
let lastLightboxTrigger = null;
const unavailablePreviews = new Set();

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function valueOrDash(value) {
  const text = String(value || "").trim();
  return text || "—";
}

function selectionIsDirty() {
  if (!inventory) return false;
  return inventory.patients.some(
    (patient) => selections.get(patient.patient_id) !== persistedSelections.get(patient.patient_id),
  );
}

function currentPatientState(patient) {
  if (patient.ready_candidate_count === 0) return "blocked";
  return selections.get(patient.patient_id) ? "selected" : "unselected";
}

function candidateHasUnavailablePreview(candidate) {
  return candidate.preview_status !== "ready"
    || !candidate.preview_path
    || !candidate.preview_url
    || unavailablePreviews.has(candidate.candidate_id);
}

function patientHasIssue(patient) {
  return patient.ready_candidate_count === 0 || patient.candidates.some(
    (candidate) => candidate.status === "not_selectable"
      || Boolean(candidate.geometry_warnings)
      || candidateHasUnavailablePreview(candidate),
  );
}

function searchText(patient) {
  const candidateFields = [
    "study_instance_uid",
    "series_instance_uid",
    "study_description",
    "series_description",
    "protocol_name",
    "source_directories",
  ];
  const values = [patient.patient_id, patient.dicom_folder];
  patient.candidates.forEach((candidate) => {
    candidateFields.forEach((field) => values.push(candidate[field]));
  });
  return values.join(" ").toLocaleLowerCase();
}

function visiblePatients() {
  const query = dom.search.value.trim().toLocaleLowerCase();
  return inventory.patients.filter((patient) => {
    const state = currentPatientState(patient);
    const matchesFilter = activeFilter === "all"
      || (activeFilter === "unselected" && state === "unselected")
      || (activeFilter === "issues" && patientHasIssue(patient));
    return matchesFilter && (!query || searchText(patient).includes(query));
  });
}

function updateHeader() {
  if (!inventory) return;
  const selectedCount = inventory.patients.filter(
    (patient) => selections.get(patient.patient_id) !== null,
  ).length;
  const counts = inventory.counts;
  dom.selectionProgress.textContent = [
    selectedCount,
    " / ",
    counts.selectable_patient_count,
    " selected",
  ].join("");
  dom.issueProgress.textContent = [
    counts.blocked_patient_count,
    " blocked / ",
    counts.issue_patient_count,
    " issues",
  ].join("");
  dom.clearSearch.disabled = !dom.search.value;
  dom.save.disabled = saving || conflict || !selectionIsDirty();
  dom.save.textContent = saving ? "Saving…" : "Save selections";
  if (!saving && !conflict) {
    dom.saveState.textContent = selectionIsDirty() ? "Unsaved changes" : "Saved";
  }
}

function setFilter(filter) {
  activeFilter = filter;
  dom.filterButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.filter === filter));
  });
  renderPatientList();
}

function addMetadataRow(list, label, value, className) {
  const term = element("dt", null, label);
  const detail = element("dd", className || null, valueOrDash(value));
  list.append(term, detail);
}

function previewPlaceholder(reason) {
  const placeholder = element("div", "preview-placeholder");
  placeholder.append(
    element("strong", null, "Preview unavailable"),
    element("span", null, valueOrDash(reason)),
  );
  return placeholder;
}

function openLightbox(candidate, trigger) {
  lastLightboxTrigger = trigger;
  const seriesNumber = valueOrDash(candidate.series_number);
  dom.lightboxTitle.textContent = `${candidate.patient_id} — Series ${seriesNumber}`;
  const priorError = dom.lightboxViewport.querySelector("[data-lightbox-error]");
  if (priorError) priorError.remove();
  dom.lightboxImage.hidden = false;
  dom.lightboxImage.src = candidate.preview_url;
  dom.lightboxImage.alt = [
    "Full-resolution CT montage for ",
    candidate.patient_id,
    ", Series ",
    seriesNumber,
  ].join("");
  setLightboxMode("fit");
  dom.lightbox.hidden = false;
  document.body.classList.add("lightbox-open");
  dom.lightboxClose.focus();
}

function closeLightbox() {
  if (dom.lightbox.hidden) return;
  dom.lightbox.hidden = true;
  dom.lightboxImage.removeAttribute("src");
  dom.lightboxImage.hidden = false;
  const priorError = dom.lightboxViewport.querySelector("[data-lightbox-error]");
  if (priorError) priorError.remove();
  document.body.classList.remove("lightbox-open");
  if (lastLightboxTrigger) lastLightboxTrigger.focus();
  lastLightboxTrigger = null;
}

function setLightboxMode(mode) {
  const fit = mode === "fit";
  dom.lightboxViewport.className = `lightbox-viewport ${fit ? "fit" : "native"}`;
  dom.lightboxImage.className = `lightbox-image ${fit ? "fit" : "native"}`;
  dom.lightboxFit.setAttribute("aria-pressed", String(fit));
  dom.lightboxNative.setAttribute("aria-pressed", String(!fit));
  dom.lightboxViewport.scrollTo(0, 0);
}

function renderPreview(candidate) {
  if (
    candidate.preview_status !== "ready"
    || !candidate.preview_path
    || !candidate.preview_url
  ) {
    const reason = candidate.preview_reason
      || (candidate.preview_status === "ready"
        ? "Preview file is missing."
        : "No preview was generated.");
    return previewPlaceholder(reason);
  }
  const button = element("button", "montage-button");
  button.type = "button";
  button.setAttribute("aria-label", `Open full-resolution montage for ${candidate.patient_id}`);
  const image = element("img");
  image.src = candidate.preview_url;
  image.alt = [
    "Complete 3 by 3 axial montage for ",
    candidate.patient_id,
    ", Series ",
    valueOrDash(candidate.series_number),
  ].join("");
  image.loading = "lazy";
  image.addEventListener("error", () => {
    unavailablePreviews.add(candidate.candidate_id);
    button.replaceWith(
      previewPlaceholder(candidate.preview_reason || "The preview file could not be loaded."),
    );
    updateHeader();
  }, {once: true});
  button.append(image);
  button.addEventListener("click", () => openLightbox(candidate, button));
  return button;
}

const primaryMetadata = [
  ["Eligibility", "status"],
  ["Eligibility reason", "reason"],
  ["Preview status", "preview_status"],
  ["Preview reason", "preview_reason"],
  ["Series number", "series_number"],
  ["Acquisition number", "acquisition_number"],
  ["Series description", "series_description"],
  ["Protocol", "protocol_name"],
  ["Study date", "study_date"],
  ["Source directories", "source_directories"],
  ["Contrast agent", "contrast_bolus_agent"],
  ["Source files", "source_file_count"],
  ["Unique files", "unique_file_count"],
  ["Duplicate files", "duplicate_file_count"],
  ["Selected slice range", "image_range"],
  ["Selected slice count", "selected_file_count"],
  ["Slice thickness (mm)", "slice_thickness_mm"],
  ["Row pixel spacing (mm)", "row_spacing_mm"],
  ["Column pixel spacing (mm)", "column_spacing_mm"],
  ["Median slice spacing (mm)", "median_slice_spacing_mm"],
  ["Maximum slice gap (mm)", "maximum_slice_gap_mm"],
  ["Geometry warnings", "geometry_warnings"],
];

const detailMetadata = [
  ["Candidate ID", "candidate_id"],
  ["Study Instance UID", "study_instance_uid"],
  ["Series Instance UID", "series_instance_uid"],
  ["SOP Class UID", "sop_class_uid"],
  ["Series SOP UID hash", "series_sop_uids_sha256"],
  ["Study description", "study_description"],
  ["Body part examined", "body_part_examined"],
  ["Rows", "rows"],
  ["Columns", "columns"],
  ["Preview path", "preview_path"],
  ["Patient ID", "patient_id"],
  ["DICOM folder", "dicom_folder"],
];

function renderCandidate(patient, candidate, patientIndex, candidateIndex) {
  const selected = selections.get(patient.patient_id) === candidate.candidate_id;
  const card = element("article", `series-card${selected ? " is-selected" : ""}`);
  const heading = element("div", "card-heading");
  const titleGroup = element("div");
  const title = element(
    "h3",
    null,
    `Series ${valueOrDash(candidate.series_number)} — ${valueOrDash(candidate.series_description)}`,
  );
  titleGroup.append(title);
  const eligibilityText = candidate.status === "ready" ? "Ready" : "Not selectable";
  const eligibility = element(
    "span",
    `eligibility-badge eligibility-${candidate.status.replace("_", "-")}`,
    eligibilityText,
  );
  heading.append(titleGroup, eligibility);
  card.append(heading);

  const selectionRow = element("div", "selection-row");
  const label = element("label", "selection-label");
  const radio = element("input");
  radio.type = "radio";
  radio.name = `patient-${patientIndex}`;
  radio.id = `candidate-${patientIndex}-${candidateIndex}`;
  radio.value = candidate.candidate_id;
  radio.checked = selected;
  radio.disabled = candidate.status !== "ready";
  radio.addEventListener("change", () => {
    selections.set(patient.patient_id, candidate.candidate_id);
    renderPatientList(candidate.candidate_id);
  });
  label.setAttribute("for", radio.id);
  label.append(radio, element("span", null, selected ? "Selected Series" : "Select this Series"));
  selectionRow.append(label);
  if (candidate.status !== "ready") {
    selectionRow.append(
      element("span", "disabled-reason", valueOrDash(candidate.reason)),
    );
  }
  card.append(selectionRow, renderPreview(candidate));

  const metadata = element("dl", "metadata");
  primaryMetadata.forEach(([labelText, field]) => {
    let displayValue = candidate[field];
    let className = "";
    if (field === "geometry_warnings" && displayValue && candidate.status === "ready") {
      displayValue = `${displayValue} (non-blocking)`;
      className = "warning-value";
    }
    addMetadataRow(metadata, labelText, displayValue, className);
  });
  card.append(metadata);

  const details = element("details", "audit-details");
  details.append(element("summary", null, "Full audit details"));
  const detailList = element("dl", "metadata");
  detailMetadata.forEach(([labelText, field]) => {
    addMetadataRow(detailList, labelText, candidate[field]);
  });
  details.append(detailList);
  card.append(details);
  return card;
}

function renderPatient(patient, patientIndex) {
  const section = element("section", "patient-section");
  section.id = `patient-${patientIndex}`;
  section.dataset.patientId = patient.patient_id;
  const heading = element("div", "patient-heading");
  const titleGroup = element("div");
  const title = element("h2", null, patient.patient_id);
  title.tabIndex = -1;
  title.dataset.patientHeading = patient.patient_id;
  titleGroup.append(title, element("p", "patient-folder", patient.dicom_folder));

  const actions = element("div", "patient-actions");
  const state = currentPatientState(patient);
  const stateLabel = state.charAt(0).toUpperCase() + state.slice(1);
  actions.append(element("span", `state-badge state-${state}`, stateLabel));
  if (patient.ready_candidate_count > 0) {
    const clear = element("button", null, "Clear selection");
    clear.type = "button";
    clear.disabled = selections.get(patient.patient_id) === null;
    clear.addEventListener("click", () => {
      selections.set(patient.patient_id, null);
      renderPatientList();
      const refreshed = document.querySelector(
        `[data-patient-heading="${CSS.escape(patient.patient_id)}"]`,
      );
      if (refreshed) refreshed.focus();
    });
    actions.append(clear);
  }
  heading.append(titleGroup, actions);
  section.append(heading);

  if (patient.ready_candidate_count === 0 && patient.candidates.length === 0) {
    section.append(element("div", "blocked-panel", patient.blocked_reason));
  } else {
    if (patient.ready_candidate_count === 0) {
      section.append(element("div", "blocked-panel", patient.blocked_reason));
    }
    const cards = element("div", "cards");
    patient.candidates.forEach((candidate, candidateIndex) => {
      cards.append(renderCandidate(patient, candidate, patientIndex, candidateIndex));
    });
    section.append(cards);
  }
  return section;
}

function renderJumpControls() {
  dom.patientNav.replaceChildren();
  dom.patientJump.replaceChildren();
  const prompt = element("option", null, "Choose patient…");
  prompt.value = "";
  dom.patientJump.append(prompt);
  inventory.patients.forEach((patient, index) => {
    const link = element("a", null, patient.patient_id);
    link.href = `#patient-${index}`;
    dom.patientNav.append(link);
    const option = element("option", null, patient.patient_id);
    option.value = String(index);
    dom.patientJump.append(option);
  });
}

function renderPatientList(focusCandidateId) {
  if (!inventory) return;
  const patients = visiblePatients();
  dom.patientList.replaceChildren();
  patients.forEach((patient) => {
    const patientIndex = inventory.patients.indexOf(patient);
    dom.patientList.append(renderPatient(patient, patientIndex));
  });
  dom.resultCount.textContent = `${patients.length} of ${inventory.patients.length} patients shown`;
  if (patients.length === 0) {
    const panel = element("div", "no-results");
    panel.append(element("p", null, "No patients match the current search and filter."));
    const reset = element("button", null, "Reset search and filters");
    reset.type = "button";
    reset.addEventListener("click", () => {
      dom.search.value = "";
      setFilter("all");
      dom.search.focus();
    });
    panel.append(reset);
    dom.patientList.append(panel);
  }
  updateHeader();
  if (focusCandidateId) {
    const radio = Array.from(dom.patientList.querySelectorAll("input[type=radio]")).find(
      (input) => input.value === focusCandidateId,
    );
    if (radio) radio.focus();
  }
}

function revealPatient(patientIndex) {
  activeFilter = "all";
  dom.search.value = "";
  dom.filterButtons.forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.filter === "all"));
  });
  renderPatientList();
  const heading = document.querySelector(
    `[data-patient-heading="${CSS.escape(inventory.patients[patientIndex].patient_id)}"]`,
  );
  if (heading) {
    heading.scrollIntoView({behavior: "smooth", block: "start"});
    heading.focus({preventScroll: true});
  }
}

function nextUnselected() {
  const total = inventory.patients.length;
  for (let offset = 1; offset <= total; offset += 1) {
    const index = (navigationIndex + offset) % total;
    const patient = inventory.patients[index];
    if (patient.ready_candidate_count > 0 && !selections.get(patient.patient_id)) {
      navigationIndex = index;
      revealPatient(index);
      dom.saveState.textContent = `Moved to unselected patient ${patient.patient_id}.`;
      return;
    }
  }
  dom.saveState.textContent = "All selectable patients reviewed.";
}

async function responsePayload(response) {
  try {
    return await response.json();
  } catch (_error) {
    return {error: `Request failed with HTTP ${response.status}.`};
  }
}

async function saveSelections() {
  if (!selectionIsDirty() || saving || conflict) return;
  saving = true;
  dom.saveState.textContent = "Saving…";
  updateHeader();
  const submitted = Object.fromEntries(selections);
  try {
    const response = await fetch("/api/selections", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({expected_sha256: revision, selections: submitted}),
    });
    const payload = await responsePayload(response);
    if (!response.ok) {
      if (response.status === 409) {
        conflict = true;
        dom.reload.hidden = false;
        dom.saveState.textContent = "File changed on disk—reload required";
      } else {
        dom.saveState.textContent = payload.error || "Save failed.";
      }
      return;
    }
    revision = payload.selection_sha256;
    persistedSelections = new Map(selections);
    dom.saveState.textContent = "Saved";
  } catch (error) {
    dom.saveState.textContent = `Save failed: ${error.message}`;
  } finally {
    saving = false;
    updateHeader();
    renderPatientList();
  }
}

function reloadInventory() {
  if (selectionIsDirty()) {
    const discard = window.confirm(
      "Reloading will discard your unsaved selection changes. Continue?",
    );
    if (!discard) return;
  }
  window.location.reload();
}

function trapLightboxFocus(event) {
  if (event.key === "Escape") {
    event.preventDefault();
    closeLightbox();
    return;
  }
  if (event.key !== "Tab") return;
  const focusable = [dom.lightboxFit, dom.lightboxNative, dom.lightboxClose];
  const currentIndex = focusable.indexOf(document.activeElement);
  const nextIndex = event.shiftKey
    ? (currentIndex <= 0 ? focusable.length - 1 : currentIndex - 1)
    : (currentIndex + 1) % focusable.length;
  event.preventDefault();
  focusable[nextIndex].focus();
}

async function loadInventory() {
  try {
    const response = await fetch(
      "/api/inventory",
      {headers: {"Accept": "application/json"}},
    );
    const payload = await responsePayload(response);
    if (!response.ok) {
      const message = payload.error || `Inventory failed with HTTP ${response.status}.`;
      throw new Error(message);
    }
    inventory = payload;
    revision = payload.selection_sha256;
    selections = new Map(
      inventory.patients.map((patient) => [patient.patient_id, patient.selected_candidate_id]),
    );
    persistedSelections = new Map(selections);
    const pathParts = inventory.selection_file.split(/[\\/]/);
    dom.selectionFile.textContent = `${pathParts.at(-1)} — ${inventory.selection_file}`;
    dom.saveState.textContent = "Saved";
    renderJumpControls();
    renderPatientList();
  } catch (error) {
    conflict = true;
    dom.selectionFile.textContent = "Inventory unavailable";
    dom.saveState.textContent = "Reload required";
    dom.patientList.replaceChildren(
      element("div", "message-panel", `Unable to load inventory: ${error.message}`),
    );
    dom.reload.hidden = false;
    updateHeader();
  }
}

dom.search.addEventListener("input", () => renderPatientList());
dom.clearSearch.addEventListener("click", () => {
  dom.search.value = "";
  renderPatientList();
  dom.search.focus();
});
dom.filterButtons.forEach((button) => {
  button.addEventListener("click", () => setFilter(button.dataset.filter));
});
dom.next.addEventListener("click", nextUnselected);
dom.save.addEventListener("click", saveSelections);
dom.reload.addEventListener("click", reloadInventory);
dom.patientJump.addEventListener("change", () => {
  if (dom.patientJump.value) revealPatient(Number(dom.patientJump.value));
  dom.patientJump.value = "";
});
dom.lightboxFit.addEventListener("click", () => setLightboxMode("fit"));
dom.lightboxNative.addEventListener("click", () => setLightboxMode("native"));
dom.lightboxClose.addEventListener("click", closeLightbox);
dom.lightbox.addEventListener("click", (event) => {
  if (event.target === dom.lightbox) closeLightbox();
});
dom.lightbox.addEventListener("keydown", trapLightboxFocus);
dom.lightboxImage.addEventListener("error", () => {
  dom.lightboxTitle.textContent = "Preview unavailable";
  dom.lightboxImage.hidden = true;
  const placeholder = previewPlaceholder("The preview file could not be loaded.");
  placeholder.dataset.lightboxError = "true";
  dom.lightboxViewport.append(placeholder);
});
window.addEventListener("beforeunload", (event) => {
  if (!selectionIsDirty()) return;
  event.preventDefault();
  event.returnValue = "";
});

loadInventory();
"""
