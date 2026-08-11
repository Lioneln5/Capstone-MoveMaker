import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(?:[A-Za-z]:)/, (m) => m.slice(1))), "..");
const outputDir = path.join(root, "outputs", "statsbomb_case_studies");
const payload = JSON.parse(await fs.readFile(path.join(outputDir, "workbook_payload.json"), "utf8"));
await fs.mkdir(path.join(outputDir, "previews"), { recursive: true });

const workbook = Workbook.create();
const COLORS = {
  navy: "#16324F",
  blue: "#1F6E8C",
  teal: "#2E8B8B",
  paleBlue: "#EAF3F7",
  paleGreen: "#EAF6F2",
  paleAmber: "#FFF4D6",
  paleRed: "#FCE8E6",
  white: "#FFFFFF",
  text: "#243447",
  muted: "#5F6B76",
  line: "#D9E2E8",
};

function colLetter(index) {
  let value = index + 1;
  let result = "";
  while (value) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

function typedValue(column, value) {
  if (value === null || value === undefined || value === "") return null;
  if (column.includes("date") && /^\d{4}-\d{2}-\d{2}/.test(String(value))) {
    return new Date(`${String(value).slice(0, 10)}T00:00:00`);
  }
  return value;
}

function setWidths(sheet, widths) {
  widths.forEach((width, index) => {
    sheet.getRange(`${colLetter(index)}:${colLetter(index)}`).format.columnWidth = width;
  });
}

function styleTitle(sheet, lastColumn, title, subtitle) {
  sheet.mergeCells(`A1:${lastColumn}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 18 },
    rowHeight: 32,
    verticalAlignment: "center",
  };
  sheet.mergeCells(`A2:${lastColumn}2`);
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: COLORS.paleBlue,
    font: { color: COLORS.muted, italic: true, size: 10 },
    rowHeight: 26,
    verticalAlignment: "center",
    wrapText: true,
  };
}

function addTableSheet({ name, title, subtitle, columns, rows, widths, tableName }) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const lastColumn = colLetter(columns.length - 1);
  styleTitle(sheet, lastColumn, title, subtitle);
  const matrix = [columns.map((column) => column.label)];
  for (const row of rows) {
    matrix.push(columns.map((column) => typedValue(column.key, row[column.key])));
  }
  sheet.getRange(`A4:${lastColumn}${3 + matrix.length}`).values = matrix;
  const header = sheet.getRange(`A4:${lastColumn}4`);
  header.format = {
    fill: COLORS.blue,
    font: { bold: true, color: COLORS.white },
    rowHeight: 26,
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: COLORS.navy },
  };
  if (rows.length) {
    const table = sheet.tables.add(`A4:${lastColumn}${4 + rows.length}`, true, tableName);
    table.style = "TableStyleMedium2";
    table.showBandedColumns = false;
    table.showFilterButton = true;
  }
  sheet.getRange(`A5:${lastColumn}${Math.max(5, 4 + rows.length)}`).format = {
    font: { color: COLORS.text, size: 9 },
    verticalAlignment: "top",
  };
  for (let index = 0; index < columns.length; index += 1) {
    const column = columns[index];
    const range = sheet.getRange(`${colLetter(index)}5:${colLetter(index)}${Math.max(5, 4 + rows.length)}`);
    if (column.format) range.format.numberFormat = column.format;
    if (column.wrap) range.format.wrapText = true;
  }
  setWidths(sheet, widths);
  if (rows.length && columns.some((column) => column.wrap)) {
    sheet.getRange(`A5:${lastColumn}${4 + rows.length}`).format.autofitRows();
  }
  sheet.freezePanes.freezeRows(4);
  return sheet;
}

// Overview
const overview = workbook.worksheets.add("Overview");
overview.showGridLines = false;
styleTitle(
  overview,
  "H",
  "StatsBomb Case-Study Extract",
  "Compact, reproducible evidence for MoveMaker case studies; not a complete modern league panel."
);
overview.getRange("A4:B4").values = [["Metric", "Value"]];
overview.getRange("A4:B4").format = {
  fill: COLORS.blue,
  font: { bold: true, color: COLORS.white },
  rowHeight: 25,
};
overview.getRange("A5:B15").values = [
  ["Cataloged matches", payload.summary.cataloged_matches],
  ["Selected matches", payload.summary.selected_matches],
  ["Selected share", null],
  ["Slim event rows", payload.summary.selected_events],
  ["Player-match rows", payload.summary.player_match_rows],
  ["Team-match rows", payload.summary.team_match_rows],
  ["Player-period rows", payload.summary.player_period_rows],
  ["Team-period rows", payload.summary.team_period_rows],
  ["Compact 360 rows", payload.summary.selected_360_context_rows],
  ["Transfer candidates", payload.summary.transfer_case_study_candidates],
  ["Blocking checks failed", payload.summary.blocking_checks_failed],
];
overview.getRange("B7").formulas = [["=B6/B5"]];
overview.getRange("B5:B6").format.numberFormat = "#,##0";
overview.getRange("B7").format.numberFormat = "0.0%";
overview.getRange("B8:B15").format.numberFormat = "#,##0";
overview.getRange("A5:B15").format.borders = {
  insideHorizontal: { style: "thin", color: COLORS.line },
  outside: { style: "thin", color: COLORS.line },
};
overview.getRange("D4:H4").merge();
overview.getRange("D4").values = [["Recommended analytical use"]];
overview.getRange("D4:H4").format = { fill: COLORS.teal, font: { bold: true, color: COLORS.white }, rowHeight: 25 };
overview.getRange("D5:H10").merge();
overview.getRange("D5").values = [[
  "Use the shortlist to choose reviewable transfer stories, player/team period tables for profile comparisons, match tables for drilldowns, and compressed event/360 files for visualizations. Keep this layer optional: selection bias and sparse club overlap make it unsuitable as the mandatory input to the main compatibility model."
]];
overview.getRange("D5:H10").format = {
  fill: COLORS.paleGreen,
  font: { color: COLORS.text, size: 11 },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: COLORS.teal },
};
overview.getRange("D12:H12").merge();
overview.getRange("D12").values = [["Selection and leakage rules"]];
overview.getRange("D12:H12").format = { fill: COLORS.blue, font: { bold: true, color: COLORS.white }, rowHeight: 25 };
overview.getRange("D13:H19").merge();
overview.getRange("D13").values = [[
  `${payload.summary.selection_rule} Cross-source player and club links remain review-only. Pre-transfer windows exclude the transfer date; post-transfer matches and ROI fields are diagnostic targets and must never be used as predictors.`
]];
overview.getRange("D13:H19").format = {
  fill: COLORS.paleAmber,
  font: { color: COLORS.text, size: 10 },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: "#D6A84B" },
};
overview.getRange("A18:B20").values = [
  ["Scope start", new Date(`${payload.summary.scope_start}T00:00:00`)],
  ["Scope end", new Date(`${payload.summary.scope_end}T00:00:00`)],
  ["Generated", new Date(payload.summary.generated_at)],
];
overview.getRange("B18:B19").format.numberFormat = "yyyy-mm-dd";
overview.getRange("B20").format.numberFormat = "yyyy-mm-dd hh:mm";
overview.getRange("A18:A20").format.font = { bold: true, color: COLORS.muted };
setWidths(overview, [28, 18, 3, 18, 18, 18, 18, 18]);
overview.freezePanes.freezeRows(2);

addTableSheet({
  name: "Coverage",
  title: "Competition and Season Coverage",
  subtitle: "Selected matches are cataloged male matches in the extraction window. A zero means the competition-season remains cataloged but was excluded.",
  tableName: "CoverageTable",
  rows: payload.coverage,
  columns: [
    { key: "country_name", label: "Country" },
    { key: "competition_name", label: "Competition" },
    { key: "season_name", label: "Season" },
    { key: "competition_gender", label: "Gender" },
    { key: "competition_international", label: "International" },
    { key: "cataloged_matches", label: "Cataloged Matches", format: "#,##0" },
    { key: "selected_matches", label: "Selected Matches", format: "#,##0" },
    { key: "selected_events", label: "Selected Events", format: "#,##0" },
    { key: "selected_360_matches", label: "Selected 360 Matches", format: "#,##0" },
    { key: "selection_share", label: "Selection Share", format: "0.0%" },
  ],
  widths: [20, 28, 13, 11, 14, 17, 16, 17, 20, 15],
});

addTableSheet({
  name: "Top Cases",
  title: "Top Transfer Case-Study Candidates",
  subtitle: "Ranked for evidence availability only. Every identity link requires manual review; post-transfer fields are outcome diagnostics.",
  tableName: "TopCasesTable",
  rows: payload.top_cases,
  columns: [
    { key: "player_name", label: "Player" },
    { key: "transfer_date", label: "Transfer Date", format: "yyyy-mm-dd" },
    { key: "from_club_name", label: "From Club" },
    { key: "to_club_name", label: "To Club" },
    { key: "player_matches_pre365", label: "Player Matches Pre-365", format: "#,##0" },
    { key: "player_matches_post365", label: "Player Matches Post-365", format: "#,##0" },
    { key: "origin_matches_pre365", label: "Origin Matches Pre-365", format: "#,##0" },
    { key: "destination_matches_pre365", label: "Destination Matches Pre-365", format: "#,##0" },
    { key: "case_study_score", label: "Evidence Score", format: "#,##0" },
    { key: "recommended_case_study_use", label: "Recommended Use", wrap: true },
    { key: "player_identity_country_agreement", label: "Country Agreement" },
    { key: "manual_review_status", label: "Review Status" },
    { key: "post24_dest_minutes", label: "Post-24 Destination Minutes", format: "#,##0.0" },
    { key: "fee_value_roi_proxy_24m", label: "24m Fee/Value Proxy", format: "0.00" },
  ],
  widths: [22, 14, 22, 22, 18, 19, 18, 20, 13, 30, 16, 15, 20, 17],
});

const qualityRows = [
  ...payload.checks.map((row) => ({
    record_type: "Check",
    name: row.check_name,
    status: row.status,
    blocking: row.blocking,
    observed: row.observed,
    expectation_or_details: row.expectation,
  })),
  ...payload.quality.map((row) => ({
    record_type: "Issue",
    name: row.issue_type,
    status: row.severity,
    blocking: false,
    observed: row.match_id,
    expectation_or_details: row.details,
  })),
];
const qualitySheet = addTableSheet({
  name: "Quality",
  title: "Extraction Checks and Known Limitations",
  subtitle: "Blocking checks must pass. Repository-level warnings are retained even when they do not affect the selected extract.",
  tableName: "QualityTable",
  rows: qualityRows,
  columns: [
    { key: "record_type", label: "Type" },
    { key: "name", label: "Check / Issue" },
    { key: "status", label: "Status" },
    { key: "blocking", label: "Blocking" },
    { key: "observed", label: "Observed" },
    { key: "expectation_or_details", label: "Expectation / Details", wrap: true },
  ],
  widths: [12, 34, 12, 12, 16, 70],
});
qualitySheet.getRange(`C5:C${4 + qualityRows.length}`).conditionalFormats.add("containsText", {
  text: "PASS",
  format: { fill: COLORS.paleGreen, font: { color: "#176B4D", bold: true } },
});
qualitySheet.getRange(`C5:C${4 + qualityRows.length}`).conditionalFormats.add("containsText", {
  text: "FAIL",
  format: { fill: COLORS.paleRed, font: { color: "#A12622", bold: true } },
});

addTableSheet({
  name: "Artifacts",
  title: "Processed Artifact Manifest",
  subtitle: "Row counts, file sizes, and SHA-256 hashes make every extract auditable and reproducible.",
  tableName: "ArtifactsTable",
  rows: payload.manifest,
  columns: [
    { key: "artifact", label: "Artifact" },
    { key: "relative_path", label: "Relative Path", wrap: true },
    { key: "format", label: "Format" },
    { key: "rows", label: "Rows", format: "#,##0" },
    { key: "bytes", label: "Bytes", format: "#,##0" },
    { key: "sha256", label: "SHA-256", wrap: true },
  ],
  widths: [36, 52, 12, 16, 17, 68],
});

addTableSheet({
  name: "Feature Guide",
  title: "Key Feature Definitions",
  subtitle: "Selected high-value fields. The complete machine-readable dictionary is in Data/processed/statsbomb_case_studies/field_dictionary.csv.",
  tableName: "FeatureGuideTable",
  rows: payload.dictionary,
  columns: [
    { key: "table_name", label: "Table" },
    { key: "column_name", label: "Column" },
    { key: "dtype", label: "Type" },
    { key: "definition", label: "Definition", wrap: true },
    { key: "modeling_role", label: "Modeling Role" },
    { key: "timing_rule", label: "Timing Rule", wrap: true },
  ],
  widths: [32, 30, 16, 58, 22, 58],
});

addTableSheet({
  name: "Event Types",
  title: "Selected Event-Type Counts",
  subtitle: "The twenty most frequent event types in the slim case-study stream.",
  tableName: "EventTypesTable",
  rows: payload.event_types,
  columns: [
    { key: "event_type", label: "Event Type" },
    { key: "selected_event_count", label: "Selected Event Count", format: "#,##0" },
  ],
  widths: [30, 24],
});

const overviewInspect = await workbook.inspect({
  kind: "table",
  range: "Overview!A1:H20",
  include: "values,formulas",
  tableMaxRows: 25,
  tableMaxCols: 10,
});
console.log(overviewInspect.ndjson);

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 },
  summary: "final formula error scan",
});
console.log(formulaErrors.ndjson);

for (const sheetName of ["Overview", "Coverage", "Top Cases", "Quality", "Artifacts", "Feature Guide", "Event Types"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(
    path.join(outputDir, "previews", `${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await preview.arrayBuffer())
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir, "StatsBomb_Case_Study_Audit.xlsx"));
console.log(JSON.stringify({ output: path.join(outputDir, "StatsBomb_Case_Study_Audit.xlsx") }));
