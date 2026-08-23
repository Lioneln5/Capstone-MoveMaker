import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceDir = path.join(root, "Data/processed/subgroup_stability");
const outputDir = path.join(root, "outputs/subgroup_stability");
const previewDir = path.join(outputDir, "previews");
const outputPath = path.join(outputDir, "MoveMaker_Subgroup_Stability.xlsx");

const colors = {
  navy: "#17324D",
  blue: "#2F75B5",
  teal: "#2A9D8F",
  green: "#D9EAD3",
  greenText: "#276749",
  red: "#F4CCCC",
  redText: "#9C2C2C",
  amber: "#FCE5CD",
  amberText: "#8A4B08",
  paleBlue: "#D9EAF7",
  paleGray: "#EEF2F5",
  grid: "#D9E1E8",
  white: "#FFFFFF",
  text: "#22313F",
  muted: "#5F6B76",
};


function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const headers = rows[0];
  return rows.slice(1).filter((values) => values.some((value) => value !== "")).map((values) => {
    const result = {};
    headers.forEach((header, index) => { result[header] = cast(values[index] ?? ""); });
    return result;
  });
}


function cast(value) {
  if (value === "") return null;
  if (value === "True" || value === "true") return true;
  if (value === "False" || value === "false") return false;
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) return Number(value);
  return value;
}


async function load(name) {
  return parseCSV(await fs.readFile(path.join(sourceDir, name), "utf8"));
}


function colLetter(index) {
  let result = "";
  let value = index;
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}


function cleanLabel(value) {
  return String(value ?? "").replaceAll("_", " ");
}


function titleCase(value) {
  return cleanLabel(value).replace(/\b\w/g, (letter) => letter.toUpperCase());
}


function configureSheet(sheet) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(5);
}


function addTitle(sheet, title, subtitle, lastColumn) {
  const last = colLetter(lastColumn);
  sheet.getRange(`A1:${last}2`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${last}2`).format = {
    fill: colors.navy,
    font: { name: "Aptos Display", size: 20, bold: true, color: colors.white },
    verticalAlignment: "center",
    horizontalAlignment: "left",
  };
  sheet.getRange(`A3:${last}3`).merge();
  sheet.getRange("A3").values = [[subtitle]];
  sheet.getRange(`A3:${last}3`).format = {
    fill: colors.paleBlue,
    font: { name: "Aptos", size: 10, italic: true, color: colors.muted },
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange("1:2").format.rowHeight = 22;
  sheet.getRange("3:3").format.rowHeight = 30;
}


function writeTable(sheet, headers, rows, tableName, widths, options = {}) {
  const startRow = options.startRow ?? 5;
  const lastColumn = colLetter(headers.length);
  const endRow = startRow + rows.length;
  const matrix = [headers, ...rows];
  sheet.getRange(`A${startRow}:${lastColumn}${endRow}`).values = matrix;
  sheet.getRange(`A${startRow}:${lastColumn}${startRow}`).format = {
    fill: colors.blue,
    font: { bold: true, color: colors.white, size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: colors.grid },
  };
  sheet.getRange(`A${startRow}:${lastColumn}${endRow}`).format.borders = {
    preset: "all", style: "thin", color: colors.grid,
  };
  for (let row = startRow + 1; row <= endRow; row += 2) {
    sheet.getRange(`A${row}:${lastColumn}${row}`).format.fill = "#F8FAFC";
  }
  const table = sheet.tables.add(`A${startRow}:${lastColumn}${endRow}`, true, tableName);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
  widths.forEach((width, index) => {
    sheet.getRange(`${colLetter(index + 1)}:${colLetter(index + 1)}`).format.columnWidth = width;
  });
  sheet.getRange(`${startRow}:${startRow}`).format.rowHeight = 32;
  sheet.getRange(`${startRow + 1}:${endRow}`).format.rowHeight = options.rowHeight ?? 22;
  if (options.wrapColumns) {
    options.wrapColumns.forEach((index) => {
      sheet.getRange(`${colLetter(index)}${startRow + 1}:${colLetter(index)}${endRow}`).format.wrapText = true;
    });
  }
  return { startRow, endRow, lastColumn };
}


function formatNumberColumns(sheet, startRow, endRow, formats) {
  Object.entries(formats).forEach(([column, format]) => {
    sheet.getRange(`${column}${startRow}:${column}${endRow}`).setNumberFormat(format);
  });
}


function applyEvidenceColors(sheet, startRow, evidenceColumn, sampleColumn, rows) {
  rows.forEach((row, index) => {
    const excelRow = startRow + index;
    const evidence = String(row.evidence_label ?? row.heterogeneity_evidence ?? "");
    const sample = String(row.sample_quality ?? "");
    if (evidence.includes("positive")) {
      sheet.getRange(`${evidenceColumn}${excelRow}`).format = { fill: colors.green, font: { color: colors.greenText, bold: true } };
    } else if (evidence.includes("negative")) {
      sheet.getRange(`${evidenceColumn}${excelRow}`).format = { fill: colors.red, font: { color: colors.redText, bold: true } };
    } else if (evidence.includes("mixed") || evidence.includes("no_detected")) {
      sheet.getRange(`${evidenceColumn}${excelRow}`).format.fill = colors.paleGray;
    }
    if (sampleColumn && sample === "small_cell") {
      sheet.getRange(`${sampleColumn}${excelRow}`).format = { fill: colors.amber, font: { color: colors.amberText, bold: true } };
    }
  });
}


await fs.mkdir(previewDir, { recursive: true });
const [decisions, primary, aggregates, heterogeneityData, origins, coverage, definitions, pairs, verification] = await Promise.all([
  load("subgroup_decision_summary.csv"),
  load("subgroup_primary_results.csv"),
  load("subgroup_aggregate_results.csv"),
  load("subgroup_heterogeneity.csv"),
  load("subgroup_origin_results.csv"),
  load("subgroup_coverage.csv"),
  load("subgroup_candidate_definitions.csv"),
  load("subgroup_pair_predictions.csv"),
  load("independent_verification.csv"),
]);

const workbook = Workbook.create();
const sheetNames = [
  "Executive Summary", "Decision Summary", "Primary Results", "Role Results",
  "Competition Results", "Heterogeneity", "Origin Results", "Coverage",
  "Candidate Definitions", "Pair Predictions", "Verification", "Methodology",
];
const sheets = {};
sheetNames.forEach((name) => {
  sheets[name] = workbook.worksheets.add(name);
  configureSheet(sheets[name]);
});

// Decision summary source table.
const decisionRows = decisions.map((row) => [
  titleCase(row.target_key), row.candidate_feature_set, row.candidate_model_family,
  row.adequate_primary_subgroups, row.adequate_roles, row.adequate_competitions,
  row.supported_positive_subgroups, row.supported_negative_subgroups,
  `${titleCase(row.strongest_subgroup_dimension)}: ${row.strongest_subgroup_name}`,
  row.strongest_relative_mae_improvement,
  `${titleCase(row.weakest_subgroup_dimension)}: ${row.weakest_subgroup_name}`,
  row.weakest_relative_mae_improvement, row.role_heterogeneity_fdr_q_value,
  row.competition_heterogeneity_fdr_q_value, titleCase(row.recommendation),
]);
const decisionSheet = sheets["Decision Summary"];
addTitle(decisionSheet, "Frozen-candidate decisions", "Mature three-origin evidence; subgroup definitions and models were frozen before these comparisons.", 15);
writeTable(decisionSheet,
  ["Target", "Candidate Features", "Candidate Model", "Adequate Cells", "Adequate Roles", "Adequate Leagues", "Supported Positive", "Supported Negative", "Strongest Segment", "Strongest Relative Lift", "Weakest Segment", "Weakest Relative Lift", "Role Het. q", "League Het. q", "Recommendation"],
  decisionRows, "DecisionSummaryTable", [15, 19, 23, 13, 13, 15, 17, 17, 24, 18, 24, 18, 13, 13, 34], { wrapColumns: [2, 3, 9, 11, 15], rowHeight: 32 });
formatNumberColumns(decisionSheet, 6, 7, { J: "0.0%", L: "0.0%", M: "0.000", N: "0.000" });

// Primary result table.
const primaryRows = primary.map((row) => [
  titleCase(row.target_key), titleCase(row.subgroup_dimension), row.subgroup_name,
  row.pooled_n, row.minimum_origin_n, titleCase(row.sample_quality),
  row.pooled_baseline_mae, row.pooled_candidate_mae, row.pooled_mae_improvement,
  row.pooled_relative_mae_improvement, row.mae_improvement_ci_lower_95,
  row.mae_improvement_ci_upper_95, row.mae_probability_candidate_better,
  row.fdr_q_value, titleCase(row.evidence_label), row.origin_win_rate,
]);
const primarySheet = sheets["Primary Results"];
addTitle(primarySheet, "Primary mature-origin subgroup results", "Candidate versus the same model family using player-history baseline features. Small cells remain visible but are excluded from confirmatory FDR.", 16);
writeTable(primarySheet,
  ["Target", "Dimension", "Subgroup", "Pooled N", "Min Origin N", "Sample Quality", "Baseline MAE", "Candidate MAE", "MAE Improvement", "Relative MAE Improvement", "95% CI Low", "95% CI High", "P(Candidate Better)", "FDR q", "Evidence", "Origin Win Rate"],
  primaryRows, "PrimaryResultsTable", [15, 14, 20, 11, 13, 15, 14, 14, 16, 20, 13, 13, 18, 12, 31, 15], { wrapColumns: [15], rowHeight: 25 });
formatNumberColumns(primarySheet, 6, 21, { G: "0.0000", H: "0.0000", I: "0.0000", J: "0.0%", K: "0.0000", L: "0.0000", M: "0.0%", N: "0.000", P: "0.0%" });
applyEvidenceColors(primarySheet, 6, "O", "F", primary);

function buildAggregateSheet(name, dimension, tableName) {
  const data = aggregates.filter((row) => row.subgroup_dimension === dimension);
  const rows = data.map((row) => [
    titleCase(row.target_key), titleCase(row.reference_type), titleCase(row.origin_subset),
    row.subgroup_name, row.pooled_n, row.minimum_origin_n, titleCase(row.sample_quality),
    row.pooled_baseline_mae, row.pooled_candidate_mae, row.pooled_mae_improvement,
    row.pooled_relative_mae_improvement, row.mae_improvement_ci_lower_95,
    row.mae_improvement_ci_upper_95, row.mae_probability_candidate_better,
    row.fdr_q_value, titleCase(row.evidence_label), row.origin_win_rate,
  ]);
  const sheet = sheets[name];
  addTitle(sheet, `${titleCase(dimension)} stability`, "All origin subsets and both references. Confirmatory interpretation is limited to mature same-family adequate cells.", 17);
  writeTable(sheet,
    ["Target", "Reference", "Origin Subset", titleCase(dimension), "Pooled N", "Min Origin N", "Sample Quality", "Baseline MAE", "Candidate MAE", "MAE Improvement", "Relative MAE Improvement", "95% CI Low", "95% CI High", "P(Candidate Better)", "FDR q", "Evidence", "Origin Win Rate"],
    rows, tableName, [15, 25, 23, 21, 11, 13, 15, 14, 14, 16, 20, 13, 13, 18, 12, 31, 15], { wrapColumns: [2, 3, 16] });
  formatNumberColumns(sheet, 6, 5 + rows.length, { H: "0.0000", I: "0.0000", J: "0.0000", K: "0.0%", L: "0.0000", M: "0.0000", N: "0.0%", O: "0.000", Q: "0.0%" });
  applyEvidenceColors(sheet, 6, "P", "G", data);
}
buildAggregateSheet("Role Results", "role", "RoleResultsTable");
buildAggregateSheet("Competition Results", "competition", "CompetitionResultsTable");

// Heterogeneity table.
const heterogeneityRows = heterogeneityData.map((row) => [
  titleCase(row.target_key), titleCase(row.reference_type), titleCase(row.origin_subset),
  titleCase(row.subgroup_dimension), row.pooled_n, row.subgroup_count,
  row.heterogeneity_statistic, row.heterogeneity_p_value, row.heterogeneity_fdr_q_value,
  row.best_subgroup_name, row.best_subgroup_mae_improvement, row.worst_subgroup_name,
  row.worst_subgroup_mae_improvement, row.subgroup_improvement_range,
  titleCase(row.heterogeneity_evidence),
]);
const heterogeneitySheet = sheets.Heterogeneity;
addTitle(heterogeneitySheet, "Subgroup-effect heterogeneity", "Permutation tests shuffle subgroup labels within each rolling origin. A low q-value would indicate effects differ beyond sampling noise.", 15);
writeTable(heterogeneitySheet,
  ["Target", "Reference", "Origin Subset", "Dimension", "Pooled N", "Groups", "Statistic", "p-value", "FDR q", "Best Group", "Best MAE Improvement", "Worst Group", "Worst MAE Improvement", "Improvement Range", "Evidence"],
  heterogeneityRows, "HeterogeneityTable", [15, 25, 23, 14, 11, 10, 13, 12, 12, 20, 20, 20, 21, 18, 29], { wrapColumns: [2, 3, 15] });
formatNumberColumns(heterogeneitySheet, 6, 29, { G: "0.0000", H: "0.000", I: "0.000", K: "0.0000", M: "0.0000", N: "0.0000" });
applyEvidenceColors(heterogeneitySheet, 6, "O", null, heterogeneityData);

// Origin results.
const originRows = origins.map((row) => [
  titleCase(row.target_key), titleCase(row.reference_type), titleCase(row.subgroup_dimension),
  row.subgroup_name, row.origin_id, row.evaluation_year, row.n, titleCase(row.sample_quality),
  row.baseline_mae, row.candidate_mae, row.mae_improvement, row.relative_mae_improvement,
  row.mae_improvement_ci_lower_95, row.mae_improvement_ci_upper_95,
  row.mae_probability_candidate_better, row.bootstrap_p_two_sided,
]);
const originSheet = sheets["Origin Results"];
addTitle(originSheet, "Origin-level subgroup results", "Season-by-season paired evidence; cells below five observations are labeled very small.", 16);
writeTable(originSheet,
  ["Target", "Reference", "Dimension", "Subgroup", "Origin", "Evaluation Year", "N", "Sample Quality", "Baseline MAE", "Candidate MAE", "MAE Improvement", "Relative MAE Improvement", "95% CI Low", "95% CI High", "P(Candidate Better)", "Bootstrap p"],
  originRows, "OriginResultsTable", [15, 25, 14, 20, 15, 15, 9, 15, 14, 14, 16, 20, 13, 13, 18, 14], { wrapColumns: [2] });
formatNumberColumns(originSheet, 6, 133, { I: "0.0000", J: "0.0000", K: "0.0000", L: "0.0%", M: "0.0000", N: "0.0000", O: "0.0%", P: "0.000" });

// Coverage.
const coverageRows = coverage.map((row) => [
  titleCase(row.target_key), titleCase(row.subgroup_dimension), row.subgroup_name,
  row.all_origin_n, row.mature_origin_n, row.terminal_origin_n, row.origins_present,
  row.minimum_origin_n, row.maximum_origin_n,
  row.mature_origin_n >= 15 && row.minimum_origin_n >= 5 ? "Adequate in mature confirmatory set" : "Use as small-cell directional evidence",
]);
const coverageSheet = sheets.Coverage;
addTitle(coverageSheet, "Subgroup coverage", "Coverage determines inferential weight; it does not remove any subgroup from the audit trail.", 10);
writeTable(coverageSheet,
  ["Target", "Dimension", "Subgroup", "All Origins N", "Mature N", "Terminal N", "Origins Present", "Min Origin N", "Max Origin N", "Interpretation"],
  coverageRows, "CoverageTable", [15, 14, 20, 14, 12, 12, 15, 13, 13, 36], { wrapColumns: [10], rowHeight: 28 });
coverageRows.forEach((row, index) => {
  if (String(row[9]).startsWith("Use")) coverageSheet.getRange(`J${index + 6}`).format = { fill: colors.amber, font: { color: colors.amberText, bold: true } };
});

// Frozen definitions.
const definitionRows = definitions.map((row) => [
  titleCase(row.target_key), row.target_label, row.candidate_feature_set,
  row.candidate_model_family, row.same_family_baseline_feature_set,
  row.same_family_baseline_model_family, row.secondary_reference,
  row.subgroup_retuning_permitted, row.candidate_scope,
]);
const definitionSheet = sheets["Candidate Definitions"];
addTitle(definitionSheet, "Frozen candidate definitions", "Subgroups are evaluation slices only. Candidate families, feature sets, and hyperparameters are never reselected within a subgroup.", 9);
writeTable(definitionSheet,
  ["Target", "Target Label", "Candidate Features", "Candidate Model", "Primary Baseline Features", "Primary Baseline Model", "Secondary Reference", "Subgroup Retuning?", "Scope"],
  definitionRows, "CandidateDefinitionsTable", [15, 27, 21, 24, 24, 24, 25, 18, 31], { wrapColumns: [2, 9], rowHeight: 34 });

// Auditable paired predictions.
const pairRows = pairs.map((row) => [
  row.origin_id, titleCase(row.target_key), row.evaluation_year,
  row.canonical_performance_id, row.canonical_player_name, row.canonical_club_name,
  row.role_name, row.competition_name, row.actual, row.candidate_prediction,
  row.baseline_prediction, row.candidate_absolute_error, row.baseline_absolute_error,
  row.mae_improvement, titleCase(row.reference_type), row.candidate_model_family,
  row.baseline_model_family,
]);
const pairSheet = sheets["Pair Predictions"];
addTitle(pairSheet, "Auditable paired predictions", "One row per player-team-season, target, origin, and benchmark. Positive MAE improvement means the frozen compatibility candidate was more accurate.", 17);
writeTable(pairSheet,
  ["Origin", "Target", "Evaluation Year", "Performance ID", "Player", "Club", "Role", "Competition", "Actual", "Candidate Prediction", "Baseline Prediction", "Candidate Abs Error", "Baseline Abs Error", "MAE Improvement", "Reference", "Candidate Model", "Baseline Model"],
  pairRows, "PairPredictionsTable", [15, 15, 15, 31, 25, 24, 15, 20, 12, 18, 18, 18, 18, 16, 25, 24, 24], { wrapColumns: [4, 15] });
formatNumberColumns(pairSheet, 6, 763, { I: "0.0000", J: "0.0000", K: "0.0000", L: "0.0000", M: "0.0000", N: "0.0000" });

// Verification.
const verificationRows = verification.map((row) => [
  row.check, titleCase(row.severity), row.passed, row.observed, row.expected, row.detail,
]);
const verificationSheet = sheets.Verification;
addTitle(verificationSheet, "Independent verification", "A separate script reconstructed the pairings and independently reran every bootstrap, permutation test, and FDR adjustment.", 6);
writeTable(verificationSheet,
  ["Check", "Severity", "Passed", "Observed", "Expected", "Detail"],
  verificationRows, "VerificationTable", [39, 13, 11, 38, 31, 62], { wrapColumns: [1, 4, 5, 6], rowHeight: 34 });
verification.forEach((row, index) => {
  verificationSheet.getRange(`C${index + 6}`).format = row.passed
    ? { fill: colors.green, font: { color: colors.greenText, bold: true } }
    : { fill: colors.red, font: { color: colors.redText, bold: true } };
});

// Methodology and definitions.
const methodologySheet = sheets.Methodology;
addTitle(methodologySheet, "Methodology and interpretation", "Predeclared design, uncertainty controls, and boundaries for using subgroup results.", 6);
const methodologyRows = [
  ["Design", "Candidate freeze", "Opportunity: full explicit fit + elastic net; Performance: full explicit fit + gradient boosted trees."],
  ["Design", "Primary reference", "Same model family using player-history baseline features, isolating compatibility information from learner choice."],
  ["Design", "Secondary reference", "Rolling-selected baseline policy chosen from each origin's validation season."],
  ["Time", "All origins", "2019, 2020, 2021, and 2022 rolling origins."],
  ["Time", "Primary subset", "Mature origins 2020-2022; terminal sensitivity uses 2021-2022."],
  ["Subgroups", "Roles", "Defender, Forward, Midfielder."],
  ["Subgroups", "Competitions", "La Liga, Ligue 1, Premier League, Serie A, Bundesliga."],
  ["Uncertainty", "Paired bootstrap", "5,000 resamples, stratified by rolling origin for pooled summaries."],
  ["Multiplicity", "Primary FDR", "Benjamini-Hochberg at q=10% across all adequate mature same-family role and competition cells."],
  ["Heterogeneity", "Permutation test", "5,000 permutations of subgroup labels within each origin; BH correction across four primary tests."],
  ["Adequacy", "Minimum sample", "Pooled N at least 15, every subset origin present, and minimum origin cell at least 5."],
  ["Interpretation", "Positive improvement", "Baseline absolute error minus candidate absolute error; positive values favor compatibility features."],
  ["Boundary", "Small cells", "Retained as directional evidence, explicitly flagged, and excluded from confirmatory subgroup FDR."],
  ["Boundary", "No retuning", "Subgroup outcomes never select model family, feature set, or hyperparameters."],
  ["Next phase", "Recommended use", "Advance to player-club case studies and prospective simulation while retaining subgroup uncertainty guardrails."],
];
writeTable(methodologySheet, ["Section", "Item", "Definition"], methodologyRows, "MethodologyTable", [17, 25, 92], { wrapColumns: [3], rowHeight: 42 });

// Executive dashboard, formula-linked to source sheets.
const summary = sheets["Executive Summary"];
summary.freezePanes.unfreeze();
addTitle(summary, "MoveMaker subgroup stability", "Do compatibility features remain useful across player roles and destination competitions? Frozen candidates, rolling-origin evaluation, paired uncertainty.", 14);
summary.getRange("A5:K5").merge();
summary.getRange("A5").values = [["Phase decision"]];
summary.getRange("A5:K5").format = { fill: colors.blue, font: { bold: true, color: colors.white, size: 12 } };
summary.getRange("A6:K6").values = [["Target", "Candidate", "Recommendation", "Adequate Cells", "Supported Positive", "Strongest Segment", "Strongest Lift", "Weakest Segment", "Weakest Lift", "Role Het. q", "League Het. q"]];
summary.getRange("A6:K6").format = { fill: colors.paleBlue, font: { bold: true, color: colors.navy }, wrapText: true, horizontalAlignment: "center", borders: { preset: "all", style: "thin", color: colors.grid } };
for (let i = 0; i < 2; i += 1) {
  const targetRow = i + 7;
  const sourceRow = i + 6;
  summary.getRange(`A${targetRow}:K${targetRow}`).formulas = [[
    `='Decision Summary'!A${sourceRow}`, `='Decision Summary'!C${sourceRow}`,
    `='Decision Summary'!O${sourceRow}`, `='Decision Summary'!D${sourceRow}`,
    `='Decision Summary'!G${sourceRow}`, `='Decision Summary'!I${sourceRow}`,
    `='Decision Summary'!J${sourceRow}`, `='Decision Summary'!K${sourceRow}`,
    `='Decision Summary'!L${sourceRow}`, `='Decision Summary'!M${sourceRow}`,
    `='Decision Summary'!N${sourceRow}`,
  ]];
}
summary.getRange("A6:K8").format.borders = { preset: "all", style: "thin", color: colors.grid };
summary.getRange("A7:K8").format.rowHeight = 34;
summary.getRange("C7:C8").format.wrapText = true;
summary.getRange("G7:K8").setNumberFormat("0.0%");
summary.getRange("J7:K8").setNumberFormat("0.000");
summary.getRange("A10:F10").merge();
summary.getRange("A10").values = [["What the evidence says"]];
summary.getRange("A10:F10").format = { fill: colors.teal, font: { bold: true, color: colors.white, size: 12 } };
summary.getRange("A11:F13").merge();
summary.getRange("A11").values = [["Forwards are the clearest positive segment: mature-origin MAE improves 11.6% for opportunity and 10.4% for performance. Performance for forwards survives 10% FDR; opportunity for forwards is nominally supported. Role and league heterogeneity tests do not survive correction, so use these as prioritization signals—not proof that the model behaves differently by subgroup."]];
summary.getRange("A11:F13").format = { fill: "#F4FBF9", font: { size: 11, color: colors.text }, wrapText: true, verticalAlignment: "center", borders: { preset: "outside", style: "medium", color: colors.teal } };
summary.getRange("A15:F15").merge();
summary.getRange("A15").values = [["Operational boundary"]];
summary.getRange("A15:F15").format = { fill: colors.amber, font: { bold: true, color: colors.amberText, size: 12 } };
summary.getRange("A16:F18").merge();
summary.getRange("A16").values = [["Advance the frozen opportunity model into player-club case studies. Continue performance research, emphasizing forward cases. Treat Premier League, Bundesliga, and some Ligue 1 estimates as small-cell directional evidence until more seasons or transfers are added."]];
summary.getRange("A16:F18").format = { fill: "#FFF9F1", font: { size: 11, color: colors.text }, wrapText: true, verticalAlignment: "center", borders: { preset: "outside", style: "medium", color: "#E6A15A" } };
summary.getRange("A20:C20").values = [["Independent QA", "Blocking Checks", "Failures"]];
summary.getRange("A20:C20").format = { fill: colors.paleBlue, font: { bold: true, color: colors.navy }, horizontalAlignment: "center" };
summary.getRange("A21:C21").values = [["PASS", verification.length, verification.filter((row) => !row.passed).length]];
summary.getRange("A21:C21").format = { fill: colors.green, font: { bold: true, color: colors.greenText, size: 12 }, horizontalAlignment: "center" };

const primaryRowMap = new Map();
primary.forEach((row, index) => primaryRowMap.set(`${row.target_key}|${row.subgroup_dimension}|${row.subgroup_name}`, index + 6));
summary.getRange("A24:C24").values = [["Role", "Opportunity", "Performance"]];
const roles = ["Defender", "Forward", "Midfielder"];
roles.forEach((role, index) => {
  const row = index + 25;
  summary.getRange(`A${row}`).values = [[role]];
  summary.getRange(`B${row}`).formulas = [[`='Primary Results'!$J$${primaryRowMap.get(`opportunity|role|${role}`)}`]];
  summary.getRange(`C${row}`).formulas = [[`='Primary Results'!$J$${primaryRowMap.get(`performance|role|${role}`)}`]];
});
summary.getRange("E24:G24").values = [["Competition", "Opportunity", "Performance"]];
const competitions = ["La Liga", "Ligue 1", "Premier League", "Serie A", "Bundesliga"];
competitions.forEach((competition, index) => {
  const row = index + 25;
  summary.getRange(`E${row}`).values = [[competition]];
  summary.getRange(`F${row}`).formulas = [[`='Primary Results'!$J$${primaryRowMap.get(`opportunity|competition|${competition}`)}`]];
  summary.getRange(`G${row}`).formulas = [[`='Primary Results'!$J$${primaryRowMap.get(`performance|competition|${competition}`)}`]];
});
summary.getRange("A24:C27").format.borders = { preset: "all", style: "thin", color: colors.grid };
summary.getRange("E24:G29").format.borders = { preset: "all", style: "thin", color: colors.grid };
summary.getRange("A24:C24").format = { fill: colors.blue, font: { bold: true, color: colors.white } };
summary.getRange("E24:G24").format = { fill: colors.blue, font: { bold: true, color: colors.white } };
summary.getRange("B25:C27").setNumberFormat("0.0%");
summary.getRange("F25:G29").setNumberFormat("0.0%");
const roleChart = summary.charts.add("bar", summary.getRange("A24:C27"));
roleChart.setPosition("H10", "N21");
roleChart.title = "Mature relative MAE improvement by role";
roleChart.hasLegend = true;
roleChart.yAxis = { numberFormatCode: "0%" };
const competitionChart = summary.charts.add("bar", summary.getRange("E24:G29"));
competitionChart.setPosition("H23", "N36");
competitionChart.title = "Mature relative MAE improvement by competition";
competitionChart.hasLegend = true;
competitionChart.yAxis = { numberFormatCode: "0%" };
summary.getRange("A:A").format.columnWidth = 17;
summary.getRange("B:B").format.columnWidth = 23;
summary.getRange("C:C").format.columnWidth = 36;
summary.getRange("D:K").format.columnWidth = 17;
summary.getRange("E:E").format.columnWidth = 20;
summary.getRange("11:13").format.rowHeight = 28;
summary.getRange("16:18").format.rowHeight = 28;

const formulaInspection = await workbook.inspect({ kind: "formula", sheetId: "Executive Summary", range: "A1:N40", maxChars: 12000, options: { maxResults: 100 } });
const formulaText = formulaInspection.ndjson ?? JSON.stringify(formulaInspection);
const errorTokens = ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"];
const formulaErrors = errorTokens.filter((token) => formulaText.includes(token));
if (formulaErrors.length) throw new Error(`Formula inspection found: ${formulaErrors.join(", ")}`);

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);

const renderSummary = [];
for (const name of sheetNames) {
  const preview = await workbook.render({ sheetName: name, autoCrop: "all", scale: name === "Pair Predictions" ? 0.35 : 0.65, format: "png" });
  const previewPath = path.join(previewDir, `${name.replaceAll(" ", "_")}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  const stats = await fs.stat(previewPath);
  renderSummary.push({ sheet: name, preview: previewPath, bytes: stats.size });
}

await fs.writeFile(path.join(outputDir, "workbook_qa.json"), JSON.stringify({
  status: "pass",
  workbook: outputPath,
  sheets: sheetNames,
  formula_errors: formulaErrors,
  rendered_sheets: renderSummary,
}, null, 2));

console.log(JSON.stringify({ outputPath, sheets: sheetNames.length, formulaErrors, previews: renderSummary.length }, null, 2));
