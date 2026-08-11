import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const workbookPath = path.join(root, "outputs/subgroup_stability/MoveMaker_Subgroup_Stability.xlsx");
const outputPath = path.join(root, "outputs/subgroup_stability/workbook_verification.json");
const expectedSheets = [
  "Executive Summary", "Decision Summary", "Primary Results", "Role Results",
  "Competition Results", "Heterogeneity", "Origin Results", "Coverage",
  "Candidate Definitions", "Pair Predictions", "Verification", "Methodology",
];

const blob = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(blob);
const observedSheets = workbook.worksheets.items.map((sheet) => sheet.name);
const missingSheets = expectedSheets.filter((name) => !observedSheets.includes(name));
const extraSheets = observedSheets.filter((name) => !expectedSheets.includes(name));

const summary = workbook.worksheets.getItem("Executive Summary");
const summaryValues = summary.getRange("A1:N40").values.flat();
const errorTokens = ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"];
const formulaErrors = summaryValues.filter((value) => errorTokens.some((token) => String(value).includes(token)));

const formulaInspection = await workbook.inspect({
  kind: "formula",
  sheetId: "Executive Summary",
  range: "A1:N40",
  maxChars: 12000,
  options: { maxResults: 100 },
});
const formulaText = formulaInspection.ndjson ?? JSON.stringify(formulaInspection);
const formulaCount = (formulaText.match(/formula/gi) ?? []).length;
const workbookInspection = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 10000,
  tableMaxRows: 3,
  tableMaxCols: 5,
  tableMaxCellChars: 60,
});

const status = missingSheets.length === 0 && extraSheets.length === 0 && formulaErrors.length === 0 ? "pass" : "fail";
const result = {
  status,
  workbook: workbookPath,
  expected_sheet_count: expectedSheets.length,
  observed_sheet_count: observedSheets.length,
  observed_sheets: observedSheets,
  missing_sheets: missingSheets,
  extra_sheets: extraSheets,
  summary_formula_error_values: formulaErrors,
  summary_formula_inspection_records: formulaCount,
  workbook_inspection_available: Boolean(workbookInspection),
};
await fs.writeFile(outputPath, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
if (status !== "pass") throw new Error("Exported workbook verification failed.");
