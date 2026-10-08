import test from "node:test";
import assert from "node:assert/strict";
import { FEATURES, NUMERIC_FEATURES, SIMPLE_CSV_FIELDS, emptyWindow, parseLocalHour, parseCSV, toCSV, validateWindow, rowError } from "../frontend/data.mjs";

function fixture() {
  return emptyWindow("2026-10-05T06:00").map(row => ({ ...row, demanda_mw: 650, temperatura_c: 19.5, humedad_pct: 78, viento_kmh: 11, radiacion_wm2: 82, precipitacion_mm: 1, precio_kwh: 25 }));
}

test("24 horas en Colombia conservan la fecha al cruzar medianoche y fin de semana", () => {
  const rows = fixture();
  assert.equal(rows.length, 24);
  assert.equal(rows[0].timestamp, "2026-10-04T07:00:00-05:00");
  assert.equal(rows[0].dia_semana, 6);
  assert.equal(rows[0].fin_semana, 1);
  assert.equal(rows[23].dia_semana, 0);
  assert.equal(rows[23].fin_semana, 0);
  assert.equal(validateWindow(rows), rows);
});
test("fechas inválidas y minutos distintos de cero se rechazan", () => {
  for (const value of ["", "2026-02-30T06:00", "2026-10-05T06:30", "2026-13-05T06:00"]) assert.throws(() => parseLocalHour(value));
});
test("CSV estándar con BOM, CRLF y columnas entre comillas conserva las 16 variables", () => {
  const rows = fixture();
  const csv = "\ufeff" + toCSV(rows).replace(FEATURES.join(","), FEATURES.map(name => `"${name}"`).join(","));
  assert.deepEqual(parseCSV(csv), rows);
});
test("CSV con punto y coma acepta decimales con coma", () => {
  const rows = fixture();
  const csv = ["timestamp", ...FEATURES].join(";") + "\n" + rows.map(row => [row.timestamp, ...FEATURES.map(name => String(row[name]).replace(".", ","))].join(";")).join("\n");
  assert.deepEqual(parseCSV(csv), rows);
});
test("CSV sin fechas conserva las entradas sin inventar timestamps", () => {
  const rows = fixture().map(({ timestamp, ...row }) => row);
  assert.deepEqual(parseCSV(toCSV(rows)), rows);
});
test("CSV admite un período elegido explícitamente, sin imponer cortes horarios", () => {
  const rows = fixture();
  rows[0].periodo_dia_manana = 0;
  rows[0].periodo_dia_madrugada = 1;
  assert.deepEqual(parseCSV(toCSV(rows)), rows);
});
test("CSV con filas, columnas o números inválidos se rechaza", () => {
  const csv = toCSV(fixture());
  assert.throws(() => parseCSV(csv.split("\r\n").slice(0, 24).join("\r\n")), /24 filas/);
  assert.throws(() => parseCSV(csv.replace("demanda_mw", "otro")), /Faltan columnas/);
  assert.throws(() => parseCSV(csv.replace("temperatura_c", "demanda_mw")), /duplicadas/);
  assert.throws(() => parseCSV(csv.replace(",650,", ",,")), /vacío/);
  assert.throws(() => parseCSV(csv.replace(",650,", ",NaN,")), /numérico/);
  assert.throws(() => parseCSV(csv.replace(",650,", ",<script>,")), /numérico/);
  assert.throws(() => parseCSV(csv.replace(",650,", ",\"650,")), /comillas/);
});
test("horas desordenadas, calendario incorrecto y fechas parciales se rechazan", () => {
  const rows = fixture();
  [rows[0], rows[1]] = [rows[1], rows[0]];
  assert.throws(() => validateWindow(rows), /consecutivos/);
  const mismatch = fixture(); mismatch[0].mes = 2;
  assert.throws(() => validateWindow(mismatch), /no coincide/);
  const partial = fixture(); delete partial[0].timestamp;
  assert.throws(() => validateWindow(partial), /todas las filas/);
});
test("entradas incompletas, humedad imposible y períodos duplicados se rechazan", () => {
  assert.ok(rowError(emptyWindow("2026-10-05T06:00")[0]));
  assert.ok(rowError({ ...fixture()[0], humedad_pct: 101 }));
  assert.ok(rowError({ ...fixture()[0], periodo_dia_noche: 1 }));
  assert.ok(rowError({ ...fixture()[0], precio_kwh: Infinity }));
  assert.equal(rowError({ ...fixture()[0], precio_kwh: -2 }), null);
});
test("datos vacíos permanecen vacíos y no se convierten en demanda cero", () => {
  for (const row of emptyWindow("2026-10-05T06:00")) {
    for (const feature of NUMERIC_FEATURES) assert.equal(row[feature], null);
  }
  assert.throws(() => parseCSV(toCSV(emptyWindow("2026-10-05T06:00"))), /vacío/);
});

test("plantilla sencilla genera calendario y conserva demanda, clima, precio y festivo", () => {
  const rows = fixture();
  rows[0].festivo = 1;
  rows[0].precio_kwh = 810;
  const csv = SIMPLE_CSV_FIELDS.join(",") + "\n" + rows.map(row => SIMPLE_CSV_FIELDS.map(name => row[name]).join(",")).join("\n");
  assert.deepEqual(parseCSV(csv), rows);
  assert.equal(parseCSV(csv)[0].precio_kwh, 810);
});

test("entrada de clima futuro sin demanda ni historial no se acepta como predicción", () => {
  const csv = "hora,temperatura_c,humedad_pct,viento_kmh,radiacion_wm2,precipitacion_mm,precio_kwh,festivo\n" + Array(24).fill("18,17.8,74,6.2,40,0.5,810,0").join("\n");
  assert.throws(() => parseCSV(csv), /demanda_mw/);
});

test("plantilla sencilla necesita 24 fechas completas, consecutivas y con zona horaria", () => {
  const rows = fixture();
  const csv = SIMPLE_CSV_FIELDS.join(",") + "\n" + rows.map(row => SIMPLE_CSV_FIELDS.map(name => row[name]).join(",")).join("\n");
  assert.throws(() => parseCSV(csv.replace(rows[0].timestamp, "")), /requiere timestamp/);
  assert.throws(() => parseCSV(csv.replace(rows[0].timestamp, "2026-10-04T07:00:00")), /zona horaria/);
  assert.throws(() => parseCSV(csv.replace(rows[0].timestamp, rows[1].timestamp)), /consecutivos/);
});
