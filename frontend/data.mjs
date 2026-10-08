export const FEATURES = ["demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh", "radiacion_wm2", "precipitacion_mm", "precio_kwh", "hora", "dia_semana", "fin_semana", "festivo", "mes", "periodo_dia_madrugada", "periodo_dia_manana", "periodo_dia_tarde", "periodo_dia_noche"];
export const NUMERIC_FEATURES = FEATURES.slice(0, 7);
export const PERIODS = ["madrugada", "manana", "tarde", "noche"];
export const WINDOW_SIZE = 24;
export const SIMPLE_CSV_FIELDS = ["timestamp", ...NUMERIC_FEATURES, "festivo"];

// Fechas de calendario como UTC ficticio: la interfaz siempre usa Colombia (UTC−5).
export function parseLocalHour(value) {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:00$/.test(value)) throw new Error("Selecciona una fecha válida y una hora exacta (minutos 00).");
  const date = new Date(`${value}:00Z`);
  if (!Number.isFinite(date.getTime()) || date.toISOString().slice(0, 16) !== value) throw new Error("La fecha no es válida.");
  return date;
}

export function calendarAt(date) {
  const hour = date.getUTCHours();
  const day = (date.getUTCDay() + 6) % 7;
  const period = Math.floor(hour / 6);
  return { timestamp: date.toISOString().slice(0, 19) + "-05:00", hora: hour, dia_semana: day, fin_semana: Number(day >= 5), festivo: 0, mes: date.getUTCMonth() + 1, ...Object.fromEntries(PERIODS.map((name, index) => [`periodo_dia_${name}`, Number(index === period)])) };
}

export function emptyWindow(endValue) {
  const end = parseLocalHour(endValue);
  return Array.from({ length: WINDOW_SIZE }, (_, index) => ({ ...Object.fromEntries(NUMERIC_FEATURES.map(name => [name, null])), ...calendarAt(new Date(end.getTime() - (23 - index) * 3600000)) }));
}

export function rowError(row) {
  for (const name of FEATURES) {
    if (typeof row[name] !== "number" || !Number.isFinite(row[name])) return `Falta un valor numérico válido en ${name}.`;
  }
  for (const name of NUMERIC_FEATURES.filter(name => !["temperatura_c", "precio_kwh"].includes(name))) {
    if (row[name] < 0) return `${name} debe ser mayor o igual a cero.`;
  }
  if (row.temperatura_c < -100 || row.temperatura_c > 100) return "La temperatura debe estar entre −100 y 100 °C.";
  if (row.humedad_pct > 100) return "La humedad debe estar entre 0 y 100 %.";
  if (!Number.isInteger(row.hora) || row.hora < 0 || row.hora > 23) return "hora debe ser un entero entre 0 y 23.";
  if (!Number.isInteger(row.dia_semana) || row.dia_semana < 0 || row.dia_semana > 6) return "dia_semana debe ser un entero entre 0 y 6.";
  if (!Number.isInteger(row.mes) || row.mes < 1 || row.mes > 12) return "mes debe ser un entero entre 1 y 12.";
  for (const name of ["fin_semana", "festivo", ...FEATURES.slice(12)]) {
    if (row[name] !== 0 && row[name] !== 1) return `${name} debe ser 0 o 1.`;
  }
  if (FEATURES.slice(12).reduce((sum, name) => sum + row[name], 0) !== 1) return "Cada hora debe tener exactamente un período del día.";
  if (row.fin_semana !== Number(row.dia_semana >= 5)) return "fin_semana no coincide con dia_semana (lunes=0).";
  return null;
}

export function validateWindow(rows) {
  if (rows.length !== WINDOW_SIZE) throw new Error("Se requieren exactamente 24 filas: una por cada hora.");
  const hasTimestamp = rows.some(row => row.timestamp != null);
  if (hasTimestamp && !rows.every(row => typeof row.timestamp === "string" && row.timestamp)) throw new Error("Incluye timestamp en todas las filas o en ninguna.");
  for (let index = 0; index < rows.length; index++) {
    const row = rows[index];
    const error = rowError(row);
    if (error) throw new Error(`Hora ${index + 1}: ${error}`);
    if (hasTimestamp) {
      if (!/T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})$/.test(row.timestamp)) throw new Error(`Hora ${index + 1}: timestamp debe ser ISO 8601 con zona horaria.`);
      const actual = new Date(row.timestamp);
      if (!Number.isFinite(actual.getTime())) throw new Error(`Hora ${index + 1}: timestamp no válido.`);
      const local = new Date(actual.getTime() - 5 * 3600000);
      const expected = calendarAt(local);
      if (local.getUTCMinutes() || local.getUTCSeconds() || local.getUTCMilliseconds()) throw new Error(`Hora ${index + 1}: timestamp debe ser una hora exacta.`);
      for (const name of ["hora", "dia_semana", "mes"]) {
        if (row[name] !== expected[name]) throw new Error(`Hora ${index + 1}: ${name} no coincide con timestamp en Colombia.`);
      }
      if (index && actual.getTime() - new Date(rows[index - 1].timestamp).getTime() !== 3600000) throw new Error("Los timestamps deben ser consecutivos, de la hora más antigua a la más reciente.");
    }
    if (index) {
      const prev = rows[index - 1];
      if (row.hora !== (prev.hora + 1) % 24 || row.dia_semana !== (prev.dia_semana + Number(prev.hora === 23)) % 7) throw new Error("Las horas y los días deben ser consecutivos, de la hora más antigua a la más reciente.");
      if (prev.hora !== 23 && prev.mes !== row.mes) throw new Error("El mes solo puede cambiar a medianoche.");
      if (prev.hora === 23 && ![prev.mes, prev.mes % 12 + 1].includes(row.mes)) throw new Error("La secuencia de meses no es válida.");
    }
  }
  return rows;
}

// Parser sin dependencias: BOM, CRLF, comillas escapadas, coma o punto y coma.
function csvCells(text, delimiter) {
  const records = []; let record = []; let cell = ""; let quoted = false; let closed = false;
  for (let index = 0; index < text.length; index++) {
    const char = text[index];
    if (quoted) {
      if (char === '"') {
        if (text[index + 1] === '"') { cell += '"'; index++; }
        else { quoted = false; closed = true; }
      } else cell += char;
    } else if (char === '"') {
      if (cell.trim() || closed) throw new Error("CSV: comillas en una posición no válida.");
      cell = ""; quoted = true;
    } else if (char === delimiter) {
      record.push(cell.trim()); cell = ""; closed = false;
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[index + 1] === "\n") index++;
      record.push(cell.trim()); if (record.some(value => value !== "")) records.push(record);
      record = []; cell = ""; closed = false;
    } else {
      if (closed && char.trim()) throw new Error("CSV: hay texto después de una comilla de cierre.");
      cell += char;
    }
  }
  if (quoted) throw new Error("CSV: hay una celda con comillas sin cerrar.");
  record.push(cell.trim()); if (record.some(value => value !== "")) records.push(record);
  return records;
}

export function parseCSV(raw) {
  if (raw.length > 262144) throw new Error("El CSV debe pesar menos de 256 KB.");
  const text = raw.replace(/^\uFEFF/, "");
  const first = text.split(/\r?\n/, 1)[0];
  const delimiter = first.includes(";") ? ";" : ",";
  const records = csvCells(text, delimiter);
  if (records.length !== WINDOW_SIZE + 1) throw new Error("El CSV debe tener una cabecera y exactamente 24 filas de datos.");
  const headers = records[0];
  if (new Set(headers).size !== headers.length) throw new Error("El CSV tiene columnas duplicadas.");
  const missing = FEATURES.filter(name => !headers.includes(name));
  const simple = missing.length > 0 && headers.length === SIMPLE_CSV_FIELDS.length && SIMPLE_CSV_FIELDS.every(name => headers.includes(name));
  if (missing.length && !simple) throw new Error(`Faltan columnas: ${missing.join(", ")}. Descarga la plantilla sencilla (fecha, mediciones y festivo) o usa las 16 variables del formato completo.`);
  const unknown = headers.filter(name => ![...FEATURES, "timestamp"].includes(name));
  if (unknown.length) throw new Error(`Columnas desconocidas: ${unknown.join(", ")}.`);
  const rows = records.slice(1).map((record, index) => {
    if (record.length !== headers.length) throw new Error(`Fila ${index + 2}: el número de columnas no coincide.`);
    let row = {};
    headers.forEach((name, column) => {
      const rawValue = record[column];
      if (name === "timestamp") { if (rawValue) row.timestamp = rawValue; return; }
      const value = delimiter === ";" ? rawValue.replace(",", ".") : rawValue;
      if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) throw new Error(`Fila ${index + 2}: ${name} está vacío o no es numérico.`);
      row[name] = Number(value);
    });
    if (simple) {
      if (!row.timestamp || !/T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})$/.test(row.timestamp)) throw new Error(`Fila ${index + 2}: el formato sencillo requiere timestamp en ISO 8601 con zona horaria.`);
      const actual = new Date(row.timestamp);
      if (!Number.isFinite(actual.getTime())) throw new Error(`Fila ${index + 2}: timestamp no válido.`);
      // Los datos existentes se conservan; solo se completan variables de calendario.
      row = { ...calendarAt(new Date(actual.getTime() - 5 * 3600000)), ...row };
    }
    return row;
  });
  return validateWindow(rows);
}

export function toCSV(rows) {
  const fields = rows.every(row => row.timestamp) ? ["timestamp", ...FEATURES] : FEATURES;
  return fields.join(",") + "\r\n" + rows.map(row => fields.map(name => row[name] ?? "").join(",")).join("\r\n") + "\r\n";
}
