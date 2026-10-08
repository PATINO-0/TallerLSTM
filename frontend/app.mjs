import { FEATURES, NUMERIC_FEATURES, PERIODS, WINDOW_SIZE, emptyWindow, parseLocalHour, calendarAt, rowError, validateWindow, parseCSV } from "./data.mjs?v=2";

const $ = id => document.getElementById(id);
const numberFormat = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 2 });
const dateFormat = new Intl.DateTimeFormat("es-CO", { timeZone: "America/Bogota", weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const state = { rows: [], selected: 0, source: "manual", result: null, ready: false, busy: false, priceReference: null };
let healthPending = false;

function defaultEnd() {
  const local = new Date(Date.now() - 5 * 3600000 - 3600000);
  local.setUTCMinutes(0, 0, 0);
  return local.toISOString().slice(0, 16);
}

function message(text, success = false) {
  $("page-message").textContent = text;
  $("page-message").classList.toggle("success", success);
  $("page-message").hidden = !text;
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 115000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal, headers: { ...options.headers, Accept: "application/json" } });
    const contentType = response.headers.get("Content-Type") || "";
    if (!contentType.includes("application/json")) throw new Error("El servidor no devolvió JSON. Comprueba que la aplicación esté ejecutándose con FastAPI.");
    const data = await response.json();
    if (!response.ok) {
      if (data.errors?.length) {
        const detail = data.errors[0];
        const index = detail.location?.indexOf("observations");
        const hour = index >= 0 && Number.isInteger(detail.location[index + 1]) ? `Hora ${detail.location[index + 1] + 1}: ` : "";
        throw new Error(hour + detail.message.replace(/^Value error, /, ""));
      }
      throw new Error(typeof data.detail === "string" ? data.detail : data.message || "No se pudo completar la solicitud.");
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("El servidor tardó demasiado. Revisa la conexión y vuelve a intentarlo; la primera carga del modelo puede tomar más tiempo.");
    if (error instanceof TypeError) throw new Error("No se pudo conectar con la API. Comprueba tu conexión y el servidor.");
    throw error;
  } finally { clearTimeout(timeout); }
}

function invalidateResult() {
  state.result = null;
  $("prediction-result").hidden = true;
  $("prediction-empty").hidden = false;
  $("prediction-warnings").hidden = true;
}

function rowLabel(row) {
  return row.timestamp ? dateFormat.format(new Date(row.timestamp)) : `${String(row.hora).padStart(2, "0")}:00 · fecha no incluida`;
}

function renderPriceNotice() {
  const price = state.rows[state.selected].precio_kwh;
  const reference = state.priceReference;
  const unusual = reference && Number.isFinite(price) && Math.abs(price - reference.mean) > 3 * reference.std;
  $("price-notice").hidden = !unusual;
  $("price-notice").textContent = unusual ? `Revisa la unidad del precio: ingresaste ${numberFormat.format(price)}, mientras el promedio del entrenamiento es ${numberFormat.format(reference.mean)}. El valor se conservará tal como lo ingresaste.` : "";
}

function renderOverview(completed) {
  const last = state.rows[23];
  $("history-summary").textContent = completed === 24 ? "Historial completo. Ya puedes predecir." : completed ? `Tienes ${completed} horas listas. Faltan ${24 - completed}.` : "Todavía no hay un historial completo.";
  $("history-range").textContent = completed ? `${rowLabel(state.rows[0])} → ${rowLabel(last)}` : "Carga un CSV o usa el ejemplo para comenzar.";
  $("forecast-time").textContent = last.timestamp ? dateFormat.format(new Date(new Date(last.timestamp).getTime() + 3600000)) : `Siguiente hora: ${String((last.hora + 1) % 24).padStart(2, "0")}:00 · sin fecha absoluta`;
}

function renderHours() {
  const grid = $("hour-grid");
  if (!grid.children.length) {
    for (let index = 0; index < WINDOW_SIZE; index++) {
      const button = document.createElement("button");
      button.type = "button"; button.className = "hour-button";
      button.addEventListener("click", () => { state.selected = index; renderEditor(); renderHours(); });
      grid.append(button);
    }
  }
  let completed = 0;
  [...grid.children].forEach((button, index) => {
    const row = state.rows[index];
    const complete = !rowError(row);
    completed += Number(complete);
    button.textContent = `${String(row.hora).padStart(2, "0")}:00`;
    button.classList.toggle("selected", index === state.selected);
    button.classList.toggle("complete", complete);
    button.setAttribute("aria-pressed", String(index === state.selected));
    button.setAttribute("aria-label", `Hora ${index + 1}, ${rowLabel(row)}, ${complete ? "completa" : "por completar"}`);
    button.title = rowLabel(row);
  });
  $("completion-badge").textContent = `${completed} / 24 horas`;
  $("progress-fill").style.width = `${completed / WINDOW_SIZE * 100}%`;
  $("progress-text").textContent = completed === 24 ? state.ready ? "Todo listo. Pulsa el botón para obtener tu resultado." : "Historial listo. Esperando la conexión del modelo…" : `Faltan ${24 - completed} horas. Carga un CSV o completa la edición manual.`;
  $("predict-button").disabled = completed !== 24 || !state.ready || state.busy;
  $("demo-notice").hidden = state.source !== "demo";
  renderOverview(completed);
}

function renderEditor() {
  const row = state.rows[state.selected];
  $("selected-index").textContent = `HORA ${String(state.selected + 1).padStart(2, "0")} DE 24`;
  $("selected-time").textContent = rowLabel(row);
  for (const name of NUMERIC_FEATURES) {
    $(name).value = row[name] ?? "";
    $(name).classList.remove("invalid");
    $(name).removeAttribute("aria-invalid");
  }
  $("festivo").value = String(row.festivo);
  $("period").value = PERIODS.find(period => row[`periodo_dia_${period}`] === 1) || "madrugada";
  $("previous-hour").disabled = state.selected === 0;
  $("next-hour").disabled = state.selected === 23;
  $("copy-next").disabled = state.selected === 23;
  $("save-next").textContent = state.selected === 23 ? "Terminar revisión del historial ✓" : "Guardar y continuar con la siguiente hora →";
  renderPriceNotice();
  const summary = $("calendar-summary"); summary.replaceChildren();
  const days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"];
  for (const text of [`Hora: ${row.hora}`, `${days[row.dia_semana]} (${row.dia_semana})`, `Mes: ${row.mes}`, `Fin de semana: ${row.fin_semana ? "Sí" : "No"}`]) {
    const chip = document.createElement("span"); chip.textContent = text; summary.append(chip);
  }
}

function renderChart() {
  const container = $("demand-chart");
  if (!state.rows.every(row => Number.isFinite(row.demanda_mw) && row.demanda_mw >= 0)) {
    const placeholder = document.createElement("p");
    placeholder.textContent = "El gráfico aparecerá cuando ingreses la demanda de las 24 horas.";
    container.replaceChildren(placeholder); return;
  }
  const values = state.rows.map(row => row.demanda_mw);
  if (state.result) values.push(state.result.predicted_demand_mw);
  const minValue = Math.min(...values), maxValue = Math.max(...values);
  const span = Math.max(maxValue - minValue, Math.abs(maxValue) * .1, 10);
  const min = minValue - span * .2, max = maxValue + span * .2;
  const left = 43, right = 401, top = 16, bottom = 161;
  const x = index => left + index / 24 * (right - left);
  const y = value => bottom - (value - min) / (max - min) * (bottom - top);
  const points = state.rows.map((row, index) => `${x(index).toFixed(2)},${y(row.demanda_mw).toFixed(2)}`).join(" ");
  const path = `M${points.split(" ").join(" L")}`;
  let grid = "";
  for (let index = 0; index < 4; index++) {
    const value = min + (max - min) * index / 3;
    grid += `<line x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}" stroke="#d7e0e5"/><text x="34" y="${y(value) + 4}" text-anchor="end" fill="#475569" font-size="14">${Math.round(value)}</text>`;
  }
  let forecast = "";
  if (state.result) {
    const last = state.rows[23].demanda_mw;
    const predicted = state.result.predicted_demand_mw;
    forecast = `<line x1="${x(23)}" y1="${y(last)}" x2="${x(24)}" y2="${y(predicted)}" stroke="#c2410c" stroke-width="3" stroke-dasharray="4 4"/><circle cx="${x(24)}" cy="${y(predicted)}" r="5" fill="#c2410c" stroke="white" stroke-width="2"/>`;
  }
  const last = state.rows[23];
  // Solo números validados llegan al SVG; nunca se interpola texto de un CSV.
  container.innerHTML = `<svg viewBox="0 0 420 195" role="img" aria-label="Demanda de las últimas 24 horas${state.result ? " y predicción de la siguiente hora" : ""}"><defs><linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#0f766e" stop-opacity=".18"/><stop offset="1" stop-color="#0f766e" stop-opacity="0"/></linearGradient></defs>${grid}<path d="${path} L${x(23)},${bottom} L${x(0)},${bottom} Z" fill="url(#chart-fill)"/><polyline points="${points}" fill="none" stroke="#0f766e" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>${forecast}<text x="${left}" y="185" fill="#475569" font-size="14">${String(state.rows[0].hora).padStart(2, "0")}:00</text><text x="${x(12)}" y="185" text-anchor="middle" fill="#475569" font-size="14">${String(state.rows[12].hora).padStart(2, "0")}:00</text><text x="${x(23)}" y="185" text-anchor="end" fill="#475569" font-size="14">${String(last.hora).padStart(2, "0")}:00</text></svg>`;
}

function setBusy(busy) {
  state.busy = busy;
  document.body.classList.toggle("busy", busy);
  $("hour-form").setAttribute("aria-busy", String(busy));
  // También bloquea la edición con teclado durante una solicitud.
  document.querySelectorAll(".input-panel button, .input-panel input, .input-panel select").forEach(element => { element.disabled = busy; });
  if (!busy) {
    $("end-time").disabled = !state.rows.every(row => row.timestamp);
    $("previous-hour").disabled = state.selected === 0;
    $("next-hour").disabled = state.selected === 23;
    $("copy-next").disabled = state.selected === 23;
  }
  renderHours();
}

function applyRows(rows, source) {
  state.rows = rows; state.source = source; state.selected = 0;
  const last = rows[23];
  if (last.timestamp) {
    const local = new Date(new Date(last.timestamp).getTime() - 5 * 3600000);
    $("end-time").value = local.toISOString().slice(0, 16);
    $("end-time").disabled = false;
  } else { $("end-time").value = ""; $("end-time").disabled = true; }
  invalidateResult(); renderEditor(); renderHours(); renderChart();
}

async function checkHealth() {
  if (healthPending) return;
  healthPending = true; state.ready = false;
  $("model-status").className = "status";
  $("status-text").textContent = "Cargando modelo…";
  renderHours();
  try {
    await api("/api/health"); state.ready = true;
    $("model-status").className = "status ready";
    $("status-text").textContent = "Modelo conectado";
    $("model-status").title = "El modelo entrenado está listo. Pulsa para comprobar de nuevo.";
  } catch (error) {
    $("model-status").className = "status error";
    $("status-text").textContent = "Reintentar conexión";
    $("model-status").title = error.message;
    message(error.message);
  } finally { healthPending = false; renderHours(); }
}

for (const name of NUMERIC_FEATURES) {
  $(name).addEventListener("input", () => {
    const input = $(name);
    state.rows[state.selected][name] = input.value.trim() === "" ? null : input.valueAsNumber;
    input.classList.remove("invalid"); input.removeAttribute("aria-invalid");
    invalidateResult(); message(""); renderHours(); renderChart();
    if (name === "precio_kwh") renderPriceNotice();
  });
  $(name).addEventListener("blur", () => {
    if ($(name).value !== "" && !$(name).checkValidity()) { $(name).classList.add("invalid"); $(name).setAttribute("aria-invalid", "true"); }
  });
}
$("festivo").addEventListener("change", () => { state.rows[state.selected].festivo = Number($("festivo").value); invalidateResult(); renderHours(); renderChart(); });
$("period").addEventListener("change", () => {
  PERIODS.forEach(period => { state.rows[state.selected][`periodo_dia_${period}`] = Number(period === $("period").value); });
  invalidateResult(); renderHours(); renderChart();
});
$("previous-hour").addEventListener("click", () => { if (state.selected > 0) state.selected--; renderEditor(); renderHours(); });
$("next-hour").addEventListener("click", () => { if (state.selected < 23) state.selected++; renderEditor(); renderHours(); });
$("manual-entry").addEventListener("click", () => {
  $("history-editor").open = true;
  const incomplete = state.rows.findIndex(row => rowError(row));
  state.selected = incomplete < 0 ? state.selected : incomplete;
  renderEditor(); renderHours();
  $("demanda_mw").focus();
});
$("save-next").addEventListener("click", () => {
  const error = rowError(state.rows[state.selected]);
  if (error) { message(error); $("hour-form").reportValidity(); return; }
  message("");
  if (state.selected < 23) { state.selected++; renderEditor(); renderHours(); $("demanda_mw").focus(); }
  else { $("history-editor").open = false; $("predict-button").focus(); }
});
$("copy-next").addEventListener("click", () => {
  if (state.selected >= 23) return;
  NUMERIC_FEATURES.forEach(name => { state.rows[state.selected + 1][name] = state.rows[state.selected][name]; });
  state.selected++; invalidateResult(); renderEditor(); renderHours(); renderChart();
});
$("end-time").addEventListener("change", () => {
  try {
    const rebuilt = emptyWindow($("end-time").value);
    state.rows = rebuilt.map((row, index) => ({ ...state.rows[index], ...calendarAt(parseLocalHour(row.timestamp.slice(0, 16))) }));
    invalidateResult(); renderEditor(); renderHours(); renderChart();
    message("Fechas y períodos actualizados. Revisa los festivos y los períodos del día para el nuevo historial.", true);
  } catch (error) { message(error.message); $("end-time").value = state.rows[23].timestamp.slice(0, 16); }
});
$("clear-data").addEventListener("click", () => {
  applyRows(emptyWindow($("end-time").value || defaultEnd()), "manual"); message("");
});
$("load-demo").addEventListener("click", async () => {
  if (state.busy) return;
  setBusy(true); message("");
  try { const data = await api("/api/example"); applyRows(validateWindow(data.observations), "demo"); $("history-editor").open = false; }
  catch (error) { message(error.message); }
  finally { setBusy(false); }
});
$("csv-file").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  if (state.busy) { event.target.value = ""; return; }
  setBusy(true);
  try {
    if (file.size > 262144) throw new Error("El CSV debe pesar menos de 256 KB.");
    const rows = parseCSV(await file.text());
    applyRows(rows, "csv");
    $("history-editor").open = false;
    message(rows.every(row => row.timestamp) ? "CSV importado: 24 horas válidas. Revisa los valores antes de predecir." : "CSV importado sin fechas. Se conserva tu calendario y se muestra la hora predicha sin una fecha absoluta.", true);
  } catch (error) { message(error.message); }
  finally { event.target.value = ""; setBusy(false); }
});

$("hour-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (state.busy) return;
  if (!state.ready) { message("El modelo todavía no está conectado. Pulsa el indicador superior para reintentar."); return; }
  try { validateWindow(state.rows); } catch (error) { message(error.message); return; }
  const payload = { observations: state.rows, source: state.source };
  setBusy(true); message(""); invalidateResult(); renderChart();
  $("predict-label").textContent = "Calculando…";
  try {
    const result = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    state.result = result;
    $("predicted-value").textContent = numberFormat.format(result.predicted_demand_mw);
    $("prediction-for").textContent = result.prediction_timestamp ? dateFormat.format(new Date(result.prediction_timestamp)) : `Siguiente hora: ${String(result.next_hour).padStart(2, "0")}:00`;
    $("change-value").textContent = `${result.change_mw >= 0 ? "+" : ""}${numberFormat.format(result.change_mw)} MW${result.change_pct === null ? "" : ` · ${numberFormat.format(result.change_pct)} %`}`;
    $("last-value").textContent = `${numberFormat.format(result.last_demand_mw)} MW`;
    $("inference-time").textContent = `${numberFormat.format(result.inference_ms)} ms`;
    const warnings = $("prediction-warnings"); warnings.replaceChildren();
    result.warnings.forEach(text => { const p = document.createElement("p"); p.textContent = text; warnings.append(p); });
    warnings.hidden = !result.warnings.length;
    $("prediction-empty").hidden = true; $("prediction-result").hidden = false;
    renderChart();
    if (window.matchMedia("(max-width: 720px)").matches) $("prediction-result").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "center" });
  } catch (error) { message(error.message); }
  finally { $("predict-label").textContent = "Predecir demanda"; setBusy(false); }
});
$("download-result").addEventListener("click", () => {
  if (!state.result) return;
  const blob = new Blob([JSON.stringify({ prediction: state.result, inputs: { source: state.source, observations: state.rows } }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob); const link = document.createElement("a");
  link.href = url; link.download = "prediccion_demanda.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("model-status").addEventListener("click", checkHealth);
$("csv-file").addEventListener("focus", () => document.querySelector(".import-button").style.outline = "3px solid #cba378");
$("csv-file").addEventListener("blur", () => document.querySelector(".import-button").style.outline = "");

$("end-time").value = defaultEnd();
state.rows = emptyWindow($("end-time").value);
renderEditor(); renderHours(); renderChart();
api("/api/metadata").then(metadata => {
  state.priceReference = metadata.price_training_reference || null;
  renderPriceNotice();
  $("model-name").textContent = metadata.model;
  $("metric-mae").textContent = `${numberFormat.format(metadata.test_mae)} MW`;
  $("metric-rmse").textContent = `${numberFormat.format(metadata.test_rmse)} MW`;
  $("metric-mape").textContent = `${numberFormat.format(metadata.test_mape)} %`;
  $("metric-r2").textContent = numberFormat.format(metadata.test_r2);
}).catch(error => message(error.message));
checkHealth();
