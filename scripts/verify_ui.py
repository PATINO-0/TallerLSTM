"""Prueba opcional con Chrome headless: pip install selenium==4.40.0.

Requiere el servidor local iniciado. Usa un perfil nuevo; no accede al navegador
personal. Guarda capturas y archivos de prueba en test-results/.
"""
from __future__ import annotations

import json
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-results"
OUTPUT.mkdir(exist_ok=True)
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--window-size=1440,1300")
options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
options.add_experimental_option("prefs", {"download.default_directory": str(OUTPUT), "download.prompt_for_download": False})


def assert_text_contrast(driver):
    report = driver.execute_script("""
      const rgb = text => (text.match(/[\\d.]+/g) || []).map(Number);
      const luminance = channels => channels.slice(0, 3).map(value => {
        const v = value / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
      }).reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
      const failures = []; let minimum = Infinity; let checked = 0;
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const node = walker.currentNode; const element = node.parentElement;
        if (!node.textContent.trim() || !element || !element.checkVisibility() || element.closest('button:disabled, script, style')) continue;
        const style = getComputedStyle(element);
        const foreground = rgb(element.tagName.toLowerCase() === 'text' ? style.fill : style.color);
        if (foreground.length < 3) continue;
        let ancestor = element; let background;
        while (ancestor) {
          const candidate = rgb(getComputedStyle(ancestor).backgroundColor);
          if (candidate.length >= 3 && (candidate.length < 4 || candidate[3] === 1)) { background = candidate; break; }
          ancestor = ancestor.parentElement;
        }
        background ||= [255, 255, 255];
        const a = luminance(foreground), b = luminance(background);
        const ratio = (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
        minimum = Math.min(minimum, ratio); checked++;
        if (ratio < 4.5) failures.push({ text: node.textContent.trim().slice(0, 70), ratio });
      }
      return { checked, minimum, failures };
    """)
    assert not report["failures"], report
    print(f"Contraste de texto: {report['checked']} nodos visibles, mínimo {report['minimum']:.2f}:1.")


with webdriver.Chrome(options=options) as driver:
    wait = WebDriverWait(driver, 120)
    driver.get("http://127.0.0.1:8000/")
    wait.until(lambda browser: browser.find_element(By.ID, "status-text").text == "Modelo conectado")
    assert not driver.find_element(By.ID, "history-editor").get_attribute("open")
    assert driver.find_element(By.ID, "predict-button").get_attribute("disabled")
    assert driver.find_element(By.ID, "completion-badge").text == "0 / 24 horas"
    assert_text_contrast(driver)
    driver.save_screenshot(str(OUTPUT / "desktop-empty.png"))

    driver.find_element(By.ID, "load-demo").click()
    wait.until(lambda browser: browser.find_element(By.ID, "completion-badge").text == "24 / 24 horas")
    wait.until(EC.element_to_be_clickable((By.ID, "predict-button"))).click()
    wait.until(EC.visibility_of_element_located((By.ID, "prediction-result")))
    assert driver.find_element(By.ID, "predicted-value").text != "—"
    assert "sintéticos" in driver.find_element(By.ID, "prediction-warnings").text
    assert driver.find_elements(By.CSS_SELECTOR, "#demand-chart svg")
    driver.execute_script("window.scrollTo(0, 0)")
    driver.save_screenshot(str(OUTPUT / "desktop-prediction.png"))
    print("Desktop: ejemplo, predicción real y gráfico correctos.")

    previous_downloads = set(OUTPUT.glob("prediccion_demanda*.json"))
    driver.find_element(By.ID, "download-result").click()
    wait.until(lambda _browser: set(OUTPUT.glob("prediccion_demanda*.json")) - previous_downloads)
    downloaded = max(OUTPUT.glob("prediccion_demanda*.json"), key=lambda item: item.stat().st_mtime)
    saved = json.loads(downloaded.read_text(encoding="utf-8"))
    assert len(saved["inputs"]["observations"]) == 24
    assert saved["prediction"]["unit"] == "MW"
    print("Descarga: respuesta y 24 entradas incluidas.")

    driver.find_element(By.ID, "manual-entry").click()
    assert driver.find_element(By.ID, "history-editor").get_attribute("open")
    humidity = driver.find_element(By.ID, "humedad_pct")
    humidity.send_keys(Keys.CONTROL, "a")
    humidity.send_keys("101")
    assert driver.find_element(By.ID, "predict-button").get_attribute("disabled")
    assert not driver.find_element(By.ID, "prediction-result").is_displayed()
    assert driver.find_element(By.ID, "completion-badge").text == "23 / 24 horas"
    print("Validación: humedad inválida bloquea la predicción y descarta la anterior.")

    price = driver.find_element(By.ID, "precio_kwh")
    price.send_keys(Keys.CONTROL, "a")
    price.send_keys("810")
    assert driver.find_element(By.ID, "price-notice").is_displayed()
    assert "810" in driver.find_element(By.ID, "price-notice").text
    assert_text_contrast(driver)
    print("Precio: aviso de unidad y valor conservado, sin conversión automática.")

    driver.find_element(By.ID, "csv-file").send_keys(str(ROOT / "examples" / "historial_demo.csv"))
    wait.until(lambda browser: "CSV importado" in browser.find_element(By.ID, "page-message").text)
    assert driver.find_element(By.ID, "completion-badge").text == "24 / 24 horas"
    wait.until(EC.element_to_be_clickable((By.ID, "predict-button"))).click()
    wait.until(EC.visibility_of_element_located((By.ID, "prediction-result")))
    print("CSV: historial completo importado y predicción correcta.")

    invalid = OUTPUT / "invalid.csv"
    invalid.write_text("bad,column\n1,2\n", encoding="utf-8")
    driver.find_element(By.ID, "csv-file").send_keys(str(invalid))
    wait.until(lambda browser: "24 filas" in browser.find_element(By.ID, "page-message").text)
    assert driver.find_element(By.ID, "completion-badge").text == "24 / 24 horas"
    assert driver.find_element(By.ID, "prediction-result").is_displayed()
    print("CSV inválido: error comprensible y datos anteriores conservados.")

    raw_csv = OUTPUT / "without_timestamps.csv"
    csv_text = (ROOT / "examples" / "historial_demo.csv").read_text(encoding="utf-8-sig")
    raw_csv.write_text("\n".join(line.split(",", 1)[1] for line in csv_text.splitlines()) + "\n", encoding="utf-8")
    driver.find_element(By.ID, "csv-file").send_keys(str(raw_csv))
    wait.until(lambda browser: "sin fechas" in browser.find_element(By.ID, "page-message").text)
    assert driver.find_element(By.ID, "end-time").get_attribute("disabled")
    wait.until(EC.element_to_be_clickable((By.ID, "predict-button"))).click()
    wait.until(EC.visibility_of_element_located((By.ID, "prediction-result")))
    assert "siguiente hora: 00:00" in driver.find_element(By.ID, "prediction-for").text.casefold()
    print("CSV sin fechas: variables preservadas sin inventar fechas absolutas.")

    fields = ["timestamp", "demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh", "radiacion_wm2", "precipitacion_mm", "precio_kwh", "festivo"]
    simple = OUTPUT / "simple.csv"
    input_rows = saved["inputs"]["observations"]
    simple.write_text(",".join(fields) + "\n" + "\n".join(",".join(str(row[name]) for name in fields) for row in input_rows) + "\n", encoding="utf-8")
    driver.find_element(By.ID, "csv-file").send_keys(str(simple))
    wait.until(lambda browser: "CSV importado: 24 horas" in browser.find_element(By.ID, "page-message").text)
    assert not driver.find_element(By.ID, "history-editor").get_attribute("open")
    wait.until(EC.element_to_be_clickable((By.ID, "predict-button"))).click()
    wait.until(EC.visibility_of_element_located((By.ID, "prediction-result")))
    print("CSV sencillo: calendario generado y predicción real correcta.")

    # Chrome headless puede imponer un mínimo al ancho de ventana; emular el
    # viewport garantiza que la comprobación use realmente 390 CSS píxeles.
    driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
    driver.execute_script("window.scrollTo(0, 0)")
    assert driver.execute_script("return window.innerWidth") == 390
    assert driver.execute_script("return document.documentElement.scrollWidth <= window.innerWidth")
    assert_text_contrast(driver)
    driver.save_screenshot(str(OUTPUT / "mobile.png"))
    driver.find_element(By.ID, "prediction-result").location_once_scrolled_into_view
    driver.save_screenshot(str(OUTPUT / "mobile-prediction.png"))
    print("Móvil: sin desbordamiento horizontal.")

    driver.execute_cdp_cmd("Emulation.clearDeviceMetricsOverride", {})
    driver.set_window_size(1440, 1300)
    driver.find_element(By.ID, "manual-entry").click()
    driver.find_element(By.ID, "clear-data").click()
    assert driver.find_element(By.ID, "completion-badge").text == "0 / 24 horas"
    assert not driver.find_element(By.ID, "prediction-result").is_displayed()
    assert driver.find_element(By.ID, "demanda_mw").get_attribute("value") == ""
    assert not driver.find_element(By.ID, "end-time").get_attribute("disabled")
    errors = [entry for entry in driver.get_log("browser") if entry["level"] == "SEVERE"]
    assert not errors, errors
    print("Limpieza correcta. Sin errores de JavaScript. Capturas: test-results/.")
