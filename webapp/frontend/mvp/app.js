// MVP de demostracion: consume la API de webapp/backend (mismo servidor).
const CLASES = ["Degeneracion_Discal", "Normal", "Protrusion"];
const ETIQ = { Degeneracion_Discal: "Degeneración", Normal: "Normal", Protrusion: "Protrusión" };
const COLOR = { Degeneracion_Discal: "var(--degeneracion)", Normal: "var(--normal)", Protrusion: "var(--protrusion)" };
const NIVELES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"];
const CORTO = { yolov8s_cls: "YOLOv8s-cls", resnet50: "ResNet-50", svm: "SVM", random_forest: "Random Forest" };

const $ = (s) => document.querySelector(s);
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v) => (100 * v).toFixed(1) + " %";

async function api(ruta, opciones) {
  const r = await fetch(ruta, opciones);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    const e = new Error(typeof j.detail === "string" ? j.detail : `Error ${r.status}`);
    e.motivo = j.motivo; e.rechazada = r.status === 422 && !!j.motivo;
    throw e;
  }
  return j;
}

// ---------- pestañas ----------
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("activa", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("activo", p.id === "tab-" + b.dataset.tab));
}));

// ---------- estado ----------
async function cargarEstado() {
  const e = $("#estado");
  try {
    const s = await api("/api/salud");
    if (s.faltan.length) {
      e.className = "estado error";
      e.textContent = `Faltan modelos: ${s.faltan.join(", ")} · ejecuta python scripts/descargar_modelos.py`;
    } else if (!s.detector_ok) {
      e.className = "estado error";
      e.textContent = "Falta models/best_disc_detector.pt (Git LFS) · el modo estudio no funcionará · ejecuta: git lfs pull";
    } else if (!s.imagenes_demo_ok) {
      e.className = "estado error";
      e.textContent = "Las imágenes de data/demo no se descargaron (falta Git LFS) · ejecuta: git lfs install y git lfs pull";
    } else {
      e.className = "estado ok";
      e.textContent = `4 modelos cargables · ${s.dispositivo.toUpperCase()} · commit ${s.commit || "?"} · ${s.casos_demo} casos`;
    }
  } catch (err) {
    e.className = "estado error";
    e.textContent = "Sin conexión con el backend";
  }
}

// ---------- piezas de interfaz ----------
function barras(puntajes) {
  return CLASES.map((c) => `
    <div class="barra"><span>${ETIQ[c]}</span>
      <div class="pista"><div class="relleno" style="width:${(100 * puntajes[c]).toFixed(1)}%;background:${COLOR[c]}"></div></div>
      <span>${(100 * puntajes[c]).toFixed(0)} %</span></div>`).join("");
}

function tarjetasPrediccion(tr) {
  const comp = Object.fromEntries((tr.comparacion || []).map((c) => [c.modelo, c]));
  const tarjetas = tr.predicciones.map((p) => {
    const c = comp[p.modelo];
    const chips = c ? `<span class="chip ${c.acierta ? "ok" : "mal"}">${c.acierta ? "✓ acierta" : "✗ falla"}</span>
      <span class="chip ${c.coincide_con_registrada ? "ok" : "mal"}" title="Igual a la predicción registrada en la evaluación oficial">${c.coincide_con_registrada ? "✓ reproduce registro" : "✗ difiere del registro"}</span>` : "";
    return `<div class="tarjeta pred"><h3>${esc(p.nombre)}</h3>
      <div class="clase" style="color:${COLOR[p.clase]}">${ETIQ[p.clase]}</div>
      ${barras(p.puntajes)}
      <div class="leyenda">${p.es_probabilidad ? "Probabilidades" : "Puntajes del SVM normalizados (no son probabilidades)"}</div>
      ${chips}</div>`;
  }).join("");
  const real = tr.caso ? ` · etiqueta real: <strong style="color:${COLOR[tr.caso.clase_real]}">${ETIQ[tr.caso.clase_real]}</strong>` : "";
  return `<div class="resumen">
      <span class="chip ${tr.coinciden ? "ok" : "neutro"}">${tr.coinciden ? "Los 4 modelos coinciden" : "Los modelos no coinciden"}</span>
      <span title="${esc(tr.ensamble.nota || "")}">Ensamble (promedio): <strong style="color:${COLOR[tr.ensamble.clase]}">${ETIQ[tr.ensamble.clase]}</strong>${real}</span>
      <span class="nota">Tiempo total: ${tr.ms_total.toFixed(0)} ms</span></div>
    <div class="predicciones">${tarjetas}</div>`;
}

function mini(valores, clase) {
  const max = Math.max(...valores.map(Math.abs), 1e-9);
  return `<div class="mini ${clase || ""}">${valores.map((v) => `<span style="height:${(100 * Math.abs(v) / max).toFixed(0)}%" title="${v}"></span>`).join("")}</div>`;
}

function diagrama(tr, urlOriginal) {
  const P = tr.pasos;
  const pred = Object.fromEntries(tr.predicciones.map((p) => [p.modelo, p]));
  const salida = (m, n) => `<div class="caja salida" style="grid-column:5;grid-row:${n}">
      <h4>Modelo ${n} · ${CORTO[m]}</h4><strong style="color:${COLOR[pred[m].clase]}">${ETIQ[pred[m].clase]}</strong>
      <div class="dato">${pct(pred[m].confianza)}</div></div>`;
  const votos = CLASES.map((c) => `${ETIQ[c]}: <strong>${P.random_forest.votos[c]}</strong>`).join("<br>");
  const decision = CLASES.map((c) => `${ETIQ[c]}: ${P.svm.decision[c].toFixed(2)}`).join("<br>");
  const capaFinal = CLASES.map((c) => `${ETIQ[c]}: ${pct(P.modelo2_capa_final.probabilidades[c])}`).join("<br>");
  return `<h2 style="margin-top:18px">Trazabilidad: cómo viaja la imagen por los 4 modelos</h2>
  <p class="nota">Valores reales de esta ejecución. La caja morada es la ResNet-50 del modelo 2: se calcula <strong>una sola vez</strong> y sus 2.048 características las reutilizan los modelos 2, 3 y 4. YOLOv8 va por un camino independiente.</p>
  <div class="flujo" style="grid-template-columns:1fr 1.15fr 1.1fr 1fr .9fr;grid-template-rows:repeat(4,auto)">
    <div class="caja" style="grid-column:1;grid-row:1/5">
      <h4>Entrada</h4>
      <div class="imgs224">${urlOriginal ? `<img src="${urlOriginal}" alt="original">` : ""}<img src="${tr.imagen_224}" alt="224"></div>
      <div class="dato">Original ${P.entrada.tamano_original.join("×")} px, intensidades ${P.entrada.rango_intensidad.join("–")}<br>
      → redimensionada a 224×224, 3 canales<br>→ normalización ${esc(P.entrada.normalizacion)}<br>
      media ${P.entrada.media_normalizada}, desv. ${P.entrada.desv_normalizada}</div></div>
    <div class="caja yolo" style="grid-column:2/5;grid-row:1">
      <h4>YOLOv8s-cls (camino independiente)</h4>
      <div class="dato">Red convolucional propia (5,1 M parámetros), entrada en [0, 1] sin normalizar · ${P.yolov8s_cls.ms} ms</div></div>
    <div class="caja compartida" style="grid-column:2;grid-row:2/5">
      <h4>ResNet-50 ajustada (compartida)</h4>
      <div class="dato">${P.resnet50.dimension} características · norma ${P.resnet50.norma} · ${(100 * P.resnet50.fraccion_ceros).toFixed(1)} % en cero (ReLU) · ${P.resnet50.ms} ms</div>
      ${mini(P.resnet50.resumen_64)}
      <div class="leyenda">Promedio de cada bloque de 32 características (64 barras)</div></div>
    <div class="caja" style="grid-column:3/5;grid-row:2">
      <h4>Capa final de la ResNet (Dropout + Linear 2048→3 + softmax)</h4><div class="dato">${capaFinal}</div></div>
    <div class="caja" style="grid-column:3;grid-row:3/5">
      <h4>Escalador + PCA</h4>
      <div class="dato">${P.escalador_pca.dimension_entrada} → ${P.escalador_pca.dimension_salida} componentes (${pct(P.escalador_pca.varianza_explicada)} de la varianza) · ${P.escalador_pca.ms} ms</div>
      ${mini(P.escalador_pca.primeras_componentes, "pca")}
      <div class="leyenda">Primeras 8 componentes: ${P.escalador_pca.primeras_componentes.join(", ")}</div></div>
    <div class="caja" style="grid-column:4;grid-row:3">
      <h4>SVM (RBF, C = ${P.svm.C})</h4><div class="dato">${P.svm.vectores_soporte} vectores de soporte<br>Puntaje de decisión:<br>${decision}</div></div>
    <div class="caja" style="grid-column:4;grid-row:4">
      <h4>Random Forest</h4><div class="dato">Votos de los ${P.random_forest.arboles} árboles:<br>${votos}<br>${P.random_forest.ms} ms</div></div>
    ${salida("yolov8s_cls", 1)}${salida("resnet50", 2)}${salida("svm", 3)}${salida("random_forest", 4)}
  </div>`;
}

function fichaCaso(c) {
  return `<div class="tarjeta ficha">
    <img src="/api/casos/${c.id}/imagen" alt="${esc(c.archivo)}">
    <dl><dt>Caso</dt><dd>${esc(c.id)}</dd><dt>Archivo</dt><dd><code>${esc(c.archivo)}</code></dd>
      <dt>Paciente</dt><dd>${esc(c.paciente)}</dd><dt>Nivel</dt><dd>${esc(c.nivel)}</dd>
      <dt>Partición</dt><dd>${esc(c.particion)} (original, no visto al entrenar)</dd>
      <dt>Etiqueta real</dt><dd><strong style="color:${COLOR[c.clase_real]}">${ETIQ[c.clase_real]}</strong></dd></dl>
    <div style="margin-left:auto"><button class="principal" id="btn-predecir">Ejecutar los 4 modelos</button></div></div>`;
}

// ---------- 1. caso a caso ----------
let CASOS = [];
async function cargarCasos() {
  CASOS = await api("/api/casos");
  $("#filtro-clase").innerHTML += CLASES.map((c) => `<option value="${c}">${ETIQ[c]}</option>`).join("");
  $("#filtro-nivel").innerHTML += NIVELES.map((n) => `<option>${n}</option>`).join("");
  pintarGaleria();
}

function pintarGaleria() {
  const fc = $("#filtro-clase").value, fn = $("#filtro-nivel").value;
  $("#galeria").innerHTML = CASOS.filter((c) => (!fc || c.clase_real === fc) && (!fn || c.nivel === fn)).map((c) => `
    <div class="miniatura" data-id="${c.id}" title="${c.id} · ${c.nivel} · ${ETIQ[c.clase_real]}">
      <img src="/api/casos/${c.id}/imagen" loading="lazy" alt="${c.id}"><span class="marca" style="background:${COLOR[c.clase_real]}"></span></div>`).join("");
  document.querySelectorAll(".miniatura").forEach((m) => m.addEventListener("click", () => elegirCaso(m.dataset.id)));
}
$("#filtro-clase").addEventListener("change", pintarGaleria);
$("#filtro-nivel").addEventListener("change", pintarGaleria);

function elegirCaso(id) {
  const c = CASOS.find((x) => x.id === id);
  document.querySelectorAll(".miniatura").forEach((m) => m.classList.toggle("sel", m.dataset.id === id));
  $("#caso-detalle").className = "";
  $("#caso-detalle").innerHTML = fichaCaso(c);
  $("#caso-resultado").innerHTML = "";
  $("#btn-predecir").addEventListener("click", () => predecirCaso(c));
}

async function predecirCaso(c) {
  const b = $("#btn-predecir");
  b.disabled = true; b.textContent = "Ejecutando…";
  try {
    const tr = await api(`/api/casos/${c.id}/predecir`, { method: "POST" });
    $("#caso-resultado").innerHTML = tarjetasPrediccion(tr) + diagrama(tr, `/api/casos/${c.id}/imagen`);
  } catch (err) {
    $("#caso-resultado").innerHTML = `<p class="nota alerta">${esc(err.message)}</p>`;
  } finally {
    b.disabled = false; b.textContent = "Ejecutar de nuevo";
  }
}

// ---------- 2. subir imagen ----------
function mensajeRechazo(err) {
  return `<p class="nota alerta">⛔ ${esc(err.message)}<br>No se muestra ninguna predicción: la imagen no se parece a un recorte de disco lumbar ni a un corte sagital de columna.</p>`;
}

$("#archivo").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const datos = () => { const d = new FormData(); d.append("imagen", f); return d; };
  const url = URL.createObjectURL(f);
  $("#subir-resultado").innerHTML = `<p class="nota">Ejecutando los 4 modelos…</p>`;
  try {
    const tr = await api("/api/predecir", { method: "POST", body: datos() });
    $("#subir-resultado").innerHTML = tarjetasPrediccion(tr) + diagrama(tr, url);
  } catch (err) {
    if (err.motivo === "corte_completo") {
      $("#subir-resultado").innerHTML = `<p class="nota">Parece un corte completo de columna: detectando los 5 discos…</p>`;
      try {
        const r = await api("/api/predecir-corte", { method: "POST", body: datos() });
        $("#subir-resultado").innerHTML = resultadoCortes([r.cortes[0]], () => url, "subir");
        enlazarTrazas("#subir-resultado");
      } catch (err2) {
        $("#subir-resultado").innerHTML = mensajeRechazo(err2);
      }
    } else {
      $("#subir-resultado").innerHTML = err.rechazada ? mensajeRechazo(err) : `<p class="nota alerta">${esc(err.message)}</p>`;
    }
  } finally {
    ev.target.value = "";     // permite volver a elegir el mismo archivo
  }
});

// ---------- 3. simulacion ----------
function matrizHTML(nombre, m, n) {
  const acc = CLASES.reduce((s, _, i) => s + m[i][i], 0);
  const maximo = Math.max(...m.flat(), 0);
  return `<div class="tarjeta matriz"><h3>${esc(nombre)}</h3>
    <table><tr><th>real \\ pred.</th>${CLASES.map((c) => `<th>${ETIQ[c].slice(0, 5)}</th>`).join("")}</tr>
    ${CLASES.map((c, i) => `<tr><th>${ETIQ[c].slice(0, 5)}</th>${m[i].map((v, j) => `<td style="background:${i === j ? `rgba(21,128,61,${maximo ? Math.min(v / maximo, 1) * 0.5 : 0})` : v ? "rgba(185,28,28,.12)" : ""}">${v}</td>`).join("")}</tr>`).join("")}
    </table><div class="leyenda">Aciertos: ${acc} / ${n}${n ? ` (${pct(acc / n)})` : ""}</div></div>`;
}

$("#btn-simular").addEventListener("click", async () => {
  const b = $("#btn-simular");
  b.disabled = true;
  const modelos = Object.keys(CORTO);
  const mats = Object.fromEntries(modelos.map((m) => [m, CLASES.map(() => CLASES.map(() => 0))]));
  const reprod = Object.fromEntries(modelos.map((m) => [m, 0]));
  const nombres = {};
  let n = 0, ens = 0;
  const t0 = performance.now();
  $("#sim-tabla").innerHTML = `<tr><th>Caso</th><th>Paciente</th><th>Nivel</th><th>Real</th>${modelos.map((m) => `<th>${CORTO[m]}</th>`).join("")}<th>Ensamble</th><th>Reproduce</th></tr>`;
  try {
    const lista = CASOS.filter((c) => (!$("#filtro-clase").value || c.clase_real === $("#filtro-clase").value)
      && (!$("#filtro-nivel").value || c.nivel === $("#filtro-nivel").value));     // respeta los filtros de la pestaña 1
    for (const c of lista) {
      $("#sim-progreso").textContent = `  ${n + 1} / ${lista.length}…`;
      const tr = await api(`/api/casos/${c.id}/predecir`, { method: "POST" });
      const real = CLASES.indexOf(c.clase_real);
      const celdas = tr.predicciones.map((p) => {
        nombres[p.modelo] = p.nombre;
        mats[p.modelo][real][CLASES.indexOf(p.clase)] += 1;
        return `<td style="color:${p.clase === c.clase_real ? "var(--ok)" : "var(--mal)"}">${p.clase === c.clase_real ? "✓" : "✗"} ${ETIQ[p.clase]}</td>`;
      }).join("");
      tr.comparacion.forEach((x) => { reprod[x.modelo] += x.coincide_con_registrada; });
      const todasRep = tr.comparacion.every((x) => x.coincide_con_registrada);
      ens += tr.ensamble.clase === c.clase_real;
      n += 1;
      $("#sim-tabla").insertAdjacentHTML("beforeend", `<tr><td>${c.id}</td><td>${c.paciente}</td><td>${c.nivel}</td>
        <td style="color:${COLOR[c.clase_real]}">${ETIQ[c.clase_real]}</td>${celdas}
        <td>${ETIQ[tr.ensamble.clase]}</td><td>${todasRep ? "✓ 4/4" : "✗"}</td></tr>`);
      $("#sim-matrices").innerHTML = modelos.map((m) => matrizHTML(nombres[m] || CORTO[m], mats[m], n)).join("");
    }
    const seg = ((performance.now() - t0) / 1000).toFixed(1);
    $("#sim-progreso").textContent = `  listo en ${seg} s`;
    $("#sim-resumen").innerHTML = `<div class="resumen">${modelos.map((m) => {
      const acc = CLASES.reduce((s, _, i) => s + mats[m][i][i], 0);
      return `<span class="chip neutro">${CORTO[m]}: ${acc}/${n} aciertos · reproduce ${reprod[m]}/${n}</span>`;
    }).join("")}<span class="chip neutro">Ensamble: ${ens}/${n}</span></div>`;
  } catch (err) {
    $("#sim-progreso").textContent = "  " + err.message;
  } finally {
    b.disabled = false;
  }
});

// ---------- 4. modelos ----------
async function cargarModelos() {
  const ms = await api("/api/modelos");
  const fila = (m) => {
    const t = m.metricas_test.test_orig, tc = m.metricas_test.test_completo, v = m.metricas_test.val_completo, cf = m.configuracion;
    const eleccion = cf.mejor_epoca ? `época ${cf.mejor_epoca}` : cf.elegido ? Object.entries(cf.elegido).map(([k, x]) => `${k} = ${x}`).join(", ") : "época 2 (ultralytics)";
    return `<tr><td><strong>${esc(m.nombre)}</strong><br><code>${esc(m.archivo)}</code></td>
      <td class="num">${pct(t.exactitud)}</td><td class="num">${t.f1_macro.toFixed(3)}</td><td class="num">${t.roc_auc_macro.toFixed(3)}</td>
      <td class="num">${pct(tc.exactitud)}</td><td class="num">${tc.f1_macro.toFixed(3)}</td><td class="num">${v.f1_macro.toFixed(3)}</td>
      <td>${esc(eleccion)}</td><td><code title="${m.sha256 || ""}">${m.sha256 ? m.sha256.slice(0, 12) + "…" : "no descargado"}</code></td></tr>`;
  };
  $("#tabla-modelos").innerHTML = `<tr><th>Modelo</th><th>Exactitud<br>test orig.</th><th>F1 macro<br>test orig.</th><th>ROC-AUC<br>test orig.</th>
    <th>Exactitud<br>test completo</th><th>F1 macro<br>test completo</th><th>F1 macro<br>validación</th><th>Elegido con validación</th><th>SHA-256 del archivo</th></tr>` + ms.map(fila).join("");
}

// enlaces directos: #tab=simulacion|modelos|subir abre esa pestana; #caso=caso_03 abre el caso y lo ejecuta;
// #simular=1 ejecuta la simulacion al abrir
function irA(tab) {
  const b = document.querySelector(`.tab[data-tab="${tab}"]`);
  if (b) b.click();
}

// ---------- 5. estudio de resonancia (y cortes completos subidos en la pestaña 2) ----------
const TRAZAS = {};      // "clave" -> traza de un ROI, para abrir el diagrama de trazabilidad al hacer clic
const NOMBRES_NIVEL = NIVELES;

function celdaModelo(p) {
  return `<td style="color:${COLOR[p.clase]}" title="${pct(p.confianza)}">${ETIQ[p.clase]}</td>`;
}

function filaNivel(nivel, info, clave) {
  const roi = info.detectado ? `<img class="roi-mini" src="${info.roi}" alt="ROI ${nivel}">` : "";
  const cabecera = `<th>${roi}${nivel}${info.detectado ? ` <span class="nota">(${pct(info.confianza)})</span>` : ""}</th>`;
  if (!info.detectado) return `<tr>${cabecera}<td colspan="5" class="nota">no detectado</td></tr>`;
  const t = info.traza;
  if (!t) return `<tr>${cabecera}<td colspan="5" class="nota">sin clasificar</td></tr>`;
  if (t.rechazado) return `<tr>${cabecera}<td colspan="5" class="rechazo">⛔ Rechazado: ${esc(t.rechazado.mensaje)}</td></tr>`;
  TRAZAS[clave] = { traza: t, roi: info.roi };
  return `<tr>${cabecera}${t.predicciones.map(celdaModelo).join("")}
    <td class="clic" data-traza="${clave}" style="color:${COLOR[t.ensamble.clase]}"><strong>${ETIQ[t.ensamble.clase]}</strong></td></tr>`;
}

function cajasSVG(c) {
  return `<svg viewBox="0 0 ${c.ancho} ${c.alto}" preserveAspectRatio="none">${NOMBRES_NIVEL.filter((n) => c.niveles[n].detectado).map((n) => {
    const [x1, y1, x2, y2] = c.niveles[n].caja;
    return `<rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" fill="none" stroke="#f59e0b" stroke-width="2"/>
      <text x="${x2 + 4}" y="${(y1 + y2) / 2}" fill="#f59e0b" font-size="${Math.max(10, c.alto / 28)}">${n}</text>`;
  }).join("")}</svg>`;
}

function resultadoCortes(cortes, urlCorte, prefijo) {
  return `<div class="cortes-res">` + cortes.map((c) => {
    const aviso = c.columna_valida ? "" : `<p class="nota alerta">El detector halló ${c.n_detectados} de 5 discos${c.orden_ok ? "" : " y no están en orden"}: este corte no parece un corte sagital lumbar y no se clasifica.</p>`;
    const filas = NOMBRES_NIVEL.map((n) => filaNivel(n, c.niveles[n], `${prefijo}-${c.corte}-${n}`)).join("");
    return `<div class="tarjeta"><h3>Corte ${c.corte + 1}</h3>${aviso}
      <div class="fila-corte"><div class="vista"><img src="${urlCorte(c.corte)}" alt="corte">${cajasSVG(c)}</div>
      <table class="tabla"><tr><th>Nivel</th><th>YOLOv8s</th><th>ResNet-50</th><th>SVM</th><th>Random Forest</th><th>Ensamble</th></tr>${filas}</table></div></div>`;
  }).join("") + `</div>`;
}

function enlazarTrazas(raiz) {
  document.querySelectorAll(`${raiz} [data-traza]`).forEach((td) => td.addEventListener("click", () => {
    const { traza, roi } = TRAZAS[td.dataset.traza];
    let d = document.querySelector(`${raiz} .traza-abierta`);
    if (!d) { d = document.createElement("div"); d.className = "traza-abierta"; document.querySelector(raiz).appendChild(d); }
    d.innerHTML = diagrama({ ...traza, imagen_224: roi }, roi);
    d.scrollIntoView({ behavior: "smooth" });
  }));
}

const EST = { id: null, serie: null, ficha: null, sel: new Set(), resultados: [] };

function equipoAviso(e) {
  const siemens = /siemens/i.test(e.fabricante), campo = parseFloat(e.campo_tesla);
  return (siemens && campo === 1.5) ? "" : `<p class="nota alerta">Equipo: ${esc(e.fabricante || "?")} ${esc(e.modelo)} ${esc(e.campo_tesla)} T. Los modelos se entrenaron con un solo equipo (Siemens, 1,5 T): con otro equipo los resultados pueden ser menos fiables.</p>`;
}

function pintarFicha() {
  const f = EST.ficha, s = f.series.find((x) => x.id === EST.serie);
  const opciones = f.series.map((x) => `<option value="${x.id}" ${x.id === EST.serie ? "selected" : ""}>${esc(x.descripcion || "(sin descripción)")} · ${x.plano} · ${x.n_cortes} cortes${x.t2_sagital ? " · T2 sagital ✓" : ""}</option>`).join("");
  const tira = Array.from({ length: s.n_cortes }, (_, i) => `
    <div class="corte ${EST.sel.has(i) ? "sel" : ""} ${s.propuestos.includes(i) ? "prop" : ""}" data-n="${i}" title="Corte ${i + 1}">
      <img loading="lazy" src="/api/estudios/${f.id}/series/${s.id}/cortes/${i}/imagen" alt="corte ${i + 1}"><span>${i + 1}</span></div>`).join("");
  $("#est-ficha").innerHTML = `${f.archivos_omitidos ? `<p class="nota">${f.archivos_omitidos} archivo(s) no eran cortes DICOM y se omitieron.</p>` : ""}
    <div class="filtros"><label>Serie: <select id="est-serie">${opciones}</select></label></div>
    ${s.plano !== "sagital" ? `<p class="nota alerta">Esta serie no es sagital: para clasificar discos necesitas una serie T2 sagital.</p>` : ""}
    ${equipoAviso(s.equipo)}
    <p class="nota">Elige los cortes midsagitales (los verdes son la propuesta: 30 %–70 % central). Clic en un corte para quitarlo o añadirlo. Seleccionados: <strong id="est-n">${EST.sel.size}</strong></p>
    <div class="tira" id="est-tira">${tira}</div>
    <div class="filtros"><button id="est-propuesta">Usar la propuesta</button><button id="est-ninguno">Quitar todos</button>
      <button id="est-png">Descargar PNG elegidos</button><button id="est-analizar" class="principal">Analizar cortes elegidos</button></div>`;
  $("#est-serie").addEventListener("change", (e) => { EST.serie = e.target.value; EST.sel = new Set(EST.ficha.series.find((x) => x.id === EST.serie).propuestos); EST.resultados = []; $("#est-resultado").innerHTML = ""; pintarFicha(); });
  document.querySelectorAll("#est-tira .corte").forEach((el) => el.addEventListener("click", () => {
    const n = +el.dataset.n; EST.sel.has(n) ? EST.sel.delete(n) : EST.sel.add(n);
    el.classList.toggle("sel", EST.sel.has(n)); $("#est-n").textContent = EST.sel.size;
  }));
  $("#est-propuesta").addEventListener("click", () => { EST.sel = new Set(s.propuestos); pintarFicha(); });
  $("#est-ninguno").addEventListener("click", () => { EST.sel = new Set(); pintarFicha(); });
  $("#est-png").addEventListener("click", descargarPNG);
  $("#est-analizar").addEventListener("click", analizarEstudio);
}

$("#est-subir").addEventListener("click", async () => {
  const archivos = $("#est-archivos").files;
  if (!archivos.length) { $("#est-progreso").textContent = "  Elige los archivos del estudio."; return; }
  const b = $("#est-subir"); b.disabled = true;
  $("#est-progreso").textContent = "  Subiendo y convirtiendo a PNG…";
  const datos = new FormData();
  for (const a of archivos) datos.append("archivos", a);
  try {
    EST.ficha = await api("/api/estudios", { method: "POST", body: datos });
    EST.id = EST.ficha.id; EST.serie = EST.ficha.serie_sugerida; EST.resultados = [];
    EST.sel = new Set(EST.ficha.series.find((x) => x.id === EST.serie).propuestos);
    $("#est-progreso").textContent = ""; $("#est-borrar").hidden = false; $("#est-resultado").innerHTML = "";
    pintarFicha();
  } catch (err) {
    $("#est-progreso").textContent = "  " + err.message;
  } finally { b.disabled = false; }
});

$("#est-borrar").addEventListener("click", async () => {
  if (!EST.id) return;
  await fetch(`/api/estudios/${EST.id}`, { method: "DELETE" }).catch(() => {});
  EST.id = null; $("#est-ficha").innerHTML = ""; $("#est-resultado").innerHTML = ""; $("#est-borrar").hidden = true;
  $("#est-progreso").textContent = "  Estudio borrado del servidor.";
});

async function descargarPNG() {
  if (!EST.sel.size) return;
  const r = await fetch(`/api/estudios/${EST.id}/png.zip`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ serie: EST.serie, cortes: [...EST.sel].sort((a, b) => a - b).slice(0, 50) }) });
  if (!r.ok) { $("#est-progreso").textContent = "  No se pudo descargar."; return; }
  const url = URL.createObjectURL(await r.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: "cortes.zip" });
  a.click(); URL.revokeObjectURL(url);
}

function resumenEstudio(resultados) {
  const filas = NOMBRES_NIVEL.map((n) => {
    const trazas = resultados.map((c) => c.niveles[n].traza).filter((t) => t && t.predicciones);
    if (!trazas.length) return `<tr><th>${n}</th><td colspan="5" class="nota">sin datos</td></tr>`;
    const moda = (xs) => { const c = {}; xs.forEach((x) => (c[x] = (c[x] || 0) + 1)); return Object.entries(c).sort((a, b) => b[1] - a[1])[0]; };
    const celdas = Object.keys(CORTO).map((m) => {
      const [cl, k] = moda(trazas.map((t) => t.predicciones.find((p) => p.modelo === m).clase));
      return `<td style="color:${COLOR[cl]}">${ETIQ[cl]} <span class="nota">${k}/${trazas.length}</span></td>`;
    }).join("");
    const [ce, ke] = moda(trazas.map((t) => t.ensamble.clase));
    return `<tr><th>${n}</th>${celdas}<td style="color:${COLOR[ce]}"><strong>${ETIQ[ce]}</strong> <span class="nota">${ke}/${trazas.length}</span></td></tr>`;
  }).join("");
  return `<h2 style="margin-top:18px">Resumen por nivel (clase más votada entre los cortes elegidos)</h2>
    <table class="tabla"><tr><th>Nivel</th>${Object.values(CORTO).map((m) => `<th>${m}</th>`).join("")}<th>Ensamble</th></tr>${filas}</table>`;
}

async function analizarEstudio() {
  const cortes = [...EST.sel].sort((a, b) => a - b);
  if (!cortes.length) { $("#est-progreso").textContent = "  Elige al menos un corte."; return; }
  const b = $("#est-analizar"); b.disabled = true;
  EST.resultados = [];
  $("#est-resultado").innerHTML = `<div id="est-cortes"></div><div id="est-resumen"></div>`;
  const url = (n) => `/api/estudios/${EST.id}/series/${EST.serie}/cortes/${n}/imagen`;
  try {
    for (const [i, n] of cortes.entries()) {
      $("#est-progreso").textContent = `  Corte ${i + 1} de ${cortes.length} (${n + 1})… cada corte tarda unos segundos`;
      const r = await api(`/api/estudios/${EST.id}/predecir`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ serie: EST.serie, cortes: [n] }) });
      EST.resultados.push(r.cortes[0]);
      $("#est-cortes").innerHTML = resultadoCortes(EST.resultados, url, "est");
      $("#est-resumen").innerHTML = resumenEstudio(EST.resultados);
      enlazarTrazas("#est-resultado");
    }
    $("#est-progreso").textContent = "  Listo. Haz clic en el ensamble de un nivel para ver la trazabilidad.";
  } catch (err) {
    $("#est-progreso").textContent = "  " + err.message;
  } finally { b.disabled = false; }
}

cargarEstado();
cargarCasos().then(() => {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("tab")) irA(h.get("tab"));
  if (h.get("simular") === "1") { irA("simulacion"); $("#btn-simular").click(); }
  if (h.get("caso") && CASOS.some((c) => c.id === h.get("caso"))) {
    irA("caso");
    elegirCaso(h.get("caso"));
    predecirCaso(CASOS.find((c) => c.id === h.get("caso")));
  }
}).catch((e) => { $("#galeria").innerHTML = `<p class="nota alerta">${esc(e.message)}</p>`; });
cargarModelos().catch((e) => { $("#tabla-modelos").innerHTML = `<tr><td>${esc(e.message)}</td></tr>`; });
