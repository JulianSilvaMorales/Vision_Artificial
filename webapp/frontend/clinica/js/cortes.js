// Paso 2: elegir la serie y los cortes centrales. El detector recomienda los 3 mas centrales; el radiologo decide.
import { postJSON } from "./api.js";
import { $, $$, E, avisar, enLista, esc, reiniciarSerie, serieActual, urlCorte } from "./estado.js";
import { habilitar, hecho, irA } from "./nav.js";
import { analizar } from "./resultados.js";
import { cajasSVG, crearVisor } from "./visor.js";

let visor, verCajas = false;
const motivo = (p) => p.motivo.replace("estan", "están");

export function abrirEstudio(ficha) {
  E.estudio = ficha;
  E.serie = ficha.serie_sugerida;
  document.dispatchEvent(new CustomEvent("estudio-nuevo"));
  abrirSerie();
  habilitar("cortes"); hecho("cargar");
  habilitar("resultados", false); habilitar("informe", false); hecho("resultados", false);
  irA("cortes");
}

// Recarga de la pagina: el estudio sigue en el servidor y el estado se recupera sin volver a puntuar
export function restaurarCortes() {
  habilitar("cortes"); hecho("cargar");
  pintar();
}

function abrirSerie() {
  reiniciarSerie();
  const s = serieActual();
  E.sel = new Set(s.propuestos);
  E.actual = s.propuestos.length ? s.propuestos[Math.floor(s.propuestos.length / 2)] : 0;
  document.dispatchEvent(new CustomEvent("serie-nueva"));
  pintar();
  puntuar();
}

async function puntuar() {
  const s = serieActual();
  if (s.plano !== "sagital" || !s.propuestos.length) return;
  const id = E.estudio.id, serie = E.serie;
  $("#recomendacion").innerHTML = `<p class="cargando">Buscando con el detector los cortes más centrales…</p>`;
  try {
    const r = await postJSON(`/api/estudios/${id}/puntuar`, { serie, cortes: s.propuestos.slice(0, 50) });
    if (!E.estudio || E.estudio.id !== id || E.serie !== serie) return;     // cambio de estudio o de serie mientras tanto
    r.cortes.forEach((p) => { E.puntos[p.corte] = p; });
    E.recomendados = r.recomendados;
    if (r.recomendados.length) { E.sel = new Set(r.recomendados); E.actual = r.recomendados[Math.floor(r.recomendados.length / 2)]; }
    else E.avisoPuntos = "El detector no encontró la columna lumbar completa en ninguno de los cortes centrales. Revisa que la serie sea T2 sagital lumbar.";
  } catch (e) {
    E.avisoPuntos = `No se pudieron evaluar los cortes (${e.message}). Se proponen los del 30 % al 70 % central.`;
  }
  pintar();
}

function avisosSerie(s) {
  const out = [];
  if (E.estudio.archivos_omitidos) out.push(`${E.estudio.archivos_omitidos} archivo(s) del estudio no eran cortes DICOM y se omitieron.`);
  if (s.plano !== "sagital") out.push("Esta serie no es sagital. Para clasificar los discos elige una serie T2 sagital.");
  const eq = s.equipo || {};
  if (!(/siemens/i.test(eq.fabricante) && parseFloat(eq.campo_tesla) === 1.5))
    out.push(`Equipo ${[eq.fabricante, eq.modelo].filter(Boolean).join(" ") || "desconocido"}${eq.campo_tesla ? ` ${eq.campo_tesla} T` : ""}: los modelos se entrenaron con un Siemens de 1,5 T, así que con otro equipo pueden ser menos fiables.`);
  return out.map((t) => `<p class="alerta">${esc(t)}</p>`).join("");
}

function textoRecomendacion() {
  if (E.avisoPuntos) return `<p class="alerta">${esc(E.avisoPuntos)}</p>`;
  const ps = Object.values(E.puntos);
  if (!ps.length) return "";
  const rec = ps.filter((p) => p.recomendado).sort((a, b) => a.corte - b.corte);
  const lejos = ps.filter((p) => !p.recomendado).sort((a, b) => a.posicion - b.posicion);
  return `<h2>Recomendación del detector</h2>
    <ul class="lista-rec">${rec.map((p) => `<li><b class="estrella">★</b> Corte ${p.corte + 1}<span>${esc(motivo(p))}</span></li>`).join("")}</ul>
    ${lejos.length ? `<p class="ayuda">Poco centrales: ${enLista(lejos.slice(0, 6).map((p) => `${p.corte + 1}`))}${lejos.length > 6 ? "…" : ""}. Pasa el mouse sobre cada miniatura para ver el motivo.</p>` : ""}
    <p class="ayuda">Es un criterio de apoyo: revisa los cortes en el visor antes de analizar.</p>`;
}

function pintarTira() {
  const s = serieActual();
  $("#tira").innerHTML = Array.from({ length: s.n_cortes }, (_, i) => {
    const p = E.puntos[i], sel = E.sel.has(i);
    const titulo = `Corte ${i + 1}${p ? ` · ${p.recomendado ? "recomendado" : "poco central"}: ${motivo(p)}` : ""}`;
    return `<div class="mini ${sel ? "sel" : ""} ${i === E.actual ? "actual" : ""} ${p && !p.recomendado ? "lejos" : ""}" role="option"
        aria-selected="${sel}" data-n="${i}" title="${esc(titulo)}">
      <img loading="lazy" src="${urlCorte(i)}" alt="">
      <button type="button" class="marcar" aria-label="${sel ? "Quitar" : "Elegir"} el corte ${i + 1}" aria-pressed="${sel}">✓</button>
      <span class="num">${i + 1}</span>${p && p.recomendado ? '<b class="estrella">★</b>' : ""}</div>`;
  }).join("");
  $$("#tira .mini").forEach((el) => {
    el.addEventListener("click", (e) => {
      const n = +el.dataset.n;
      if (e.target.closest(".marcar")) alternar(n);
      else ver(n);
    });
  });
  const actual = $("#tira .mini.actual");
  if (actual) actual.scrollIntoView({ block: "nearest", inline: "center" });
}

function pintarLateral() {
  const s = serieActual();
  $("#sel-serie").innerHTML = E.estudio.series.map((x) => `<option value="${x.id}" ${x.id === E.serie ? "selected" : ""}>
    ${esc(x.descripcion || "Serie sin descripción")} · ${x.plano} · ${x.n_cortes} cortes${x.t2_sagital ? " · T2 sagital" : ""}</option>`).join("");
  $("#serie-avisos").innerHTML = avisosSerie(s);
  $("#recomendacion").innerHTML = textoRecomendacion();
  $("#btn-recomendados").disabled = !E.recomendados.length;
  const n = E.sel.size;
  $("#btn-analizar").disabled = !n;
  $("#btn-analizar").textContent = n ? `Analizar ${n} corte${n === 1 ? "" : "s"}` : "Elige al menos un corte";
  $("#analizar-info").textContent = n ? `Elegidos: ${enLista([...E.sel].sort((a, b) => a - b).map((x) => x + 1))}. Cada corte tarda unos segundos.` : "";
}

function pintarVisor() {
  const s = serieActual(), p = E.puntos[E.actual];
  const marcas = [p && p.recomendado ? "★ recomendado" : "", E.sel.has(E.actual) ? "elegido" : ""].filter(Boolean).join(" · ");
  visor.mostrar(urlCorte(E.actual), { esquina: `Corte ${E.actual + 1} de ${s.n_cortes}${marcas ? ` · ${marcas}` : ""}` });
  const b = $("#btn-elegir");
  b.setAttribute("aria-pressed", String(E.sel.has(E.actual)));
  b.querySelector("span").textContent = E.sel.has(E.actual) ? "Elegido" : "Elegir este corte";
  dibujarCajas();
}

async function dibujarCajas() {
  visor.dibujar("");
  $("#btn-cajas").setAttribute("aria-pressed", String(verCajas));
  if (!verCajas) return;
  const n = E.actual;
  try {
    if (!E.cajas[n]) E.cajas[n] = (await postJSON(`/api/estudios/${E.estudio.id}/recortar`, { serie: E.serie, cortes: [n] })).cortes[0];
    if (verCajas && E.actual === n) visor.dibujar(cajasSVG(E.cajas[n]));
  } catch (e) { avisar(`No se pudieron dibujar las cajas: ${e.message}`, "error"); }
}

function pintar() { pintarLateral(); pintarTira(); pintarVisor(); }

function ver(n) {
  const s = serieActual();
  E.actual = Math.max(0, Math.min(s.n_cortes - 1, n));
  $$("#tira .mini").forEach((el) => el.classList.toggle("actual", +el.dataset.n === E.actual));
  const el = $("#tira .mini.actual");
  if (el) el.scrollIntoView({ block: "nearest", inline: "nearest" });
  pintarVisor();
}

function alternar(n) {
  E.sel.has(n) ? E.sel.delete(n) : E.sel.add(n);
  pintarLateral(); pintarTira(); pintarVisor();
}

export function iniciarCortes() {
  visor = crearVisor($("#visor-cortes"), {
    alRodar: (d) => ver(E.actual + d),
    extraBarra: `<span class="separa"></span>
      <button type="button" id="btn-cajas" aria-pressed="false" title="Cajas del detector (B)">Cajas del detector</button>
      <button type="button" id="btn-elegir" aria-pressed="false" title="Elegir o quitar este corte (Espacio)"><span>Elegir este corte</span></button>`,
  });
  visor.vacio("Carga un estudio para ver sus cortes");
  $("#btn-cajas").addEventListener("click", () => { verCajas = !verCajas; dibujarCajas(); });
  $("#btn-elegir").addEventListener("click", () => alternar(E.actual));
  $("#sel-serie").addEventListener("change", (e) => { E.serie = e.target.value; abrirSerie(); });
  $("#btn-recomendados").addEventListener("click", () => { E.sel = new Set(E.recomendados); pintar(); });
  $("#btn-ninguno").addEventListener("click", () => { E.sel = new Set(); pintar(); });
  $("#btn-analizar").addEventListener("click", () => analizar([...E.sel].sort((a, b) => a - b)));
  document.addEventListener("keydown", (e) => {
    if (!$("#paso-cortes").classList.contains("activo") || !E.estudio) return;
    if (e.target.closest("input, select, textarea, dialog")) return;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") ver(E.actual + 1);
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") ver(E.actual - 1);
    else if (e.key === " " && !e.target.closest("button")) alternar(E.actual);
    else if (e.key.toLowerCase() === "b") { verCajas = !verCajas; dibujarCajas(); }
    else return;
    e.preventDefault();
  });
}
