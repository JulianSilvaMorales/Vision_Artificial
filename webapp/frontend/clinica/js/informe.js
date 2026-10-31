// Paso 4: informe con el resultado del modelo y, al lado, la valoracion y las observaciones del radiologo.
// Las observaciones no salen del navegador: se guardan en localStorage (borrador) y en el PDF que se imprima.
import { fraseGeneral, resumenEstudio } from "./analisis.js";
import { $, $$, E, ETIQ, avisar, esc, guardado, serieActual, urlCorte } from "./estado.js";
import { alEntrarEn, irA } from "./nav.js";
import { claseColor } from "./resultados.js";
import { cajasSVG } from "./visor.js";

const CAMPOS = [
  { id: "estudio", etiqueta: "Identificación del estudio", tipo: "text", ayuda: "Opcional. Los datos del DICOM no se usan.", ej: "Estudio 2026-118, hombre de 54 años" },
  { id: "hallazgos", etiqueta: "Hallazgos", frases: ["Hallazgos concordantes con el análisis automático.", "Sin estenosis significativa del canal central.",
    "Disminución de la señal en T2 (deshidratación) del disco ", "Fisura anular en "] },
  { id: "impresion", etiqueta: "Impresión", frases: ["Cambios degenerativos discales en ", "Protrusión discal en ", "Sin alteraciones discales significativas."] },
  { id: "recomendacion", etiqueta: "Recomendación", frases: ["Se sugiere correlación clínica.", "Control por imagen en 6 meses.", "Valoración por especialista en columna."] },
  { id: "radiologo", etiqueta: "Radiólogo que revisa", tipo: "text", ej: "Nombre y apellido" },
];
const MAX = 2000;
let inf = { campos: {}, firma: null };

const clave = () => `deepspine.informe.${E.estudio.id}`;
const guardar = () => E.estudio && guardado.escribir(clave(), { ...inf, valoracion: E.valoracion });

function pintarFormulario() {
  const firmado = !!inf.firma;
  const sinRevisar = resumenEstudio(E.resultados).filter((r) => !(E.valoracion[r.n] || {}).estado).length;
  $("#form-informe").innerHTML = `<h1 id="t-informe">Informe</h1>
    <p class="estado-informe ${firmado ? "firmado" : ""}">${firmado ? `Finalizado por ${esc(inf.firma.nombre)} el ${esc(inf.firma.fecha)}` : "Borrador · se guarda solo en este navegador"}</p>
    ${sinRevisar ? `<p class="alerta">${sinRevisar} disco${sinRevisar === 1 ? "" : "s"} sin tu valoración. <button type="button" class="enlace" id="ir-resultados">Revisar en Resultados</button></p>` : ""}
    ${CAMPOS.map((c) => `<div class="campo-informe">
      <label class="campo" for="f-${c.id}">${c.etiqueta}</label>
      ${c.tipo === "text"
        ? `<input type="text" id="f-${c.id}" data-campo="${c.id}" maxlength="200" placeholder="${esc(c.ej)}" value="${esc(inf.campos[c.id] || "")}" ${firmado ? "disabled" : ""}>`
        : `<textarea id="f-${c.id}" data-campo="${c.id}" maxlength="${MAX}" rows="3" ${firmado ? "disabled" : ""}>${esc(inf.campos[c.id] || "")}</textarea>`}
      ${c.ayuda ? `<p class="ayuda">${c.ayuda}</p>` : ""}
      ${c.frases && !firmado ? `<div class="frases">${c.frases.map((f, i) => `<button type="button" data-frase="${c.id}:${i}">${esc(f.trim())}</button>`).join("")}</div>` : ""}
    </div>`).join("")}
    <div class="fila-botones">
      ${firmado ? `<button type="button" class="secundario" id="btn-editar">Editar (quita la firma)</button>`
        : `<button type="button" class="principal" id="btn-finalizar">Finalizar informe</button>`}
      <button type="button" class="secundario" id="btn-imprimir">Imprimir o guardar PDF</button>
    </div>
    <div class="fila-botones">
      <button type="button" class="secundario" id="btn-png">Descargar cortes en PNG</button>
      <button type="button" class="secundario peligro" id="btn-borrar">Borrar estudio</button>
    </div>`;
  if ($("#ir-resultados")) $("#ir-resultados").addEventListener("click", () => irA("resultados"));
  $$("#form-informe [data-campo]").forEach((el) => el.addEventListener("input", () => {
    inf.campos[el.dataset.campo] = el.value.slice(0, MAX); guardar(); pintarHoja();
  }));
  $$("#form-informe [data-frase]").forEach((b) => b.addEventListener("click", () => {
    const [id, i] = b.dataset.frase.split(":"), el = $(`#f-${id}`);
    const frase = CAMPOS.find((c) => c.id === id).frases[+i];
    el.value = (el.value && !/\s$/.test(el.value) ? `${el.value} ` : el.value) + frase;
    el.dispatchEvent(new Event("input"));
    el.focus();
    el.setSelectionRange(el.value.length, el.value.length);
  }));
  if ($("#btn-finalizar")) $("#btn-finalizar").addEventListener("click", () => finalizar(sinRevisar));
  if ($("#btn-editar")) $("#btn-editar").addEventListener("click", () => { inf.firma = null; guardar(); pintarInforme(); });
  $("#btn-imprimir").addEventListener("click", () => window.print());
  $("#btn-png").addEventListener("click", descargarPNG);
  $("#btn-borrar").addEventListener("click", borrarEstudio);
}

function finalizar(sinRevisar) {
  const nombre = (inf.campos.radiologo || "").trim();
  if (!nombre) { avisar("Escribe el nombre del radiólogo que revisa antes de finalizar.", "error"); $("#f-radiologo").focus(); return; }
  if (sinRevisar && !confirm(`Hay ${sinRevisar} disco(s) sin tu valoración. En el informe figurarán como "Sin revisar". ¿Finalizar igual?`)) return;
  inf.firma = { nombre, fecha: new Date().toLocaleString("es", { dateStyle: "long", timeStyle: "short" }) };
  guardar(); pintarInforme();
  avisar("Informe finalizado. Ya puedes imprimirlo o guardarlo como PDF.", "ok");
}

function valoracionCelda(r) {
  const v = E.valoracion[r.n] || {};
  if (v.estado === "concuerda") return `<td>Concuerda</td><td></td>`;
  if (v.estado === "discrepa") return `<td><strong style="color:${claseColor(v.clase === "Otro" ? "Otro" : v.clase)}">Discrepa: ${ETIQ[v.clase]}</strong></td><td>${esc(v.nota || "")}</td>`;
  return `<td class="tenue">Sin revisar</td><td></td>`;
}

function imagenHoja() {
  const validos = E.resultados.filter((c) => c.columna_valida);
  const c = validos[Math.floor(validos.length / 2)] || E.resultados[0];
  const color = (n) => { const t = c.niveles[n].traza; return t && t.ensamble ? claseColor(t.ensamble.clase) : "var(--aviso)"; };
  return `<figure class="hoja-figura"><div class="hoja-img" style="aspect-ratio:${c.ancho}/${c.alto}"><img src="${urlCorte(c.corte)}" alt="Corte ${c.corte + 1}">
    <svg viewBox="0 0 ${c.ancho} ${c.alto}" preserveAspectRatio="none">${cajasSVG(c, { color, escalaTexto: 1.8 })}</svg></div>
    <figcaption>Corte ${c.corte + 1}: cada caja tiene el color de la clase del ensamble en este corte.</figcaption></figure>`;
}

function pintarHoja() {
  if (!E.resultados.length) { $("#hoja").innerHTML = ""; return; }
  const s = serieActual(), res = resumenEstudio(E.resultados), t = (id) => esc(inf.campos[id] || "").replace(/\n/g, "<br>") || '<span class="tenue">—</span>';
  $("#hoja").innerHTML = `<header class="hoja-cab"><div><strong>DeepSpine</strong> · Apoyo al diagnóstico lumbar</div><div>${new Date().toLocaleDateString("es", { dateStyle: "long" })}</div></header>
    <h2>Informe de resonancia lumbar · T2 sagital</h2>
    <dl class="hoja-datos"><dt>Estudio</dt><dd>${t("estudio")}</dd><dt>Serie</dt><dd>${esc(s.descripcion || "sin descripción")} · ${s.n_cortes} cortes</dd>
      <dt>Cortes analizados</dt><dd>${E.resultados.map((c) => c.corte + 1).join(", ")}</dd></dl>
    <h3>Análisis automático</h3><p>${esc(fraseGeneral(res, E.resultados))}</p>
    <table class="hoja-tabla"><thead><tr><th>Disco</th><th>Modelo</th><th>Cortes</th><th>Votos</th><th>Radiólogo</th><th>Nota</th></tr></thead><tbody>
      ${res.map((r) => `<tr><th>${r.n}</th><td><strong style="color:${claseColor(r.clase)}">${r.clase ? ETIQ[r.clase] : r.estado === "mixto" ? "No concluyente" : "No evaluable"}</strong></td>
        <td>${r.total ? `${r.k ?? 0}/${r.total}` : "—"}</td><td>${r.modelos ? `${r.modelos.k}/${r.modelos.total}` : "—"}</td>${valoracionCelda(r)}</tr>`).join("")}
    </tbody></table>
    <div class="hoja-cuerpo"><div><h3>Hallazgos</h3><p>${t("hallazgos")}</p><h3>Impresión</h3><p>${t("impresion")}</p>
      <h3>Recomendación</h3><p>${t("recomendacion")}</p></div>${imagenHoja()}</div>
    <footer class="hoja-pie"><p class="firma">${inf.firma ? `Revisado por <strong>${esc(inf.firma.nombre)}</strong> · ${esc(inf.firma.fecha)}` : "<strong>Borrador</strong> · sin finalizar"}</p>
      <p>Herramienta de apoyo con fines académicos: clasificación automática de discos L1-S1 con 4 modelos (YOLOv8s-cls, ResNet-50, SVM y Random Forest).
      La decisión diagnóstica corresponde al radiólogo.</p></footer>`;
}

export function pintarInforme() { pintarFormulario(); pintarHoja(); }

async function descargarPNG() {
  const cortes = E.resultados.map((c) => c.corte);
  try {
    const r = await fetch(`/api/estudios/${E.estudio.id}/png.zip`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ serie: E.serie, cortes }) });
    if (!r.ok) throw new Error(`Error ${r.status}`);
    const url = URL.createObjectURL(await r.blob());
    Object.assign(document.createElement("a"), { href: url, download: "cortes.zip" }).click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) { avisar(`No se pudieron descargar los cortes (${e.message}). El estudio se borra a los 30 minutos.`, "error"); }
}

async function borrarEstudio() {
  if (!confirm("¿Borrar el estudio del servidor y el borrador del informe? No se puede deshacer.")) return;
  await fetch(`/api/estudios/${E.estudio.id}`, { method: "DELETE" }).catch(() => {});
  guardado.borrar(clave());
  E.estudio = null;                    // asi la sesion no se vuelve a guardar al recargar
  location.reload();
}

export function iniciarInforme() {
  document.addEventListener("estudio-nuevo", () => { inf = { campos: {}, firma: null }; });
  document.addEventListener("analisis-nuevo", () => {
    // un borrador del mismo estudio se conserva (los campos de texto); la firma no, porque cambio el analisis
    const previo = guardado.leer(clave());
    inf = { campos: (previo && previo.campos) || {}, firma: null };
  });
  document.addEventListener("valoracion-cambio", () => {
    if (inf.firma) { inf.firma = null; avisar("Se quitó la firma del informe porque cambió una valoración.", "info"); }
    guardar();
  });
  alEntrarEn("informe", () => E.resultados.length && pintarInforme());
}

// Al recargar la pagina con un estudio abierto se recupera tambien el borrador
export function restaurarInforme() {
  const previo = guardado.leer(clave());
  if (previo) inf = { campos: previo.campos || {}, firma: previo.firma || null };
}
