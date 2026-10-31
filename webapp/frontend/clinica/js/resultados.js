// Paso 3: analiza los cortes elegidos y muestra, por disco, la clase, el consenso entre cortes, el acuerdo entre
// modelos y la valoracion del radiologo (concuerda o discrepa).
import { resumenEstudio, fraseGeneral, fraseNivel } from "./analisis.js";
import { postJSON } from "./api.js";
import { $, $$, CLASES, E, ETIQ, MODELOS, NIVELES, avisar, esc, pct, urlCorte } from "./estado.js";
import { habilitar, hecho, irA } from "./nav.js";
import { cajasSVG, crearVisor } from "./visor.js";

let visor, verCorte = null, activo = null;
let corrida = 0;      // cada analisis tiene su numero: uno viejo que sigue en curso ya no escribe resultados
const ESTADO_TXT = { no_evaluable: "No evaluable", mixto: "No concluyente" };
const CONF_TXT = { alta: "Confianza alta", media: "Confianza media", baja: "Confianza baja" };

export const claseColor = (cl) => (cl ? `var(--${cl})` : "var(--aviso)");

function colorNivel(c, n) {
  const t = c.niveles[n].traza;
  return t && t.ensamble ? claseColor(t.ensamble.clase) : "var(--aviso)";
}

export async function analizar(cortes) {
  const esta = ++corrida, id = E.estudio.id, serie = E.serie;
  E.resultados = []; E.valoracion = {}; activo = null; verCorte = null;
  document.dispatchEvent(new CustomEvent("analisis-nuevo"));
  habilitar("resultados"); hecho("cortes"); habilitar("informe", false); hecho("resultados", false);
  $("#btn-ir-informe").disabled = true;
  irA("resultados");
  const prog = $("#analisis-progreso");
  prog.hidden = false;
  pintar();
  try {
    for (const [i, n] of cortes.entries()) {
      prog.innerHTML = `<span>Analizando el corte ${n + 1} (${i + 1} de ${cortes.length})…</span><progress max="${cortes.length}" value="${i}"></progress>`;
      const r = await postJSON(`/api/estudios/${id}/predecir`, { serie, cortes: [n] });
      if (esta !== corrida) return;
      E.resultados.push(r.cortes[0]);
      if (verCorte === null) verCorte = 0;
      pintar();
    }
    prog.hidden = true;
    $("#btn-ir-informe").disabled = false;
    habilitar("informe"); hecho("resultados");
    avisar("Análisis terminado.", "ok");
  } catch (e) {
    if (esta !== corrida) return;
    prog.innerHTML = `<p class="error">No se pudo analizar: ${esc(e.message)}</p>`;
  }
}

function pintarChips() {
  $("#chips-cortes").innerHTML = E.resultados.map((c, i) => `<button type="button" role="tab" aria-selected="${i === verCorte}"
    data-i="${i}" class="${c.columna_valida ? "" : "con-aviso"}" title="${c.columna_valida ? "" : "No parece un corte sagital lumbar: no se clasificó"}">
    Corte ${c.corte + 1}${c.columna_valida ? "" : " ⚠"}</button>`).join("");
  $$("#chips-cortes button").forEach((b) => b.addEventListener("click", () => { verCorte = +b.dataset.i; pintarVisor(); pintarChips(); }));
}

function pintarVisor() {
  if (verCorte === null || !E.resultados[verCorte]) { visor.vacio("Analizando…"); return; }
  const c = E.resultados[verCorte];
  visor.mostrar(urlCorte(c.corte), { esquina: `Corte ${c.corte + 1}${c.columna_valida ? "" : " · no se clasificó: no parece sagital lumbar"}` });
  visor.dibujar(cajasSVG(c, { color: (n) => colorNivel(c, n), activo }));
}

function medidor(nivel) {
  const k = { alta: 3, media: 2, baja: 1 }[nivel] || 0;
  return `<span class="medidor" title="${CONF_TXT[nivel] || ""}" aria-label="${CONF_TXT[nivel] || "Sin confianza"}">${[1, 2, 3].map((i) => `<i class="${i <= k ? "on" : ""}"></i>`).join("")}</span>`;
}

function valoracionHTML(r) {
  const v = E.valoracion[r.n] || {};
  const opciones = [...CLASES, "Otro"].map((cl) => `<option value="${cl}" ${v.clase === cl ? "selected" : ""}>${ETIQ[cl]}</option>`).join("");
  return `<div class="valoracion" data-nivel="${r.n}">
    <span>Tu valoración</span>
    <div class="segmentado" role="group" aria-label="Valoración del disco ${r.n}">
      <button type="button" data-v="concuerda" aria-pressed="${v.estado === "concuerda"}">Concuerdo</button>
      <button type="button" data-v="discrepa" aria-pressed="${v.estado === "discrepa"}">Discrepo</button>
    </div>
    ${v.estado === "discrepa" ? `<div class="discrepa">
      <label class="campo">Clase que ves<select data-campo="clase">${opciones}</select></label>
      <label class="campo">Nota<input type="text" data-campo="nota" maxlength="200" value="${esc(v.nota || "")}" placeholder="Ej.: protrusión foraminal izquierda"></label></div>` : ""}
  </div>`;
}

function tarjeta(r) {
  const cl = r.clase || "nada", abierto = activo === r.n;
  const rois = E.resultados.map((c) => {
    const d = c.niveles[r.n], t = d.traza;
    if (!d.detectado) return `<figure class="roi vacio"><div>no detectado</div><figcaption>Corte ${c.corte + 1}</figcaption></figure>`;
    const txt = t && t.ensamble ? ETIQ[t.ensamble.clase] : t && t.rechazado ? "descartado" : "sin clasificar";
    return `<figure class="roi c-${t && t.ensamble ? t.ensamble.clase : "nada"}"><img src="${d.roi}" alt="Disco ${r.n}, corte ${c.corte + 1}">
      <figcaption>Corte ${c.corte + 1} · <b>${txt}</b></figcaption></figure>`;
  }).join("");
  return `<article class="disco c-${cl} ${abierto ? "abierto" : ""}" data-nivel="${r.n}">
    <button type="button" class="disco-cab" aria-expanded="${abierto}">
      <span class="nivel">${r.n}</span>
      <span class="etiqueta">${r.clase ? ETIQ[r.clase] : ESTADO_TXT[r.estado]}</span>
      ${r.estado === "no_evaluable" ? "" : medidor(r.confianza)}
    </button>
    ${r.estado === "no_evaluable" ? "" : `<div class="metricas">
      <span><b>${r.k} de ${r.total}</b> cortes</span><span><b>${r.modelos.k} de ${r.modelos.total}</b> votos de modelos</span>
      ${r.prob !== null ? `<span>Puntaje medio <b>${pct(r.prob)}</b></span>` : ""}</div>`}
    <p class="frase">${esc(fraseNivel(r))}</p>
    ${abierto ? `<div class="rois">${rois}</div>` : ""}
    ${valoracionHTML(r)}
  </article>`;
}

// La capa 3 del rechazo solo actua con models/umbral_rechazo.json vigente
const sinFiltro = () => E.resultados.some((c) => Object.values(c.niveles).some((d) =>
  ["sin_umbral", "umbral_obsoleto"].includes(d.traza?.pasos?.validacion?.capa3)));

function pintarDiscos() {
  if (!E.resultados.length) { $("#conclusion").innerHTML = ""; $("#discos").innerHTML = ""; return; }
  const res = resumenEstudio(E.resultados);
  $("#conclusion").innerHTML = `<h2>Conclusión</h2><p>${esc(fraseGeneral(res, E.resultados))}</p>
    <p class="ayuda">Clase del ensamble de los 4 modelos en cada corte. La confianza es orientativa: una regla fija según cuántos cortes y modelos coinciden, no una probabilidad calibrada. Haz clic en un disco para ver sus recortes y resaltarlo en la imagen.</p>
    ${sinFiltro() ? `<p class="alerta">El filtro de recortes atípicos no está calibrado: un disco muy distinto a los del entrenamiento no se descarta.</p>` : ""}`;
  $("#discos").innerHTML = res.map(tarjeta).join("");
  $$("#discos .disco-cab").forEach((b) => b.addEventListener("click", () => {
    const n = b.closest(".disco").dataset.nivel;
    activo = activo === n ? null : n;
    pintarDiscos(); pintarVisor();
  }));
  $$("#discos .valoracion").forEach((el) => {
    const n = el.dataset.nivel;
    el.querySelectorAll("[data-v]").forEach((b) => b.addEventListener("click", () => {
      const v = E.valoracion[n] || {};
      E.valoracion[n] = v.estado === b.dataset.v ? {} : { ...v, estado: b.dataset.v, clase: v.clase || CLASES[0] };
      cambioValoracion(); pintarDiscos();
    }));
    el.querySelectorAll("[data-campo]").forEach((c) => c.addEventListener("input", () => {
      E.valoracion[n][c.dataset.campo] = c.value; cambioValoracion();
    }));
  });
}

const cambioValoracion = () => document.dispatchEvent(new CustomEvent("valoracion-cambio"));

function pintarTecnico() {
  $("#tecnico").innerHTML = E.resultados.map((c) => `<h3>Corte ${c.corte + 1}</h3><table class="tabla"><thead><tr><th>Disco</th>
    ${Object.values(MODELOS).map((m) => `<th>${m}</th>`).join("")}<th>Ensamble</th></tr></thead><tbody>
    ${NIVELES.map((n) => {
      const d = c.niveles[n], t = d.traza;
      if (!t || !t.predicciones) return `<tr><th>${n}</th><td colspan="5" class="tenue">${!d.detectado ? "no detectado" : t && t.rechazado ? `descartado: ${esc(t.rechazado.mensaje)}` : "sin clasificar"}</td></tr>`;
      return `<tr><th>${n} <small>${pct(d.confianza)}</small></th>${t.predicciones.map((p) => `<td class="c-${p.clase}"><span class="etiqueta">${ETIQ[p.clase]}</span> <small>${pct(p.confianza)}</small></td>`).join("")}
        <td class="c-${t.ensamble.clase}"><strong class="etiqueta">${ETIQ[t.ensamble.clase]}</strong></td></tr>`;
    }).join("")}</tbody></table>`).join("") + `<p class="ayuda">Debajo de cada disco, la confianza del detector. El puntaje del SVM es un softmax de distancias, no una probabilidad calibrada.</p>`;
}

function pintar() { pintarChips(); pintarVisor(); pintarDiscos(); pintarTecnico(); }

export function restaurarResultados() {
  if (!E.resultados.length) return;
  verCorte = 0;
  habilitar("resultados"); hecho("cortes"); habilitar("informe"); hecho("resultados");
  $("#btn-ir-informe").disabled = false;
  pintar();
}

export function iniciarResultados() {
  visor = crearVisor($("#visor-resultados"));
  visor.vacio("Elige cortes y pulsa Analizar");
  $("#btn-ir-informe").addEventListener("click", () => irA("informe"));
  document.addEventListener("serie-nueva", () => {
    corrida += 1;           // cancela el analisis en curso: era de la serie o del estudio anterior
    E.resultados = []; verCorte = null; activo = null;
    $("#analisis-progreso").hidden = true;
    $("#btn-ir-informe").disabled = true;
    habilitar("resultados", false); habilitar("informe", false); hecho("cortes", false); hecho("resultados", false);
    pintar(); visor.vacio("Elige cortes y pulsa Analizar");
  });
}
