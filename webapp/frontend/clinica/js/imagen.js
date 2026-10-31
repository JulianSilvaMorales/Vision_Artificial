// Atajo "Analizar una imagen": un recorte de disco o un corte completo suelto, sin pasar por el estudio.
import { fraseNivel, resumenNivel } from "./analisis.js";
import { api } from "./api.js";
import { $, ETIQ, MODELOS, NIVELES, esc, pct } from "./estado.js";
import { claseColor } from "./resultados.js";
import { cajasSVG } from "./visor.js";

const datos = (f) => { const d = new FormData(); d.append("imagen", f); return d; };

function rechazo(e) {
  return `<div class="rechazo"><h3>No se puede analizar esta imagen</h3><p>${esc(e.message)}</p>
    <p class="ayuda">Usa una resonancia T2 sagital lumbar en escala de grises: un corte completo o el recorte de un disco.</p></div>`;
}

function resultadoROI(tr, url) {
  const cl = tr.ensamble.clase, prob = tr.ensamble.puntajes[cl];
  const votos = tr.predicciones.filter((p) => p.clase === cl).length;
  return `<div class="res-roi"><img src="${url}" alt="Imagen analizada">
    <div><p class="ayuda">Recorte de un disco</p>
      <p class="clase-grande" style="color:${claseColor(cl)}">${ETIQ[cl]}</p>
      <p>${votos} de 4 modelos coinciden · puntaje medio ${pct(prob)}</p>
      <ul class="votos">${tr.predicciones.map((p) => `<li><span>${MODELOS[p.modelo]}</span><strong style="color:${claseColor(p.clase)}">${ETIQ[p.clase]}</strong><small>${pct(p.confianza)}</small></li>`).join("")}</ul>
    </div></div>`;
}

function resultadoCorte(c, url) {
  const color = (n) => { const t = c.niveles[n].traza; return t && t.ensamble ? claseColor(t.ensamble.clase) : "var(--aviso)"; };
  const filas = NIVELES.map((n) => {
    const r = resumenNivel(n, [c]);
    return `<li><strong>${n}</strong> <span style="color:${claseColor(r.clase)}">${r.clase ? ETIQ[r.clase] : "No evaluable"}</span>
      <small>${esc(fraseNivel(r).replace(/^[^.]*\.\s?/, ""))}</small></li>`;
  }).join("");
  return `<div class="res-corte"><div class="hoja-img" style="aspect-ratio:${c.ancho}/${c.alto}"><img src="${url}" alt="Corte analizado">
    <svg viewBox="0 0 ${c.ancho} ${c.alto}" preserveAspectRatio="none">${cajasSVG(c, { color, escalaTexto: 1.4 })}</svg></div>
    <div><p class="ayuda">Corte completo: el detector halló ${c.n_detectados} de 5 discos</p><ul class="lista-niveles">${filas}</ul>
    <p class="ayuda">Para un resultado más fiable, carga el estudio completo y analiza varios cortes centrales.</p></div></div>`;
}

async function analizarImagen(f) {
  const caja = $("#imagen-resultado"), url = URL.createObjectURL(f);
  caja.innerHTML = `<p class="cargando">Analizando con los 4 modelos…</p>`;
  try {
    caja.innerHTML = resultadoROI(await api("/api/predecir", { method: "POST", body: datos(f) }), url);
  } catch (e) {
    if (e.motivo !== "corte_completo") { caja.innerHTML = e.rechazada || e.estado === 400 ? rechazo(e) : `<p class="error">${esc(e.message)}</p>`; return; }
    caja.innerHTML = `<p class="cargando">Es un corte completo: detectando los 5 discos…</p>`;
    try {
      const r = await api("/api/predecir-corte", { method: "POST", body: datos(f) });
      caja.innerHTML = resultadoCorte(r.cortes[0], url);
    } catch (e2) {
      caja.innerHTML = e2.rechazada ? rechazo(e2) : `<p class="error">${esc(e2.message)}</p>`;
    }
  }
}

export function iniciarImagen() {
  const dlg = $("#dlg-imagen"), zona = $("#zona-imagen"), input = $("#in-imagen");
  $("#btn-imagen").addEventListener("click", () => { $("#imagen-resultado").innerHTML = ""; dlg.showModal(); });
  dlg.querySelector("[data-cerrar]").addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });       // clic fuera del cuadro
  input.addEventListener("change", () => { const f = input.files[0]; input.value = ""; if (f) analizarImagen(f); });
  ["dragenter", "dragover"].forEach((t) => zona.addEventListener(t, (e) => { e.preventDefault(); zona.classList.add("encima"); }));
  ["dragleave", "drop"].forEach((t) => zona.addEventListener(t, () => zona.classList.remove("encima")));
  zona.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) analizarImagen(f); });
}
