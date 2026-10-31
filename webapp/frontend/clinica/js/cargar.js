// Paso 1: elegir archivos o una carpeta, descartar lo que no es DICOM y subir el estudio con progreso.
import { subirEstudio } from "./api.js";
import { abrirEstudio } from "./cortes.js";
import { $, $$, E, avisar, esc } from "./estado.js";

const MAX_ARCHIVOS = 500, MAX_BYTES = 300 * 1024 * 1024;       // los mismos limites que el servidor
// Lo que suele venir en la carpeta exportada de un resonador y no es un corte: el indice DICOMDIR, visores, imagenes
const NO_DICOM = /\.(jpe?g|png|gif|bmp|tiff?|txt|xml|html?|pdf|exe|dll|inf|ini|bat|cmd|db|js|css|json|log|md|csv|cab|msi)$/i;
const NOMBRES_NO = new Set(["dicomdir", "thumbs.db", "desktop.ini", "autorun.inf", "lockfile", "version"]);

const tamano = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1).replace(".", ",")} MB` : `${Math.ceil(b / 1e3)} KB`);

export function filtrar(archivos) {
  const ok = [], omitidos = [];
  for (const a of archivos) {
    const n = a.name.toLowerCase();
    if (n.startsWith(".") || NOMBRES_NO.has(n) || NO_DICOM.test(n)) omitidos.push(a.name);
    else ok.push(a);
  }
  return { ok, omitidos };
}

// Recorre una carpeta arrastrada (File System API de los navegadores Chromium/Firefox/Safari)
async function leerEntrada(entrada, salida) {
  if (entrada.isFile) {
    salida.push(await new Promise((res, rej) => entrada.file(res, rej)));
  } else if (entrada.isDirectory) {
    const lector = entrada.createReader();
    for (;;) {
      const lote = await new Promise((res, rej) => lector.readEntries(res, rej));
      if (!lote.length) break;
      for (const e of lote) await leerEntrada(e, salida);
    }
  }
}

function etapa(nombre, estado) {
  const li = $(`#progreso [data-etapa="${nombre}"]`);
  li.className = estado;
}

async function subir(lista) {
  const { ok, omitidos } = filtrar(lista);
  const err = $("#cargar-error");
  err.hidden = true;
  const bytes = ok.reduce((s, a) => s + a.size, 0);
  $("#seleccion").hidden = false;
  $("#seleccion").innerHTML = `<strong>${ok.length} archivo${ok.length === 1 ? "" : "s"}</strong> · ${tamano(bytes)}`
    + (omitidos.length ? `<span>Se omiten ${omitidos.length} que no son cortes (${esc(omitidos.slice(0, 3).join(", "))}${omitidos.length > 3 ? "…" : ""})</span>` : "");
  if (!ok.length) return mostrarError("No hay archivos DICOM en la selección. Elige los .IMA/.dcm del estudio, su carpeta o un .zip.");
  if (ok.length > MAX_ARCHIVOS || bytes > MAX_BYTES)
    return mostrarError(`El estudio supera el límite (${MAX_ARCHIVOS} archivos o 300 MB). Exporta solo la serie T2 sagital y vuelve a intentarlo.`);

  $$("#zona, #btn-archivos, #btn-carpeta").forEach((b) => b.classList.add("ocupado"));
  $("#progreso").hidden = false;
  $("#progreso progress").value = 0;
  ["subida", "conversion", "series"].forEach((x) => etapa(x, ""));
  etapa("subida", "en-curso");
  try {
    const ficha = await subirEstudio(ok, (fase, v) => {
      if (fase === "subida") $("#progreso progress").value = v;
      else { etapa("subida", "hecha"); etapa("conversion", "en-curso"); }
    });
    etapa("subida", "hecha"); etapa("conversion", "hecha"); etapa("series", "hecha");
    if (E.estudio) fetch(`/api/estudios/${E.estudio.id}`, { method: "DELETE" }).catch(() => {});   // el estudio anterior
    avisar(`Estudio convertido: ${ficha.series.length} serie${ficha.series.length === 1 ? "" : "s"}.`, "ok");
    abrirEstudio(ficha);
  } catch (e) {
    $$("#progreso li.en-curso").forEach((li) => (li.className = "fallo"));
    mostrarError(e.message);
  } finally {
    $$("#zona, #btn-archivos, #btn-carpeta").forEach((b) => b.classList.remove("ocupado"));
  }
}

function mostrarError(texto) {
  const err = $("#cargar-error");
  err.textContent = texto;
  err.hidden = false;
}

export function iniciarCargar() {
  const zona = $("#zona");
  $("#btn-archivos").addEventListener("click", () => $("#in-archivos").click());
  $("#btn-carpeta").addEventListener("click", () => $("#in-carpeta").click());
  for (const id of ["#in-archivos", "#in-carpeta"]) {
    $(id).addEventListener("change", (e) => {
      const lista = [...e.target.files];
      e.target.value = "";               // permite volver a elegir lo mismo
      if (lista.length) subir(lista);
    });
  }
  ["dragenter", "dragover"].forEach((t) => zona.addEventListener(t, (e) => { e.preventDefault(); zona.classList.add("encima"); }));
  ["dragleave", "drop"].forEach((t) => zona.addEventListener(t, () => zona.classList.remove("encima")));
  zona.addEventListener("drop", async (e) => {
    e.preventDefault();
    const entradas = [...e.dataTransfer.items].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
    let lista = [];
    if (entradas.length) for (const en of entradas) await leerEntrada(en, lista);
    else lista = [...e.dataTransfer.files];
    if (lista.length) subir(lista);
  });
  // soltar fuera de la zona no debe abrir el archivo en el navegador
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
}
