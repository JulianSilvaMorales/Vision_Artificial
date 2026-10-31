// Arranque de la aplicacion: tema, estado del sistema, navegacion y cada paso.
import { api } from "./api.js";
import { iniciarCargar } from "./cargar.js";
import { iniciarCortes, restaurarCortes } from "./cortes.js";
import { $, E, avisar, guardado } from "./estado.js";
import { iniciarImagen } from "./imagen.js";
import { iniciarInforme, restaurarInforme } from "./informe.js";
import { iniciarNav, irA } from "./nav.js";
import { iniciarResultados, restaurarResultados } from "./resultados.js";

const SOL = '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
const LUNA = '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>';

function pintarBotonTema() {
  const oscuro = document.documentElement.dataset.theme !== "claro";
  const b = $("#btn-tema");
  b.innerHTML = oscuro ? SOL : LUNA;
  b.setAttribute("aria-label", oscuro ? "Cambiar a tema claro" : "Cambiar a tema oscuro");
}

function iniciarTema() {
  pintarBotonTema();
  $("#btn-tema").addEventListener("click", () => {
    const nuevo = document.documentElement.dataset.theme === "claro" ? "oscuro" : "claro";
    document.documentElement.dataset.theme = nuevo;
    guardado.escribir("deepspine.tema", nuevo);
    pintarBotonTema();
  });
}

// Punto de estado: verde si los 4 modelos y el detector estan listos; si no, dice que falta y como arreglarlo
async function cargarEstado() {
  const e = $("#estado"), t = e.querySelector("span");
  try {
    const s = await api("/api/salud");
    if (s.faltan.length) {
      e.className = "estado mal";
      t.textContent = `Faltan modelos: ${s.faltan.join(", ")}`;
      e.title = "Ejecuta: python scripts/descargar_modelos.py y reinicia";
    } else if (!s.detector_ok) {
      e.className = "estado mal";
      t.textContent = "Falta el detector de discos";
      e.title = "Falta models/best_disc_detector.pt (Git LFS). Ejecuta: git lfs pull y reinicia";
    } else {
      e.className = "estado ok";
      t.textContent = `Sistema listo · ${s.dispositivo.toUpperCase()}`;
      e.title = `4 modelos y detector cargados${s.commit ? ` · versión ${s.commit}` : ""}`;
    }
  } catch {
    e.className = "estado mal";
    t.textContent = "Sin conexión con el servidor";
    e.title = "Abre webapp/iniciar_mvp.bat (o .sh) y recarga la página";
  }
}

// Sesion: si se recarga la pagina, se recupera el estudio abierto (si sigue en el servidor), el analisis y el borrador
const SESION = "deepspine.sesion";

function guardarSesion() {
  if (!E.estudio) { guardado.borrar(SESION); return; }
  const paso = document.querySelector(".pasos [aria-current]")?.dataset.paso || "cortes";
  guardado.escribir(SESION, { estudio: E.estudio, serie: E.serie, actual: E.actual, sel: [...E.sel], puntos: E.puntos,
    recomendados: E.recomendados, avisoPuntos: E.avisoPuntos, resultados: E.resultados, valoracion: E.valoracion, paso });
}

async function restaurarSesion() {
  const s = guardado.leer(SESION);
  if (!s || !s.estudio) return;
  try {
    await api(`/api/estudios/${s.estudio.id}`);
  } catch {
    guardado.borrar(SESION);
    guardado.borrar(`deepspine.informe.${s.estudio.id}`);
    avisar("El estudio anterior ya no está en el servidor (se borra a los 30 minutos). Cárgalo de nuevo.");
    return;
  }
  Object.assign(E, { ...s, sel: new Set(s.sel), cajas: {} });
  delete E.paso;
  restaurarCortes();
  restaurarResultados();
  restaurarInforme();
  irA(s.paso === "resultados" || s.paso === "informe" ? (E.resultados.length ? s.paso : "cortes") : s.paso);
}

window.addEventListener("pagehide", guardarSesion);
document.addEventListener("visibilitychange", () => document.visibilityState === "hidden" && guardarSesion());

iniciarTema();
iniciarNav();
iniciarCargar();
iniciarCortes();
iniciarResultados();
iniciarInforme();
iniciarImagen();
cargarEstado();
restaurarSesion();
