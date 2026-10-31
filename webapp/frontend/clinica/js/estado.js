// Constantes, estado de la sesion y utilidades comunes de la aplicacion para el radiologo.

export const CLASES = ["Normal", "Degeneracion_Discal", "Protrusion"];
export const ETIQ = { Degeneracion_Discal: "Degeneración", Normal: "Normal", Protrusion: "Protrusión", Otro: "Otro hallazgo" };
export const NIVELES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"];
export const MODELOS = { yolov8s_cls: "YOLOv8s-cls", resnet50: "ResNet-50", svm: "SVM", random_forest: "Random Forest" };

// Estado del estudio abierto. Nada de esto sale del navegador salvo las llamadas a la API.
export const E = {
  estudio: null,          // ficha que devuelve POST /api/estudios
  serie: null,            // id de la serie elegida
  actual: 0,              // corte que se ve en el visor de "Cortes"
  sel: new Set(),         // cortes elegidos para analizar
  puntos: {},             // corte -> puntuacion del detector (POST /puntuar)
  recomendados: [],
  avisoPuntos: "",
  cajas: {},              // corte -> resultado de /recortar (cajas del detector, se pide al activar el interruptor)
  resultados: [],         // cortes analizados (POST /predecir), en orden de corte
  valoracion: {},         // nivel -> { estado: "concuerda" | "discrepa", clase, nota }
};

export function serieActual() {
  return E.estudio ? E.estudio.series.find((s) => s.id === E.serie) : null;
}

export function reiniciarSerie() {
  Object.assign(E, { actual: 0, sel: new Set(), puntos: {}, recomendados: [], avisoPuntos: "", cajas: {}, resultados: [], valoracion: {} });
}

export const $ = (s, raiz = document) => raiz.querySelector(s);
export const $$ = (s, raiz = document) => [...raiz.querySelectorAll(s)];
export const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const pct = (v) => `${Math.round(100 * v)} %`;
export const enLista = (xs) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} y ${xs[xs.length - 1]}`);
export const urlCorte = (n) => `/api/estudios/${E.estudio.id}/series/${E.serie}/cortes/${n}/imagen`;

// localStorage puede fallar (modo privado, almacenamiento bloqueado): la app funciona igual sin el
export const guardado = {
  leer(clave) { try { return JSON.parse(localStorage.getItem(clave)); } catch { return null; } },
  escribir(clave, valor) { try { localStorage.setItem(clave, JSON.stringify(valor)); } catch { /* sin almacenamiento */ } },
  borrar(clave) { try { localStorage.removeItem(clave); } catch { /* sin almacenamiento */ } },
};

// Confirmacion dentro de la pagina (no la ventana del navegador). Devuelve una promesa con true si se acepta;
// Cancelar, Esc o cerrar el dialogo dan false.
let respuesta = null;       // solo hay una confirmacion pendiente a la vez

export function confirmar({ titulo, texto, aceptar, peligro = false }) {
  const d = $("#dlg-confirmar"), si = $("#confirmar-si");
  $("#confirmar-titulo").textContent = titulo;
  $("#confirmar-texto").textContent = texto;
  si.textContent = aceptar;
  si.className = peligro ? "secundario peligro" : "principal";
  if (respuesta) respuesta(false);      // la anterior se cancelo y su "close" aun no llega
  d.returnValue = "";
  d.onclose = () => {
    if (d.open) return;                 // ese "close" atrasado: este dialogo sigue abierto
    const r = respuesta;
    respuesta = null;
    if (r) r(d.returnValue === "si");
  };
  d.showModal();
  return new Promise((resolver) => { respuesta = resolver; });
}

// Avisos breves abajo a la derecha
export function avisar(texto, tipo = "info") {
  const caja = $("#avisos");
  const el = document.createElement("div");
  el.className = `aviso aviso-${tipo}`;
  el.setAttribute("role", tipo === "error" ? "alert" : "status");
  el.textContent = texto;
  caja.appendChild(el);
  setTimeout(() => el.remove(), tipo === "error" ? 8000 : 4000);
}
