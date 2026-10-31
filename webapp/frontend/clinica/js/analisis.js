// Consolidacion de los cortes analizados: por disco, la clase del ensamble en cada corte, el consenso entre cortes y
// el acuerdo entre los 4 modelos. Funciones puras (sin DOM) para poder probarlas y reutilizarlas en el informe.
import { ETIQ, MODELOS, NIVELES, enLista } from "./estado.js";

const cortesTxt = (cs) => `${cs.length === 1 ? "el corte" : "los cortes"} ${enLista(cs)}`;

export function resumenNivel(n, resultados) {
  const evaluados = resultados.filter((c) => c.niveles[n].traza && c.niveles[n].traza.predicciones);
  const sinDisco = resultados.filter((c) => !c.niveles[n].detectado).map((c) => c.corte + 1);
  const rechazados = resultados.filter((c) => c.niveles[n].traza && c.niveles[n].traza.rechazado).map((c) => c.corte + 1);
  const sinColumna = resultados.filter((c) => c.niveles[n].detectado && !c.niveles[n].traza).map((c) => c.corte + 1);
  const base = { n, total: evaluados.length, sinDisco, rechazados, sinColumna, cortes: resultados.map((c) => c.corte + 1) };
  if (!evaluados.length) return { ...base, estado: "no_evaluable", clase: null };

  const votos = {};
  evaluados.forEach((c) => (votos[c.niveles[n].traza.ensamble.clase] ||= []).push(c.corte + 1));
  const [clase, cortes] = Object.entries(votos).sort((a, b) => b[1].length - a[1].length)[0];
  const k = cortes.length, total = evaluados.length;
  const estado = k === total ? "unanime" : 2 * k > total ? "mayoria" : "mixto";

  let modelosA = 0;
  const discrepan = {};
  evaluados.forEach((c) => c.niveles[n].traza.predicciones.forEach((p) => {
    if (p.clase === clase) modelosA += 1;
    else (discrepan[p.modelo] ||= []).push(p.clase);
  }));
  const probs = evaluados.map((c) => c.niveles[n].traza.ensamble.puntajes?.[clase]).filter((x) => typeof x === "number");
  const prob = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : null;
  const acuerdo = modelosA / (4 * total);
  // Confianza orientativa: alta si todos los cortes y casi todos los votos coinciden; baja si no hay mayoria
  const confianza = estado === "mixto" ? "baja"
    : estado === "unanime" && acuerdo >= 0.9 && (prob === null || prob >= 0.6) ? "alta" : "media";
  return { ...base, estado, clase: estado === "mixto" ? null : clase, claseTop: clase, k, votos, discrepan,
    modelos: { k: modelosA, total: 4 * total }, prob, confianza };
}

export const resumenEstudio = (resultados) => NIVELES.map((n) => resumenNivel(n, resultados));

// Frase de un disco, en lenguaje natural
export function fraseNivel(r) {
  const extra = [r.sinDisco.length && `no se detectó en ${cortesTxt(r.sinDisco)}`,
    r.rechazados.length && `se descartó por no parecerse a un disco en ${cortesTxt(r.rechazados)}`,
    r.sinColumna.length && `no se clasificó en ${cortesTxt(r.sinColumna)} (el corte no parece sagital lumbar)`]
    .filter(Boolean);
  if (r.estado === "no_evaluable") return `No se pudo evaluar${extra.length ? `: ${extra.join("; ")}` : ""}.`;
  const otros = Object.entries(r.votos).filter(([cl]) => cl !== r.claseTop).map(([cl, cs]) => `${ETIQ[cl]} en ${cortesTxt(cs)}`);
  let t = r.estado === "unanime"
    ? `${ETIQ[r.clase]} en ${r.total === 1 ? "el único corte analizado" : `todos los cortes analizados (${r.total} de ${r.total})`}`
    : r.estado === "mayoria"
      ? `${ETIQ[r.clase]} en ${r.k} de ${r.total} cortes; ${otros.join("; ")}`
      : `Resultado no concluyente: ${Object.entries(r.votos).map(([cl, cs]) => `${ETIQ[cl]} en ${cortesTxt(cs)}`).join("; ")}`;
  if (r.estado !== "mixto") {
    // modelos con la misma discrepancia se agrupan: "los 4 modelos indican Protrusion en 1 de 3"
    const grupos = {};
    Object.entries(r.discrepan).forEach(([m, cls]) => {
      const k = `${enLista([...new Set(cls)].map((c) => ETIQ[c]))} en ${cls.length} de ${r.total}`;
      (grupos[k] ||= []).push(MODELOS[m]);
    });
    const dis = Object.entries(grupos).map(([k, ms]) => `${ms.length === 4 ? "los 4 modelos indican" : ms.length > 1 ? `${enLista(ms)} indican` : `${ms[0]} indica`} ${k}`);
    t += dis.length ? `. Discrepa: ${dis.join("; ")}` : ". Los 4 modelos coinciden";
  }
  return `${t}${extra.length ? `. Además, ${extra.join("; ")}` : ""}.`;
}

// Frase general: "Se analizaron 3 cortes (7, 8 y 9). Protrusión en L4-L5 y L5-S1; Normal en L1-L2 ..."
export function fraseGeneral(resumen, resultados) {
  const cs = resultados.map((c) => c.corte + 1);
  const grupos = {};
  resumen.filter((r) => r.clase).forEach((r) => (grupos[r.clase] ||= []).push(r.n));
  const partes = Object.entries(grupos).map(([cl, ns]) => `${ETIQ[cl]} en ${enLista(ns)}`);
  const dudosos = resumen.filter((r) => !r.clase).map((r) => r.n);
  if (dudosos.length) partes.push(`sin conclusión clara en ${enLista(dudosos)}`);
  return `Se ${cs.length === 1 ? "analizó 1 corte" : `analizaron ${cs.length} cortes`} (${cs.length === 1 ? "corte" : "cortes"} ${enLista(cs)}). ${partes.join("; ")}.`;
}
