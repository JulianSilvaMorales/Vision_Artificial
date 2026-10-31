// Llamadas al backend. Todas devuelven JSON o lanzan un Error con el mensaje del servidor.

export async function api(ruta, opciones) {
  const r = await fetch(ruta, opciones);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    const e = new Error(typeof j.detail === "string" ? j.detail : `Error ${r.status} del servidor`);
    e.estado = r.status;
    e.motivo = j.motivo;
    e.rechazada = r.status === 422 && !!j.motivo;
    throw e;
  }
  return j;
}

export const postJSON = (ruta, cuerpo) =>
  api(ruta, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cuerpo) });

// Subida del estudio con progreso: fetch no informa del avance de la subida, XMLHttpRequest si.
export function subirEstudio(archivos, alAvanzar) {
  return new Promise((resolver, rechazar) => {
    const datos = new FormData();
    for (const a of archivos) datos.append("archivos", a, a.name);
    const x = new XMLHttpRequest();
    x.open("POST", "/api/estudios");
    x.responseType = "json";
    x.upload.onprogress = (ev) => ev.lengthComputable && alAvanzar("subida", ev.loaded / ev.total);
    x.upload.onload = () => alAvanzar("conversion", 1);
    x.onload = () => {
      const j = x.response || {};
      if (x.status >= 200 && x.status < 300) resolver(j);
      else rechazar(new Error(typeof j.detail === "string" ? j.detail : `Error ${x.status} del servidor`));
    };
    x.onerror = () => rechazar(new Error("Sin conexión con el servidor. Revisa que la ventana de iniciar_mvp siga abierta."));
    x.send(datos);
  });
}
