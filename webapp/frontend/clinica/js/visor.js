// Visor de cortes al estilo de un PACS: brillo y contraste arrastrando, zoom, desplazamiento y una capa SVG encima
// (cajas del detector) en coordenadas de pixel del corte.
//
// Raton: arrastrar = brillo/contraste (o mover, segun la herramienta) · boton derecho o Shift = mover ·
// Ctrl + rueda = zoom · rueda = cambiar de corte (si se pasa alRodar) · doble clic = restablecer.
// Teclado con el visor enfocado: W brillo/contraste, M mover, + / - zoom, R restablecer.

const ICONOS = {
  wl: '<path d="M12 3a9 9 0 1 0 0 18V3z" fill="currentColor"/><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/>',
  mover: '<path d="M12 2v20M2 12h20M12 2l-3 3M12 2l3 3M12 22l-3-3M12 22l3-3M2 12l3-3M2 12l3 3M22 12l-3-3M22 12l-3 3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  mas: '<circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="2"/><path d="M15 15l6 6M7 10h6M10 7v6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  menos: '<circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="2"/><path d="M15 15l6 6M7 10h6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  reset: '<path d="M4 12a8 8 0 1 0 2.3-5.7M4 4v4h4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
};
export const icono = (nombre, tam = 18) => `<svg width="${tam}" height="${tam}" viewBox="0 0 24 24" aria-hidden="true">${ICONOS[nombre]}</svg>`;

export function crearVisor(raiz, { alRodar, extraBarra = "" } = {}) {
  raiz.classList.add("visor");
  raiz.innerHTML = `
    <div class="visor-barra" role="toolbar" aria-label="Herramientas del visor">
      <button type="button" class="herr activa" data-h="wl" title="Brillo y contraste (W): arrastra sobre la imagen" aria-pressed="true">${icono("wl")}<span>Brillo/contraste</span></button>
      <button type="button" class="herr" data-h="mover" title="Mover (M)" aria-pressed="false">${icono("mover")}<span>Mover</span></button>
      <button type="button" data-z="1.25" title="Acercar (+)" aria-label="Acercar">${icono("mas")}</button>
      <button type="button" data-z="0.8" title="Alejar (-)" aria-label="Alejar">${icono("menos")}</button>
      <button type="button" data-reset title="Restablecer (R o doble clic)" aria-label="Restablecer vista">${icono("reset")}</button>
      ${extraBarra}
    </div>
    <div class="visor-lienzo" tabindex="0" aria-label="Imagen del corte">
      <div class="visor-capa"><img alt="" draggable="false"><svg class="visor-svg" preserveAspectRatio="none"></svg></div>
      <div class="visor-esquina sup"></div>
      <div class="visor-esquina inf"></div>
      <div class="visor-vacio"></div>
    </div>`;
  const lienzo = raiz.querySelector(".visor-lienzo"), capa = raiz.querySelector(".visor-capa");
  const img = capa.querySelector("img"), svg = capa.querySelector("svg");
  const v = { zoom: 1, x: 0, y: 0, brillo: 1, contraste: 1, herr: "wl", ancho: 0, alto: 0 };

  function encajar() {
    if (!v.ancho) return;
    const r = lienzo.getBoundingClientRect();
    const k = Math.min((r.width - 16) / v.ancho, (r.height - 16) / v.alto);
    capa.style.width = `${v.ancho * k}px`;
    capa.style.height = `${v.alto * k}px`;
  }
  function aplicar() {
    capa.style.transform = `translate(-50%, -50%) translate(${v.x}px, ${v.y}px) scale(${v.zoom})`;
    img.style.filter = `brightness(${v.brillo}) contrast(${v.contraste})`;
    raiz.querySelector(".visor-esquina.inf").textContent =
      `Brillo ${Math.round(v.brillo * 100)} % · Contraste ${Math.round(v.contraste * 100)} % · Zoom ${Math.round(v.zoom * 100)} %`;
  }
  function herramienta(h) {
    v.herr = h;
    raiz.querySelectorAll(".herr").forEach((b) => {
      b.classList.toggle("activa", b.dataset.h === h);
      b.setAttribute("aria-pressed", String(b.dataset.h === h));
    });
    lienzo.dataset.herr = h;
  }
  function restablecer() { Object.assign(v, { zoom: 1, x: 0, y: 0, brillo: 1, contraste: 1 }); aplicar(); }
  const zoom = (f) => { v.zoom = Math.min(8, Math.max(0.5, v.zoom * f)); aplicar(); };

  raiz.querySelectorAll(".herr").forEach((b) => b.addEventListener("click", () => herramienta(b.dataset.h)));
  raiz.querySelectorAll("[data-z]").forEach((b) => b.addEventListener("click", () => zoom(+b.dataset.z)));
  raiz.querySelector("[data-reset]").addEventListener("click", restablecer);
  new ResizeObserver(encajar).observe(lienzo);

  let arrastre = null;
  lienzo.addEventListener("contextmenu", (e) => e.preventDefault());
  lienzo.addEventListener("pointerdown", (e) => {
    if (!v.ancho) return;
    lienzo.setPointerCapture(e.pointerId);
    const modo = e.button === 2 || e.button === 1 || e.shiftKey || v.herr === "mover" ? "mover" : "wl";
    arrastre = { modo, px: e.clientX, py: e.clientY };
  });
  lienzo.addEventListener("pointermove", (e) => {
    if (!arrastre) return;
    const dx = e.clientX - arrastre.px, dy = e.clientY - arrastre.py;
    arrastre.px = e.clientX; arrastre.py = e.clientY;
    if (arrastre.modo === "mover") { v.x += dx; v.y += dy; }
    else {
      v.contraste = Math.min(4, Math.max(0.2, v.contraste + dx * 0.005));
      v.brillo = Math.min(4, Math.max(0.2, v.brillo - dy * 0.005));
    }
    aplicar();
  });
  const soltar = () => { arrastre = null; };
  lienzo.addEventListener("pointerup", soltar);
  lienzo.addEventListener("pointercancel", soltar);
  lienzo.addEventListener("dblclick", restablecer);
  lienzo.addEventListener("wheel", (e) => {
    if (!v.ancho) return;
    e.preventDefault();
    if (e.ctrlKey || !alRodar) zoom(e.deltaY < 0 ? 1.1 : 1 / 1.1);
    else alRodar(e.deltaY > 0 ? 1 : -1);
  }, { passive: false });
  lienzo.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (k === "w") herramienta("wl");
    else if (k === "m") herramienta("mover");
    else if (k === "+" || k === "=") zoom(1.25);
    else if (k === "-") zoom(0.8);
    else if (k === "r") restablecer();
    else return;
    e.preventDefault();
  });

  aplicar();
  return {
    raiz,
    lienzo,
    // Muestra un corte. Conserva brillo, contraste y zoom para comparar cortes vecinos con la misma ventana.
    mostrar(url, { esquina = "", alCargar } = {}) {
      raiz.querySelector(".visor-vacio").textContent = "";
      raiz.querySelector(".visor-esquina.sup").textContent = esquina;
      img.onload = () => {
        v.ancho = img.naturalWidth; v.alto = img.naturalHeight;
        svg.setAttribute("viewBox", `0 0 ${v.ancho} ${v.alto}`);
        encajar();
        if (alCargar) alCargar();
      };
      img.alt = esquina || "Corte";
      img.src = url;
    },
    dibujar(contenidoSVG) { svg.innerHTML = contenidoSVG; },
    vacio(texto) {
      v.ancho = 0; img.removeAttribute("src"); svg.innerHTML = "";
      raiz.querySelector(".visor-vacio").textContent = texto;
      raiz.querySelector(".visor-esquina.sup").textContent = "";
    },
    restablecer,
  };
}

// Cajas de los discos sobre el corte. `color(nivel)` decide el trazo; `activo` resalta un nivel y atenua el resto.
export function cajasSVG(corte, { color = () => "var(--acento)", activo = null, escalaTexto = 1 } = {}) {
  const tam = Math.max(11, corte.alto / 30) * escalaTexto;
  return Object.entries(corte.niveles).filter(([, d]) => d.detectado).map(([n, d]) => {
    const [x1, y1, x2, y2] = d.caja, c = color(n), apagado = activo && activo !== n;
    // colores en style (no en atributos): los atributos de presentacion de SVG no aceptan var()
    return `<g opacity="${apagado ? 0.35 : 1}"><rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" rx="3" fill="none"
      style="stroke:${c}" stroke-width="${activo === n ? 4 : 2}" vector-effect="non-scaling-stroke"/>
      <text x="${x2 + 6}" y="${(y1 + y2) / 2 + tam / 3}" style="fill:${c}" font-size="${tam}" font-weight="600"
      paint-order="stroke" stroke="#000" stroke-width="${tam / 5}">${n}</text></g>`;
  }).join("");
}
