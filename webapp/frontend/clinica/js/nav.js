// Navegacion entre los 4 pasos (Cargar, Cortes, Resultados, Informe).
import { $, $$ } from "./estado.js";

const alEntrar = {};

export function alEntrarEn(paso, fn) { alEntrar[paso] = fn; }

export function irA(paso) {
  $$(".pasos button").forEach((b) => {
    if (b.dataset.paso === paso) b.setAttribute("aria-current", "step");
    else b.removeAttribute("aria-current");
  });
  $$(".paso").forEach((s) => s.classList.toggle("activo", s.id === `paso-${paso}`));
  if (alEntrar[paso]) alEntrar[paso]();
}

export function habilitar(paso, si = true) {
  $(`.pasos button[data-paso="${paso}"]`).disabled = !si;
}

export function hecho(paso, si = true) {
  $(`.pasos button[data-paso="${paso}"]`).classList.toggle("hecho", si);
}

export function iniciarNav() {
  $$(".pasos button").forEach((b) => b.addEventListener("click", () => irA(b.dataset.paso)));
}
