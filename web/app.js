"use strict";

const byId = (id) => document.getElementById(id);
const pages = ["inicio", "personas", "capturar", "entrenar", "reconocer"];
const pageTitles = {
  inicio: "Inicio",
  personas: "Personas",
  capturar: "Registrar rostro",
  entrenar: "Entrenar modelo",
  reconocer: "Reconocer",
};

let currentPage = "inicio";
let people = [];
let stream = null;
let cameraMode = null;
let cameraGeneration = 0;
let previewFrame = 0;
let recognitionTimer = 0;
let recognitionBusySession = null;
let toastTimer = 0;
let deleteTarget = null;

async function api(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  if (!response.ok) {
    let message = "No se pudo completar la solicitud.";
    try {
      const body = await response.json();
      if (body.error) message = body.error;
    } catch (_error) {
      // Keep a local, generic message if the server did not return JSON.
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response;
}

function showToast(message, type = "") {
  const toast = byId("toast");
  toast.textContent = message;
  toast.className = `toast show ${type}`.trim();
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 3600);
}

function navigate(page) {
  if (!pages.includes(page)) return;
  currentPage = page;
  document.querySelectorAll(".page").forEach((element) => {
    element.classList.toggle("active", element.id === `page-${page}`);
  });
  document.querySelectorAll(".nav-item").forEach((element) => {
    const active = element.dataset.page === page;
    element.classList.toggle("active", active);
    if (active) element.setAttribute("aria-current", "page");
    else element.removeAttribute("aria-current");
  });
  byId("page-label").textContent = pageTitles[page];
  byId("sidebar").classList.remove("open");
  if (page === "personas") refreshPeople();
  if (page === "entrenar") refreshStatus();
  if (page !== "capturar" && page !== "reconocer") stopCamera();
  history.replaceState(null, "", `#${page}`);
}

document.querySelectorAll("[data-page]").forEach((button) => {
  button.addEventListener("click", () => navigate(button.dataset.page));
});
document.querySelectorAll("[data-go]").forEach((button) => {
  button.addEventListener("click", () => navigate(button.dataset.go));
});
document.querySelector(".brand").addEventListener("click", (event) => {
  event.preventDefault();
  navigate("inicio");
});
byId("mobile-menu").addEventListener("click", () => byId("sidebar").classList.toggle("open"));
document.addEventListener("click", (event) => {
  if (window.innerWidth <= 780 && !event.target.closest(".sidebar") && !event.target.closest("#mobile-menu")) {
    byId("sidebar").classList.remove("open");
  }
});

function setModelStatus(ready) {
  const label = ready ? "Modelo listo" : "Entrenamiento pendiente";
  byId("sidebar-model").textContent = label;
  byId("sidebar-dot").className = `status-dot ${ready ? "ready" : "pending"}`;
  byId("stat-model").textContent = ready ? "Listo para reconocer" : "Pendiente de entrenamiento";
  byId("stat-model").classList.toggle("is-ready", ready);
  byId("stat-model").classList.toggle("is-pending", !ready);
  byId("train-status").textContent = ready ? "Listo" : "Pendiente";
  byId("train-status").classList.toggle("is-ready", ready);
  byId("train-status").classList.toggle("is-pending", !ready);
}

async function refreshStatus() {
  try {
    const response = await api("/api/status");
    const data = await response.json();
    byId("stat-people").textContent = String(data.people);
    byId("stat-images").textContent = String(data.images);
    byId("train-people").textContent = String(data.people);
    byId("train-images").textContent = String(data.images);
    setModelStatus(data.model_ready);
    if (data.operation && data.operation !== "training") {
      byId("sidebar-model").textContent = "Operación en curso";
    }
    if (data.training.state === "running") {
      setTrainingRunning(true, data.training.message || "Entrenamiento en curso…");
    } else if (data.training.state === "complete" || data.training.state === "error") {
      setTrainingRunning(false, data.training.message, data.training.state === "error");
    }
  } catch (error) {
    byId("sidebar-model").textContent = "Servidor no disponible";
    byId("sidebar-dot").className = "status-dot";
    if (currentPage === "entrenar") byId("train-message").textContent = error.message;
  }
}

async function refreshPeople() {
  try {
    const response = await api("/api/people");
    const data = await response.json();
    people = data.people;
    renderPeople(byId("people-search").value);
    const stats = await api("/api/status");
    const status = await stats.json();
    byId("stat-people").textContent = String(status.people);
    byId("stat-images").textContent = String(status.images);
    setModelStatus(status.model_ready);
  } catch (error) {
    showToast(error.message, "error");
  }
}

function renderPeople(query = "") {
  const grid = byId("people-grid");
  grid.replaceChildren();
  renderDatasetBars();
  const normalized = query.trim().toLocaleLowerCase();
  const visible = people.filter((person) => person.name.toLocaleLowerCase().includes(normalized));
  byId("people-count").textContent = `${visible.length} ${visible.length === 1 ? "perfil" : "perfiles"}`;
  byId("people-empty").classList.toggle("hidden", people.length !== 0);
  if (!people.length) return;
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "empty-search";
    empty.textContent = "No hay perfiles que coincidan con la búsqueda.";
    grid.append(empty);
    return;
  }
  for (const person of visible) {
    const card = document.createElement("article");
    card.className = "person-card";
    const avatar = document.createElement("span");
    avatar.className = "person-avatar";
    avatar.textContent = person.name.trim().slice(0, 2).toLocaleUpperCase();
    const meta = document.createElement("div");
    meta.className = "person-meta";
    const name = document.createElement("strong");
    name.textContent = person.name;
    const count = document.createElement("span");
    count.textContent = `${person.images} ${person.images === 1 ? "recorte" : "recortes"}`;
    meta.append(name, count);
    const actions = document.createElement("div");
    actions.className = "person-actions";
    const add = document.createElement("button");
    add.type = "button";
    add.textContent = "Añadir fotos";
    add.addEventListener("click", () => {
      byId("person-name").value = person.name;
      navigate("capturar");
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "delete-person";
    remove.textContent = "Eliminar";
    remove.addEventListener("click", () => openDeleteModal(person.name));
    actions.append(add, remove);
    card.append(avatar, meta, actions);
    grid.append(card);
  }
}

function renderDatasetBars() {
  const container = byId("dataset-bars");
  if (!container) return;
  container.replaceChildren();
  const coverage = people.filter((person) => person.images >= 15).length;
  byId("coverage-count").textContent = String(coverage);
  byId("coverage-label").textContent = `de ${people.length} ${people.length === 1 ? "perfil" : "perfiles"} con 15 o más capturas`;
  const sorted = [...people].sort((a, b) => b.images - a.images || a.name.localeCompare(b.name)).slice(0, 5);
  if (!sorted.length) {
    const empty = document.createElement("p");
    empty.className = "dataset-empty";
    empty.textContent = "Cuando registres personas, aquí verás el balance de sus capturas.";
    container.append(empty);
    return;
  }
  for (const person of sorted) {
    const row = document.createElement("div");
    row.className = "dataset-bar-row";
    const label = document.createElement("div");
    label.className = "dataset-bar-label";
    const name = document.createElement("span");
    name.textContent = person.name;
    const count = document.createElement("strong");
    count.textContent = `${person.images}/15`;
    label.append(name, count);
    const track = document.createElement("progress");
    track.className = "dataset-track";
    track.max = 15;
    track.value = Math.min(person.images, 15);
    track.setAttribute("aria-label", `${person.name}: ${person.images} de 15 capturas recomendadas`);
    track.dataset.level = person.images >= 15 ? "complete" : "incomplete";
    row.append(label, track);
    container.append(row);
  }
  if (people.length > sorted.length) {
    const more = document.createElement("p");
    more.className = "dataset-more";
    more.textContent = `y ${people.length - sorted.length} perfiles más`;
    container.append(more);
  }
}

byId("people-search").addEventListener("input", (event) => renderPeople(event.target.value));

async function startCamera(mode) {
  stopCamera();
  const session = cameraGeneration;
  try {
    const statusResponse = await api("/api/status");
    const status = await statusResponse.json();
    if (session !== cameraGeneration) return;
    if (mode === "recognize" && !status.model_ready) {
      showToast("Entrena primero el modelo para reconocer personas.", "error");
      return;
    }
    if (mode === "capture" && !status.face_model_ready) {
      showToast("Falta el modelo facial. Ejecuta python 1_setup.py.", "error");
      return;
    }
  } catch (error) {
    if (session !== cameraGeneration) return;
    showToast(error.message, "error");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast("Este navegador no permite acceder a la cámara en esta dirección.", "error");
    return;
  }
  try {
    const newStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
    });
    if (session !== cameraGeneration) {
      newStream.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = newStream;
    cameraMode = mode;
    const video = byId(mode === "capture" ? "capture-video" : "recognize-video");
    video.srcObject = stream;
    await video.play();
    if (mode === "capture") {
      byId("capture-placeholder").classList.add("hidden");
      byId("capture-live").classList.remove("hidden");
      byId("capture-save").disabled = false;
      byId("capture-connect").textContent = "Desconectar cámara";
      drawCapturePreview(session);
    } else {
      byId("recognize-placeholder").classList.add("hidden");
      byId("recognize-canvas").classList.add("visible");
      byId("recognize-connect").classList.add("hidden");
      byId("recognize-stop").classList.remove("hidden");
      byId("recognition-state").classList.add("active");
      byId("recognition-state").lastChild.textContent = " En vivo";
      runRecognitionLoop(session);
    }
  } catch (error) {
    if (session !== cameraGeneration) return;
    stopCamera();
    showToast(error.name === "NotAllowedError" ? "Permite el acceso a la cámara para continuar." : `No se pudo iniciar la cámara: ${error.message}`, "error");
  }
}

function stopCamera() {
  cameraGeneration += 1;
  window.cancelAnimationFrame(previewFrame);
  window.clearTimeout(recognitionTimer);
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  const oldMode = cameraMode;
  cameraMode = null;
  if (oldMode === "capture" || !oldMode) {
    byId("capture-video").srcObject = null;
    byId("capture-placeholder").classList.remove("hidden");
    byId("capture-live").classList.add("hidden");
    byId("capture-canvas").classList.remove("visible");
    byId("capture-save").disabled = true;
    byId("capture-connect").textContent = "Conectar cámara";
  }
  if (oldMode === "recognize" || !oldMode) {
    byId("recognize-video").srcObject = null;
    byId("recognize-placeholder").classList.remove("hidden");
    byId("recognize-canvas").classList.remove("visible");
    byId("recognize-connect").classList.remove("hidden");
    byId("recognize-stop").classList.add("hidden");
    byId("recognition-state").classList.remove("active");
    byId("recognition-state").lastChild.textContent = " En espera";
    byId("faces-found").textContent = "";
  }
}

function drawCapturePreview(session) {
  if (session !== cameraGeneration || cameraMode !== "capture" || !stream) return;
  const video = byId("capture-video");
  const canvas = byId("capture-canvas");
  if (video.videoWidth && video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.classList.add("visible");
    byId("capture-placeholder").classList.add("hidden");
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  }
  previewFrame = window.requestAnimationFrame(() => drawCapturePreview(session));
}

function captureBlob(videoElement, maxWidth = 960) {
  return new Promise((resolve, reject) => {
    if (!videoElement.videoWidth || !videoElement.videoHeight) {
      reject(new Error("La cámara aún no tiene una imagen disponible."));
      return;
    }
    const scale = Math.min(1, maxWidth / videoElement.videoWidth);
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(videoElement.videoWidth * scale);
    canvas.height = Math.round(videoElement.videoHeight * scale);
    canvas.getContext("2d").drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("No se pudo preparar la imagen.")), "image/jpeg", .86);
  });
}

byId("capture-connect").addEventListener("click", () => {
  if (cameraMode === "capture") stopCamera();
  else startCamera("capture");
});

byId("capture-save").addEventListener("click", async () => {
  if (cameraMode !== "capture") return;
  const name = byId("person-name").value.trim();
  if (!name) {
    byId("person-name").focus();
    showToast("Escribe el nombre de la persona antes de capturar.", "error");
    return;
  }
  const button = byId("capture-save");
  button.disabled = true;
  byId("capture-message").textContent = "Buscando y guardando el rostro…";
  try {
    const profileResponse = await api("/api/people", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const profile = await profileResponse.json();
    byId("person-name").value = profile.name;
    const blob = await captureBlob(byId("capture-video"));
    const response = await api(`/api/people/${encodeURIComponent(profile.name)}/capture`, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    const result = await response.json();
    byId("capture-message").textContent = `Recorte guardado. Este perfil tiene ${result.images} ${result.images === 1 ? "imagen" : "imágenes"}.`;
    showToast("Rostro registrado en este dispositivo.");
    await refreshPeople();
    await refreshStatus();
  } catch (error) {
    byId("capture-message").textContent = error.message;
    showToast(error.message, "error");
  } finally {
    button.disabled = cameraMode !== "capture";
  }
});

byId("recognize-connect").addEventListener("click", () => startCamera("recognize"));
byId("recognize-stop").addEventListener("click", stopCamera);

async function runRecognitionLoop(session) {
  if (session !== cameraGeneration || cameraMode !== "recognize" || !stream || recognitionBusySession === session) return;
  recognitionBusySession = session;
  try {
    const blob = await captureBlob(byId("recognize-video"), 720);
    const options = new URLSearchParams({
      mesh: byId("option-mesh").checked ? "1" : "0",
      pose: byId("option-pose").checked ? "1" : "0",
    });
    const response = await api(`/api/recognize?${options}`, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    const bitmap = await createImageBitmap(await response.blob());
    if (session === cameraGeneration && cameraMode === "recognize") {
      const canvas = byId("recognize-canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext("2d").drawImage(bitmap, 0, 0);
      byId("faces-found").textContent = `${response.headers.get("X-Face-Count") || 0} rostros detectados`;
      byId("recognition-state").lastChild.textContent = " Analizando";
    }
    bitmap.close();
  } catch (error) {
    if (session !== cameraGeneration) {
      return;
    } else if (error.status === 409) {
      byId("recognition-state").lastChild.textContent = " Esperando";
    } else if (session === cameraGeneration && cameraMode === "recognize") {
      byId("recognition-state").lastChild.textContent = " Revisa el modelo";
      byId("faces-found").textContent = error.message;
    }
  } finally {
    if (recognitionBusySession === session) recognitionBusySession = null;
    if (session === cameraGeneration && cameraMode === "recognize") {
      recognitionTimer = window.setTimeout(() => runRecognitionLoop(session), 450);
    }
  }
}

byId("recognize-file").addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  event.target.value = "";
  if (!file) return;
  stopCamera();
  if (file.size > 5 * 1024 * 1024) {
    showToast("La imagen supera el límite de 5 MB.", "error");
    return;
  }
  byId("recognize-placeholder").classList.add("hidden");
  byId("recognize-canvas").classList.add("visible");
  byId("recognition-state").lastChild.textContent = " Analizando imagen";
  try {
    const options = new URLSearchParams({ mesh: byId("option-mesh").checked ? "1" : "0", pose: byId("option-pose").checked ? "1" : "0" });
    const response = await api(`/api/recognize?${options}`, { method: "POST", headers: { "Content-Type": file.type }, body: file });
    const bitmap = await createImageBitmap(await response.blob());
    const canvas = byId("recognize-canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0);
    byId("faces-found").textContent = `${response.headers.get("X-Face-Count") || 0} rostros detectados`;
    byId("recognition-state").lastChild.textContent = " Imagen analizada";
    bitmap.close();
  } catch (error) {
    byId("recognize-placeholder").classList.remove("hidden");
    byId("recognize-canvas").classList.remove("visible");
    byId("recognition-state").lastChild.textContent = " En espera";
    showToast(error.message, "error");
  }
});

function setTrainingRunning(running, message = "", isError = false) {
  byId("train-start").disabled = running;
  byId("train-start").textContent = running ? "Entrenando modelo…" : "Iniciar entrenamiento";
  byId("train-progress").classList.toggle("hidden", !running);
  byId("train-message").textContent = message;
  byId("train-message").classList.toggle("error", isError);
}

byId("train-start").addEventListener("click", async () => {
  const button = byId("train-start");
  button.disabled = true;
  try {
    const response = await api("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const result = await response.json();
    setTrainingRunning(true, result.message);
    pollTraining();
  } catch (error) {
    button.disabled = false;
    setTrainingRunning(false, error.message, true);
  }
});

async function pollTraining() {
  try {
    const response = await api("/api/status");
    const data = await response.json();
    if (data.training.state === "running") {
      setTrainingRunning(true, data.training.message || "El entrenamiento está en curso…");
      window.setTimeout(pollTraining, 1100);
      return;
    }
    await refreshStatus();
    const failed = data.training.state === "error";
    setTrainingRunning(false, data.training.message || (failed ? "No se pudo completar." : "El modelo está listo."), failed);
    showToast(data.training.message || (failed ? "No se pudo completar el entrenamiento." : "Modelo entrenado."), failed ? "error" : "");
  } catch (error) {
    setTrainingRunning(false, error.message, true);
  }
}

function openDeleteModal(name) {
  deleteTarget = name;
  byId("delete-description").textContent = `Se eliminarán los recortes de “${name}” en este dispositivo. El modelo quedará pendiente de entrenamiento.`;
  byId("delete-confirm").value = "";
  byId("delete-submit").disabled = true;
  byId("delete-modal").classList.remove("hidden");
  byId("delete-confirm").focus();
}

function closeDeleteModal() {
  byId("delete-modal").classList.add("hidden");
  deleteTarget = null;
}

byId("delete-confirm").addEventListener("input", (event) => {
  byId("delete-submit").disabled = event.target.value.trim() !== "ELIMINAR";
});
byId("delete-close").addEventListener("click", closeDeleteModal);
byId("delete-cancel").addEventListener("click", closeDeleteModal);
byId("delete-modal").addEventListener("click", (event) => {
  if (event.target === byId("delete-modal")) closeDeleteModal();
});
byId("delete-submit").addEventListener("click", async () => {
  if (!deleteTarget) return;
  const button = byId("delete-submit");
  button.disabled = true;
  try {
    await api(`/api/people/${encodeURIComponent(deleteTarget)}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmation: byId("delete-confirm").value.trim() }),
    });
    closeDeleteModal();
    showToast("Perfil y recortes eliminados.");
    await refreshPeople();
    await refreshStatus();
  } catch (error) {
    showToast(error.message, "error");
    button.disabled = false;
  }
});

window.addEventListener("beforeunload", stopCamera);
const initialPage = location.hash.slice(1);
navigate(pages.includes(initialPage) ? initialPage : "inicio");
refreshStatus();
refreshPeople();
window.setInterval(() => {
  if (currentPage === "inicio" || currentPage === "entrenar") refreshStatus();
}, 5000);
