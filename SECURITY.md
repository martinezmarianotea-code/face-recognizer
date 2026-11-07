# Seguridad y privacidad

## Datos faciales

Las imágenes, los modelos entrenados y los resultados pueden revelar información biométrica. Se guardan localmente y no están cifrados. No los adjuntes a incidencias, pull requests ni capturas públicas. Si compartes registros de errores, elimina nombres, rutas privadas e imágenes.

La interfaz web escucha únicamente en `127.0.0.1` y valida el host y el origen de las solicitudes. No cambies el enlace a `0.0.0.0`, no la expongas en una red local y no la publiques detrás de un proxy. Las imágenes para reconocimiento web se procesan en memoria; al registrar, solo se guarda el recorte facial. El acceso a la cámara lo concede el navegador para el origen local.

Antes de enviar cambios, comprueba que `data/`, `models/`, `output/`, los entornos virtuales y cualquier imagen o video personal no estén incluidos en el commit.

## Reportar una vulnerabilidad

No publiques detalles explotables en una incidencia pública. Si el repositorio tiene habilitados los informes privados de vulnerabilidades de GitHub, utiliza la pestaña **Security → Report a vulnerability**. Si no está disponible, abre una incidencia sin incluir datos sensibles y solicita un canal privado de contacto.

Este proyecto es experimental y no debe exponerse como servicio de autenticación ni usarse para decisiones de seguridad.
