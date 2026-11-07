"""
Interfaz gráfica — Face Recognizer
python app.py
"""

import os, sys, shutil, threading, subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Usar el Python del venv del proyecto si existe; evita que sys.executable
# apunte al intérprete base de pyenv/sistema sin los paquetes instalados.
_venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
PYTHON = _venv_python if os.path.isfile(_venv_python) else sys.executable

DATASET_DIR = "data/known_faces"
MODEL_FILE  = "models/face_recognizer.yml"

# ─── Paleta ────────────────────────────────────────────────────────────────────
BG      = "#0d0d1a"
SIDEBAR = "#12122b"
CARD    = "#1a1a30"
CARD_HL = "#222240"
ACCENT  = "#7c3aed"
ACCENT_H= "#6d28d9"
CYAN    = "#06b6d4"
CYAN_H  = "#0891b2"
GREEN   = "#10b981"
RED     = "#ef4444"
ORANGE  = "#f59e0b"
TEXT    = "#f1f5f9"
TEXT2   = "#cbd5e1"
MUTED   = "#475569"
BORDER  = "#1e1e40"

F_H1   = ("Helvetica", 20, "bold")
F_H2   = ("Helvetica", 14, "bold")
F_BODY = ("Helvetica", 11)
F_SM   = ("Helvetica", 9)
F_BTN  = ("Helvetica", 11, "bold")
F_MONO = ("Courier", 10)

# ─── Utilidades ────────────────────────────────────────────────────────────────
def get_personas():
    if not os.path.exists(DATASET_DIR): return []
    return sorted([d for d in os.listdir(DATASET_DIR)
                   if os.path.isdir(os.path.join(DATASET_DIR, d))])

def count_fotos(nombre):
    d = os.path.join(DATASET_DIR, nombre)
    return len([f for f in os.listdir(d)
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])

def modelo_listo():
    return os.path.exists(MODEL_FILE)

def make_avatar(nombre, size=52):
    d = os.path.join(DATASET_DIR, nombre)
    fotos = [f for f in os.listdir(d)
             if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
    if not fotos:
        return None
    try:
        img = Image.open(os.path.join(d, fotos[0])).convert("RGBA")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
        img = img.resize((size, size), Image.LANCZOS)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
        out = Image.new("RGBA", (size, size), (0,0,0,0))
        out.paste(img, mask=mask)
        return ImageTk.PhotoImage(out)
    except Exception:
        return None

def darken(hex_c, amt=25):
    h = hex_c.lstrip("#")
    r,g,b = [int(h[i:i+2],16) for i in (0,2,4)]
    return "#{:02x}{:02x}{:02x}".format(
        max(0,r-amt), max(0,g-amt), max(0,b-amt))

# ─── Widgets base ──────────────────────────────────────────────────────────────
class Btn(tk.Button):
    def __init__(self, p, text, cmd, bg=ACCENT, fg="white",
                 icon="", pad_x=18, pad_y=9, **kw):
        label = f"{icon}  {text}" if icon else text
        super().__init__(p, text=label, command=cmd,
                         bg=bg, fg=fg, font=F_BTN, relief="flat",
                         cursor="hand2", activeforeground="white",
                         activebackground=darken(bg),
                         padx=pad_x, pady=pad_y, **kw)
        self._bg = bg
        self.bind("<Enter>", lambda e: self.config(bg=darken(bg)))
        self.bind("<Leave>", lambda e: self.config(bg=bg))

class NavBtn(tk.Frame):
    def __init__(self, parent, icon, label, cmd):
        super().__init__(parent, bg=SIDEBAR, cursor="hand2")
        self._active = False
        self._cmd = cmd

        self.bar = tk.Frame(self, bg=SIDEBAR, width=4)
        self.bar.pack(side="left", fill="y")

        inner = tk.Frame(self, bg=SIDEBAR, padx=14, pady=13)
        inner.pack(fill="both", expand=True)

        self.ico = tk.Label(inner, text=icon, font=("Helvetica",15),
                            bg=SIDEBAR, fg=MUTED)
        self.ico.pack(side="left")
        self.lbl = tk.Label(inner, text=label, font=F_BODY,
                            bg=SIDEBAR, fg=MUTED, padx=10)
        self.lbl.pack(side="left")

        for w in (self, inner, self.ico, self.lbl):
            w.bind("<Button-1>", lambda e: cmd())
            w.bind("<Enter>",    self._on)
            w.bind("<Leave>",    self._off)

    def activate(self, v):
        self._active = v
        c = ACCENT if v else SIDEBAR
        self.bar.config(bg=c)
        self.lbl.config(fg=TEXT if v else MUTED)
        self.ico.config(fg=TEXT if v else MUTED)

    def _on(self, e):
        if not self._active:
            self.config(bg=CARD)
    def _off(self, e):
        if not self._active:
            self.config(bg=SIDEBAR)

class Toast(tk.Frame):
    """Notificación flotante temporal."""
    def __init__(self, root, msg, kind="ok"):
        color = GREEN if kind=="ok" else RED if kind=="err" else ORANGE
        super().__init__(root, bg=color, padx=18, pady=10)
        tk.Label(self, text=msg, bg=color, fg="white", font=F_BODY).pack()
        self.place(relx=1.0, rely=1.0, anchor="se", x=-20, y=-20)
        root.after(2800, self.destroy)

class Entry(tk.Frame):
    """Entry estilizado con label y línea inferior."""
    def __init__(self, parent, label, **kw):
        super().__init__(parent, bg=BG)
        tk.Label(self, text=label, font=F_SM, bg=BG, fg=MUTED).pack(anchor="w")
        self.var = tk.StringVar()
        e = tk.Entry(self, textvariable=self.var, font=("Helvetica",12),
                     bg=CARD, fg=TEXT, insertbackground=TEXT,
                     relief="flat", bd=0, **kw)
        e.pack(fill="x", ipady=9, ipadx=8)
        tk.Frame(self, bg=ACCENT, height=2).pack(fill="x")

    def get(self): return self.var.get().strip()
    def set(self, v): self.var.set(v)

class LogBox(tk.Text):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CARD, fg=TEXT, font=F_MONO,
                         relief="flat", bd=0, state="disabled", **kw)
        self.tag_config("ok",  foreground=GREEN)
        self.tag_config("err", foreground=RED)
        self.tag_config("dim", foreground=MUTED)

    def write(self, msg, tag=None):
        self.config(state="normal")
        self.insert("end", msg+"\n", tag or "")
        self.see("end")
        self.config(state="disabled")

    def clear(self):
        self.config(state="normal")
        self.delete("1.0","end")
        self.config(state="disabled")

# ─── Páginas ───────────────────────────────────────────────────────────────────
class Page(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app

    def header(self, title, sub=""):
        f = tk.Frame(self, bg=BG)
        f.pack(fill="x", padx=30, pady=(24,0))
        tk.Label(f, text=title, font=F_H1, bg=BG, fg=TEXT, anchor="w").pack(fill="x")
        if sub:
            tk.Label(f, text=sub, font=F_SM, bg=BG, fg=MUTED, anchor="w").pack(fill="x", pady=(2,0))
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=30, pady=(10,0))

    def on_show(self): pass
    def on_resize(self, w, h): pass


# ── Personas ───────────────────────────────────────────────────────────────────
class PersonasPage(Page):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.header("👥  Personas registradas",
                    "Lista de personas disponibles para el reconocimiento")

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=30, pady=(12,4))
        self.lbl_stats = tk.Label(top, text="", font=F_SM, bg=BG, fg=MUTED, anchor="w")
        self.lbl_stats.pack(side="left")
        Btn(top, "Agregar persona", lambda: app.show("agregar"),
            bg=ACCENT, icon="➕", pad_x=14, pad_y=7).pack(side="right")

        # Scroll area
        wrap = tk.Frame(self, bg=BG)
        wrap.pack(fill="both", expand=True, padx=30, pady=8)

        self.canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=BG)
        self.inner.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._resize_job = None
        self._cards_data = []
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        # Windows/Mac
        self.canvas.bind("<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        # Linux
        self.canvas.bind("<Button-4>",
            lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind("<Button-5>",
            lambda e: self.canvas.yview_scroll(1, "units"))

    def _on_canvas_resize(self, e):
        self.canvas.itemconfig(self._win, width=e.width)
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(150, self._draw_cards)

    def on_show(self): self.refresh()

    def refresh(self):
        for w in self.inner.winfo_children(): w.destroy()
        personas = get_personas()
        total = sum(count_fotos(p) for p in personas)
        self.lbl_stats.config(text=f"{len(personas)} personas  ·  {total} fotos")

        if not personas:
            tk.Label(self.inner, text="Sin personas registradas.\nUsá el botón 'Agregar persona'.",
                     font=F_BODY, bg=BG, fg=MUTED, justify="center"
                     ).pack(pady=60)
            return

        # Grid de cards responsive
        self._cards_data = personas
        self._draw_cards()

    def _draw_cards(self):
        for w in self.inner.winfo_children(): w.destroy()
        personas = self._cards_data
        cols = max(1, self.canvas.winfo_width() // 320)
        for i, p in enumerate(personas):
            r, c = divmod(i, cols)
            card = self._make_card(self.inner, p)
            card.grid(row=r, column=c, padx=8, pady=8, sticky="nsew")
            self.inner.columnconfigure(c, weight=1)

    def _make_card(self, parent, nombre):
        card = tk.Frame(parent, bg=CARD, padx=16, pady=14,
                        highlightbackground=BORDER, highlightthickness=1)

        # Avatar + info
        top = tk.Frame(card, bg=CARD)
        top.pack(fill="x")

        thumb = make_avatar(nombre, 48)
        if thumb:
            lbl = tk.Label(top, image=thumb, bg=CARD)
            lbl.image = thumb
            lbl.pack(side="left", padx=(0,12))
        else:
            tk.Label(top, text="👤", font=("Helvetica",24), bg=CARD
                     ).pack(side="left", padx=(0,12))

        info = tk.Frame(top, bg=CARD)
        info.pack(side="left", fill="both", expand=True)
        tk.Label(info, text=nombre, font=F_H2, bg=CARD, fg=TEXT, anchor="w"
                 ).pack(fill="x")
        n = count_fotos(nombre)
        fc = GREEN if n >= 15 else ORANGE if n >= 6 else RED
        tk.Label(info, text=f"📸  {n} fotos", font=F_SM,
                 bg=CARD, fg=fc, anchor="w").pack(fill="x", pady=(2,0))

        # Barra de fotos
        bar_bg = tk.Frame(card, bg=BORDER, height=4)
        bar_bg.pack(fill="x", pady=(10,8))
        pct = min(n/25, 1.0)
        tk.Frame(bar_bg, bg=fc, height=4).place(relx=0, rely=0,
                                                  relwidth=pct, relheight=1)

        # Consejo
        hint = "Listo ✔" if n >= 15 else f"Agregar {max(0,15-n)} más para mejor precisión"
        tk.Label(card, text=hint, font=F_SM, bg=CARD,
                 fg=GREEN if n>=15 else MUTED).pack(anchor="w")

        # Botón eliminar
        Btn(card, "Eliminar", lambda p=nombre: self._del(p),
            bg=CARD, fg=RED, pad_x=10, pad_y=5).pack(anchor="e", pady=(8,0))

        return card

    def _del(self, nombre):
        if messagebox.askyesno("Confirmar", f"¿Eliminar a '{nombre}' y todas sus fotos?"):
            shutil.rmtree(os.path.join(DATASET_DIR, nombre))
            Toast(self.app, f"'{nombre}' eliminado", "err")
            self.refresh()


# ── Agregar persona ────────────────────────────────────────────────────────────
class AgregarPage(Page):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.header("➕  Agregar persona",
                    "Registrá a alguien con fotos existentes o capturá desde la webcam")

        # Layout dos columnas
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=30, pady=16)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        # ── Columna izquierda: formulario ──────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,12))

        self.entry_nombre = Entry(left, "NOMBRE DE LA PERSONA")
        self.entry_nombre.pack(fill="x", pady=(0,16))

        # Fuente
        tk.Label(left, text="FUENTE DE FOTOS", font=F_SM,
                 bg=BG, fg=MUTED).pack(anchor="w")
        rb = tk.Frame(left, bg=BG)
        rb.pack(fill="x", pady=(6,16))
        self.fuente = tk.StringVar(value="carpeta")

        for val, txt, clr in [("carpeta","📁  Desde carpeta", ACCENT),
                               ("webcam", "📷  Desde webcam",  CYAN)]:
            f = tk.Frame(rb, bg=CARD, padx=14, pady=10,
                         highlightbackground=BORDER, highlightthickness=1,
                         cursor="hand2")
            f.pack(side="left", padx=(0,8), fill="x", expand=True)
            r = tk.Radiobutton(f, text=txt, variable=self.fuente, value=val,
                               bg=CARD, fg=TEXT, selectcolor=CARD,
                               activebackground=CARD, font=F_BODY,
                               indicatoron=False, relief="flat",
                               cursor="hand2")
            r.pack(fill="x")

        # Carpeta
        self.carpeta_frame = tk.Frame(left, bg=BG)
        self.carpeta_frame.pack(fill="x", pady=(0,16))
        tk.Label(self.carpeta_frame, text="CARPETA CON FOTOS",
                 font=F_SM, bg=BG, fg=MUTED).pack(anchor="w")
        row = tk.Frame(self.carpeta_frame, bg=BG)
        row.pack(fill="x", pady=(6,0))
        self.entry_carpeta = Entry(row, "")
        self.entry_carpeta.pack(side="left", fill="x", expand=True)
        Btn(row, "Buscar", self._browse, bg=CARD_HL,
            icon="📂", pad_x=12, pad_y=10).pack(side="left", padx=(8,0), anchor="s")

        # Botón principal
        Btn(left, "Agregar persona", self._agregar,
            bg=ACCENT, icon="✅", pad_x=20, pad_y=11).pack(anchor="w", pady=(4,0))

        # ── Columna derecha: log ───────────────────────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(12,0))
        right.rowconfigure(1, weight=1)

        tk.Label(right, text="ACTIVIDAD", font=F_SM,
                 bg=BG, fg=MUTED).pack(anchor="w")
        self.log = LogBox(right, height=18)
        self.log.pack(fill="both", expand=True, pady=(6,0))
        self.log.write("Esperando acción...", "dim")

    def on_show(self): pass

    def _browse(self):
        self.app.lift()
        ruta = filedialog.askdirectory(
            title="Seleccioná la carpeta con fotos",
            parent=self.app)
        if ruta:
            self.entry_carpeta.set(ruta)
            self.log.write(f"📂 Carpeta: {ruta}", "dim")

    def _agregar(self):
        nombre = self.entry_nombre.get()
        if not nombre:
            Toast(self.app, "Ingresá un nombre primero", "warn")
            return
        dest = os.path.join(DATASET_DIR, nombre)
        os.makedirs(dest, exist_ok=True)

        if self.fuente.get() == "carpeta":
            ruta = self.entry_carpeta.get()
            if not os.path.isdir(ruta):
                Toast(self.app, "Carpeta no encontrada", "err")
                return
            exts = (".jpg",".jpeg",".png",".bmp",".webp")
            archivos = [f for f in os.listdir(ruta) if f.lower().endswith(exts)]
            if not archivos:
                Toast(self.app, "No hay imágenes en esa carpeta", "warn")
                return
            self.log.clear()
            for f in archivos:
                shutil.copy2(os.path.join(ruta, f), dest)
                self.log.write(f"  ✔ {f}", "ok")
            self.log.write(f"\n✅  {len(archivos)} fotos copiadas para '{nombre}'", "ok")
            Toast(self.app, f"'{nombre}' agregado con {len(archivos)} fotos")
        else:
            self.log.clear()
            self.log.write(f"🎥 Abriendo webcam para '{nombre}'...", "dim")
            self.log.write("   Espacio = captura manual   |   A = auto   |   Q = salir", "dim")
            def run():
                result = subprocess.run(
                    [PYTHON, "2_collect_faces.py",
                     "--name", nombre, "--source", "webcam", "--max", "30"])
                def _done():
                    if result.returncode == 0:
                        self.log.write(f"\n✅  Captura finalizada para '{nombre}'", "ok")
                        Toast(self.app, f"Captura de '{nombre}' completada")
                    else:
                        self.log.write(f"\n❌  Error en la captura (código {result.returncode})", "err")
                        Toast(self.app, "Error en la captura", "err")
                self.app.after(0, _done)
            threading.Thread(target=run, daemon=True).start()


# ── Entrenar ───────────────────────────────────────────────────────────────────
class EntrenarPage(Page):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.header("🏋️  Entrenar modelo",
                    "Genera el clasificador LBPH con las fotos de cada persona")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=30, pady=16)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        # Stats row
        stats = tk.Frame(body, bg=BG)
        stats.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,16))
        self.c_personas = self._stat(stats, "👥", "Personas", "0")
        self.c_fotos    = self._stat(stats, "📸", "Fotos",    "0")
        self.c_estado   = self._stat(stats, "🤖", "Modelo",   "—")

        # Log
        left = tk.Frame(body, bg=BG)
        left.grid(row=1, column=0, sticky="nsew", padx=(0,12))
        left.rowconfigure(0, weight=1)
        tk.Label(left, text="SALIDA DEL PROCESO", font=F_SM,
                 bg=BG, fg=MUTED).pack(anchor="w")
        self.log = LogBox(left, height=16)
        self.log.pack(fill="both", expand=True, pady=(6,0))

        # Derecha: opciones + botón
        right = tk.Frame(body, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", padx=(12,0))

        # Card info
        info_card = tk.Frame(right, bg=CARD, padx=20, pady=18,
                             highlightbackground=BORDER, highlightthickness=1)
        info_card.pack(fill="x", pady=(0,16))
        tk.Label(info_card, text="ℹ️  Sobre el entrenamiento",
                 font=F_H2, bg=CARD, fg=TEXT).pack(anchor="w")
        for t in [
            "• Mínimo 5 fotos por persona",
            "• 15-30 fotos = mejor precisión",
            "• Variá ángulos e iluminación",
            "• Re-entrenà al agregar personas",
        ]:
            tk.Label(info_card, text=t, font=F_SM, bg=CARD,
                     fg=TEXT2, anchor="w").pack(fill="x", pady=2)

        # Barra progreso
        self.prog = ttk.Progressbar(right, mode="indeterminate")
        self.prog.pack(fill="x", pady=(0,12))
        ttk.Style().configure("TProgressbar",
                              troughcolor=CARD, background=ACCENT,
                              thickness=6)

        self.btn = Btn(right, "Iniciar entrenamiento", self._entrenar,
                       bg=ACCENT, icon="🚀", pad_x=20, pad_y=13)
        self.btn.pack(fill="x")

        Btn(right, "Ver personas", lambda: app.show("personas"),
            bg=CARD_HL, pad_x=20, pad_y=10).pack(fill="x", pady=(8,0))

    def _stat(self, parent, icon, label, val):
        c = tk.Frame(parent, bg=CARD, padx=22, pady=16,
                     highlightbackground=BORDER, highlightthickness=1)
        c.pack(side="left", padx=(0,10), fill="x", expand=True)
        tk.Label(c, text=icon, font=("Helvetica",20), bg=CARD).pack()
        v = tk.Label(c, text=val, font=("Helvetica",26,"bold"), bg=CARD, fg=TEXT)
        v.pack(pady=2)
        tk.Label(c, text=label, font=F_SM, bg=CARD, fg=MUTED).pack()
        return v

    def on_show(self): self.refresh()

    def refresh(self):
        ps = get_personas()
        total = sum(count_fotos(p) for p in ps)
        self.c_personas.config(text=str(len(ps)))
        self.c_fotos.config(text=str(total))
        ok = modelo_listo()
        self.c_estado.config(text="✅ listo" if ok else "⚠ pendiente",
                             fg=GREEN if ok else ORANGE)

    def _entrenar(self):
        if not get_personas():
            Toast(self.app, "No hay personas registradas", "warn"); return
        self.btn.config(state="disabled", text="  Entrenando...")
        self.log.clear()
        self.prog.start(10)
        def run():
            proc = subprocess.Popen(
                [PYTHON, "3_train.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                l = line.rstrip()
                tag = "ok" if "✔" in l or "completado" in l.lower() else \
                      "err" if "error" in l.lower() else "dim"
                self.app.after(0, lambda l=l, tag=tag: self.log.write(l, tag))
            rc = proc.wait()
            def _done():
                self.prog.stop()
                if rc == 0:
                    self.log.write("\n✅  Modelo listo para usar.", "ok")
                    Toast(self.app, "Modelo entrenado correctamente")
                else:
                    self.log.write(f"\n❌  Entrenamiento falló (código {rc})", "err")
                    Toast(self.app, "Error en el entrenamiento", "err")
                self.btn.config(state="normal", text="🚀  Iniciar entrenamiento")
                self.refresh()
            self.app.after(0, _done)
        threading.Thread(target=run, daemon=True).start()


# ── Reconocer ──────────────────────────────────────────────────────────────────
class Toggle(tk.Frame):
    """Switch ON/OFF estilizado."""
    def __init__(self, parent, label, default=True, **kw):
        super().__init__(parent, bg=CARD, **kw)
        self.var = tk.BooleanVar(value=default)

        self.lbl = tk.Label(self, text=label, font=F_SM, bg=CARD, fg=TEXT2)
        self.lbl.pack(side="left", padx=(0, 10))

        self.btn = tk.Label(self, font=F_SM, cursor="hand2",
                            padx=10, pady=4, relief="flat")
        self.btn.pack(side="left")
        self.btn.bind("<Button-1>", self._toggle)
        self._draw()

    def _draw(self):
        on = self.var.get()
        self.btn.config(
            text="  ON  " if on else "  OFF  ",
            bg=GREEN if on else MUTED,
            fg="white")

    def _toggle(self, e=None):
        self.var.set(not self.var.get())
        self._draw()

    def get(self): return self.var.get()


class ReconocerPage(Page):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.header("📷  Reconocimiento facial",
                    "Detecta e identifica personas en tiempo real o en imágenes")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=30, pady=16)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Cards acción
        self._big_card(body, col=0,
            icon="📷", title="Cámara en vivo",
            desc="Reconocimiento en tiempo real.\nDetecta y etiqueta automáticamente.",
            btn_text="Encender cámara", btn_color=ACCENT, cmd=self._webcam)

        self._big_card(body, col=1,
            icon="🖼️", title="Imagen / Carpeta",
            desc="Analiza fotos existentes.\nPodés guardar el resultado anotado.",
            btn_text="Seleccionar archivo", btn_color=CYAN, cmd=self._imagen)

        # Panel opciones
        opts = tk.Frame(self, bg=CARD, padx=24, pady=14,
                        highlightbackground=BORDER, highlightthickness=1)
        opts.pack(fill="x", padx=30, pady=(10, 4))

        tk.Label(opts, text="OPCIONES DE ANÁLISIS", font=F_SM,
                 bg=CARD, fg=MUTED).pack(anchor="w", pady=(0, 8))

        row = tk.Frame(opts, bg=CARD)
        row.pack(fill="x")

        self.tog_mesh = Toggle(row, "🕸  Malla facial (468 puntos)", default=True)
        self.tog_mesh.pack(side="left", padx=(0, 24))

        self.tog_pose = Toggle(row, "🦴  Esqueleto corporal", default=True)
        self.tog_pose.pack(side="left", padx=(0, 24))

        self.tog_save = Toggle(row, "💾  Guardar resultado", default=False)
        self.tog_save.pack(side="left")

        # Personas conocidas
        bot = tk.Frame(self, bg=CARD, padx=24, pady=12,
                       highlightbackground=BORDER, highlightthickness=1)
        bot.pack(fill="x", padx=30, pady=(4, 8))
        tk.Label(bot, text="Personas en el modelo:", font=F_SM,
                 bg=CARD, fg=MUTED).pack(anchor="w")
        


        
        self.lbl_pers = tk.Label(bot, text="", font=F_BODY,
                                 bg=CARD, fg=TEXT, wraplength=700, anchor="w")
        self.lbl_pers.pack(fill="x", pady=(4, 0))

        tk.Label(self, text="Presioná  Q  para salir de la cámara",
                 font=F_SM, bg=BG, fg=MUTED).pack(pady=(0, 4))

        self._proc = None
        self.btn_detener = Btn(self, "Detener cámara", self._detener,
                               bg=RED, icon="⏹", pad_x=16, pad_y=7)
        # se muestra solo cuando hay proceso activo

    def _big_card(self, parent, col, icon, title, desc, btn_text, btn_color, cmd):
        card = tk.Frame(parent, bg=CARD, padx=30, pady=24,
                        highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=col, padx=8, pady=4, sticky="nsew")
        tk.Label(card, text=icon, font=("Helvetica", 46), bg=CARD).pack(pady=(0, 10))
        tk.Label(card, text=title, font=F_H2, bg=CARD, fg=TEXT).pack()
        tk.Label(card, text=desc, font=F_SM, bg=CARD, fg=MUTED,
                 justify="center").pack(pady=(6, 18))
        Btn(card, btn_text, cmd, bg=btn_color, pad_x=22, pad_y=11).pack()

    def on_show(self): self.refresh()

    def refresh(self):
        ps = get_personas()
        self.lbl_pers.config(
            text="  ·  ".join(ps) if ps else "Sin personas registradas.",
            fg=TEXT if ps else MUTED)

    def _build_cmd(self, source=None):
        cmd = [PYTHON, "4_recognize.py"]
        if source:
            cmd += ["--source", source]
        if not self.tog_mesh.get():
            cmd += ["--no-mesh"]
        if not self.tog_pose.get():
            cmd += ["--no-pose"]
        if self.tog_save.get():
            cmd += ["--save"]
        return cmd

    def _launch(self, cmd):
        proc = subprocess.Popen(cmd)
        self._proc = proc
        self.app.register_proc(proc)
        self._set_running(True)
        def _watch():
            proc.wait()
            self.app.after(0, lambda: self._set_running(False))
        threading.Thread(target=_watch, daemon=True).start()

    def _set_running(self, active):
        if active:
            self.btn_detener.pack(pady=(0, 8))
        else:
            self.btn_detener.pack_forget()

    def _detener(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _webcam(self):
        if not modelo_listo():
            Toast(self.app, "Entrenà el modelo primero", "warn"); return
        if self._proc and self._proc.poll() is None:
            Toast(self.app, "La cámara ya está activa", "warn"); return
        self._launch(self._build_cmd())

    def _imagen(self):
        if not modelo_listo():
            Toast(self.app, "Entrenà el modelo primero", "warn"); return
        self.app.lift()
        ruta = filedialog.askopenfilename(
            parent=self.app,
            title="Seleccioná una imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("Todos", "*.*")])
        if not ruta:
            ruta = filedialog.askdirectory(parent=self.app,
                                           title="O seleccioná una carpeta")
        if not ruta:
            return
        self._launch(self._build_cmd(ruta))


# ─── App principal ─────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognizer")
        self.geometry("1000x660")
        self.minsize(820, 560)
        self.configure(bg=BG)

        # Estilo scrollbar
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Vertical.TScrollbar",
                    background=CARD, troughcolor=BG,
                    bordercolor=BG, arrowcolor=MUTED)

        self._nav = {}
        self._subprocs = []
        self._build()
        self.show("personas")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def register_proc(self, proc):
        self._subprocs = [p for p in self._subprocs if p.poll() is None]
        self._subprocs.append(proc)

    def _on_close(self):
        for proc in self._subprocs:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        self.destroy()

    def _build(self):
        # Sidebar
        sb = tk.Frame(self, bg=SIDEBAR, width=220)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # Logo
        logo = tk.Frame(sb, bg=SIDEBAR, pady=24, padx=18)
        logo.pack(fill="x")
        tk.Label(logo, text="🎭", font=("Helvetica",30),
                 bg=SIDEBAR, fg=ACCENT).pack()
        tk.Label(logo, text="Face Recognizer",
                 font=("Helvetica",12,"bold"), bg=SIDEBAR, fg=TEXT).pack(pady=(6,2))
        tk.Label(logo, text="YOLO  ·  LBPH",
                 font=F_SM, bg=SIDEBAR, fg=MUTED).pack()
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", padx=18, pady=10)

        for key, ico, lbl in [
            ("personas",  "👥", "Personas"),
            ("agregar",   "➕", "Agregar persona"),
            ("entrenar",  "🏋️",  "Entrenar modelo"),
            ("reconocer", "📷", "Reconocer"),
        ]:
            btn = NavBtn(sb, ico, lbl, lambda k=key: self.show(k))
            btn.pack(fill="x")
            self._nav[key] = btn

        # Estado modelo
        tk.Frame(sb, bg=BORDER, height=1).pack(fill="x", padx=18, pady=8, side="bottom")
        self.lbl_estado = tk.Label(sb, text="", font=F_SM, bg=SIDEBAR, pady=12)
        self.lbl_estado.pack(side="bottom")
        self._tick()

        # Contenido
        self.cont = tk.Frame(self, bg=BG)
        self.cont.pack(fill="both", expand=True)

        self.pages = {
            "personas":  PersonasPage(self.cont, self),
            "agregar":   AgregarPage(self.cont, self),
            "entrenar":  EntrenarPage(self.cont, self),
            "reconocer": ReconocerPage(self.cont, self),
        }
        for p in self.pages.values():
            p.place(relwidth=1, relheight=1)

    def show(self, key):
        for k, b in self._nav.items():
            b.activate(k == key)
        self.pages[key].tkraise()
        self.pages[key].on_show()

    def _tick(self):
        ok = modelo_listo()
        self.lbl_estado.config(
            text="✅  Modelo listo" if ok else "⚠️  Sin entrenar",
            fg=GREEN if ok else ORANGE)
        self.after(2000, self._tick)


if __name__ == "__main__":
    App().mainloop()
