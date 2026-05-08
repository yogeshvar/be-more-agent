from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from beemboy.config.settings import Settings
from beemboy.ui.controller import UIController


class DesktopUIApp:
    """Tkinter desktop surface for camera preview + identity state."""

    def __init__(self, settings: Settings, controller: UIController) -> None:
        self._settings = settings
        self._controller = controller
        self._root = tk.Tk()
        self._root.title("Beemboy Desktop")
        self._root.geometry("1100x700")
        self._root.protocol("WM_DELETE_WINDOW", self._on_safe_exit)

        self._assets_root = Path(settings.ui_assets_path).expanduser()
        self._capture = None
        self._loop_after_id: str | None = None
        self._camera_available = False
        self._preview_photo: tk.PhotoImage | None = None
        self._identity_photo: tk.PhotoImage | None = None

        self._status_var = tk.StringVar(value="Idle")
        self._camera_var = tk.StringVar(value="Camera not started")

        self._preview_label = tk.Label(self._root, text="Camera preview unavailable", anchor="center")
        self._identities_tree = ttk.Treeview(
            self._root,
            columns=("name", "person_id", "source", "last_seen"),
            show="headings",
            height=12,
        )
        self._status_text = tk.Text(self._root, height=10, state="disabled", wrap="word")
        self._identity_image_label = tk.Label(self._root, text="No thumbnail")

        self._build_layout()
        self._load_assets()
        self._refresh_identities()
        self._refresh_status_text()
        self._camera_var.set("Ready")

    def run(self) -> None:
        self._root.mainloop()

    def _build_layout(self) -> None:
        container = ttk.Frame(self._root, padding=12)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x")
        ttk.Label(header, text="Beemboy Desktop UI", font=("TkDefaultFont", 13, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self._status_var).pack(side="right")

        body = ttk.Frame(container)
        body.pack(fill="both", expand=True, pady=(12, 8))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        preview_wrap = ttk.LabelFrame(body, text="Live Preview")
        preview_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._preview_label.pack(in_=preview_wrap, fill="both", expand=True, padx=8, pady=8)
        ttk.Label(preview_wrap, textvariable=self._camera_var).pack(anchor="w", padx=8, pady=(0, 8))

        identities_wrap = ttk.LabelFrame(body, text="Known Identities")
        identities_wrap.grid(row=0, column=1, sticky="nsew")
        for idx, heading in enumerate(("Name", "Person ID", "Source", "Last seen")):
            col = self._identities_tree["columns"][idx]
            self._identities_tree.heading(col, text=heading)
            self._identities_tree.column(col, width=120 if idx else 140, stretch=True)
        self._identities_tree.bind("<<TreeviewSelect>>", self._on_identity_selected)
        self._identities_tree.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self._identity_image_label.pack(fill="x", padx=8, pady=(0, 8))

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Start", command=self._on_start).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Stop", command=self._on_stop).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Save now", command=self._on_save_now).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Safe Exit", command=self._on_safe_exit).pack(side="left")

        status_wrap = ttk.LabelFrame(container, text="Recognition Status")
        status_wrap.pack(fill="both", expand=False)
        self._status_text.pack(in_=status_wrap, fill="both", expand=True, padx=8, pady=8)

    def _load_assets(self) -> None:
        icon_path = self._first_asset(
            [
                "idle/01-frame.png",
                "idle/02-frame.png",
                "loading/01-frame.png",
            ]
        )
        if icon_path:
            try:
                icon = tk.PhotoImage(file=str(icon_path))
                self._root.iconphoto(True, icon)
                self._identity_photo = icon
                self._identity_image_label.configure(image=self._identity_photo, text="")
            except tk.TclError:
                self._identity_image_label.configure(text="Thumbnail asset could not be loaded")
        else:
            self._identity_image_label.configure(text="No assets found; running without thumbnails")

    def _on_start(self) -> None:
        self._controller.start()
        self._status_var.set("Running")
        self._ensure_camera()
        self._schedule_loop()
        self._refresh_status_text()

    def _on_stop(self) -> None:
        self._controller.stop()
        self._status_var.set("Stopped")
        self._refresh_status_text()

    def _on_save_now(self) -> None:
        self._controller.save_now()
        self._refresh_identities()
        self._refresh_status_text()

    def _on_safe_exit(self) -> None:
        if self._loop_after_id:
            self._root.after_cancel(self._loop_after_id)
            self._loop_after_id = None
        self._controller.safe_exit()
        self._release_camera()
        self._refresh_status_text()
        self._root.destroy()

    def _schedule_loop(self) -> None:
        if self._loop_after_id:
            return
        self._loop_after_id = self._root.after(self._settings.ui_poll_interval_ms, self._poll_camera)

    def _poll_camera(self) -> None:
        self._loop_after_id = None
        if not self._controller.running:
            return
        if self._camera_available and self._capture is not None:
            self._process_capture_frame()
        else:
            self._camera_var.set("Camera unavailable")
        self._refresh_status_text()
        self._schedule_loop()

    def _process_capture_frame(self) -> None:
        if self._capture is None:
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._camera_var.set("Camera read failed")
            return
        self._camera_var.set("Camera online")
        self._set_preview_image(frame)
        jpeg = self._encode_jpeg(frame)
        if not jpeg:
            return
        statuses = self._controller.process_frame(jpeg)
        if statuses:
            self._refresh_identities()

    def _set_preview_image(self, frame) -> None:  # noqa: ANN001
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        ppm_header = f"P6 {w} {h} 255 ".encode("ascii")
        ppm_data = ppm_header + rgb.tobytes()
        try:
            self._preview_photo = tk.PhotoImage(data=ppm_data, format="PPM")
            self._preview_label.configure(image=self._preview_photo, text="")
        except tk.TclError:
            self._preview_label.configure(text="Preview rendering unavailable")

    @staticmethod
    def _encode_jpeg(frame) -> bytes:  # noqa: ANN001
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return b""
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return b""
        return bytes(encoded)

    def _refresh_status_text(self) -> None:
        self._status_text.configure(state="normal")
        self._status_text.delete("1.0", "end")
        self._status_text.insert("1.0", self._controller.get_status_text())
        self._status_text.configure(state="disabled")

    def _refresh_identities(self) -> None:
        for item in self._identities_tree.get_children():
            self._identities_tree.delete(item)
        for identity in self._controller.list_identities():
            self._identities_tree.insert(
                "",
                "end",
                values=(identity.name, identity.person_id, identity.source, identity.last_seen_at),
            )

    def _on_identity_selected(self, _event: tk.Event[tk.Misc]) -> None:
        image_path = self._first_asset(
            [
                "idle/03-frame.png",
                "listening/01-frame.png",
                "thinking/01-frame.png",
            ]
        )
        if image_path is None:
            return
        try:
            self._identity_photo = tk.PhotoImage(file=str(image_path))
            self._identity_image_label.configure(image=self._identity_photo, text="")
        except tk.TclError:
            self._identity_image_label.configure(text="Selected identity thumbnail unavailable")

    def _ensure_camera(self) -> None:
        if self._capture is not None:
            self._camera_available = True
            return
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            self._camera_available = False
            self._camera_var.set("OpenCV not installed; preview disabled")
            return
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self._camera_available = False
            capture.release()
            self._camera_var.set("No camera available")
            return
        self._capture = capture
        self._camera_available = True
        self._camera_var.set("Camera initialized")

    def _release_camera(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._camera_available = False

    def _first_asset(self, candidates: list[str]) -> Path | None:
        for rel in candidates:
            path = self._assets_root / rel
            if path.is_file():
                return path
        return None
