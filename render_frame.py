"""Captura frames 'bonitos' de un MJCF sin renderizar la simulación.

El bucle de física solo llama a `mj_step`. El `mujoco.Renderer` de alta
calidad (sombras, MSAA, texturas; Filament si la lib se compiló así) se
crea de forma perezosa y solo se usa en los pasos listados en
`BEAUTY_STEPS`. Esos PNG son los que luego entran en un vídeo corto.

SSAO no tiene flag público en Python; aquí van las opciones que sí expone
`Renderer`. No uses este renderer dentro de un bucle a 30–60 fps.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
from PIL import Image

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
SCENE_XML = "scene.xml"
OUTPUT_DIR = Path("output/beauty_frames")
HIGHLIGHT_VIDEO = Path("output/beauty_frames/highlight.mp4")
MAKE_VIDEO = True  # ffmpeg solo con los PNG especiales, no con cada mj_step

# Resolución de los stills (no de un preview de simulación).
WIDTH = 1920
HEIGHT = 1080

# Física sin dibujar nada.
SIM_STEPS = 200

# Pasos (después de N mj_step) que sí se renderizan en alta calidad.
# Ejemplo: asentar + un instante posterior. Vacío = solo el último paso.
BEAUTY_STEPS = (16, 200)

# Cámara XML, o None = cámara libre.
CAMERA_NAME = None  # p. ej. "track"

LOOKAT = [0.0, 0.0, 0.5]
DISTANCE = 2.5
AZIMUTH = 135.0  # ángulo horizontal de la cámara libre
ELEVATION = -20.0


class BeautyCapture:
    """Renderer caro, creado al primer snapshot. Ciérralo al terminar."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        width: int = WIDTH,
        height: int = HEIGHT,
        camera_name: str | None = CAMERA_NAME,
        lookat=LOOKAT,
        distance: float = DISTANCE,
        azimuth: float = AZIMUTH,
        elevation: float = ELEVATION,
        max_geom: int = 50_000,
    ) -> None:
        self.model = model
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.lookat = lookat
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.max_geom = max_geom
        self._renderer: mujoco.Renderer | None = None
        self._option: mujoco.MjvOption | None = None

    def _ensure_renderer(self) -> mujoco.Renderer:
        if self._renderer is not None:
            return self._renderer
        # vis.quality se lee al crear el contexto GL: no tocarlo durante mj_step.
        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), self.width
        )
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), self.height
        )
        self.model.vis.quality.shadowsize = max(
            int(self.model.vis.quality.shadowsize), 4096
        )
        self.model.vis.quality.offsamples = max(
            int(self.model.vis.quality.offsamples), 8
        )
        self.model.vis.quality.numslices = max(
            int(self.model.vis.quality.numslices), 28
        )
        self.model.vis.quality.numstacks = max(
            int(self.model.vis.quality.numstacks), 16
        )
        self.model.vis.quality.numquads = max(
            int(self.model.vis.quality.numquads), 8
        )
        renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width, max_geom=self.max_geom
        )
        flags = mujoco.mjtRndFlag
        renderer.scene.flags[flags.mjRND_SHADOW] = 1
        renderer.scene.flags[flags.mjRND_REFLECTION] = 1
        renderer.scene.flags[flags.mjRND_SKYBOX] = 1
        renderer.scene.flags[flags.mjRND_HAZE] = 1
        renderer.scene.flags[flags.mjRND_CULL_FACE] = 1
        renderer.scene.flags[flags.mjRND_WIREFRAME] = 0
        option = mujoco.MjvOption()
        option.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
        self._renderer = renderer
        self._option = option
        return renderer

    def camera(self, azimuth: float | None = None) -> mujoco.MjvCamera | str:
        if self.camera_name:
            return self.camera_name
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = self.lookat
        camera.distance = self.distance
        camera.azimuth = self.azimuth if azimuth is None else azimuth
        camera.elevation = self.elevation
        return camera

    def snapshot(self, data: mujoco.MjData, path: Path, azimuth: float | None = None):
        """Un solo frame de alta calidad. Llamar solo en instantes elegidos."""
        renderer = self._ensure_renderer()
        renderer.update_scene(data, camera=self.camera(azimuth), scene_option=self._option)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(renderer.render()).save(path)
        print(f"beauty frame -> {path.resolve()}")
        return path

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def simulate(model: mujoco.MjModel, data: mujoco.MjData, n_steps: int) -> None:
    """Solo física. Sin GL, sin sombras, sin leer píxeles."""
    for _ in range(n_steps):
        mujoco.mj_step(model, data)


def frames_to_video(pngs: list[Path], video: Path, fps: int = 2) -> None:
    """Monta un MP4 corto solo con los stills bonitos (no con el trace de sim)."""
    if not pngs:
        return
    video = Path(video)
    video.parent.mkdir(parents=True, exist_ok=True)
    list_file = video.with_suffix(".ffmpeg.txt")
    # Un segundo por imagen si fps=1; con fps=2 cada still dura 0.5 s.
    list_file.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\nduration {1.0 / fps}\n" for p in pngs)
        + f"file '{pngs[-1].resolve().as_posix()}'\n",
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-vsync", "vfr", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video),
    ]
    if subprocess.run(cmd, check=False).returncode != 0:
        raise RuntimeError("ffmpeg failed while encoding highlight video")
    print(f"highlight video -> {video.resolve()}  ({len(pngs)} frames, {fps} fps)")


def main() -> None:
    scene_path = Path(SCENE_XML)
    if not scene_path.is_file():
        raise FileNotFoundError(f"No se encontró {scene_path.resolve()}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    capture_at = set(BEAUTY_STEPS) if BEAUTY_STEPS else {SIM_STEPS}
    capture_at.add(SIM_STEPS)
    pngs: list[Path] = []
    beauty = BeautyCapture(model)  # todavía no abre GL
    try:
        for step in range(1, SIM_STEPS + 1):
            mujoco.mj_step(model, data)
            if step not in capture_at:
                continue
            pngs.append(beauty.snapshot(data, OUTPUT_DIR / f"frame_{step:06d}.png"))
    finally:
        beauty.close()

    if MAKE_VIDEO and pngs:
        frames_to_video(pngs, HIGHLIGHT_VIDEO)


if __name__ == "__main__":
    main()
