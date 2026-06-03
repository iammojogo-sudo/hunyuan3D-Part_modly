import io
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress


# Redirect our print calls to stderr so stdout stays clean for the runner protocol.
_print = print


def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)


# Make stdout/stderr safe for upstream code that emits CJK debug prints on Windows.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except (AttributeError, io.UnsupportedOperation):
    pass


_HF_REPO_ID  = "tencent/Hunyuan3D-Part"
_GLB_MAGIC   = b"glTF"
_LOG         = "[Hunyuan3DPartGenerator]"

# SDP patch is a one-time global; track at module level so it survives
# multiple generator instances in the same process.
_SDP_PATCHED = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(val, default):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _fast_geometric_aabb(mesh, n):
    """Axis-split fallback AABB — used if SAM prediction fails entirely."""
    mn, mx = mesh.bounds[0].copy(), mesh.bounds[1].copy()
    span   = mx - mn
    axis   = int(span.argmax())
    boxes  = []
    for i in range(n):
        lo, hi       = mn.copy(), mx.copy()
        lo[axis]     = mn[axis] + i       * span[axis] / n
        hi[axis]     = mn[axis] + (i + 1) * span[axis] / n
        boxes.append([lo, hi])
    return np.array(boxes)


def _weld_and_clean(geom):
    """Weld duplicate verts and strip NaN/inf/degenerate/duplicate faces."""
    try:
        geom.remove_infinite_values()
    except Exception:
        pass
    try:
        geom.merge_vertices()
    except Exception:
        pass
    try:
        geom.update_faces(geom.nondegenerate_faces())
        geom.update_faces(geom.unique_faces())
        geom.remove_unreferenced_vertices()
    except Exception:
        pass
    return geom


def _cluster_decimate(vertices, faces, target_faces):
    """
    Pure-numpy vertex-clustering decimation. Always runs with only numpy
    installed (no fast-simplification / open3d backend required). Snaps verts
    onto an adaptive grid, welds per cell, drops collapsed faces. Used as the
    guaranteed fallback when simplify_quadric_decimation is unavailable.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) <= target_faces or len(v) == 0:
        return v, f

    mn = v.min(axis=0)
    mx = v.max(axis=0)
    span = np.maximum(mx - mn, 1e-9)

    ratio = float(target_faces) / float(len(f))
    res = max(2, int(round((len(v) * ratio) ** (1.0 / 3.0)) * 2))

    for _ in range(8):
        cell = np.floor((v - mn) / span * res).astype(np.int64)
        cell = np.clip(cell, 0, res - 1)
        key = (cell[:, 0] * res + cell[:, 1]) * res + cell[:, 2]
        uniq, inv = np.unique(key, return_inverse=True)
        inv = inv.reshape(-1)  # numpy 2.0 shape-safety

        counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
        new_v = np.zeros((len(uniq), 3), dtype=np.float64)
        for axis in range(3):
            new_v[:, axis] = np.bincount(inv, weights=v[:, axis], minlength=len(uniq)) / counts

        new_f = inv[f]
        good = (
            (new_f[:, 0] != new_f[:, 1])
            & (new_f[:, 1] != new_f[:, 2])
            & (new_f[:, 0] != new_f[:, 2])
        )
        new_f = new_f[good]

        if len(new_f) <= target_faces or res <= 2:
            return new_v, new_f
        res = max(2, int(res * 0.75))

    return new_v, new_f


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Hunyuan3DPartGenerator(BaseGenerator):
    MODEL_ID     = "hunyuan3d-part"
    DISPLAY_NAME = "Hunyuan3D-Part"
    VRAM_GB      = 12

    # ------------------------------------------------------------------
    # Download check
    # ------------------------------------------------------------------

    def is_downloaded(self) -> bool:
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return (self.model_dir / "p3sam" / "p3sam.safetensors").exists()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _ensure_xpart_on_path(self):
        repo_dir  = Path(__file__).parent / "Hunyuan3D-Part"
        xpart_dir = repo_dir / "XPart"
        if not repo_dir.exists():
            raise RuntimeError(
                "%s Hunyuan3D-Part repo not found at %s.\n"
                "Please reinstall or repair the extension." % (_LOG, repo_dir)
            )
        if not xpart_dir.exists():
            raise RuntimeError(
                "%s XPart/ directory not found inside %s.\n"
                "The cloned repo may be incomplete — please reinstall." % (_LOG, repo_dir)
            )
        if str(xpart_dir) not in sys.path:
            sys.path.insert(0, str(xpart_dir))

        # auto_mask_api.py does `sys.path.append("../P3-SAM"); from model import ...`,
        # but that path is relative to the CWD and doesn't resolve in Modly's
        # subprocess. Put the P3-SAM folder on sys.path explicitly so its
        # model.py imports as top-level `model`. (model.py then bootstraps its
        # own deps via a __file__-relative path, so nothing else is needed.)
        p3sam_dir = repo_dir / "P3-SAM"
        if p3sam_dir.exists() and str(p3sam_dir) not in sys.path:
            sys.path.insert(0, str(p3sam_dir))

    # ------------------------------------------------------------------
    # SDP patch — applied once globally per process
    # ------------------------------------------------------------------

    def _apply_sdp_patch(self):
        global _SDP_PATCHED
        if _SDP_PATCHED:
            return
        import torch
        _orig = torch.backends.cuda.sdp_kernel

        def _fast_sdp(*args, **kwargs):
            # Keep flash off (no Blackwell-Windows kernel), enable mem_efficient
            # (O(N) memory — fast), keep math as safe fallback.
            return _orig(enable_flash=False, enable_math=True, enable_mem_efficient=True)

        torch.backends.cuda.sdp_kernel = _fast_sdp
        _SDP_PATCHED = True
        print("%s SDP kernel patched: mem_efficient + math, flash off." % _LOG)

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self):
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_xpart_on_path()

        import torch
        from partgen.partformer_pipeline import PartFormerPipeline

        self._device             = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype              = torch.float16 if self._device == "cuda" else torch.float32
        self._PartFormerPipeline = PartFormerPipeline
        self._pipeline           = None
        self._current_max_parts  = 3

        self._apply_sdp_patch()

        self._model = True
        print("%s Ready on %s (%s)." % (_LOG, self._device, str(self._dtype).replace("torch.", "")))

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        # Modly already points model_dir at the per-node bundle
        # (…/models/hunyuan3d-part/decompose-mesh) — the same folder the HF repo
        # downloaded into (model/, conditioner/, shapevae/, scheduler/, p3sam/).
        # Do NOT append "decompose-mesh" again.
        bundle_root = self.model_dir
        if not (bundle_root / "p3sam" / "p3sam.safetensors").exists():
            raise RuntimeError(
                "%s Weights not found under %s.\n"
                "Expected subfolders: model/, conditioner/, shapevae/, scheduler/, p3sam/.\n"
                "Run the extension's setup again to re-download." % (_LOG, bundle_root)
            )

        print("%s Loading PartFormerPipeline from %s ..." % (_LOG, bundle_root))
        import torch
        device = torch.device(self._device)

        # Sonata needs runtime fixes to load here (flash off on both import
        # copies, plus a hardcoded download path). Apply before the conditioner
        # and bbox predictor are built.
        self._patch_sonata()

        self._pipeline = self._PartFormerPipeline.from_pretrained(
            str(bundle_root),
            device=device,
            dtype=self._dtype,
        )
        self._pipeline.to(device=device, dtype=self._dtype)

        self._patch_encoder_pc_sizes()
        self._patch_bbox_predictor()

        print("%s PartFormerPipeline loaded." % _LOG)

    def _patch_sonata(self):
        """
        Make the Sonata point encoder load in this environment. Patches three
        upstream issues on every loaded copy of the module:

        1. enable_flash. Sonata configs set enable_flash=True, which asserts
           flash_attn is installed — but flash-attn has no Windows wheel. We
           force enable_flash=False (standard attention, identical weights).
        2. Dual import. Sonata is imported under two module names depending on
           the caller — partgen.models.sonata.model (the conditioner) and
           models.sonata.model (P3-SAM's build_P3SAM, via its own sys.path).
           Python keeps those as separate class objects, so we patch both.
        3. download_root. P3-SAM calls sonata.load(download_root='/root/sonata'),
           a hardcoded Linux path. We redirect it to the standard HF cache.
        """
        import importlib

        # P3-SAM adds XPart/partgen to sys.path and does `from models import
        # sonata`; make that alias importable now so both copies exist before
        # anything instantiates them.
        partgen_dir = Path(__file__).parent / "Hunyuan3D-Part" / "XPart" / "partgen"
        if partgen_dir.exists() and str(partgen_dir) not in sys.path:
            sys.path.insert(0, str(partgen_dir))

        for modname in ("partgen.models.sonata", "models.sonata",
                        "partgen.models.sonata.model", "models.sonata.model"):
            try:
                importlib.import_module(modname)
            except Exception:
                pass

        patched_flash = False
        for name, mod in list(sys.modules.items()):
            if mod is None:
                continue

            # 1 + 2: force enable_flash=False on every PointTransformerV3 copy
            if name.endswith("sonata.model"):
                cls = getattr(mod, "PointTransformerV3", None)
                if cls is not None and not getattr(cls, "_modly_noflash", False):
                    _orig_init = cls.__init__

                    def _init_noflash(inner_self, *a, __orig=_orig_init, **kw):
                        kw["enable_flash"] = False
                        return __orig(inner_self, *a, **kw)

                    cls.__init__ = _init_noflash
                    cls._modly_noflash = True
                    patched_flash = True

            # 3: redirect the hardcoded '/root/sonata' download path
            if name.endswith(".sonata"):
                load_fn = getattr(mod, "load", None)
                if callable(load_fn) and not getattr(load_fn, "_modly_pathfix", False):

                    def _load_safe(*a, __load=load_fn, **kw):
                        dr = kw.get("download_root")
                        if dr and str(dr).replace("\\", "/").startswith("/root"):
                            kw["download_root"] = None  # -> ~/.cache/sonata/ckpt
                        return __load(*a, **kw)

                    _load_safe._modly_pathfix = True
                    mod.load = _load_safe

        if patched_flash:
            print("%s Sonata patched (flash off on all copies, download path fixed)." % _LOG)
        else:
            print("%s Could not patch Sonata — flash_attn may still be required." % _LOG)

    def _patch_encoder_pc_sizes(self):
        """Cap point-cloud sizes in the conditioner encoders for 12 GB VRAM."""
        for _, module in self._pipeline.conditioner.named_modules():
            if hasattr(module, "pc_size") and int(module.pc_size) > 1024:
                module.pc_size = 1024
            if hasattr(module, "pc_sharpedge_size"):
                module.pc_sharpedge_size = 0
        print("%s Encoder pc_size capped at 1024." % _LOG)

    def _patch_bbox_predictor(self):
        """
        Replace the slow SAM bbox predictor with a decimation-first version.

        The upstream SAM processes full-resolution meshes (sometimes 1M+ faces)
        one viewpoint at a time. On a 12 GB GPU with shared-memory overflow this
        takes hours.  Instead we decimate to a manageable face count first, run
        SAM on the lighter mesh, then hand the resulting semantic AABB back to
        XPart which uses the original full-resolution mesh for generation.

        Falls back to a fast geometric axis-split AABB if SAM fails for any
        reason, so the pipeline always completes.
        """
        _orig      = self._pipeline.predict_bbox
        _generator = self  # closure

        def _fast_predict_bbox(mesh, seed=None, **kwargs):
            import trimesh

            target = int(getattr(_generator, "_current_bbox_decimate", 100000))
            work   = mesh

            if target > 0 and len(mesh.faces) > target:
                print("%s Decimating %d → %d faces for bbox prediction..."
                      % (_LOG, len(mesh.faces), target))
                try:
                    work = mesh.simplify_quadric_decimation(target)
                    print("%s Decimation done (%d faces)." % (_LOG, len(work.faces)))
                except Exception as e:
                    # Quadric decimation needs fast-simplification/open3d, which
                    # may not be installed. Fall back to the pure-numpy clustering
                    # decimator so SAM never has to run on the full dense mesh
                    # (running SAM on 500k+ faces can exhaust memory and kill the
                    # subprocess outright).
                    print("%s Quadric decimation unavailable (%s) — using numpy clustering fallback."
                          % (_LOG, e))
                    try:
                        nv, nf = _cluster_decimate(mesh.vertices, mesh.faces, target)
                        work = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
                        print("%s Clustering decimation done (%d faces)." % (_LOG, len(work.faces)))
                    except Exception as e2:
                        print("%s Clustering fallback also failed (%s) — using original mesh."
                              % (_LOG, e2))
                        work = mesh

            try:
                result = _orig(work, seed=seed, **kwargs)
            except Exception as e:
                print("%s SAM bbox prediction failed (%s) — using geometric AABB." % (_LOG, e))
                result = _fast_geometric_aabb(mesh, getattr(_generator, "_current_max_parts", 3))

            n = getattr(_generator, "_current_max_parts", 3)
            if hasattr(result, "shape") and result.shape[0] > n:
                result = result[:n]
            return result

        self._pipeline.predict_bbox = _fast_predict_bbox
        print("%s BBox predictor patched (decimation + SAM + geometric fallback)." % _LOG)

    def unload(self):
        self._pipeline = None
        self._model    = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # generate() — entry point called by runner.py
    # ------------------------------------------------------------------

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        params   = params or {}
        tmp_path = None

        try:
            # Detect mesh bytes (GLB magic = "glTF")
            mesh_path = None
            if isinstance(image_bytes, (bytes, bytearray)) and image_bytes[:4] == _GLB_MAGIC:
                with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                    f.write(image_bytes)
                    tmp_path = f.name
                mesh_path = tmp_path

            # Fallback: mesh path in params
            if mesh_path is None:
                for key in ("mesh_path", "input_mesh_path", "primary_mesh_path", "file_path"):
                    v = params.get(key)
                    if v and os.path.isfile(str(v)):
                        mesh_path = str(v)
                        break

            if mesh_path is None:
                raise RuntimeError(
                    "No mesh input found. Connect a GLB mesh as the primary input "
                    "or set mesh_path in params."
                )

            return self._decompose(mesh_path, params, progress_cb, cancel_event)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Output sanitation / cap
    # ------------------------------------------------------------------

    def _sanitize_and_cap_scene(self, scene, max_total_faces):
        """
        Clean every part and cap total face count before export so the host
        renderer can load the GLB without exhausting the V8 heap. Marching-cubes
        parts at high octree resolutions can otherwise reach tens of millions of
        faces across max_parts geometries.
        """
        import trimesh

        geoms = dict(scene.geometry)
        if not geoms:
            return scene

        cleaned = {}
        for name, geom in geoms.items():
            if not hasattr(geom, "faces"):
                continue
            _weld_and_clean(geom)
            if len(geom.faces) > 0 and len(geom.vertices) > 0:
                cleaned[name] = geom

        if not cleaned:
            raise RuntimeError("All part geometries were empty after cleaning.")

        total = sum(len(g.faces) for g in cleaned.values())
        print("%s Output mesh: %d part(s), %d faces total." % (_LOG, len(cleaned), total))

        if total <= max_total_faces:
            out = trimesh.Scene()
            for name, geom in cleaned.items():
                out.add_geometry(geom, geom_name=name)
            return out

        print("%s Over budget (%d > %d) — decimating parts..." % (_LOG, total, max_total_faces))
        out = trimesh.Scene()
        for name, geom in cleaned.items():
            share  = max(1, int(max_total_faces * len(geom.faces) / total))
            target = min(len(geom.faces), share)
            done   = False

            if len(geom.faces) > target:
                try:
                    simplified = geom.simplify_quadric_decimation(target)
                    if (simplified is not None
                            and len(simplified.faces) > 0
                            and len(simplified.faces) <= len(geom.faces)):
                        geom = simplified
                        done = True
                except Exception as e:
                    print("%s Quadric decimation unavailable for %s (%s)." % (_LOG, name, e))

            if not done and len(geom.faces) > target:
                nv, nf = _cluster_decimate(geom.vertices, geom.faces, target)
                geom = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
                _weld_and_clean(geom)

            if len(geom.faces) > 0:
                out.add_geometry(geom, geom_name=name)

        new_total = sum(len(g.faces) for g in out.geometry.values())
        print("%s Decimated output: %d faces total." % (_LOG, new_total))
        return out

    # ------------------------------------------------------------------
    # Core decomposition
    # ------------------------------------------------------------------

    def _decompose(self, mesh_path, params, progress_cb=None, cancel_event=None):
        import torch

        steps   = _safe_int(params.get("num_inference_steps"), 6)
        octree  = _safe_int(params.get("octree_resolution"),   128)
        chunks  = _safe_int(params.get("num_chunks"),          8192)
        n_parts = _safe_int(params.get("max_parts"),           3)
        seed    = _safe_int(params.get("seed"),                42)
        decimate_target = _safe_int(params.get("bbox_decimate_faces"), 100000)

        print("%s params: steps=%d octree=%d chunks=%d max_parts=%d decimate=%d seed=%d"
              % (_LOG, steps, octree, chunks, n_parts, decimate_target, seed))

        # Store per-request config on self so the bbox closure can read it.
        self._current_max_parts      = n_parts
        self._current_bbox_decimate  = decimate_target

        self._report(progress_cb, 5, "Loading model...")
        self.load()
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 15, "Loading pipeline...")
        self._load_pipeline()
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 30, "Decomposing mesh...")
        stop_evt = threading.Event()
        t        = None
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 30, 95, "Decomposing mesh...", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            generator = torch.Generator(device=self._device).manual_seed(seed)
            with torch.inference_mode():
                result = self._pipeline(
                    mesh_path=str(mesh_path),
                    aabb=None,
                    num_inference_steps=steps,
                    octree_resolution=octree,
                    num_chunks=chunks,
                    enable_pbar=False,
                    seed=seed,
                    generator=generator,
                    output_type="trimesh",
                )
        finally:
            stop_evt.set()
            if t:
                t.join(timeout=1.0)

        self._check_cancelled(cancel_event)

        scene = result[0] if isinstance(result, (list, tuple)) else result

        if not hasattr(scene, "geometry") or not scene.geometry:
            raise RuntimeError("Pipeline returned an empty scene with no part geometries.")

        n_out = len(scene.geometry)
        print("%s Decomposition complete: %d part(s)." % (_LOG, n_out))

        out_max_faces = _safe_int(params.get("output_max_faces"), 750000)
        scene = self._sanitize_and_cap_scene(scene, out_max_faces)

        self._report(progress_cb, 97, "Exporting...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / (
            "%d_%s_parts.glb" % (int(time.time()), uuid.uuid4().hex[:8])
        )
        scene.export(str(out_path))
        print("%s Exported to %s" % (_LOG, out_path))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._report(progress_cb, 100, "Done")
        return str(out_path)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("%s Downloading weights from %s ..." % (_LOG, _HF_REPO_ID))
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=["*.md", "*.txt", "LICENSE", ".gitattributes"],
        )
        print("%s Weights downloaded." % _LOG)import io
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress


# Redirect our print calls to stderr so stdout stays clean for the runner protocol.
_print = print


def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)


# Make stdout/stderr safe for upstream code that emits CJK debug prints on Windows.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except (AttributeError, io.UnsupportedOperation):
    pass


_HF_REPO_ID  = "tencent/Hunyuan3D-Part"
_GLB_MAGIC   = b"glTF"
_LOG         = "[Hunyuan3DPartGenerator]"

# SDP patch is a one-time global; track at module level so it survives
# multiple generator instances in the same process.
_SDP_PATCHED = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(val, default):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _fast_geometric_aabb(mesh, n):
    """Axis-split fallback AABB — used if SAM prediction fails entirely."""
    mn, mx = mesh.bounds[0].copy(), mesh.bounds[1].copy()
    span   = mx - mn
    axis   = int(span.argmax())
    boxes  = []
    for i in range(n):
        lo, hi       = mn.copy(), mx.copy()
        lo[axis]     = mn[axis] + i       * span[axis] / n
        hi[axis]     = mn[axis] + (i + 1) * span[axis] / n
        boxes.append([lo, hi])
    return np.array(boxes)


def _weld_and_clean(geom):
    """Weld duplicate verts and strip NaN/inf/degenerate/duplicate faces."""
    try:
        geom.remove_infinite_values()
    except Exception:
        pass
    try:
        geom.merge_vertices()
    except Exception:
        pass
    try:
        geom.update_faces(geom.nondegenerate_faces())
        geom.update_faces(geom.unique_faces())
        geom.remove_unreferenced_vertices()
    except Exception:
        pass
    return geom


def _cluster_decimate(vertices, faces, target_faces):
    """
    Pure-numpy vertex-clustering decimation. Always runs with only numpy
    installed (no fast-simplification / open3d backend required). Snaps verts
    onto an adaptive grid, welds per cell, drops collapsed faces. Used as the
    guaranteed fallback when simplify_quadric_decimation is unavailable.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) <= target_faces or len(v) == 0:
        return v, f

    mn = v.min(axis=0)
    mx = v.max(axis=0)
    span = np.maximum(mx - mn, 1e-9)

    ratio = float(target_faces) / float(len(f))
    res = max(2, int(round((len(v) * ratio) ** (1.0 / 3.0)) * 2))

    for _ in range(8):
        cell = np.floor((v - mn) / span * res).astype(np.int64)
        cell = np.clip(cell, 0, res - 1)
        key = (cell[:, 0] * res + cell[:, 1]) * res + cell[:, 2]
        uniq, inv = np.unique(key, return_inverse=True)
        inv = inv.reshape(-1)  # numpy 2.0 shape-safety

        counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
        new_v = np.zeros((len(uniq), 3), dtype=np.float64)
        for axis in range(3):
            new_v[:, axis] = np.bincount(inv, weights=v[:, axis], minlength=len(uniq)) / counts

        new_f = inv[f]
        good = (
            (new_f[:, 0] != new_f[:, 1])
            & (new_f[:, 1] != new_f[:, 2])
            & (new_f[:, 0] != new_f[:, 2])
        )
        new_f = new_f[good]

        if len(new_f) <= target_faces or res <= 2:
            return new_v, new_f
        res = max(2, int(res * 0.75))

    return new_v, new_f


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Hunyuan3DPartGenerator(BaseGenerator):
    MODEL_ID     = "hunyuan3d-part"
    DISPLAY_NAME = "Hunyuan3D-Part"
    VRAM_GB      = 12

    # ------------------------------------------------------------------
    # Download check
    # ------------------------------------------------------------------

    def is_downloaded(self) -> bool:
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return (self.model_dir / "p3sam" / "p3sam.safetensors").exists()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _ensure_xpart_on_path(self):
        repo_dir  = Path(__file__).parent / "Hunyuan3D-Part"
        xpart_dir = repo_dir / "XPart"
        if not repo_dir.exists():
            raise RuntimeError(
                "%s Hunyuan3D-Part repo not found at %s.\n"
                "Please reinstall or repair the extension." % (_LOG, repo_dir)
            )
        if not xpart_dir.exists():
            raise RuntimeError(
                "%s XPart/ directory not found inside %s.\n"
                "The cloned repo may be incomplete — please reinstall." % (_LOG, repo_dir)
            )
        if str(xpart_dir) not in sys.path:
            sys.path.insert(0, str(xpart_dir))

        # auto_mask_api.py does `sys.path.append("../P3-SAM"); from model import ...`,
        # but that path is relative to the CWD and doesn't resolve in Modly's
        # subprocess. Put the P3-SAM folder on sys.path explicitly so its
        # model.py imports as top-level `model`. (model.py then bootstraps its
        # own deps via a __file__-relative path, so nothing else is needed.)
        p3sam_dir = repo_dir / "P3-SAM"
        if p3sam_dir.exists() and str(p3sam_dir) not in sys.path:
            sys.path.insert(0, str(p3sam_dir))

    # ------------------------------------------------------------------
    # SDP patch — applied once globally per process
    # ------------------------------------------------------------------

    def _apply_sdp_patch(self):
        global _SDP_PATCHED
        if _SDP_PATCHED:
            return
        import torch
        _orig = torch.backends.cuda.sdp_kernel

        def _fast_sdp(*args, **kwargs):
            # Keep flash off (no Blackwell-Windows kernel), enable mem_efficient
            # (O(N) memory — fast), keep math as safe fallback.
            return _orig(enable_flash=False, enable_math=True, enable_mem_efficient=True)

        torch.backends.cuda.sdp_kernel = _fast_sdp
        _SDP_PATCHED = True
        print("%s SDP kernel patched: mem_efficient + math, flash off." % _LOG)

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self):
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_xpart_on_path()

        import torch
        from partgen.partformer_pipeline import PartFormerPipeline

        self._device             = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype              = torch.float16 if self._device == "cuda" else torch.float32
        self._PartFormerPipeline = PartFormerPipeline
        self._pipeline           = None
        self._current_max_parts  = 3

        self._apply_sdp_patch()

        self._model = True
        print("%s Ready on %s (%s)." % (_LOG, self._device, str(self._dtype).replace("torch.", "")))

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        # Modly already points model_dir at the per-node bundle
        # (…/models/hunyuan3d-part/decompose-mesh) — the same folder the HF repo
        # downloaded into (model/, conditioner/, shapevae/, scheduler/, p3sam/).
        # Do NOT append "decompose-mesh" again.
        bundle_root = self.model_dir
        if not (bundle_root / "p3sam" / "p3sam.safetensors").exists():
            raise RuntimeError(
                "%s Weights not found under %s.\n"
                "Expected subfolders: model/, conditioner/, shapevae/, scheduler/, p3sam/.\n"
                "Run the extension's setup again to re-download." % (_LOG, bundle_root)
            )

        print("%s Loading PartFormerPipeline from %s ..." % (_LOG, bundle_root))
        import torch
        device = torch.device(self._device)

        # Sonata needs runtime fixes to load here (flash off on both import
        # copies, plus a hardcoded download path). Apply before the conditioner
        # and bbox predictor are built.
        self._patch_sonata()

        self._pipeline = self._PartFormerPipeline.from_pretrained(
            str(bundle_root),
            device=device,
            dtype=self._dtype,
        )
        self._pipeline.to(device=device, dtype=self._dtype)

        self._patch_encoder_pc_sizes()
        self._patch_bbox_predictor()

        print("%s PartFormerPipeline loaded." % _LOG)

    def _patch_sonata(self):
        """
        Make the Sonata point encoder load in this environment. Patches three
        upstream issues on every loaded copy of the module:

        1. enable_flash. Sonata configs set enable_flash=True, which asserts
           flash_attn is installed — but flash-attn has no Windows wheel. We
           force enable_flash=False (standard attention, identical weights).
        2. Dual import. Sonata is imported under two module names depending on
           the caller — partgen.models.sonata.model (the conditioner) and
           models.sonata.model (P3-SAM's build_P3SAM, via its own sys.path).
           Python keeps those as separate class objects, so we patch both.
        3. download_root. P3-SAM calls sonata.load(download_root='/root/sonata'),
           a hardcoded Linux path. We redirect it to the standard HF cache.
        """
        import importlib

        # P3-SAM adds XPart/partgen to sys.path and does `from models import
        # sonata`; make that alias importable now so both copies exist before
        # anything instantiates them.
        partgen_dir = Path(__file__).parent / "Hunyuan3D-Part" / "XPart" / "partgen"
        if partgen_dir.exists() and str(partgen_dir) not in sys.path:
            sys.path.insert(0, str(partgen_dir))

        for modname in ("partgen.models.sonata", "models.sonata",
                        "partgen.models.sonata.model", "models.sonata.model"):
            try:
                importlib.import_module(modname)
            except Exception:
                pass

        patched_flash = False
        for name, mod in list(sys.modules.items()):
            if mod is None:
                continue

            # 1 + 2: force enable_flash=False on every PointTransformerV3 copy
            if name.endswith("sonata.model"):
                cls = getattr(mod, "PointTransformerV3", None)
                if cls is not None and not getattr(cls, "_modly_noflash", False):
                    _orig_init = cls.__init__

                    def _init_noflash(inner_self, *a, __orig=_orig_init, **kw):
                        kw["enable_flash"] = False
                        return __orig(inner_self, *a, **kw)

                    cls.__init__ = _init_noflash
                    cls._modly_noflash = True
                    patched_flash = True

            # 3: redirect the hardcoded '/root/sonata' download path
            if name.endswith(".sonata"):
                load_fn = getattr(mod, "load", None)
                if callable(load_fn) and not getattr(load_fn, "_modly_pathfix", False):

                    def _load_safe(*a, __load=load_fn, **kw):
                        dr = kw.get("download_root")
                        if dr and str(dr).replace("\\", "/").startswith("/root"):
                            kw["download_root"] = None  # -> ~/.cache/sonata/ckpt
                        return __load(*a, **kw)

                    _load_safe._modly_pathfix = True
                    mod.load = _load_safe

        if patched_flash:
            print("%s Sonata patched (flash off on all copies, download path fixed)." % _LOG)
        else:
            print("%s Could not patch Sonata — flash_attn may still be required." % _LOG)

    def _patch_encoder_pc_sizes(self):
        """Cap point-cloud sizes in the conditioner encoders for 12 GB VRAM."""
        for _, module in self._pipeline.conditioner.named_modules():
            if hasattr(module, "pc_size") and int(module.pc_size) > 1024:
                module.pc_size = 1024
            if hasattr(module, "pc_sharpedge_size"):
                module.pc_sharpedge_size = 0
        print("%s Encoder pc_size capped at 1024." % _LOG)

    def _patch_bbox_predictor(self):
        """
        Replace the slow SAM bbox predictor with a decimation-first version.

        The upstream SAM processes full-resolution meshes (sometimes 1M+ faces)
        one viewpoint at a time. On a 12 GB GPU with shared-memory overflow this
        takes hours.  Instead we decimate to a manageable face count first, run
        SAM on the lighter mesh, then hand the resulting semantic AABB back to
        XPart which uses the original full-resolution mesh for generation.

        Falls back to a fast geometric axis-split AABB if SAM fails for any
        reason, so the pipeline always completes.
        """
        _orig      = self._pipeline.predict_bbox
        _generator = self  # closure

        def _fast_predict_bbox(mesh, seed=None, **kwargs):
            import trimesh

            target = int(getattr(_generator, "_current_bbox_decimate", 100000))
            work   = mesh

            if target > 0 and len(mesh.faces) > target:
                print("%s Decimating %d → %d faces for bbox prediction..."
                      % (_LOG, len(mesh.faces), target))
                try:
                    work = mesh.simplify_quadric_decimation(target)
                    print("%s Decimation done (%d faces)." % (_LOG, len(work.faces)))
                except Exception as e:
                    print("%s Decimation failed (%s) — using original mesh." % (_LOG, e))
                    work = mesh

            try:
                result = _orig(work, seed=seed, **kwargs)
            except Exception as e:
                print("%s SAM bbox prediction failed (%s) — using geometric AABB." % (_LOG, e))
                result = _fast_geometric_aabb(mesh, getattr(_generator, "_current_max_parts", 3))

            n = getattr(_generator, "_current_max_parts", 3)
            if hasattr(result, "shape") and result.shape[0] > n:
                result = result[:n]
            return result

        self._pipeline.predict_bbox = _fast_predict_bbox
        print("%s BBox predictor patched (decimation + SAM + geometric fallback)." % _LOG)

    def unload(self):
        self._pipeline = None
        self._model    = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # generate() — entry point called by runner.py
    # ------------------------------------------------------------------

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        params   = params or {}
        tmp_path = None

        try:
            # Detect mesh bytes (GLB magic = "glTF")
            mesh_path = None
            if isinstance(image_bytes, (bytes, bytearray)) and image_bytes[:4] == _GLB_MAGIC:
                with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                    f.write(image_bytes)
                    tmp_path = f.name
                mesh_path = tmp_path

            # Fallback: mesh path in params
            if mesh_path is None:
                for key in ("mesh_path", "input_mesh_path", "primary_mesh_path", "file_path"):
                    v = params.get(key)
                    if v and os.path.isfile(str(v)):
                        mesh_path = str(v)
                        break

            if mesh_path is None:
                raise RuntimeError(
                    "No mesh input found. Connect a GLB mesh as the primary input "
                    "or set mesh_path in params."
                )

            return self._decompose(mesh_path, params, progress_cb, cancel_event)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Output sanitation / cap
    # ------------------------------------------------------------------

    def _sanitize_and_cap_scene(self, scene, max_total_faces):
        """
        Clean every part and cap total face count before export so the host
        renderer can load the GLB without exhausting the V8 heap. Marching-cubes
        parts at high octree resolutions can otherwise reach tens of millions of
        faces across max_parts geometries.
        """
        import trimesh

        geoms = dict(scene.geometry)
        if not geoms:
            return scene

        cleaned = {}
        for name, geom in geoms.items():
            if not hasattr(geom, "faces"):
                continue
            _weld_and_clean(geom)
            if len(geom.faces) > 0 and len(geom.vertices) > 0:
                cleaned[name] = geom

        if not cleaned:
            raise RuntimeError("All part geometries were empty after cleaning.")

        total = sum(len(g.faces) for g in cleaned.values())
        print("%s Output mesh: %d part(s), %d faces total." % (_LOG, len(cleaned), total))

        if total <= max_total_faces:
            out = trimesh.Scene()
            for name, geom in cleaned.items():
                out.add_geometry(geom, geom_name=name)
            return out

        print("%s Over budget (%d > %d) — decimating parts..." % (_LOG, total, max_total_faces))
        out = trimesh.Scene()
        for name, geom in cleaned.items():
            share  = max(1, int(max_total_faces * len(geom.faces) / total))
            target = min(len(geom.faces), share)
            done   = False

            if len(geom.faces) > target:
                try:
                    simplified = geom.simplify_quadric_decimation(target)
                    if (simplified is not None
                            and len(simplified.faces) > 0
                            and len(simplified.faces) <= len(geom.faces)):
                        geom = simplified
                        done = True
                except Exception as e:
                    print("%s Quadric decimation unavailable for %s (%s)." % (_LOG, name, e))

            if not done and len(geom.faces) > target:
                nv, nf = _cluster_decimate(geom.vertices, geom.faces, target)
                geom = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
                _weld_and_clean(geom)

            if len(geom.faces) > 0:
                out.add_geometry(geom, geom_name=name)

        new_total = sum(len(g.faces) for g in out.geometry.values())
        print("%s Decimated output: %d faces total." % (_LOG, new_total))
        return out

    # ------------------------------------------------------------------
    # Core decomposition
    # ------------------------------------------------------------------

    def _decompose(self, mesh_path, params, progress_cb=None, cancel_event=None):
        import torch

        steps   = _safe_int(params.get("num_inference_steps"), 6)
        octree  = _safe_int(params.get("octree_resolution"),   128)
        chunks  = _safe_int(params.get("num_chunks"),          8192)
        n_parts = _safe_int(params.get("max_parts"),           3)
        seed    = _safe_int(params.get("seed"),                42)
        decimate_target = _safe_int(params.get("bbox_decimate_faces"), 100000)

        print("%s params: steps=%d octree=%d chunks=%d max_parts=%d decimate=%d seed=%d"
              % (_LOG, steps, octree, chunks, n_parts, decimate_target, seed))

        # Store per-request config on self so the bbox closure can read it.
        self._current_max_parts      = n_parts
        self._current_bbox_decimate  = decimate_target

        self._report(progress_cb, 5, "Loading model...")
        self.load()
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 15, "Loading pipeline...")
        self._load_pipeline()
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 30, "Decomposing mesh...")
        stop_evt = threading.Event()
        t        = None
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 30, 95, "Decomposing mesh...", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            generator = torch.Generator(device=self._device).manual_seed(seed)
            with torch.inference_mode():
                result = self._pipeline(
                    mesh_path=str(mesh_path),
                    aabb=None,
                    num_inference_steps=steps,
                    octree_resolution=octree,
                    num_chunks=chunks,
                    enable_pbar=False,
                    seed=seed,
                    generator=generator,
                    output_type="trimesh",
                )
        finally:
            stop_evt.set()
            if t:
                t.join(timeout=1.0)

        self._check_cancelled(cancel_event)

        scene = result[0] if isinstance(result, (list, tuple)) else result

        if not hasattr(scene, "geometry") or not scene.geometry:
            raise RuntimeError("Pipeline returned an empty scene with no part geometries.")

        n_out = len(scene.geometry)
        print("%s Decomposition complete: %d part(s)." % (_LOG, n_out))

        out_max_faces = _safe_int(params.get("output_max_faces"), 750000)
        scene = self._sanitize_and_cap_scene(scene, out_max_faces)

        self._report(progress_cb, 97, "Exporting...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / (
            "%d_%s_parts.glb" % (int(time.time()), uuid.uuid4().hex[:8])
        )
        scene.export(str(out_path))
        print("%s Exported to %s" % (_LOG, out_path))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._report(progress_cb, 100, "Done")
        return str(out_path)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("%s Downloading weights from %s ..." % (_LOG, _HF_REPO_ID))
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=["*.md", "*.txt", "LICENSE", ".gitattributes"],
        )
        print("%s Weights downloaded." % _LOG)
