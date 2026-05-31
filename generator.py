"""
generator.py — VGGT scene/room reconstruction for Modly.

Feed-forward multi-view geometry (Meta's VGGT-1B): takes a set of overlapping
photos of a scene and reconstructs the observed geometry — flat walls, real
corners — then meshes it. Unlike object generators, this is built for scenes:
point a folder of overlapping room photos at it via the "images_dir" param and
do NOT remove backgrounds (the walls/floor ARE the subject).
"""
import contextlib
import io
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress


# Route stray print() to stderr so any stdout JSON protocol stays clean.
_print = print

def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)


_HF_REPO_ID = "facebook/VGGT-1B"
_GLB_MAGIC  = b"glTF"
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _safe_int(val, default):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _safe_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _strip_quotes(value):
    if not isinstance(value, str):
        return ""
    return value.strip().strip(chr(34) + chr(39))


class VGGTSceneGenerator(BaseGenerator):
    MODEL_ID     = "vggt_scene"
    DISPLAY_NAME = "VGGT Scene Reconstruction"
    VRAM_GB      = 8

    # ------------------------------------------------------------------
    # Download checks
    # ------------------------------------------------------------------

    def is_downloaded(self) -> bool:
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return (self.model_dir / "model.pt").exists()

    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download

        repo_id = self.hf_repo or _HF_REPO_ID
        ignore  = ["*.md", "*.txt", "LICENSE", "NOTICE", ".gitattributes"]
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[VGGTSceneGenerator] Downloading VGGT weights from %s ..." % repo_id)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(self.model_dir),
            ignore_patterns=ignore,
        )
        print("[VGGTSceneGenerator] Weights downloaded.")

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def _ensure_vggt_on_path(self):
        # setup.py installs the package editable, but add the cloned repo to
        # sys.path as a fallback so `import vggt` resolves regardless.
        repo_dir = Path(__file__).parent / "vggt"
        if repo_dir.exists() and str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

    def load(self):
        if getattr(self, "_model", None) is not None:
            return

        import torch

        self._ensure_vggt_on_path()
        from vggt.models.vggt import VGGT

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float16

        print("[VGGTSceneGenerator] Loading VGGT on %s (%s) ..." % (self._device, self._dtype))
        try:
            model = VGGT.from_pretrained(str(self.model_dir))
        except Exception as exc:
            print("[VGGTSceneGenerator] from_pretrained failed (%s); loading model.pt directly." % exc)
            model = VGGT()
            state = torch.load(str(self.model_dir / "model.pt"), map_location="cpu")
            model.load_state_dict(state)

        model.eval()
        self._model = model.to(self._device)
        print("[VGGTSceneGenerator] Model ready.")

    def unload(self):
        import torch
        if getattr(self, "_model", None) is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        import numpy as np
        import torch
        import trimesh

        params        = params or {}
        images_dir    = _strip_quotes(params.get("images_dir"))
        max_views     = _safe_int(params.get("max_views"), 24)
        conf_pct      = _safe_float(params.get("conf_percentile"), 50.0)
        output_mode   = params.get("output_mode") or "mesh"
        voxel_size    = _safe_float(params.get("voxel_size"), 0.01)
        poisson_depth = _safe_int(params.get("poisson_depth"), 9)
        density_q     = _safe_float(params.get("density_quantile"), 0.02)
        clean_float   = (params.get("clean_floaters") or "true") == "true"
        postprocess   = params.get("postprocess") or "none"
        plane_tol     = _safe_float(params.get("plane_tolerance"), 0.012)
        max_planes    = _safe_int(params.get("max_planes"), 10)

        print("[VGGTSceneGenerator] images_dir=%s max_views=%s conf_pct=%.1f mode=%s post=%s "
              "voxel=%.4f poisson_depth=%s density_q=%.3f clean=%s max_planes=%s"
              % (images_dir, max_views, conf_pct, output_mode, postprocess,
                 voxel_size, poisson_depth, density_q, clean_float, max_planes))

        # ---- Collect input images -----------------------------------
        self._report(progress_cb, 4, "Collecting images...")
        work = Path(tempfile.mkdtemp(prefix="vggt_"))
        image_paths = self._collect_images(image_bytes, images_dir, work)

        if len(image_paths) == 0:
            raise RuntimeError("No input images found. Wire a photo into the node or set 'images_dir'.")
        if max_views > 0 and len(image_paths) > max_views:
            image_paths = self._evenly_sample(image_paths, max_views)
        if len(image_paths) < 2:
            print("[VGGTSceneGenerator] WARNING: only 1 view — room geometry needs several "
                  "overlapping photos to reconstruct accurately.")
        print("[VGGTSceneGenerator] Using %d views." % len(image_paths))
        self._check_cancelled(cancel_event)

        # ---- Load model ---------------------------------------------
        self._report(progress_cb, 10, "Loading model...")
        self.load()
        self._check_cancelled(cancel_event)

        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        # ---- Inference ----------------------------------------------
        self._report(progress_cb, 18, "Reconstructing scene...")
        stop_evt = threading.Event()
        thread   = None
        if progress_cb:
            thread = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 18, 70, "Reconstructing scene...", stop_evt),
                daemon=True,
            )
            thread.start()

        try:
            images = load_and_preprocess_images([str(p) for p in image_paths]).to(self._device)
            with torch.no_grad():
                if self._device == "cuda":
                    autocast = torch.cuda.amp.autocast(dtype=self._dtype)
                else:
                    autocast = contextlib.nullcontext()
                with autocast:
                    batch = images[None]  # [1, S, 3, H, W]
                    tokens, ps_idx = self._model.aggregator(batch)
                pose_enc = self._model.camera_head(tokens)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
                depth_map, depth_conf = self._model.depth_head(tokens, batch, ps_idx)

            depth_np = depth_map.squeeze(0).float().cpu().numpy()   # [S, H, W, 1]
            conf_np  = depth_conf.squeeze(0).float().cpu().numpy()  # [S, H, W]
            extr_np  = extrinsic.squeeze(0).float().cpu().numpy()   # [S, 3, 4]
            intr_np  = intrinsic.squeeze(0).float().cpu().numpy()   # [S, 3, 3]
            imgs_np  = batch.squeeze(0).float().cpu().numpy()       # [S, 3, H, W]
        finally:
            stop_evt.set()
            if thread:
                thread.join(timeout=1.0)

        self._check_cancelled(cancel_event)

        # ---- Unproject depth -> world point cloud -------------------
        self._report(progress_cb, 74, "Building point cloud...")
        world_pts = unproject_depth_map_to_point_map(depth_np, extr_np, intr_np)  # [S, H, W, 3]
        pts  = world_pts.reshape(-1, 3)
        cols = np.transpose(imgs_np, (0, 2, 3, 1)).reshape(-1, 3)                 # [N, 3] in 0..1
        conf = conf_np.reshape(-1)

        finite = np.isfinite(pts).all(axis=1)
        pts, cols, conf = pts[finite], cols[finite], conf[finite]

        if conf.size and 0 < conf_pct < 100:
            keep = conf >= np.percentile(conf, conf_pct)
            pts, cols = pts[keep], cols[keep]

        cols = np.clip(cols, 0.0, 1.0)
        if pts.shape[0] == 0:
            raise RuntimeError("All points were filtered out — lower the Confidence Filter.")
        print("[VGGTSceneGenerator] %d points after filtering." % pts.shape[0])

        # ---- Mesh (or export raw point cloud) -----------------------
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        stamp = "%d_%s" % (int(time.time()), uuid.uuid4().hex[:8])

        out_obj  = None
        out_kind = "points"

        if postprocess in ("ransac_planes", "manhattan"):
            self._report(progress_cb, 82, "Fitting planes...")
            out_obj = self._planar_mesh(
                pts, cols,
                manhattan=(postprocess == "manhattan"),
                tol_frac=plane_tol,
                max_planes=max_planes,
            )
            if out_obj is not None:
                out_kind = "planar"
        elif output_mode != "pointcloud":
            self._report(progress_cb, 82, "Meshing (Poisson)...")
            out_obj = self._poisson_mesh(pts, cols, voxel_size, poisson_depth, density_q, clean_float)
            if out_obj is not None:
                out_kind = "scene"

        if out_obj is None:
            out_obj = trimesh.PointCloud(pts, (cols * 255).astype("uint8"))

        out_path = self.outputs_dir / ("%s_%s.glb" % (stamp, out_kind))

        self._report(progress_cb, 96, "Exporting...")
        out_obj.export(str(out_path))
        print("[VGGTSceneGenerator] Exported: %s" % out_path)

        self._report(progress_cb, 100, "Done")
        return str(out_path)

    # ------------------------------------------------------------------
    # Meshing
    # ------------------------------------------------------------------

    def _poisson_mesh(self, pts, cols, voxel_size, poisson_depth, density_q, clean_floaters):
        """
        Point cloud -> mesh via Open3D screened Poisson. Returns a
        trimesh.Trimesh, or None if Open3D is missing / meshing fails — the
        caller then exports the raw point cloud so a result still comes out.
        """
        import numpy as np
        try:
            import open3d as o3d
        except Exception as exc:
            print("[VGGTSceneGenerator] open3d unavailable (%s) — exporting point cloud." % exc)
            return None

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

            if voxel_size and voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size)

            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            pcd.orient_normals_consistent_tangent_plane(k=15)

            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth
            )

            densities = np.asarray(densities)
            if density_q and density_q > 0 and densities.size:
                mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_q))

            if clean_floaters:
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                tri_ids, counts, _ = mesh.cluster_connected_triangles()
                tri_ids = np.asarray(tri_ids)
                counts  = np.asarray(counts)
                if counts.size:
                    mesh.remove_triangles_by_mask(tri_ids != int(counts.argmax()))
                    mesh.remove_unreferenced_vertices()

            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            if verts.shape[0] == 0 or faces.shape[0] == 0:
                print("[VGGTSceneGenerator] Poisson produced an empty mesh — exporting point cloud.")
                return None

            vcols = None
            if mesh.has_vertex_colors():
                vcols = (np.asarray(mesh.vertex_colors) * 255.0).astype("uint8")

            import trimesh
            return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vcols, process=False)
        except Exception as exc:
            import traceback
            print("[VGGTSceneGenerator] Meshing failed (%s) — exporting point cloud." % exc)
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Planar cleanup (RANSAC + Manhattan snap)
    # ------------------------------------------------------------------

    def _planar_mesh(self, pts, cols, manhattan, tol_frac, max_planes):
        """
        Fit dominant flat planes to the cloud via iterative RANSAC and rebuild
        each as a flat coloured quad. With manhattan=True the planes are first
        rotated onto orthogonal axes and snapped, so walls/floor/ceiling meet at
        true 90 degrees. Returns a trimesh.Trimesh, or None (caller then exports
        the raw point cloud so a result still comes out).
        """
        import numpy as np
        try:
            import open3d as o3d
            import trimesh
        except Exception as exc:
            print("[VGGTSceneGenerator] planar cleanup needs open3d (%s) — exporting points." % exc)
            return None

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

            extent = pcd.get_axis_aligned_bounding_box().get_extent()
            diag = float(np.linalg.norm(extent)) or 1.0
            dist_thresh = max(diag * tol_frac, 1e-6)

            # RANSAC on millions of points is slow; thin large clouds first.
            if len(pcd.points) > 300000:
                pcd = pcd.voxel_down_sample(max(diag * 0.0025, 1e-6))

            total   = len(pcd.points)
            min_pts = max(int(0.005 * total), 150)

            remaining = pcd
            planes = []
            for _ in range(max(1, max_planes)):
                if len(remaining.points) < min_pts:
                    break
                model, inliers = remaining.segment_plane(
                    distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000)
                if len(inliers) < min_pts:
                    break
                inlier = remaining.select_by_index(inliers)
                normal = np.array(model[:3], dtype=np.float64)
                norm = np.linalg.norm(normal)
                if norm < 1e-9:
                    break
                pcolors = np.asarray(inlier.colors)
                planes.append({
                    "normal": normal / norm,
                    "points": np.asarray(inlier.points),
                    "color":  pcolors.mean(axis=0) if pcolors.size else np.array([0.72, 0.72, 0.72]),
                    "count":  len(inliers),
                })
                remaining = remaining.select_by_index(inliers, invert=True)

            if not planes:
                print("[VGGTSceneGenerator] No planes found — exporting points.")
                return None
            print("[VGGTSceneGenerator] Fitted %d planes (%s)."
                  % (len(planes), "manhattan" if manhattan else "raw"))

            if manhattan:
                R = self._manhattan_rotation(planes)
                for p in planes:
                    p["points"] = p["points"] @ R.T
                    n = R @ p["normal"]
                    axis = int(np.argmax(np.abs(n)))
                    p["axis"]   = axis
                    p["offset"] = float(np.mean(p["points"][:, axis]))

            quads = [self._plane_quad(p, manhattan) for p in planes]
            quads = [q for q in quads if q is not None]
            if not quads:
                return None
            return trimesh.util.concatenate(quads)
        except Exception as exc:
            import traceback
            print("[VGGTSceneGenerator] Planar cleanup failed (%s) — exporting points." % exc)
            traceback.print_exc()
            return None

    def _manhattan_rotation(self, planes):
        """
        Orthonormal rotation that aligns the dominant plane normals to the world
        axes: the largest plane sets the first axis, the most-perpendicular
        normal sets the second, their cross gives the third. Not gravity-aware —
        the room may sit rotated relative to world up, which is fine for corners.
        """
        import numpy as np
        order = sorted(planes, key=lambda p: p["count"], reverse=True)
        n1 = order[0]["normal"] / (np.linalg.norm(order[0]["normal"]) + 1e-12)

        n2 = None
        best = 1.0
        for p in order[1:]:
            n = p["normal"] / (np.linalg.norm(p["normal"]) + 1e-12)
            dot = abs(float(np.dot(n, n1)))
            if dot < best:
                best, n2 = dot, n
        if n2 is None:
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(ref, n1))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            n2 = ref

        n2 = n2 - np.dot(n2, n1) * n1
        n2 /= (np.linalg.norm(n2) + 1e-12)
        n3 = np.cross(n1, n2)
        n3 /= (np.linalg.norm(n3) + 1e-12)

        R = np.vstack([n1, n2, n3])
        if np.linalg.det(R) < 0:
            R[2] = -R[2]
        return R

    def _plane_quad(self, plane, manhattan):
        import numpy as np
        import trimesh
        pts = plane["points"]
        if len(pts) < 3:
            return None
        color = np.clip(plane["color"], 0.0, 1.0)
        vcol = np.tile((color * 255).astype("uint8"), (4, 1))

        if manhattan:
            axis = plane["axis"]
            val  = plane["offset"]
            o = [i for i in range(3) if i != axis]
            amin, amax = float(pts[:, o[0]].min()), float(pts[:, o[0]].max())
            bmin, bmax = float(pts[:, o[1]].min()), float(pts[:, o[1]].max())
            combos = [(amin, bmin), (amax, bmin), (amax, bmax), (amin, bmax)]
            corners = np.zeros((4, 3))
            for i, (ca, cb) in enumerate(combos):
                corners[i, axis] = val
                corners[i, o[0]] = ca
                corners[i, o[1]] = cb
        else:
            normal = plane["normal"] / (np.linalg.norm(plane["normal"]) + 1e-12)
            centroid = pts.mean(axis=0)
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(ref, normal))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            u = np.cross(normal, ref); u /= (np.linalg.norm(u) + 1e-12)
            v = np.cross(normal, u);  v /= (np.linalg.norm(v) + 1e-12)
            rel = pts - centroid
            cu, cv = rel @ u, rel @ v
            umin, umax = float(cu.min()), float(cu.max())
            vmin, vmax = float(cv.min()), float(cv.max())
            corners = np.array([
                centroid + umin * u + vmin * v,
                centroid + umax * u + vmin * v,
                centroid + umax * u + vmax * v,
                centroid + umin * u + vmax * v,
            ])

        # Both windings so each surface is visible from inside the room too.
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])
        return trimesh.Trimesh(vertices=corners, faces=faces, vertex_colors=vcol, process=False)

    # ------------------------------------------------------------------
    # Image collection
    # ------------------------------------------------------------------

    def _collect_images(self, image_bytes, images_dir, work):
        paths = []
        if images_dir:
            folder = Path(images_dir)
            if folder.is_dir():
                for p in sorted(folder.iterdir()):
                    if p.suffix.lower() in _IMAGE_EXTS:
                        paths.append(p)
            else:
                print("[VGGTSceneGenerator] images_dir is not a folder: %s" % images_dir)

        # Fall back to the single wired image only if the folder gave nothing.
        if not paths and image_bytes:
            primary = work / "view_000.png"
            Image.open(io.BytesIO(image_bytes)).convert("RGB").save(str(primary), format="PNG")
            paths.append(primary)
        return paths

    def _evenly_sample(self, items, n):
        if n <= 0 or len(items) <= n:
            return items
        step = len(items) / float(n)
        return [items[int(i * step)] for i in range(n)]
