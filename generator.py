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
        grid_stride   = _safe_int(params.get("grid_stride"), 2)
        rec_mode      = params.get("reconstruction_mode") or "joint"

        print("[VGGTSceneGenerator] images_dir=%s max_views=%s conf_pct=%.1f mode=%s post=%s "
              "voxel=%.4f poisson_depth=%s density_q=%.3f clean=%s max_planes=%s rec=%s"
              % (images_dir, max_views, conf_pct, output_mode, postprocess,
                 voxel_size, poisson_depth, density_q, clean_float, max_planes, rec_mode))

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

        # ---- Inference (joint or incremental) ----------------------
        world_pts     = None   # only set in joint mode; used by grid-mesh fallback
        inc_grid_mesh = None   # only set in incremental mode

        if rec_mode == "incremental":
            pts, cols, inc_grid_mesh = self._incremental_build(
                image_paths, conf_pct, grid_stride, output_mode,
                progress_cb, cancel_event,
            )
        else:
            self._report(progress_cb, 18, "Reconstructing scene (joint)...")
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

                depth_np = depth_map.squeeze(0).float().cpu().numpy()
                conf_np  = depth_conf.squeeze(0).float().cpu().numpy()
                extr_np  = extrinsic.squeeze(0).float().cpu().numpy()
                intr_np  = intrinsic.squeeze(0).float().cpu().numpy()
                imgs_np  = batch.squeeze(0).float().cpu().numpy()
            finally:
                stop_evt.set()
                if thread:
                    thread.join(timeout=1.0)

            self._check_cancelled(cancel_event)

            self._report(progress_cb, 74, "Building point cloud...")
            world_pts = unproject_depth_map_to_point_map(depth_np, extr_np, intr_np)
            world_pts = world_pts * np.array([1.0, -1.0, -1.0])
            pts  = world_pts.reshape(-1, 3)
            cols = np.transpose(imgs_np, (0, 2, 3, 1)).reshape(-1, 3)
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
            self._report(progress_cb, 82, "Meshing...")
            out_obj = self._poisson_mesh(pts, cols, voxel_size, poisson_depth, density_q, clean_float)
            if out_obj is None:
                self._report(progress_cb, 86, "Meshing (depth grid)...")
                if inc_grid_mesh is not None:
                    out_obj = inc_grid_mesh
                elif world_pts is not None:
                    out_obj = self._depth_grid_mesh(world_pts, imgs_np, conf_np, conf_pct, grid_stride)
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
    # Incremental reconstruction
    # ------------------------------------------------------------------

    def _incremental_build(self, image_paths, conf_pct, grid_stride, output_mode,
                           progress_cb, cancel_event):
        import numpy as np
        import trimesh
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        acc_pts, acc_cols = None, None
        do_grid = (output_mode != "pointcloud")
        n = len(image_paths)

        # Write each aligned mesh to a temp file immediately rather than keeping
        # them all in RAM — avoids both memory pile-up and concatenation colour
        # issues with large in-memory lists.
        tmp_dir = Path(tempfile.mkdtemp(prefix="vggt_inc_"))
        temp_paths = []

        for i, path in enumerate(image_paths):
            self._report(progress_cb, 18 + int(64 * i / n),
                         "Image %d/%d..." % (i + 1, n))
            self._check_cancelled(cancel_event)

            images = load_and_preprocess_images([str(path)]).to(self._device)
            with torch.no_grad():
                if self._device == "cuda":
                    autocast = torch.cuda.amp.autocast(dtype=self._dtype)
                else:
                    autocast = contextlib.nullcontext()
                with autocast:
                    batch  = images[None]  # [1, 1, 3, H, W]
                    tokens, ps_idx = self._model.aggregator(batch)
                pose_enc  = self._model.camera_head(tokens)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
                depth_map, depth_conf = self._model.depth_head(tokens, batch, ps_idx)

            depth_np = depth_map.squeeze(0).float().cpu().numpy()
            conf_np  = depth_conf.squeeze(0).float().cpu().numpy()
            extr_np  = extrinsic.squeeze(0).float().cpu().numpy()
            intr_np  = intrinsic.squeeze(0).float().cpu().numpy()
            imgs_np  = batch.squeeze(0).float().cpu().numpy()

            world_pts = unproject_depth_map_to_point_map(depth_np, extr_np, intr_np)
            world_pts = world_pts * np.array([1.0, -1.0, -1.0])

            pts_i  = world_pts.reshape(-1, 3)
            cols_i = np.transpose(imgs_np, (0, 2, 3, 1)).reshape(-1, 3)
            conf_i = conf_np.reshape(-1)

            finite = np.isfinite(pts_i).all(axis=1)
            pts_i, cols_i, conf_i = pts_i[finite], cols_i[finite], conf_i[finite]
            if conf_i.size and 0 < conf_pct < 100:
                keep = conf_i >= np.percentile(conf_i, conf_pct)
                pts_i, cols_i = pts_i[keep], cols_i[keep]
            cols_i = np.clip(cols_i, 0.0, 1.0)

            print("[VGGTSceneGenerator] Image %d/%d: %d points" % (i + 1, n, len(pts_i)))

            R = t = s = None

            if acc_pts is not None and len(pts_i) > 3:
                n_sub   = min(8000, len(pts_i), len(acc_pts))
                rng     = np.random.default_rng(i)
                src_idx = rng.choice(len(pts_i),   n_sub, replace=False)
                ref_idx = rng.choice(len(acc_pts), n_sub, replace=False)
                R, t, s = self._align_clouds(
                    pts_i[src_idx], cols_i[src_idx],
                    acc_pts[ref_idx], acc_cols[ref_idx],
                )
                if R is not None:
                    pts_i = s * (pts_i @ R.T) + t
                    print("[VGGTSceneGenerator] Image %d aligned (scale=%.4f)." % (i + 1, s))
                else:
                    print("[VGGTSceneGenerator] Image %d alignment failed — appending in local space." % (i + 1))

            # Build mesh in this image's local space then apply the same transform
            if do_grid:
                mesh_i = self._depth_grid_mesh(world_pts, imgs_np, conf_np, conf_pct, grid_stride)
                if mesh_i is not None:
                    if R is not None:
                        verts = np.asarray(mesh_i.vertices, dtype=np.float64)
                        mesh_i.vertices = s * (verts @ R.T) + t
                    tmp = tmp_dir / ("mesh_%04d.glb" % i)
                    try:
                        mesh_i.export(str(tmp))
                        temp_paths.append(tmp)
                        print("[VGGTSceneGenerator] Saved temp mesh %d: %d verts %d faces"
                              % (i, len(mesh_i.vertices), len(mesh_i.faces)))
                    except Exception as exc:
                        print("[VGGTSceneGenerator] Temp mesh save failed: %s" % exc)
                    del mesh_i

            if acc_pts is None:
                acc_pts, acc_cols = pts_i, cols_i
            else:
                acc_pts  = np.concatenate([acc_pts,  pts_i])
                acc_cols = np.concatenate([acc_cols, cols_i])
                if len(acc_pts) > 800000:
                    sel = np.random.default_rng(i).choice(len(acc_pts), 800000, replace=False)
                    acc_pts, acc_cols = acc_pts[sel], acc_cols[sel]

        # Load temp files fresh and concatenate — avoids any in-memory state issues
        combined_mesh = None
        if temp_paths:
            print("[VGGTSceneGenerator] Loading %d temp meshes for final combine..." % len(temp_paths))
            parts = []
            for p in temp_paths:
                try:
                    m = trimesh.load(str(p), process=False)
                    if isinstance(m, trimesh.scene.Scene):
                        geoms = list(m.geometry.values())
                        if geoms:
                            m = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
                    parts.append(m)
                except Exception as exc:
                    print("[VGGTSceneGenerator] Failed to reload temp mesh %s: %s" % (p.name, exc))
            if parts:
                combined_mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
                print("[VGGTSceneGenerator] Combined: %d verts, %d faces from %d meshes"
                      % (len(combined_mesh.vertices), len(combined_mesh.faces), len(parts)))

        pts  = acc_pts  if acc_pts  is not None else np.zeros((0, 3))
        cols = acc_cols if acc_cols is not None else np.zeros((0, 3))
        return pts, cols, combined_mesh

    def _align_clouds(self, src_pts, src_cols, ref_pts, ref_cols,
                      color_thresh=0.15, iters=1500):
        import numpy as np
        try:
            from scipy.spatial import KDTree
        except ImportError:
            print("[VGGTSceneGenerator] scipy missing — skipping alignment.")
            return None, None, None

        # Per-pixel brightness normalisation: divide each pixel's RGB by its own
        # mean brightness. A dark red and a bright red become the same normalised
        # colour. Per-channel std normalisation was the previous approach but made
        # all colours look similar (7999/8000 hit rate = pure noise).
        def brightness_norm(c):
            lum = c.mean(axis=1, keepdims=True) + 0.01
            return c / lum  # values in ~[0, 3]

        src_n = brightness_norm(src_cols)
        ref_n = brightness_norm(ref_cols)

        tree = KDTree(ref_n)
        dists, idxs = tree.query(src_n, k=1, workers=-1)
        mask = dists < color_thresh
        n_cand = int(mask.sum())
        print("[VGGTSceneGenerator]   colour candidates: %d / %d" % (n_cand, len(src_pts)))
        if n_cand < 6:
            return None, None, None

        cs = src_pts[mask]
        cr = ref_pts[idxs[mask]]
        if len(cs) > 2000:
            sel = np.random.default_rng(0).choice(len(cs), 2000, replace=False)
            cs, cr = cs[sel], cr[sel]

        diag   = float(np.linalg.norm(cr.max(0) - cr.min(0))) or 1.0
        thresh = max(diag * 0.05, 1e-4)
        n      = len(cs)
        best_n, best = 0, (None, None, None)
        rng = np.random.default_rng(7)

        for _ in range(iters):
            idx = rng.choice(n, 4, replace=False)
            R, t, s = self._umeyama(cs[idx], cr[idx])
            # VGGT produces metric depth, so single-image runs on the same scene
            # should need only a small scale correction. Anything outside 0.5-2.0
            # is a degenerate colour-noise solution and must be rejected.
            if R is None or not (0.5 < s < 2.0):
                continue
            res = np.linalg.norm(s * (cs @ R.T) + t - cr, axis=1)
            c = int((res < thresh).sum())
            if c > best_n:
                best_n, best = c, (R, t, s)

        R, t, s = best
        if R is None:
            print("[VGGTSceneGenerator]   RANSAC: no valid transform within scale bounds (0.5-2.0).")
            return None, None, None

        res = np.linalg.norm(s * (cs @ R.T) + t - cr, axis=1)
        inliers = res < thresh
        if inliers.sum() >= 4:
            Rf, tf, sf = self._umeyama(cs[inliers], cr[inliers])
            if Rf is not None and (0.3 < sf < 3.0):
                R, t, s = Rf, tf, sf

        print("[VGGTSceneGenerator]   RANSAC inliers: %d / %d  scale=%.4f"
              % (int(inliers.sum()), n, s))
        return R, t, s

    def _umeyama(self, src, dst):
        import numpy as np
        n = src.shape[0]
        mu_s, mu_d = src.mean(0), dst.mean(0)
        sc, dc = src - mu_s, dst - mu_d
        var_s = (sc ** 2).sum() / n
        if var_s < 1e-12:
            return None, None, None
        cov = dc.T @ sc / n
        U, S, Vt = np.linalg.svd(cov)
        d = np.linalg.det(U @ Vt)
        D = np.diag([1.0, 1.0, d])
        R = U @ D @ Vt
        s = float((S * D.diagonal()).sum() / var_s)
        t = mu_d - s * R @ mu_s
        return R, t, s

    # ------------------------------------------------------------------
    # Meshing
    # ------------------------------------------------------------------

    def _poisson_mesh(self, pts, cols, voxel_size, poisson_depth, density_q, clean_floaters):
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
    # Depth-grid meshing (open3d-free, always produces faces)
    # ------------------------------------------------------------------

    def _depth_grid_mesh(self, world_pts, imgs_np, conf_np, conf_pct, stride=2, cull_factor=8.0):
        import numpy as np
        import trimesh

        S, H, W = world_pts.shape[:3]
        cols_4d = np.transpose(imgs_np, (0, 2, 3, 1))  # [S, H, W, 3]

        all_verts, all_cols, all_faces = [], [], []
        vert_offset = 0

        for s in range(S):
            pts_s  = world_pts[s]   # [H, W, 3]
            col_s  = cols_4d[s]     # [H, W, 3]
            conf_s = conf_np[s]     # [H, W]

            ys = np.arange(0, H, stride)
            xs = np.arange(0, W, stride)
            h, w = len(ys), len(xs)

            pts_g  = pts_s[np.ix_(ys, xs)]   # [h, w, 3]
            col_g  = col_s[np.ix_(ys, xs)]   # [h, w, 3]
            conf_g = conf_s[np.ix_(ys, xs)]  # [h, w]

            valid = np.isfinite(pts_g).all(axis=2)
            if conf_pct > 0:
                valid &= conf_g >= np.percentile(conf_g, conf_pct)

            # Quad corner flat indices
            ig, jg = np.meshgrid(np.arange(h - 1), np.arange(w - 1), indexing='ij')
            ig, jg = ig.ravel(), jg.ravel()
            tl = ig * w + jg
            tr = ig * w + (jg + 1)
            bl = (ig + 1) * w + jg
            br = (ig + 1) * w + (jg + 1)

            vf = valid.ravel()
            ok = vf[tl] & vf[tr] & vf[bl] & vf[br]
            tl, tr, bl, br = tl[ok], tr[ok], bl[ok], br[ok]
            if len(tl) == 0:
                continue

            verts = pts_g.reshape(-1, 3)
            vcols = np.clip(col_g.reshape(-1, 3), 0.0, 1.0)

            # Cull quads that span a depth discontinuity
            if cull_factor > 0:
                p = verts
                e = np.maximum.reduce([
                    np.linalg.norm(p[tr] - p[tl], axis=1),
                    np.linalg.norm(p[bl] - p[tl], axis=1),
                    np.linalg.norm(p[br] - p[tr], axis=1),
                    np.linalg.norm(p[br] - p[bl], axis=1),
                ])
                med_e = float(np.median(e))
                keep = e < (cull_factor * med_e)
                tl, tr, bl, br = tl[keep], tr[keep], bl[keep], br[keep]
            if len(tl) == 0:
                continue

            faces = np.vstack([
                np.column_stack([tl, tr, bl]),
                np.column_stack([tr, br, bl]),
            ])

            used, inv = np.unique(faces, return_inverse=True)
            faces = inv.reshape(faces.shape)
            all_verts.append(verts[used])
            all_cols.append(vcols[used])
            all_faces.append(faces + vert_offset)
            vert_offset += len(used)

        if not all_verts:
            print("[VGGTSceneGenerator] depth-grid mesh: no faces produced.")
            return None

        v = np.concatenate(all_verts)
        c = (np.concatenate(all_cols) * 255).astype(np.uint8)
        f = np.concatenate(all_faces)
        print("[VGGTSceneGenerator] depth-grid mesh: %d verts, %d faces (stride=%d)" % (len(v), len(f), stride))
        return trimesh.Trimesh(vertices=v, faces=f, vertex_colors=c, process=False)

    # ------------------------------------------------------------------
    # Planar cleanup (RANSAC + Manhattan snap)
    # ------------------------------------------------------------------

    def _planar_mesh(self, pts, cols, manhattan, tol_frac, max_planes):
        import numpy as np
        try:
            import trimesh
        except Exception as exc:
            print("[VGGTSceneGenerator] planar cleanup needs trimesh (%s)." % exc)
            return None

        try:
            P = np.ascontiguousarray(pts, dtype=np.float64)
            C = np.clip(cols, 0.0, 1.0)

            diag = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0
            dist_thresh = max(diag * tol_frac, 1e-6)

            # RANSAC doesn't need every point; cap the working set for speed.
            rng = np.random.default_rng(0)
            if len(P) > 60000:
                sel = rng.choice(len(P), 60000, replace=False)
                P, C = P[sel], C[sel]

            min_pts = max(int(0.01 * len(P)), 100)
            idx_all = np.arange(len(P))
            alive   = np.ones(len(P), dtype=bool)

            planes = []
            for _ in range(max(1, max_planes)):
                live_idx = idx_all[alive]
                if live_idx.size < min_pts:
                    break
                model, inlier_local = self._ransac_plane(P[live_idx], dist_thresh, rng)
                if inlier_local is None or int(inlier_local.sum()) < min_pts:
                    break
                inlier_idx = live_idx[inlier_local]
                normal, _d = model
                planes.append({
                    "normal": normal,
                    "points": P[inlier_idx],
                    "color":  C[inlier_idx].mean(axis=0),
                    "count":  int(inlier_idx.size),
                })
                alive[inlier_idx] = False

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

    def _ransac_plane(self, P, thresh, rng, iters=500):
        import numpy as np
        n = len(P)
        if n < 3:
            return (None, None)
        best_count = 0
        best = (None, None)
        for _ in range(iters):
            i = rng.choice(n, 3, replace=False)
            p1, p2, p3 = P[i]
            normal = np.cross(p2 - p1, p3 - p1)
            ln = float(np.linalg.norm(normal))
            if ln < 1e-9:
                continue
            normal = normal / ln
            d = -float(np.dot(normal, p1))
            mask = np.abs(P @ normal + d) < thresh
            c = int(mask.sum())
            if c > best_count:
                best_count, best = c, ((normal, d), mask)

        model, mask = best
        if mask is not None and int(mask.sum()) >= 3:
            Q = P[mask]
            centroid = Q.mean(axis=0)
            _, _, vh = np.linalg.svd(Q - centroid, full_matrices=False)
            normal = vh[-1] / (np.linalg.norm(vh[-1]) + 1e-12)
            d = -float(np.dot(normal, centroid))
            mask = np.abs(P @ normal + d) < thresh
            model = (normal, d)
        return (model, mask)

    def _manhattan_rotation(self, planes):
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
            amin, amax = np.percentile(pts[:, o[0]], [2, 98])
            bmin, bmax = np.percentile(pts[:, o[1]], [2, 98])
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
            umin, umax = np.percentile(cu, [2, 98])
            vmin, vmax = np.percentile(cv, [2, 98])
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
    # Open3D-free dense meshing
    # ------------------------------------------------------------------

    def _depth_grid_mesh(self, world_pts, imgs_np, conf_np, conf_pct, cull_factor=8.0):
        import numpy as np
        try:
            import trimesh
        except Exception as exc:
            print("[VGGTSceneGenerator] grid mesh needs trimesh (%s)." % exc)
            return None

        try:
            S, H, W, _ = world_pts.shape
            imgs = np.transpose(imgs_np, (0, 2, 3, 1))            # [S, H, W, 3]
            finite_all = np.isfinite(world_pts).all(axis=-1)      # [S, H, W]
            Psafe = np.where(np.isfinite(world_pts), world_pts, 0.0)

            cflat = conf_np.reshape(-1)
            if cflat.size and 0 < conf_pct < 100:
                thr_conf = np.percentile(cflat, conf_pct)
            else:
                thr_conf = -np.inf

            idx = np.arange(H * W).reshape(H, W)
            tl = idx[:-1, :-1].ravel(); tr = idx[:-1, 1:].ravel()
            bl = idx[1:, :-1].ravel();  br = idx[1:, 1:].ravel()

            parts = []
            for s in range(S):
                P = Psafe[s].reshape(-1, 3)
                C = imgs[s].reshape(-1, 3)
                V = finite_all[s].reshape(-1) & (conf_np[s].reshape(-1) >= thr_conf)

                quad_ok = V[tl] & V[tr] & V[bl] & V[br]
                if not quad_ok.any():
                    continue

                def elen(a, b):
                    return np.linalg.norm(P[a] - P[b], axis=1)
                e = np.maximum.reduce([elen(tl, bl), elen(tl, tr), elen(tr, br),
                                       elen(bl, br), elen(tl, br), elen(tr, bl)])
                typ = float(np.median(elen(tl, bl)[quad_ok]))
                if not np.isfinite(typ) or typ <= 0:
                    typ = float(np.median(e[quad_ok])) or 1.0
                ok = quad_ok & (e < cull_factor * typ)
                if not ok.any():
                    continue

                f = np.vstack([
                    np.stack([tl[ok], bl[ok], tr[ok]], axis=1),
                    np.stack([tr[ok], bl[ok], br[ok]], axis=1),
                ])
                used = np.unique(f)
                remap = np.full(H * W, -1, dtype=np.int64)
                remap[used] = np.arange(used.size)
                parts.append(trimesh.Trimesh(
                    vertices=P[used],
                    faces=remap[f],
                    vertex_colors=(np.clip(C[used], 0.0, 1.0) * 255).astype("uint8"),
                    process=False,
                ))

            if not parts:
                print("[VGGTSceneGenerator] grid mesh produced no faces — exporting points.")
                return None
            mesh = trimesh.util.concatenate(parts)
            print("[VGGTSceneGenerator] grid mesh: %d verts, %d faces."
                  % (len(mesh.vertices), len(mesh.faces)))
            return mesh
        except Exception as exc:
            import traceback
            print("[VGGTSceneGenerator] grid meshing failed (%s) — exporting points." % exc)
            traceback.print_exc()
            return None

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
