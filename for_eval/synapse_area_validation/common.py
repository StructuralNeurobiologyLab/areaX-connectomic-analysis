"""Shared constants and geometry helpers for the synaptic-area validation."""

import numpy as np

# j0251 mag1 voxel size in nm, and dataset bounds in mag1 voxels.
VOXEL_NM = np.array([10.0, 10.0, 25.0])
DATASET_SHAPE = np.array([27119, 27350, 15494])

API = "https://syconn.esc.mpcdf.mpg.de/j0251/synapses/json"
# Dataset version of the SyConn server (v6 cell types; CELLTYPES below is the matching ID table).
VERSION = "72_seg_20210127_agglo2_syn_20220811_celltypes_20230822"
RAW_LAYER = ("precomputed://https://syconn.esc.mpcdf.mpg.de/"
             "j0251_72_seg_20210127_agglo2/image")
SEG_LAYER = ("precomputed://https://syconn.esc.mpcdf.mpg.de/"
             "j0251_72_seg_20210127_agglo2/segmentation")

# NB: index 12 is MIGR.  Omitting it shifts every later index by one and
# silently relabels MSN as OLIGO -- verify against the guide before editing.
CELLTYPES = ['ASTRO', 'DA', 'FRAG', 'GPe', 'GPi', 'HVC', 'INT1', 'INT2',
             'INT3', 'LMAN', 'LTS', 'MICRO', 'MIGR', 'MSN', 'OLIGO', 'STN',
             'TAN', 'unclassified']
assert len(CELLTYPES) == 18 and CELLTYPES[13] == 'MSN'
MORPH = {0: 'spine_neck', 1: 'spine_head', 2: 'shaft', 3: 'soma',
         255: 'unknown'}

# The paper's analysis pool.
PROB_MIN = 0.6
SIZE_MIN = 0.1

# Orientation of a tracing stack: the axis we step along, and the two in-plane
# axes.  'xy' means slices normal to z, i.e. the native cutting plane.
ORIENTATIONS = {'xy': dict(normal=2, inplane=(0, 1)),
                'xz': dict(normal=1, inplane=(0, 2)),
                'yz': dict(normal=0, inplane=(1, 2))}


def polyline_length_nm(vertices, axes):
    """Length of a polyline in nm.

    `vertices` are voxel coordinates restricted to the two in-plane `axes`,
    which carry different nm/voxel: x,y are 10 nm but z is 25 nm, so segment
    lengths must be taken in nm, never in voxels.
    """
    v = np.asarray(vertices, dtype=float)
    if len(v) < 2:
        return 0.0
    v_nm = v * VOXEL_NM[list(axes)]
    return float(np.sum(np.linalg.norm(np.diff(v_nm, axis=0), axis=1)))


def section_step(traces):
    """Spacing between traced sections, in sections.

    Hand-clicked lines jitter by a voxel or so.  At 10 nm section spacing
    (xz/yz) that jitter rivals the synapse's real drift between neighbouring
    sections, and because surface area is inflated by noise and never reduced
    by it, tracing every section overestimates.  Tracing every Nth section
    instead is the fix, so the estimators must read the step back off the data
    rather than assume 1.  Median, so one accidentally skipped section does not
    change it.
    """
    idxs = sorted(int(k) for k, v in traces.items() if v)
    if len(idxs) < 2:
        return 1.0
    return max(1.0, float(np.median(np.diff(idxs))))


def trace_surface(traces, orientation, n_resample=32):
    """Loft a triangle mesh through the traced lines.  Returns (verts, faces).

    Each line is resampled to a common number of points, a half-section ring is
    extrapolated past each end so the sheet spans the same extent the Riemann
    sum credits, and consecutive rings are joined by quads split into triangles.

    This is the single source of truth for the manual surface: both the area
    estimator and the 3D renders build on it, so a render can never disagree
    with the number it is labelled with.
    """
    o = ORIENTATIONS[orientation]
    idxs = sorted(int(k) for k, v in traces.items() if v)
    if len(idxs) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    def resample(pl):
        p = np.asarray(pl, dtype=float) * VOXEL_NM
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0),
                                                            axis=1))])
        if d[-1] == 0:
            return np.repeat(p[:1], n_resample, axis=0)
        t = np.linspace(0, d[-1], n_resample)
        return np.column_stack([np.interp(t, d, p[:, i]) for i in range(3)])

    # One ring per slice: the longest line, so a slice with stray fragments
    # still contributes a single consistent cross-section.
    rings = [resample(max(traces[i], key=len)) for i in idxs]
    # Orient consecutive rings consistently.  Annotators naturally alternate
    # stroke direction from section to section; lofting point i to point i
    # across a reversed pair produces a bow-tied ribbon whose area is wildly
    # inflated (observed: 9.7x on a real trace).
    for k in range(1, len(rings)):
        a, b = rings[k - 1], rings[k]
        same = np.linalg.norm(a[0] - b[0]) + np.linalg.norm(a[-1] - b[-1])
        rev = np.linalg.norm(a[0] - b[-1]) + np.linalg.norm(a[-1] - b[0])
        if rev < same:
            rings[k] = b[::-1]
    # Lofting only between traced sections spans (n-1) intervals, while the
    # Riemann sum credits each of the n sections a full section thickness.
    # Extrapolate a half-section ring past each end so the two estimators
    # cover the same extent and differ only by obliquity.
    rings = ([rings[0] + 0.5 * (rings[0] - rings[1])] + rings
             + [rings[-1] + 0.5 * (rings[-1] - rings[-2])])

    verts = np.concatenate(rings)
    faces = []
    n = n_resample
    for k in range(len(rings) - 1):
        o0, o1 = k * n, (k + 1) * n
        for j in range(n - 1):
            faces.append([o0 + j, o0 + j + 1, o1 + j])
            faces.append([o0 + j + 1, o1 + j + 1, o1 + j])
    return verts, np.array(faces, dtype=int)


def trace_area_mesh_um2(traces, orientation, n_resample=32):
    """Area of the lofted surface, in um^2.

    The classical sum(L x t) implicitly assumes the junction runs perpendicular
    to the sectioning plane.  A synapse tilted with respect to that plane has
    more surface than sum(L x t) credits it with -- the traced lines are the
    surface's cross-sections, not its width.  The ratio of the two is a direct
    measure of how oblique the synapse is.
    """
    v, f = trace_surface(traces, orientation, n_resample)
    if len(f) == 0:
        return 0.0
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2 / 1e6)


def trace_area_um2(traces, orientation):
    """Fuse per-slice cleft lines into one apposition area.

    `traces` maps a slice index (along the stack normal) to a list of
    polylines, each a list of full 3D voxel coordinates.  Returns both the
    plain Riemann sum -- sum of (traced length x section spacing), the
    classical serial-section estimator -- and a trapezoidal variant that ramps
    the profile to zero one section beyond each end of the traced run.  The two
    should agree closely; a large gap means the synapse was not traced to its
    ends.
    """
    o = ORIENTATIONS[orientation]
    axes = o['inplane']
    spacing_nm = VOXEL_NM[o['normal']] * section_step(traces)

    per_slice = {}
    for idx, polylines in traces.items():
        total = 0.0
        for pl in polylines:
            total += polyline_length_nm([[p[axes[0]], p[axes[1]]] for p in pl],
                                        axes)
        if total > 0:
            per_slice[int(idx)] = total
    if not per_slice:
        return dict(riemann_um2=0.0, trapz_um2=0.0, n_slices=0,
                    extent_nm=0.0, total_length_nm=0.0)

    step = int(round(section_step(traces)))
    lo, hi = min(per_slice), max(per_slice)
    profile = np.array([per_slice.get(i, 0.0) for i in range(lo, hi + 1, step)])
    riemann = profile.sum() * spacing_nm
    padded = np.concatenate([[0.0], profile, [0.0]])
    trapz = float(np.trapezoid(padded, dx=spacing_nm))
    return dict(riemann_um2=riemann / 1e6,
                trapz_um2=trapz / 1e6,
                mesh_um2=trace_area_mesh_um2(traces, orientation),
                n_slices=len(per_slice),
                extent_nm=(hi - lo + 1) * spacing_nm,
                total_length_nm=float(profile.sum()))
