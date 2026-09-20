"""Export the beauty-branch orchard (terrain, wood, leaves, fruit) for a three.js viewer."""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from treesim.config import FoliageParams, FruitParams
from treesim.kiwi_material import STEM_LENGTH
from treesim.orchard_terrain import floor_kwargs_for_plantation, sample_orchard_floor
from treesim.pergola import generate, place_fruit
from treesim.foliage import LEAF_SIZE_CLASSES, leaf_blade_arrays, leaf_blade_style, place_canopy_leaves, place_leaves
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
seed, rows, cols, spacing, fruit_count, canopy_spacing = 42, int(sys.argv[2]), int(sys.argv[3]), 5.0, int(sys.argv[4]), float(sys.argv[5])
SUN = np.array([0.26, 0.42, -1.0]); SUN /= np.linalg.norm(SUN)
cover = floor_kwargs_for_plantation(rows, cols, spacing)
floor = sample_orchard_floor(seed, canopy_height_m=1.6, slope_deg=0., noise_m=0.04, rut_depth_m=0.05, rut_width_m=0.40, friction=1., slope_azimuth_deg=38., landform_m=0., landform_wavelength_m=18., **cover)
skel = generate(height=1.6, seed=seed, rows=rows, columns=cols, spacing=spacing, ground_z=floor.ground_z, canopy_z=floor.canopy_z)
fruit_all = place_fruit(skel, FruitParams(max_count=fruit_count, joint="free", colors=((0.42,0.28,0.10),(0.55,0.38,0.14),(0.33,0.22,0.08))), seed=seed)
fruit = [f for f in fruit_all if skel[f.parent_seg].supported]
keep = [i for i, sg in enumerate(skel) if not (sg.order == 2 and not sg.supported)]
fp = FoliageParams(enabled=True, leaves_per_terminal=8, min_order_for_leaves=2, leaf_length=0.22, leaf_width=0.17, leaf_shape="cordate", leaf_color=(0.14,0.36,0.10), canopy_spacing_m=canopy_spacing)
pl = place_leaves(skel, fp, seed=seed, height_z=floor.canopy_z)
if canopy_spacing: pl.extend(place_canopy_leaves(skel, fp, seed=seed, height_z=floor.canopy_z))
style = leaf_blade_style(fp.leaf_shape)
blades = []
for scale in LEAF_SIZE_CLASSES:
    v, f = leaf_blade_arrays(fp.leaf_length*scale, fp.leaf_width*scale, **style)
    blades.append(dict(verts=np.asarray(v, np.float32).reshape(-1,3), faces=np.asarray(f, np.uint32).reshape(-1,3)))
nominal = fp.leaf_length
leaf_inst = np.zeros((len(pl), 8), np.float32)   # x y z qx qy qz qw cls
for i, p in enumerate(pl):
    leaf_inst[i,:3] = p.attach; leaf_inst[i,3:7] = p.frame
    leaf_inst[i,7] = int(np.argmin([abs(p.length/nominal - s) for s in LEAF_SIZE_CLASSES]))
wood = np.array([[*s.start, *s.end, s.mean_radius, min(s.order,2)] for i, s in enumerate(skel) if i in set(keep)], np.float32)
fr = np.array([[*(f.attach - np.array([0,0,STEM_LENGTH+float(f.radii[2])])), *f.radii, *f.color, *f.attach] for f in fruit], np.float32)
H = np.asarray(floor.heights_m, np.float32)
blob = bytearray(); meta = dict(seed=seed, rows=rows, cols=cols, spacing=spacing, half=float(floor.half_extent_m), canopy_height=1.6, stem_length=float(STEM_LENGTH), sun=SUN.tolist(),
    terrain=dict(n=int(H.shape[0]), min_z=float(H.min()), max_z=float(H.max()), off=0, ground_at_origin=float(floor.ground_z(0.,0.))), blades=[], leaf_color=list(fp.leaf_color))
blob += H.tobytes()
meta['wood'] = dict(off=len(blob), n=len(wood)); blob += wood.tobytes()
meta['leaves'] = dict(off=len(blob), n=len(leaf_inst)); blob += leaf_inst.tobytes()
meta['fruit'] = dict(off=len(blob), n=len(fr)); blob += fr.tobytes()
for b in blades:
    meta['blades'].append(dict(voff=len(blob), nv=len(b['verts']), foff=len(blob)+b['verts'].nbytes, nf=len(b['faces']))); blob += b['verts'].tobytes() + b['faces'].tobytes()
(out/'orchard.bin').write_bytes(blob); (out/'orchard.json').write_text(json.dumps(meta))
(out/'ground.png').write_bytes(floor.texture_png_bytes(skel, sun_dir=SUN, seed=seed))
print('fruit radii', np.round(fr[:5,3:6],3).tolist(), 'attach z', np.round(fr[:,11],2).min(), np.round(fr[:,11],2).max(), 'center z', np.round(fr[:,2],2).min(), np.round(fr[:,2],2).max())
print(dict(segments=len(wood), leaves=len(pl), fruit=len(fr), terrain=H.shape, half=floor.half_extent_m, zrange=(float(H.min()), float(H.max())), canopy=float(floor.canopy_z(0,0)), bytes=len(blob)))
posts = [s for s in skel if s.order == 0]; print('order0 segments', len(posts), 'sample', posts[0].start.round(2).tolist(), posts[0].end.round(2).tolist()); print('orders', np.unique(wood[:,7], return_counts=True))
