"""Chassis-mounted open basket with independent, collidable kiwi bodies."""
import numpy as np

# Coordinates relative to Spot's body. Front is +X; arm mount is at X=0.292.
CENTER = np.array([-.20, 0., .145])
SIZE = np.array([.54, .38, .28])
WALL = .009


def payload_layout(mass, seed):
    """Non-overlapping spawn positions; all fruit are free rigid bodies."""
    if not np.isfinite(mass) or not 0 <= mass <= 6:
        raise ValueError('Fruit payload must be between 0 and 6 kg')
    if mass == 0:
        return np.empty((0, 3)), np.empty((0,)), np.array([.0259, .028, .03249])
    rng = np.random.default_rng(seed)
    radii = np.array([.0259, .028, .03249])
    count = int(np.ceil(mass / .10))
    # Fill bottom layers first; varied occupancy creates
    # unequal loads without overlapping fruit or moving COM outside the basket.
    grid = np.array([(x, y, z) for z in (.044, .114, .184)
                     for x in np.linspace(-.175, .175, 7)
                     for y in np.linspace(-.126, .126, 5)])
    selected = []
    for layer in np.array_split(grid, 3):
        rng.shuffle(layer)
        selected.extend(layer[:max(0, min(len(layer), count-len(selected)))])
    positions = np.asarray(selected) + CENTER
    # Normal payloads represent whole Hayward-sized fruit. Payloads below one
    # fruit remain synthetic load tests; scale their volume to preserve density.
    weights = np.full(count, mass/count)
    density = weights[0]/(4*np.pi*np.prod(radii)/3)
    if density < 940.:
        radii = radii*(density/990.)**(1/3)
    return positions, weights, radii


def add_basket(builder, chassis, mass, payload, seed, spawn=None):
    from .kiwi_material import FRUIT_FRICTION, LINER_FRICTION, RESTITUTION
    import newton
    import warp as wp
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('Empty basket mass must be positive')
    positions, masses, radii = payload_layout(payload, seed)
    if spawn is None:
        spawn = wp.transform_identity()
    yellow, black = (1., .72, .025), (.065, .075, .085)
    lx, ly, height = SIZE
    cx, cy, base = CENTER
    parts = [((cx, cy, base), (lx/2, ly/2, WALL/2))]
    for sign in (-1, 1):
        parts += [((cx+sign*(lx-WALL)/2, cy, base+height/2), (WALL/2, (ly-2*WALL)/2, height/2)),
                  ((cx, sign*(ly-WALL)/2, base+height/2), (lx/2, WALL/2, height/2))]
    volume = sum(8*np.prod(half) for _, half in parts)
    before = float(builder.body_mass[chassis])
    first_shape = builder.shape_count
    # Collision shell is a hollow, open-top box; the visible lattice is a proxy
    # for a basket with a thin liner. All mass/inertia comes from actual shapes.
    for i, (pos, half) in enumerate(parts):
        builder.add_shape_box(chassis, xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
            hx=float(half[0]), hy=float(half[1]), hz=float(half[2]),
            cfg=builder.ShapeConfig(density=float(mass/volume), mu=LINER_FRICTION, is_visible=i == 0),
            color=black, label='basket_floor' if i == 0 else 'basket_liner')
    visual = builder.ShapeConfig(density=0., has_shape_collision=False, has_particle_collision=False)
    # Flat, angular yellow side plates, trapezoidal vents, black bumpers and
    # mounting feet reproduce the reference silhouette rather than wire rails.
    def box(pos, half, color, cfg=visual, label=None):
        return builder.add_shape_box(chassis,
            xform=wp.transform(wp.vec3(*map(float, pos)), wp.quat_identity()),
            hx=float(half[0]), hy=float(half[1]), hz=float(half[2]), cfg=cfg,
            color=color, label=label)

    def plate(points, y, thickness, color):
        # Extrude a polygon in the XZ plane. Individual convex strips surround
        # real visual openings; the thin collision liner stays behind them.
        vertices = np.array([(cx+x, y+side*thickness/2, base+z)
                             for side in (-1, 1) for x, z in points], dtype=np.float32)
        n = len(points)
        triangles = []
        for i in range(1, n-1):
            triangles.extend([0, i+1, i, n, n+i, n+i+1])
        for i in range(n):
            j = (i+1) % n
            triangles.extend([i, j, n+j, i, n+j, n+i])
        mesh = newton.Mesh(vertices, np.asarray(triangles, dtype=np.int32))
        builder.add_shape_mesh(chassis, mesh=mesh, cfg=visual, color=color)

    for sy in (-1, 1):
        y = sy*ly/2
        # Bevelled outer frame with thick side panels and open centre windows.
        plate([(-.265,.055),(-.24,.018),(.235,.018),(.265,.055),(.23,.09),(-.23,.09)],y,.019,yellow)
        plate([(-.25,.235),(-.215,.278),(.215,.278),(.255,.24),(.225,.22),(-.225,.22)],y,.023,yellow)
        plate([(-.25,.05),(-.245,.24),(-.16,.25),(-.12,.18),(-.15,.065)],y,.024,yellow)
        plate([(.15,.06),(.12,.18),(.16,.255),(.24,.24),(.255,.055)],y,.024,yellow)
        plate([(-.155,.14),(.15,.14),(.15,.162),(-.155,.162)],y,.016,yellow)
        for x in (-.095,.085):
            plate([(x-.027,.078),(x+.065,.222),(x+.09,.222),(x,.078)],y,.014,yellow)
        plate([(-.257,.058),(-.229,.009),(-.11,.009),(-.07,.039),(.07,.039),(.11,.009),(.228,.009),(.265,.054),(.245,.068),(.22,.03),(.12,.03),(.085,.06),(-.085,.06),(-.12,.03),(-.218,.03),(-.237,.072)],y+sy*.016,.025,black)
        # Deck rails and short supports bridge the former visible air gap.
        box((cx,sy*.105,.112),(.25,.026,.018),black,label='basket_mount_rail')
        for x in (cx-.17,cx+.17):
            box((x,sy*.105,.128),(.038,.025,.025),black,label='basket_mount_foot')
            box((x,sy*(ly/2+.012),base+.145),(.018,.018,.126),black)
    for sx in (-1, 1):
        x=cx+sx*lx/2
        box((x,0,base+.093),(.014,ly/2,.074),yellow)
        box((x,0,base+height-.02),(.02,ly/2,.018),yellow)
        # Black carrying handles at the ends, above the rim.
        box((x,0,base+height+.024),(.018,.12,.013),black)
        for sy in (-1,1):
            box((x,sy*.12,base+height+.006),(.018,.012,.027),black)
    # Reference-style capacity labels on both front side panels.
    from PIL import Image, ImageDraw, ImageFont
    label = Image.new('RGB', (256, 384), (255, 184, 6))
    draw = ImageDraw.Draw(label)
    try:
        large = ImageFont.truetype('DejaVuSansCondensed-Bold.ttf', 66)
        small = ImageFont.truetype('DejaVuSansCondensed.ttf', 24)
    except OSError:
        large, small = ImageFont.load_default(size=58), ImageFont.load_default(size=23)
    draw.multiline_text((20, 18), 'KIWI\n6KG', font=large, fill=(12, 16, 17), spacing=0)
    draw.multiline_text((22, 204), 'FIELD\nHARVEST', font=small, fill=(12, 16, 17), spacing=3)
    draw.line((22, 334, 180, 334), fill=(12, 16, 17), width=10)
    for sy in (-1, 1):
        y = sy*(ly/2+.013)
        vertices = np.array([(cx+x,y,base+z) for x,z in ((.158,.091),(.231,.091),(.231,.216),(.158,.216))], dtype=np.float32)
        uv = np.array([(0,1),(1,1),(1,0),(0,0)], dtype=np.float32)
        mesh = newton.Mesh(vertices, np.array([0,1,2,0,2,3,2,1,0,3,2,0], dtype=np.int32),
                           uvs=uv, texture=np.asarray(label), compute_inertia=False)
        builder.add_shape_mesh(chassis, mesh=mesh, cfg=visual, color=(1.,1.,1.))
    basket_shape_end = builder.shape_count
    fruit_bodies = []
    fruit_volume = 4*np.pi*np.prod(radii)/3
    for i, (pos, fruit_mass) in enumerate(zip(positions, masses)):
        pose = wp.transform_multiply(spawn,
                                     wp.transform(wp.vec3(*map(float, pos)), wp.quat_identity()))
        body = builder.add_link(xform=pose, label=f'basket_kiwi_{i}')
        cfg = builder.ShapeConfig(density=float(fruit_mass/fruit_volume), mu=FRUIT_FRICTION,
                                   restitution=RESTITUTION, collision_group=1)
        builder.add_shape_ellipsoid(body,
                                   rx=float(radii[0]), ry=float(radii[1]), rz=float(radii[2]), cfg=cfg,
                                   color=(.40+.025*(i%4), .27+.012*(i%3), .12), label='loose_kiwi')
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint], label=f'basket_kiwi_{i}')
        fruit_bodies.append(body)
    added = float(builder.body_mass[chassis])-before
    if not np.isclose(added, mass, atol=1e-5):
        raise RuntimeError(f'Basket mass mismatch: {added} versus {mass}')
    return dict(empty_mass_kg=mass, payload_kg=payload, seed=seed,
                payload_com_body_m=(np.average(positions, axis=0, weights=masses).tolist() if len(masses) else None),
                fruit_count=len(masses), size_m=SIZE.tolist(), first_shape=first_shape,
                shape_end=basket_shape_end, mass_added_kg=added,
                fruit_bodies=fruit_bodies, fruit_masses_kg=masses.tolist(),
                chassis_com_body_m=list(map(float, builder.body_com[chassis])))


class SpillTracker:
    """One negative reward per newly lost fruit; no repeated penalty every step."""
    def __init__(self, basket):
        self.basket = basket
        self.spilled = np.zeros(len(basket['fruit_bodies']), dtype=bool)
        self.total_penalty = 0.

    def update(self, body_q, chassis):
        pose = body_q[chassis]
        relative = body_q[self.basket['fruit_bodies'], :3]-pose[:3]
        u, w = -pose[3:6], pose[6]
        local = relative + 2*np.cross(u, np.cross(u, relative)+w*relative)
        # Must clear the outside of a wall, or pass below the floor. A bouncing
        # fruit still over the open mouth is not counted as lost.
        outside = (np.any(np.abs(local[:,:2]-CENTER[:2]) > SIZE[:2]/2+.04, axis=1)
                   | (local[:,2] < CENTER[2]-.045))
        newly = outside & ~self.spilled
        self.spilled |= outside
        penalty = -float(np.count_nonzero(newly))
        self.total_penalty += penalty
        return penalty
