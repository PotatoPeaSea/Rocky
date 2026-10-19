"""Generate rocky_ctrl.kicad_sym, rocky_ctrl.kicad_sch and project files from design.py.

Run inside an environment with the KiCad 9 symbol libraries (KICAD_SYMBOL_DIR or /usr/share/kicad/symbols).
"""
import copy
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from sexp import parse, dump, find, find1, Sym  # noqa: E402
import design  # noqa: E402

PRJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SYMDIR = os.environ.get('KICAD_SYMBOL_DIR', '/usr/share/kicad/symbols')
NAME = 'rocky_ctrl'
ROOT_UUID = str(uuid.uuid5(uuid.NAMESPACE_URL, 'rocky_ctrl/root'))
POWER_NETS = {'GND': 'power:GND', '+3V3': 'power:+3V3', '+5V': 'power:+5V'}
_u = [0]


def uid(tag=None):
    _u[0] += 1
    return str(uuid.uuid5(uuid.NAMESPACE_URL, 'rocky_ctrl/%s/%d' % (tag or '', _u[0])))


_libs = {}


def lib(name):
    if name not in _libs:
        path = os.path.join(PRJ, NAME + '.kicad_sym') if name == NAME else os.path.join(SYMDIR, name + '.kicad_sym')
        _libs[name] = parse(open(path).read())
    return _libs[name]


def raw_symbol(libname, symname):
    for s in find(lib(libname), 'symbol'):
        if s[1] == symname:
            return s
    raise KeyError(libname + ':' + symname)


def flat_symbol(libid):
    """Return a self-contained copy of a library symbol (derived symbols flattened), named libid."""
    libname, symname = libid.split(':')
    s = copy.deepcopy(raw_symbol(libname, symname))
    ext = find1(s, 'extends')
    if ext:
        parent = flat_symbol(libname + ':' + ext[1])
        pname = ext[1]
        child_props = {p[1]: p for p in find(s, 'property')}
        out = [Sym('symbol'), libid]
        for item in parent[2:]:
            if isinstance(item, list) and item[0] == 'property' and item[1] in child_props:
                out.append(child_props.pop(item[1]))
            elif isinstance(item, list) and item[0] == 'symbol':
                sub = copy.deepcopy(item)
                sub[1] = sub[1].replace(pname, symname, 1)
                out.append(sub)
            else:
                out.append(item)
        # insert child-only properties before the first sub-symbol
        idx = next(i for i, x in enumerate(out) if isinstance(x, list) and x[0] == 'symbol')
        for p in child_props.values():
            out.insert(idx, p); idx += 1
        return out
    s[1] = libid
    return s


def sym_pins(symdef):
    """[(unit, number, name, type, x, y, angle)] in library coordinates (y up)."""
    pins = []
    for sub in find(symdef, 'symbol'):
        unit = int(sub[1].split('_')[-2])
        for p in find(sub, 'pin'):
            at = find1(p, 'at')
            pins.append((unit, find1(p, 'number')[1], find1(p, 'name')[1], str(p[1]),
                         float(at[1]), float(at[2]), int(float(at[3]))))
    return pins


def sym_bbox(symdef):
    xs, ys = [], []
    for sub in find(symdef, 'symbol'):
        for g in sub[2:]:
            if not isinstance(g, list):
                continue
            if g[0] == 'pin':
                at = find1(g, 'at'); xs.append(float(at[1])); ys.append(float(at[2]))
            for key in ('start', 'end', 'center'):
                e = find1(g, key)
                if e:
                    xs.append(float(e[1])); ys.append(float(e[2]))
            pts = find1(g, 'pts')
            if pts:
                for xy in find(pts, 'xy'):
                    xs.append(float(xy[1])); ys.append(float(xy[2]))
    if not xs:
        return (-2.54, -2.54, 2.54, 2.54)
    return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------- custom symbols
def build_custom_lib():
    lv = flat_symbol('Logic_LevelTranslator:SN74LV1T125DBV')
    lv = copy.deepcopy(lv)
    lv[1] = 'SN74LV1T126DBV'
    for e in lv:
        if isinstance(e, list) and e[0] == 'symbol':
            e[1] = e[1].replace('SN74LV1T125DBV', 'SN74LV1T126DBV')
    for e in lv:
        if isinstance(e, list) and e[0] == 'property':
            if e[1] == 'Value':
                e[2] = 'SN74LV1T126DBV'
            elif e[1] == 'Description':
                e[2] = 'Single buffer, 3-state, active-HIGH output enable, 1.6-5.5 V, TTL-level 5.5 V tolerant inputs (level up-shift)'
            elif e[1] == 'Datasheet':
                e[2] = 'https://www.ti.com/lit/ds/symlink/sn74lv1t126.pdf'
    for sub in find(lv, 'symbol'):
        for p in find(sub, 'pin'):
            nm = find1(p, 'name')
            if nm[1] == '~{OE}':
                nm[1] = 'OE'
                p[2] = Sym('line')

    pm = copy.deepcopy(flat_symbol('Device:Q_PMOS'))
    pm[1] = 'Q_PMOS_SSSGD'
    for e in pm:
        if isinstance(e, list) and e[0] == 'symbol':
            e[1] = e[1].replace('Q_PMOS', 'Q_PMOS_SSSGD')
        if isinstance(e, list) and e[0] == 'property':
            if e[1] == 'Description':
                e[2] = 'P-MOSFET, 8-pin power package (pins 1-3 source, 4 gate, 5-8/tab drain), e.g. Infineon PG-TDSON-8'
            if e[1] == 'Footprint':
                e[2] = 'Package_TO_SOT_SMD:TDSON-8-1'
    for sub in find(pm, 'symbol'):
        newpins = []
        for p in find(sub, 'pin'):
            num = find1(p, 'number')
            name = find1(p, 'name')[1]
            if name == 'G':
                num[1] = '4'
            elif name == 'D':
                num[1] = '5'
            elif name == 'S':
                num[1] = '1'
                for extra in ('2', '3'):
                    q = copy.deepcopy(p)
                    find1(q, 'number')[1] = extra
                    q.append([Sym('hide'), Sym('yes')])
                    newpins.append(q)
        sub.extend(newpins)
    lv[1] = 'SN74LV1T126DBV'
    out = [Sym('kicad_symbol_lib'), [Sym('version'), 20241209], [Sym('generator'), 'rocky_gen'],
           [Sym('generator_version'), '9.0'], pm, lv]
    open(os.path.join(PRJ, NAME + '.kicad_sym'), 'w').write(dump(out) + '\n')
    _libs.pop(NAME, None)


# --------------------------------------------------------------------------- schematic helpers
def fnt(size=1.27, hide=False, justify=None, bold=False):
    f = [Sym('font'), [Sym('size'), size, size]]
    if bold:
        f.append([Sym('bold'), Sym('yes')])
    e = [Sym('effects'), f]
    if justify:
        e.append([Sym('justify')] + [Sym(j) for j in justify.split()])
    if hide:
        e.append([Sym('hide'), Sym('yes')])
    return e


def prop(name, val, x, y, hide=False, justify=None, angle=0):
    return [Sym('property'), name, val, [Sym('at'), x, y, angle], fnt(hide=hide, justify=justify)]


ITEMS = []
PWR_COUNT = [0]
LIB_USED = {}


def r2(v):
    return round(v * 100) / 100.0


def add_symbol(libid, ref, value, fp, x, y, rot=0, unit=1, props=None, pins=(), in_bom=True, on_board=True,
               dnp=False, power=False):
    LIB_USED.setdefault(libid, flat_symbol(libid))
    s = [Sym('symbol'), [Sym('lib_id'), libid], [Sym('at'), x, y, rot], [Sym('unit'), unit],
         [Sym('exclude_from_sim'), Sym('no')], [Sym('in_bom'), Sym('yes' if in_bom else 'no')],
         [Sym('on_board'), Sym('yes' if on_board else 'no')], [Sym('dnp'), Sym('yes' if dnp else 'no')],
         [Sym('uuid'), uid('sym')]]
    s.append(prop('Reference', ref, x + 2.54, y - 1.27, hide=power, justify='left'))
    s.append(prop('Value', value, x + 2.54, y + 1.27, hide=False, justify='left'))
    s.append(prop('Footprint', fp, x, y, hide=True))
    s.append(prop('Datasheet', '', x, y, hide=True))
    for k, v in (props or {}).items():
        s.append(prop(k, v, x, y, hide=True))
    for num in pins:
        s.append([Sym('pin'), num, [Sym('uuid'), uid('pin')]])
    s.append([Sym('instances'), [Sym('project'), NAME, [Sym('path'), '/' + ROOT_UUID, [Sym('reference'), ref],
                                                         [Sym('unit'), unit]]]])
    ITEMS.append(s)


def wire(x1, y1, x2, y2):
    ITEMS.append([Sym('wire'), [Sym('pts'), [Sym('xy'), x1, y1], [Sym('xy'), x2, y2]],
                  [Sym('stroke'), [Sym('width'), 0], [Sym('type'), Sym('default')]], [Sym('uuid'), uid('w')]])


def label(net, x, y, d):
    ang = {(-1, 0): 180, (1, 0): 0, (0, -1): 90, (0, 1): 270}[d]
    just = 'left bottom' if ang in (0, 90) else 'right bottom'
    ITEMS.append([Sym('label'), net, [Sym('at'), x, y, ang], [Sym('fields_autoplaced'), Sym('yes')],
                  fnt(justify=just), [Sym('uuid'), uid('lbl')]])


def power_sym(net, x, y, d):
    PWR_COUNT[0] += 1
    libid = POWER_NETS[net]
    if net == 'GND':
        rot = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}[d]
    else:
        rot = {(0, -1): 0, (-1, 0): 90, (0, 1): 180, (1, 0): 270}[d]
    add_symbol(libid, '#PWR%03d' % PWR_COUNT[0], net, '', x, y, rot, pins=['1'], in_bom=False, on_board=False,
               power=True)


def pwr_flag(x, y):
    PWR_COUNT[0] += 1
    add_symbol('power:PWR_FLAG', '#FLG%03d' % PWR_COUNT[0], 'PWR_FLAG', '', x, y, 0, pins=['1'], in_bom=False,
               on_board=False, power=True)


def no_connect(x, y):
    ITEMS.append([Sym('no_connect'), [Sym('at'), x, y], [Sym('uuid'), uid('nc')]])


def text(s, x, y, size=1.27, bold=False):
    ITEMS.append([Sym('text'), s, [Sym('exclude_from_sim'), Sym('no')], [Sym('at'), x, y, 0],
                  fnt(size, justify='left bottom', bold=bold), [Sym('uuid'), uid('txt')]])


def rect(x1, y1, x2, y2):
    ITEMS.append([Sym('rectangle'), [Sym('start'), x1, y1], [Sym('end'), x2, y2],
                  [Sym('stroke'), [Sym('width'), 0.3], [Sym('type'), Sym('dash')]], [Sym('fill'), [Sym('type'), Sym('none')]],
                  [Sym('uuid'), uid('rect')]])


def snap(v, g=2.54):
    return round(v / g) * g


# --------------------------------------------------------------------------- main
def main():
    build_custom_lib()
    # which nets have a power_out driver (to decide PWR_FLAGs)
    net_types = {}
    part_syms = {}
    for p in design.PARTS:
        sd = flat_symbol(p['lib'])
        part_syms[p['ref']] = sd
        for unit, num, name, typ, px, py, ang in sym_pins(sd):
            net = p['pins'].get(num)
            if net:
                net_types.setdefault(net, set()).add(typ)

    # place parts per block
    cursor = {}
    flagged = set()
    for b, info in design.BLOCKS.items():
        rect(info['x'], info['y'], info['x'] + info['w'], info['y'] + info['h'])
        text(info['title'], info['x'] + 2, info['y'] + 6, 2.5, bold=True)
        cursor[b] = [info['x'] + 6, info['y'] + 14, 0]   # x, y, row height
    overflow = []
    for p in design.PARTS:
        sd = part_syms[p['ref']]
        pins = sym_pins(sd)
        units = sorted(set(u for u, *_ in pins)) or [1]
        x0, y0, x1, y1 = sym_bbox(sd)
        # screen bbox (y flipped) + label allowance
        maxlab = max([len(n or '') for n in p['pins'].values()] + [3]) * 1.0 + 5.08
        left = -x0 + (maxlab if any(a == 0 for *_, a in pins) else 2)
        right = x1 + (maxlab if any(a == 180 for *_, a in pins) else 2) + 14
        top = y1 + (maxlab if any(a == 270 for *_, a in pins) else 3)
        bottom = -y0 + (maxlab if any(a == 90 for *_, a in pins) else 3) + 2
        if len(units) > 1:
            right += (x1 - x0) + 30
        b = design.BLOCKS[p['block']]
        cx, cy, rh = cursor[p['block']]
        if cx + left + right > b['x'] + b['w'] - 2:
            cx = b['x'] + 6
            cy = cy + rh
            rh = 0
        ox, oy = snap(cx + left), snap(cy + top)
        if oy + bottom > b['y'] + b['h']:
            overflow.append(p['ref'])
        cursor[p['block']] = [cx + left + right, cy, max(rh, top + bottom)]
        props = {'LCSC': p['lcsc'], 'MPN': p['mpn']}
        if p['note']:
            props['Note'] = p['note']
        for k, unit in enumerate(units):
            ux = ox + k * snap((x1 - x0) + 30)
            unit_pins = [pp for pp in pins if pp[0] == unit or pp[0] == 0]
            add_symbol(p['lib'], p['ref'], p['value'], p['fp'], ux, oy, 0, unit=unit, props=props,
                       pins=[pp[1] for pp in unit_pins], dnp=p['dnp'])
            done = {}
            for (_, num, name, typ, px, py, ang) in unit_pins:
                sx, sy = r2(ux + px), r2(oy - py)
                net = p['pins'].get(num)
                key = (sx, sy)
                if key in done:
                    if done[key] != net:
                        raise ValueError('stacked pins with different nets %s %s' % (p['ref'], num))
                    continue
                done[key] = net
                if num not in p['pins']:
                    if typ != 'no_connect':
                        print('WARN unassigned pin', p['ref'], num, name)
                    continue
                if not net:
                    if typ != 'no_connect':
                        no_connect(sx, sy)
                    continue
                d = {0: (-1, 0), 180: (1, 0), 90: (0, 1), 270: (0, -1)}[ang]
                ex, ey = r2(sx + d[0] * 2.54), r2(sy + d[1] * 2.54)
                wire(sx, sy, ex, ey)
                if net in POWER_NETS:
                    power_sym(net, ex, ey, d)
                else:
                    label(net, ex, ey, d)
                types = net_types.get(net, set()) | ({'power_in'} if net in POWER_NETS else set())
                if net not in flagged and 'power_in' in types and 'power_out' not in types:
                    # power input with no power output on the net: tie a PWR_FLAG to it (drawn in one cluster)
                    flagged.add(net)
                    FLAGS.append(net)
    if overflow:
        print('BLOCK OVERFLOW:', overflow)
    # PWR_FLAG cluster, each tied by a label/power symbol
    fx, fy = 660.0, 560.0
    text('Power flags (ERC: these nets are driven by regulators through passives)', fx - 10, fy - 8, 1.5)
    for i, net in enumerate(FLAGS):
        x = snap(fx + (i % 6) * 25.4)
        y = snap(fy + (i // 6) * 12.7)
        pwr_flag(x, y)
        wire(x, y, x, r2(y + 2.54))
        if net in POWER_NETS:
            power_sym(net, x, r2(y + 2.54), (0, 1))
        else:
            label(net, x, r2(y + 2.54), (0, 1))

    # notes
    text('Rocky control board v1 - see DESIGN.md for the current budget, pin map and placement.', 20, 590, 2.0)

    lib_symbols = [Sym('lib_symbols')] + [LIB_USED[k] for k in sorted(LIB_USED)]
    sch = [Sym('kicad_sch'), [Sym('version'), 20250114], [Sym('generator'), 'eeschema'],
           [Sym('generator_version'), '9.0'], [Sym('uuid'), ROOT_UUID], [Sym('paper'), 'A1'],
           [Sym('title_block'), [Sym('title'), 'Rocky pentapod control board'], [Sym('date'), '2026-10-19'],
            [Sym('rev'), 'v1'], [Sym('company'), 'Matthew Lin'],
            [Sym('comment'), 1, 'ESP32-WROOM-32E + 2x half-duplex servo bus + 20 A battery input + 5 V/5 A Pi supply'],
            [Sym('comment'), 2, 'Generated by tools/gen_sch.py from tools/design.py']],
           lib_symbols] + ITEMS + [[Sym('sheet_instances'), [Sym('path'), '/', [Sym('page'), '1']]],
                                   [Sym('embedded_fonts'), Sym('no')]]
    open(os.path.join(PRJ, NAME + '.kicad_sch'), 'w').write(dump(sch) + '\n')

    # library tables (project-relative custom libs; standard libs come from KiCad's global tables)
    open(os.path.join(PRJ, 'sym-lib-table'), 'w').write(
        '(sym_lib_table\n\t(version 7)\n\t(lib (name "rocky_ctrl")(type "KiCad")(uri "${KIPRJMOD}/rocky_ctrl.kicad_sym")'
        '(options "")(descr "Rocky custom symbols"))\n)\n')
    print('symbols placed:', len([i for i in ITEMS if i[0] == 'symbol']), 'flags:', FLAGS)


FLAGS = []
if __name__ == '__main__':
    main()
