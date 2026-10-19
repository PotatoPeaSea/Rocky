# PCB work paused (2026-10-19 16:27)

Status when stopped:
- Done: research, schematic, DESIGN.md, 4-layer ring outline, placement, hand-routed power paths (tools/power_route.py).
- Interrupted: autoroute of signal nets (tools/route.sh). The .kicad_pcb holds the routing from before that pass; expect unrouted nets.
- Not done: finish routing, ERC/DRC to zero, STEP export (tools/export.sh, make_3d.py), JLCPCB gerbers/BOM/CPL, copy STEP to hardware/rocky_control_pcb.step for the Fusion v6 stage_pcb() import, independent review.

Rebuild: tools/rebuild.sh (KiCad 9, Freerouting 1.9.0 jar, Java 21). Generators in tools/ are the source of truth.
