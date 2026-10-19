#!/bin/bash
# regenerate schematic -> netlist -> placed board
set -e
cd /home/claude/rocky/pcb/rocky_ctrl
python3 tools/gen_sch.py >/dev/null
kicad-cli sch export netlist --format kicadsexpr -o /tmp/rocky.net rocky_ctrl.kicad_sch >/dev/null
python3 tools/gen_pcb.py /tmp/rocky.net
