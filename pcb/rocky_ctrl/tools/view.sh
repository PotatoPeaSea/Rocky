#!/bin/bash
# render the board to PNG for review: view.sh <out.png> [layers]
L=${2:-F.Cu,B.Cu,F.SilkS,Edge.Cuts,F.Fab,F.Courtyard}
kicad-cli pcb export svg --mode-single --exclude-drawing-sheet --fit-page-to-board -l "$L" -o /tmp/view.svg /home/claude/rocky/pcb/rocky_ctrl/rocky_ctrl.kicad_pcb >/dev/null && echo ok
