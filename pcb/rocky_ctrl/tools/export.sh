#!/bin/bash
# Fab + documentation outputs. Run from anywhere (inside the KiCad 9 environment).
set -e
P=/home/claude/rocky/pcb/rocky_ctrl
cd $P
mkdir -p fab fab/gerbers
kicad-cli sch erc --severity-all -o fab/erc.rpt rocky_ctrl.kicad_sch | tail -1
kicad-cli pcb drc --schematic-parity --severity-all -o fab/drc.rpt rocky_ctrl.kicad_pcb | tail -1
rm -f fab/gerbers/*
kicad-cli pcb export gerbers -o fab/gerbers/ -l F.Cu,In1.Cu,In2.Cu,B.Cu,F.Paste,B.Paste,F.SilkS,B.SilkS,F.Mask,B.Mask,Edge.Cuts \
  --subtract-soldermask rocky_ctrl.kicad_pcb
kicad-cli pcb export drill -o fab/gerbers/ --format excellon --excellon-separate-th --generate-map --map-format gerberx2 rocky_ctrl.kicad_pcb
(cd fab/gerbers && rm -f ../rocky_ctrl_gerbers.zip && zip -q ../rocky_ctrl_gerbers.zip *)
kicad-cli pcb export pos -o fab/rocky_ctrl_pos.csv --format csv --units mm --side both rocky_ctrl.kicad_pcb
kicad-cli sch export bom -o fab/rocky_ctrl_bom_full.csv --fields 'Reference,Value,Footprint,LCSC,MPN,${QUANTITY}' \
  --group-by 'Value,Footprint,LCSC,MPN' --ref-range-delimiter '' rocky_ctrl.kicad_sch
python3 tools/jlc.py
kicad-cli sch export pdf -o fab/rocky_ctrl_schematic.pdf rocky_ctrl.kicad_sch
for L in F.Cu In1.Cu In2.Cu B.Cu; do
  kicad-cli pcb export svg --mode-single --exclude-drawing-sheet --fit-page-to-board -l $L,Edge.Cuts -o fab/layer_$L.svg rocky_ctrl.kicad_pcb >/dev/null
done
kicad-cli pcb export svg --mode-single --exclude-drawing-sheet --fit-page-to-board -l F.Cu,F.SilkS,F.Mask,Edge.Cuts -o fab/top_assembly.svg rocky_ctrl.kicad_pcb >/dev/null
kicad-cli pcb export step --subst-models --user-origin 150x150mm --force -o rocky_ctrl.step rocky_ctrl.kicad_pcb 2>&1 | grep -v -i "^$" | tail -4
xvfb-run -a kicad-cli pcb render --side top --quality high --width 1800 --height 1800 --background opaque -o fab/rocky_ctrl_top.png rocky_ctrl.kicad_pcb >/dev/null 2>&1 || echo "render top failed"
xvfb-run -a kicad-cli pcb render --side bottom --quality high --width 1800 --height 1800 --background opaque -o fab/rocky_ctrl_bottom.png rocky_ctrl.kicad_pcb >/dev/null 2>&1 || echo "render bottom failed"
xvfb-run -a kicad-cli pcb render --side top --rotate '-45,0,30' --perspective --quality high --width 1800 --height 1400 --background opaque -o fab/rocky_ctrl_iso.png rocky_ctrl.kicad_pcb >/dev/null 2>&1 || echo "render iso failed"
ls -la fab rocky_ctrl.step
