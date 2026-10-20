# rocky_ctrl v1: independent pre-order review (2026-10-19)

**Scope:** the schematic source (`tools/design.py`), the netlist and pads in `rocky_ctrl.kicad_pcb`, the fab outputs, and a fresh ERC and DRC run.

**Result:** 1 BLOCKER (a process item, not a design bug), 4 SHOULD-FIX, 11 NOTES. No wiring or pinout error was found that would make the board dead or dangerous.

## Re-run of ERC and DRC (KiCad 9 kicad-cli)

- **ERC** (`--severity-all`): 0 errors, 0 warnings.
- **DRC** (`--schematic-parity --severity-all`): 0 unconnected items and 0 parity issues. There are 29 violations, all silkscreen: 21 `silk_overlap`, 5 `silk_edge_clearance` and 3 `silk_over_copper`. This matches `fab/drc.rpt`.

---

## BLOCKER

### B1. About 30 BOM lines have no LCSC number, so JLC will not place the core parts (process)
- **Parts without an LCSC number:** U1 INA180A3, U2 TPS54560B, U3 AP63203, U6 '3157, U8-U11 LV1T125/126, Q1/Q2 BSC030P03NS3, D1, D2, D5-D9, DZ1/DZ2, L1, L2, R6, R14, R16, R17, R19, C2/C8/C11/C12/C23, C3, C15, C16 and C39-C43.
- **Evidence:** the `LCSC` column is empty for these lines in `fab/rocky_ctrl_bom_jlc.csv`, and DESIGN.md §7 says so too.
- **What happens if you order now:** an assembly order ships without its regulators, FETs, current sense and bus buffers.
- **Fix:** before you pay, assign a part to every line in JLC's BOM tool and check that each one is in stock. If a part is not stocked, use the drop-in alternative from DESIGN.md §9.6. Check that the package of every substitute matches its footprint, in particular:
  - TDSON-8 / SuperSO8 for Q1 and Q2;
  - HSOP-8 DDA with PowerPAD for U2;
  - TSOT-23-6 for U3;
  - SC-70-6 (DCK) for U6.

---

## SHOULD-FIX

### S1. Q2's Miller capacitor turns the servo rail on at every battery plug-in
- **Parts and nets:** Q2, C3 (220 nF, Q2_G to VSERVO_SW), R2 (100 k), DZ2.
- **The mechanism:**
  1. When the XT60 mates, VBAT_P steps to 12.6 V in microseconds. VSERVO_SW is held at 0 V by C6/C7 (2 mF), so C3 holds the gate near 0 V. Q2's own Ciss of a few nF cannot fight 220 nF.
  2. The source-gate voltage (Vsg) therefore jumps to the DZ2 clamp of about 10 V, and Q2 turns fully on while SERVO_EN is low.
  3. The drain then rises and drags the gate up through C3 until Vsg falls to Vth. That happens after a rise of roughly 10 − Vth ≈ 7-8 V.
- **The result:**
  - Every battery connect dumps an uncontrolled, roughly 100 A pulse of about 100 µs into 2 mF. That is the spark the soft start was meant to remove.
  - The servo rail then sits at about 8 V and decays over several seconds through LED3 and R9, because the rail is meant to be "default OFF" and nothing actively discharges it.
  - Q2 and F1 will very likely survive, but the intended behaviour is defeated. DZ2 does not prevent this, and neither would removing DZ2.
- **Evidence:** `design.py` lines 86-96 (Q2, R2, DZ2, C3). Vsg at the step ≈ 12.6 V × C3 / (C3 + C_GS), where C_GS is the gate-source capacitance. Today C_GS ≈ Ciss only.
- **Fix:** add a gate-source capacitor from Q2_G to VBAT_P, in parallel with R2, so the step divider keeps Vsg below Vth(min), which is about 1 V.
  - **Option 1:** C_GS = 4.7 µF / 25 V X7R gives Vsg ≈ 0.56 V. Also cut C3 to 100 nF, which halves the plateau current to keep inrush similar, and drop R2 to 47 k so turn-off stays near 0.2 s.
  - **Option 2:** keep C3 at 220 nF and use C_GS ≥ 3.3 µF. Turn-off then becomes R2 × C_GS, about 0.35 s or more.
  - **Verify in LTspice**, and lengthen the firmware's 30 ms soft-start wait to cover the extra threshold delay of about 10-20 ms.
  - **If no layout change is made:** at minimum place an 0805 footprint for C_GS next to R2 so the cap can be bodged on.

### S2. The unidirectional TVS ahead of Q1 turns reverse polarity into a fuse-blowing short
- **Parts and nets:** D1 SMBJ15A (K = VBAT_F, A = GND), F1, Q1.
- **The problem:** with the battery reversed, D1 is forward-biased straight across the pack, before Q1. Q1's reverse protection never acts. The fuse (or the TVS, which fails short) takes the hit.
- **Evidence:** `design.py` line 70, which places D1 on VBAT_F (the fuse side of Q1).
- **On the positive side the part is fine:** the SMBJ15A has VR = 15 V and VBR(min) = 16.7 V, so it does not conduct at 12.6 V, and its clamp is about 24.4 V.
- **Fix:** fit an **SMBJ15CA** (bidirectional, same SMB footprint, no layout change). At −12.6 V it then stays below standoff and Q1 blocks the reverse battery.
- **Alternative:** move D1 to VBAT_P, which needs a layout change.

### S3. The TPS54560B input hot loop is large: D2 sits about 19 mm from the input-cap ground
- **Parts and nets:** U2, C11/C12/C13, D2.
- **Positions (KiCad mm):**
  - U2 VIN pin 2 is at (147.6, 192.1), and the C11/C12 GND pads are at (151, 189) and (154, 189);
  - U2 SW pin 8 is at (142.4, 193.4);
  - D2 cathode is at (140.0, 197.6) and D2 anode (GND) at (140.0, 204.4).
- **Why it matters:** the high-di/dt loop Cin → high-side FET → D2 → GND → Cin encloses about 11 × 15 mm. Expect SW-node ringing and EMI. The board will probably still work, but this goes against the TPS54560B datasheet's layout guidance (§11, Layout), which puts the catch-diode anode next to the Cin GND and the IC GND pins.
- **Fix:**
  - **Best:** move D2 (and L1 with it) so that D2's cathode touches pin 8 and its anode sits beside the C11/C12 GND pads, with several GND vias at the anode.
  - **Minimum:** add a 10 µF + 100 nF pair from VBAT_P to GND right at D2's anode, plus 2-4 GND vias there.

### S4. Every power zone connects to pads through 0.6 mm thermal-relief spokes
- **Zones affected:** VBAT_P, VSERVO_SW, VSERVO (F.Cu), +5V and all GND pours, with 0.6 mm spokes. The VBAT_F and VBAT_IN zones have 1.2 mm spokes.
- **Pads affected:** Q1 and Q2 source pins 1-3 and drain tab, R6 force pads, D2, L1 and the C17-C20 output caps.
- **Why it matters for VSERVO_SW:** the Q2 → R6 path is effectively one 2 mm track plus a zone that is reached only through 4 × 0.6 mm spokes at each end. That is about 4.4 mm of effective 1 oz copper, at the edge of the 15-20 A target. It also adds heat to the FET and shunt pads.
- **Evidence:** every non-keepout zone has `connect_pads` with no `yes`, and `thermal_bridge_width` is 0.6 or 1.2.
- **Fix:** for the VBAT_IN, VBAT_F, VBAT_P, VSERVO_SW, VSERVO (F.Cu) and +5V zones, set Zone properties → Pad connections → **Solid**. Do the same for the F.Cu/B.Cu GND pour, or add pad-level overrides on the power pads: J1 pin 1, D1, D2, C6/C7, C1 and U2 EP. Refill and re-run DRC. Hand-solderability of THT pins in a solid plane is the only cost, and the XT60 and fuse holder can take it with a good iron.

---

## NOTES (checked; fine or worth knowing)

1. **CPL rotations** are raw KiCad angles. Check JLC's preview for each of these:
   - SOT-23 / SOT-23-5 / SOT-23-6 / SC-70-6 / TSOT parts (U1, U3, U4, U6, U8-U11, Q3-Q8), which are commonly off by 180°;
   - U5 SOIC-16, often off by ±90°;
   - U2 HSOP-8, Q1/Q2 TDSON-8, U7 (CPL 270°), J3 and J10;
   - polarity marks on D1-D9, DZ1/DZ2 and LED1-LED5.

   Every SMD part in the BOM is present in the CPL; I checked both directions with a set comparison.
2. **TPS54560B values check out:**
   - Vout = 0.8 × (1 + 53.6/10) = 5.09 V;
   - RT 191 k gives ≈ 507 kHz (RT = 101756 / f^1.008);
   - L: the calculation gives 4.0 µH for K_ind 0.3 at 12.6 V, so 4.7 µH is fine. Ripple is 1.3 A pk-pk and peak 5.6 A, below the 6.3 A minimum current limit;
   - BOOT 100 nF is correct, and the B540C (40 V / 5 A) has margin over the 24.4 V clamp;
   - EN divider 100 k / 18 k with Ip 1.2 µA and Ih 3.4 µA: start ≈ 7.75 V and stop ≈ 7.2 V (DESIGN says 7.9 / 7.6, close enough). EN at the clamp is ≈ 3.7 V, under the 8.4 V absolute maximum.
   - **Compensation:** 16 k / 3.9 nF / 39 pF gives about 27 kHz crossover with 88 µF nominal. With the real DC-bias derating of 0805 22 µF 25 V X5R parts (about 10 µF each at 5 V), crossover is about 55-60 kHz, which is above fsw/10. Use 1206/1210 22 µF parts, or add two more, and do the load step DESIGN already lists.
3. **Pi supply margin:** 5.09 V minus the drops across the VH, 8 cm of 20 AWG and the USB-C plug leaves about 4.85 V at 5 A. That is OK, but tight against the Pi 5's ≈ 4.63 V under-voltage flag. Consider R17 = 54.9 k (5.19 V).
4. **AP63203:**
   - Pinout and support parts check out: 1 FB = Vout (fixed part), 2 EN, 3 VIN, 4 GND, 5 SW, 6 BST, with BST 100 nF, 4.7 µH and 2 × 22 µF out.
   - **Layout:** C23's GND pad (193.5, 146.0) is on the opposite side of the package from GND pin 4 (198.1, 147.9). L2 is about 9 mm from SW. Cin loop and SW trace are longer than ideal for 1.1 MHz, but acceptable at ≤ 0.5 A. If you touch the layout, put a 100 nF from pin 3 to pin 4 directly.
5. **Current paths:**
   - VBAT_P feeds U2 through a 1 mm F.Cu track plus a 1 mm B.Cu track of about 55 mm (≈ 13 mΩ, about 3 A). That is fine.
   - The VSERVO feed into In2 is 9 × 0.8/0.4 mm vias near R6 (with 18 VSERVO vias in total), about 2 A each at 20 A. That is fine for bursts. Add 4-6 more vias if space allows.
   - Losses at 20 A are about 1.2 W each in Q1 and Q2 and 0.4 W in R6. Both FETs sit on 1 oz pours; that is acceptable for bursts. Sustained 15 A or more will run hot, so S4 helps.
6. **The TVS clamp vs the ceramic caps:** C39-C43 and C17-C20 are 25 V parts, and SMBJ15A clamps at up to 24.4 V at full Ipp. That is thin margin on the VSERVO caps. Prefer 35-50 V for C39-C43.
7. **The servo bus is correct:**
   - '126: OE = DIR, VCC = VIO. '125: OE# = DIR, VCC = 3V3, and its 5.5 V-tolerant input reads the bus.
   - DIR high means TX (receiver tri-stated, R40 holds RX idle high). DIR low means RX (driver Hi-Z, R41 pulls the bus to VIO). R42/R46 hold DIR low at reset.
   - TX goes to the '126 A input and the '125 Y output goes to RX.
   - The 22 µF caps (C39-C43) are on VSERVO-GND, not on SIG.
   - The RC on the line (22 + 33 Ω, about 300 pF of cable, ESD diodes and servos) is around 30 ns, so 1 Mbps is fine.
   - **Bring-up risk:** in the IDF RS485 half-duplex mode, RTS is dropped in the TX-done ISR, which adds a few to tens of µs. If the HX-30HM replies faster than that, the first reply byte is lost. Scope it during bring-up.
8. **Pin and connector checks (all OK):**
   - **Leg headers (J5-J9):** 1 GND, 2 VSERVO, 3 SIG, as DESIGN.md states. With VIN in the middle, a 180° mis-plug only swaps GND and SIG and never puts 12.6 V on SIG. Still meter one servo first.
   - **XT60 (J1):** the footprint's "−" silk is at pad 1 (GND) and "+" at pad 2 (VBAT_IN). Verify against the physical part before soldering.
9. **Pinouts verified against the datasheets (all OK):**
   - BSC030P03NS3 (1-3 S, 4 G, 5 D tab);
   - Q1 orientation: D = VBAT_F (battery side), S = VBAT_P, gate to GND via 10 k, 10 V zener S→G;
   - Q2: S = VBAT_P, D = VSERVO_SW, so the body diode blocks battery → rail when off;
   - TPS54560B DDA (1 BOOT, 2 VIN, 3 EN, 4 RT, 5 FB, 6 COMP, 7 GND, 8 SW, EP GND);
   - INA180 DBV (1 OUT, 2 GND, 3 IN+, 4 IN−, 5 VS);
   - LV1T125/126 DBV (1 OE, 2 A, 3 GND, 4 Y, 5 VCC);
   - 74LVC1G3157 DCK (1 B2, 2 GND, 3 B1, 4 A, 5 VCC, 6 S);
   - CH340C (4 V3 = VCC = 3V3 is correct for 3.3 V operation, 5 UD+ = USB_DP, 6 UD−);
   - USBLC6-2SC6;
   - USB-C (A6/B6 = D+, A7/B7 = D−, CC1/CC2 with 5.1 k each to GND);
   - WROOM-32E pin numbers;
   - the R6 Kelvin pads (pads 2/3 are sense, on the correct sides);
   - diode, LED and electrolytic polarities;
   - MMBT3904 / 2N7002 SOT-23 B/E/C and G/S/D.
10. **ESP32 and boot checks (all OK):**
    - **Straps:** IO0 has a 10 k pull-up; IO2 goes through an LED and 1 k to GND, so it is low; IO12 is NC (internal pull-down); IO5 and IO15 float.
    - **EN:** 10 k / 1 µF.
    - **Auto-reset:** the classic DevKitC two-transistor arrangement, with DTR → base of the EN transistor and RTS → base of the IO0 transistor, cross-connected emitters.
    - **IO34-39** are used only as ADC1 inputs, and no ADC2 pin is used for analog. IO16/IO17 are free on the N8 (no PSRAM).
    - **Antenna:** the module antenna spans KiCad x 225.5-231.7. Copper on all layers stops at x ≈ 225.5, under only the module body. About 2 mm of bare FR4 lies under the antenna before the board edge, and the rest overhangs. That is OK.
11. **Pi link:**
    - The J4 pin n = Pi pin n+4 mapping is correct.
    - When the Pi is powered off through Q4, ESP_TX0 (through R29) and PI_ATTN (through R38) still drive about 3 mA into the unpowered RP1 pins through 1 k. Firmware should hold those lines low or tri-stated while the Pi is off.
    - In the 2×5 socket on the Pi, pin 1 must sit on Pi pin 5. Mark the stripe.
