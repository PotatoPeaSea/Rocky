# Rocky pentapod – v5 design notes (Oct 2026)

Fusion document: "Pentapod Walker v5". Build script: `fusion/pentapod_build_v5.py`; vendor CAD in Documents/Pentapod/hardware (not in repo).

## Controls and power (v5, superseded by the integrated control PCB in v6)
- ESP32-DevKitC-32E drives a Hiwonder BusLinker V3.0 over TTL.
- Servos: 15 × HX-30HM, IDs 1-15, each leg daisy-chained hip → femur → knee.
- Power hub: 6 × WAGO 221-415 on a printed hex holder; 1000 µF bulk cap.
- Battery → 20 A inline ATM fuse → XT60 loop key → hub.
- LM2596 5 V for ESP32; Pololu D36V50F5 5 V for Pi 5 via right-angle USB-C pigtail.

## Logic deck (r82, z 52-55.5)
- Pi 5 + Active Cooler centre; ESP32 carrier back; BusLinker front; Pololu left; BNO085 back-left.

## Joint ranges
- Knee inward 75°, femur down 45°, hip yaw ±75°, femur up free at yaw 0°.

## Open items (v5)
- BusLinker TTL logic level, HX-30HM bus voltage and baud to measure.
