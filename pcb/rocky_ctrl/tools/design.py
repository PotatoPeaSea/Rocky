"""Rocky integrated control board: parts, nets and schematic grouping.

Single source of truth for gen_sch.py (schematic) and, through the netlist that
kicad-cli exports from that schematic, for gen_pcb.py (board).

Conventions
  - pins = {pin_number: net}. '' or None means no-connect (an X flag is drawn).
  - GND, +3V3 and +5V are drawn with power symbols; every other net uses labels.
  - lcsc = '' where the part number could not be verified offline (see DESIGN.md).
"""

PARTS = []
BLOCKS = {}


def block(name, title, x, y, w, h):
    BLOCKS[name] = dict(title=title, x=x, y=y, w=w, h=h)


def P(ref, lib, value, fp, pins, blk, lcsc='', mpn='', dnp=False, unit_pins=None, note=''):
    PARTS.append(dict(ref=ref, lib=lib, value=value, fp=fp, pins=pins, block=blk,
                      lcsc=lcsc, mpn=mpn or value, dnp=dnp, note=note))


# --------------------------------------------------------------------------- footprints
R0603 = 'Resistor_SMD:R_0603_1608Metric'
C0603 = 'Capacitor_SMD:C_0603_1608Metric'
C0805 = 'Capacitor_SMD:C_0805_2012Metric'
C1206 = 'Capacitor_SMD:C_1206_3216Metric'
LED0603 = 'LED_SMD:LED_0603_1608Metric'
SOT23 = 'Package_TO_SOT_SMD:SOT-23'
SOT235 = 'Package_TO_SOT_SMD:SOT-23-5'
TDSON = 'Package_TO_SOT_SMD:TDSON-8-1'
TP = 'TestPoint:TestPoint_Pad_D1.5mm'


def R(ref, val, a, b, blk, lcsc='', fp=R0603, mpn=''):
    P(ref, 'Device:R', val, fp, {'1': a, '2': b}, blk, lcsc, mpn or ('%s 1%% 0603' % val if fp == R0603 else val))


def C(ref, val, a, b, blk, lcsc='', fp=C0603, mpn=''):
    P(ref, 'Device:C', val, fp, {'1': a, '2': b}, blk, lcsc, mpn or val)


# Common JLC basic parts (LCSC numbers known for these values)
RL = {'10k': 'C25804', '100k': 'C25803', '1k': 'C21190', '5.1k': 'C23186', '4.7k': 'C23162',
      '22k': 'C31850', '22': 'C23345', '33': 'C23140', '470': 'C23179', '2.2k': 'C4190',
      '18k': '', '33k': 'C4216', '16k': '', '53.6k': '', '191k': ''}
CL = {'100n': 'C14663', '1u': 'C15849', '10n': 'C57112', '22p': 'C1653', '39p': '', '3.9n': ''}

# --------------------------------------------------------------------------- blocks (A1 sheet, mm)
block('pwr_in', 'Battery input: XT60, 20 A fuse, TVS, reverse-polarity P-FET', 20, 30, 250, 120)
block('servo_pwr', 'Servo rail: ESP32-switched P-FET (soft start), current shunt, bulk', 280, 30, 270, 120)
block('buck5', '5 V / 5 A buck for the Raspberry Pi 5 (TPS54560B)', 20, 160, 250, 140)
block('buck3', '3.3 V logic buck (diode-OR battery / USB)', 280, 160, 270, 140)
block('usb', 'USB-C, CH340C, auto-reset, UART0 source mux (USB wins)', 560, 30, 260, 160)
block('mcu', 'ESP32-WROOM-32E', 560, 195, 260, 165)
block('bus', 'Half-duplex servo buses A (UART1) and B (UART2)', 20, 310, 400, 150)
block('legs', 'Leg connectors (Molex SPOX 5267 = mates the servo 5264 cable)', 430, 370, 390, 90)
block('pi', 'Raspberry Pi 5 link (IDC 2x5 to Pi header pins 5-14)', 20, 470, 260, 110)
block('io', 'IMU (STEMMA QT), spare header, LEDs, sense dividers', 290, 470, 340, 110)
block('mech', 'Test points and mounting holes', 640, 470, 180, 110)

# =========================================================================== POWER INPUT
P('J1', 'Connector_Generic:Conn_01x02', 'XT60PW-M', 'Connector_AMASS:AMASS_XT60PW-M_1x02_P7.20mm_Horizontal',
  {'1': 'GND', '2': 'VBAT_IN'}, 'pwr_in', '', 'AMASS XT60PW-M',
  note='XT60 pin 1 = minus (left, flat side), pin 2 = plus; verify against the silk before soldering')
P('F1', 'Device:Fuse', '20A ATM', 'Fuse:Fuseholder_Blade_Mini_Keystone_3568',
  {'1': 'VBAT_IN', '2': 'VBAT_F'}, 'pwr_in', '', 'Keystone 3568 + 20 A mini blade fuse')
P('D1', 'Device:D_Zener', 'SMBJ15A', 'Diode_SMD:D_SMB', {'1': 'VBAT_F', '2': 'GND'}, 'pwr_in', '',
  'SMBJ15A (15 V stand-off unidirectional TVS)')
P('Q1', 'rocky_ctrl:Q_PMOS_SSSGD', 'BSC030P03NS3', TDSON,
  {'1': 'VBAT_P', '2': 'VBAT_P', '3': 'VBAT_P', '4': 'Q1_G', '5': 'VBAT_F'}, 'pwr_in', '',
  'Infineon BSC030P03NS3G (-30 V, 3.0 mOhm)')
R('R1', '10k', 'Q1_G', 'GND', 'pwr_in', RL['10k'])
P('DZ1', 'Device:D_Zener', 'BZT52C10', 'Diode_SMD:D_SOD-123', {'1': 'VBAT_P', '2': 'Q1_G'}, 'pwr_in', '',
  'BZT52C10 (10 V Vgs clamp)')
P('C1', 'Device:C_Polarized', '220u 35V', 'Capacitor_THT:CP_Radial_D8.0mm_P3.50mm',
  {'1': 'VBAT_P', '2': 'GND'}, 'pwr_in', '', 'Panasonic EEU-FR1V221 (220 uF 35 V, 8x11.5)')
C('C2', '10u 50V', 'VBAT_P', 'GND', 'pwr_in', '', C1206, 'GRM31CR61H106KA12 (10 uF 50 V X5R 1206)')
R('R11', '100k', 'VBAT_P', 'VBAT_SENSE', 'pwr_in', RL['100k'])
R('R12', '22k', 'VBAT_SENSE', 'GND', 'pwr_in', RL['22k'])
C('C10', '100n', 'VBAT_SENSE', 'GND', 'pwr_in', CL['100n'])

# =========================================================================== SERVO RAIL
P('Q2', 'rocky_ctrl:Q_PMOS_SSSGD', 'BSC030P03NS3', TDSON,
  {'1': 'VBAT_P', '2': 'VBAT_P', '3': 'VBAT_P', '4': 'Q2_G', '5': 'VSERVO_SW'}, 'servo_pwr', '',
  'Infineon BSC030P03NS3G (-30 V, 3.0 mOhm)')
R('R2', '100k', 'VBAT_P', 'Q2_G', 'servo_pwr', RL['100k'])
P('DZ2', 'Device:D_Zener', 'BZT52C10', 'Diode_SMD:D_SOD-123', {'1': 'VBAT_P', '2': 'Q2_G'}, 'servo_pwr', '',
  'BZT52C10 (10 V Vgs clamp)')
R('R3', '33k', 'Q2_G', 'Q3_C', 'servo_pwr', RL['33k'])
C('C3', '220n 50V', 'Q2_G', 'VSERVO_SW', 'servo_pwr', '', C0805, '220 nF 50 V X7R 0805 (Miller soft-start)')
P('Q3', 'Transistor_BJT:MMBT3904', 'MMBT3904', SOT23, {'1': 'Q3_B', '2': 'GND', '3': 'Q3_C'}, 'servo_pwr', 'C20526')
R('R4', '10k', 'SERVO_EN', 'Q3_B', 'servo_pwr', RL['10k'])
R('R5', '100k', 'Q3_B', 'GND', 'servo_pwr', RL['100k'])
P('R6', 'Device:R_Shunt', '1m', 'Resistor_SMD:R_Shunt_Vishay_WSK2512_6332Metric_T1.19mm',
  {'1': 'VSERVO_SW', '2': 'ISH_P', '3': 'ISH_N', '4': 'VSERVO'}, 'servo_pwr', '',
  'Vishay WSK25121L000FEA (1 mOhm 1 % 2512 4-terminal)')
P('U1', 'Amplifier_Current:INA180A3', 'INA180A3', SOT235,
  {'1': 'ISNS_RAW', '2': 'GND', '3': 'ISH_P', '4': 'ISH_N', '5': '+3V3'}, 'servo_pwr', '', 'INA180A3IDBVR (100 V/V)')
R('R7', '1k', 'ISNS_RAW', 'ISERVO', 'servo_pwr', RL['1k'])
C('C4', '10n', 'ISERVO', 'GND', 'servo_pwr', CL['10n'])
C('C5', '100n', '+3V3', 'GND', 'servo_pwr', CL['100n'])
for i, ref in enumerate(('C6', 'C7')):
    P(ref, 'Device:C_Polarized', '1000u 35V', 'Capacitor_THT:CP_Radial_D12.5mm_P5.00mm',
      {'1': 'VSERVO', '2': 'GND'}, 'servo_pwr', '', 'Panasonic EEU-FR1V102 (1000 uF 35 V, 12.5x20, 2.2 A ripple)')
C('C8', '10u 50V', 'VSERVO', 'GND', 'servo_pwr', '', C1206, 'GRM31CR61H106KA12 (10 uF 50 V X5R 1206)')
R('R9', '100k', 'VSERVO', 'VSRV_SENSE', 'servo_pwr', RL['100k'])
R('R10', '22k', 'VSRV_SENSE', 'GND', 'servo_pwr', RL['22k'])
C('C9', '100n', 'VSRV_SENSE', 'GND', 'servo_pwr', CL['100n'])

# =========================================================================== 5 V BUCK
P('U2', 'Regulator_Switching:TPS54560BDDA', 'TPS54560B', 'Package_SO:Texas_R-PDSO-G8_EP2.95x4.9mm_Mask2.4x3.1mm_ThermalVias',
  {'1': 'B5_BOOT', '2': 'VBAT_P', '3': 'PI_EN', '4': 'B5_RT', '5': 'B5_FB', '6': 'B5_COMP', '7': 'GND', '8': 'B5_SW', '9': 'GND'},
  'buck5', '', 'TPS54560BDDAR')
C('C11', '10u 50V', 'VBAT_P', 'GND', 'buck5', '', C1206, 'GRM31CR61H106KA12 (10 uF 50 V X5R 1206)')
C('C12', '10u 50V', 'VBAT_P', 'GND', 'buck5', '', C1206, 'GRM31CR61H106KA12 (10 uF 50 V X5R 1206)')
C('C13', '100n', 'VBAT_P', 'GND', 'buck5', CL['100n'])
R('R13', '100k', 'VBAT_P', 'PI_EN', 'buck5', RL['100k'])
R('R14', '18k', 'PI_EN', 'GND', 'buck5', RL['18k'])
P('Q4', 'Transistor_FET:2N7002', '2N7002', SOT23, {'1': 'PI_PWR_OFF', '2': 'GND', '3': 'PI_EN'}, 'buck5', 'C8545')
R('R15', '100k', 'PI_PWR_OFF', 'GND', 'buck5', RL['100k'])
R('R16', '191k', 'B5_RT', 'GND', 'buck5', RL['191k'])
C('C14', '100n', 'B5_BOOT', 'B5_SW', 'buck5', CL['100n'])
P('D2', 'Device:D_Schottky', 'B540C', 'Diode_SMD:D_SMC', {'1': 'B5_SW', '2': 'GND'}, 'buck5', '',
  'B540C-13-F (40 V 5 A Schottky, SMC)')
P('L1', 'Device:L', '4.7u', 'Inductor_SMD:L_Bourns_SRP1245A', {'1': 'B5_SW', '2': '+5V'}, 'buck5', '',
  'Bourns SRP1245A-4R7M (4.7 uH, Isat 21 A)')
R('R17', '53.6k', '+5V', 'B5_FB', 'buck5', RL['53.6k'])
R('R18', '10k', 'B5_FB', 'GND', 'buck5', RL['10k'])
R('R19', '16k', 'B5_COMP', 'B5_CZ', 'buck5', RL['16k'])
C('C15', '3.9n', 'B5_CZ', 'GND', 'buck5', CL['3.9n'])
C('C16', '39p', 'B5_COMP', 'GND', 'buck5', CL['39p'])
for ref in ('C17', 'C18', 'C19', 'C20'):
    C(ref, '22u', '+5V', 'GND', 'buck5', 'C45783', C0805, '22 uF 25 V X5R 0805')
P('J2', 'Connector_Generic:Conn_01x03', 'Pi 5V', 'Connector_JST:JST_VH_B3P-VH-B_1x03_P3.96mm_Vertical',
  {'1': '+5V', '2': 'GND', '3': 'PI_CC'}, 'buck5', '', 'JST B3P-VH-B (to USB-C pigtail into the Pi 5)')
R('R20', '10k', '+5V', 'PI_CC', 'buck5', RL['10k'])
R('R21', '22k', '+5V', 'V5_SENSE', 'buck5', RL['22k'])
R('R22', '10k', 'V5_SENSE', 'GND', 'buck5', RL['10k'])
C('C22', '100n', 'V5_SENSE', 'GND', 'buck5', CL['100n'])

# =========================================================================== 3V3 BUCK
P('D3', 'Device:D_Schottky', 'SS34', 'Diode_SMD:D_SMA', {'1': 'V_LOGIC', '2': 'VBAT_P'}, 'buck3', 'C8678')
P('D4', 'Device:D_Schottky', 'SS34', 'Diode_SMD:D_SMA', {'1': 'V_LOGIC', '2': 'USB_VBUS'}, 'buck3', 'C8678')
P('U3', 'Regulator_Switching:AP63203WU', 'AP63203WU', 'Package_TO_SOT_SMD:TSOT-23-6',
  {'1': '+3V3', '2': 'V_LOGIC', '3': 'V_LOGIC', '4': 'GND', '5': 'B3_SW', '6': 'B3_BST'}, 'buck3', '', 'AP63203WU-7')
C('C23', '10u 50V', 'V_LOGIC', 'GND', 'buck3', '', C1206, 'GRM31CR61H106KA12 (10 uF 50 V X5R 1206)')
C('C24', '100n', 'V_LOGIC', 'GND', 'buck3', CL['100n'])
C('C25', '100n', 'B3_BST', 'B3_SW', 'buck3', CL['100n'])
P('L2', 'Device:L', '4.7u', 'Inductor_SMD:L_Bourns_SRN6045TA', {'1': 'B3_SW', '2': '+3V3'}, 'buck3', '',
  'Bourns SRN6045TA-4R7M (4.7 uH 3.4 A)')
C('C26', '22u', '+3V3', 'GND', 'buck3', 'C45783', C0805, '22 uF 25 V X5R 0805')
C('C27', '22u', '+3V3', 'GND', 'buck3', 'C45783', C0805, '22 uF 25 V X5R 0805')

# =========================================================================== USB / UART0
P('J3', 'Connector:USB_C_Receptacle_USB2.0_16P', 'USB-C', 'Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12',
  {'A1': 'GND', 'A12': 'GND', 'B1': 'GND', 'B12': 'GND', 'A4': 'USB_VBUS', 'A9': 'USB_VBUS', 'B4': 'USB_VBUS',
   'B9': 'USB_VBUS', 'A5': 'USB_CC1', 'B5': 'USB_CC2', 'A7': 'USB_DN', 'B7': 'USB_DN', 'A6': 'USB_DP',
   'B6': 'USB_DP', 'A8': '', 'B8': '', 'S1': 'GND'}, 'usb', 'C165948', 'HRO TYPE-C-31-M-12')
R('R23', '5.1k', 'USB_CC1', 'GND', 'usb', RL['5.1k'])
R('R24', '5.1k', 'USB_CC2', 'GND', 'usb', RL['5.1k'])
P('U4', 'Power_Protection:USBLC6-2SC6', 'USBLC6-2SC6', 'Package_TO_SOT_SMD:SOT-23-6',
  {'1': 'USB_DP', '6': 'USB_DP', '3': 'USB_DN', '4': 'USB_DN', '5': 'USB_VBUS', '2': 'GND'}, 'usb', 'C7519')
P('U5', 'Interface_USB:CH340C', 'CH340C', 'Package_SO:SOIC-16_3.9x9.9mm_P1.27mm',
  {'1': 'GND', '2': 'CH_TXD', '3': 'ESP_TX0', '4': '+3V3', '5': 'USB_DP', '6': 'USB_DN', '7': '', '8': '',
   '9': '', '10': '', '11': '', '12': '', '13': 'DTR', '14': 'RTS', '15': 'GND', '16': '+3V3'}, 'usb', 'C84681')
C('C28', '100n', '+3V3', 'GND', 'usb', CL['100n'])
P('Q5', 'Transistor_BJT:MMBT3904', 'MMBT3904', SOT23, {'1': 'Q5_B', '2': 'RTS', '3': 'EN'}, 'usb', 'C20526')
P('Q6', 'Transistor_BJT:MMBT3904', 'MMBT3904', SOT23, {'1': 'Q6_B', '2': 'DTR', '3': 'BOOT'}, 'usb', 'C20526')
R('R25', '10k', 'DTR', 'Q5_B', 'usb', RL['10k'])
R('R26', '10k', 'RTS', 'Q6_B', 'usb', RL['10k'])
P('U6', '74xGxx:74LVC1G3157', '74LVC1G3157', 'Package_TO_SOT_SMD:SOT-363_SC-70-6',
  {'1': 'CH_TXD', '2': 'GND', '3': 'PI_TXD', '4': 'ESP_RX0', '5': '+3V3', '6': 'USB_PRES'}, 'usb', '',
  'SN74LVC1G3157DCKR (S=L: B1=Pi, S=H: B2=USB)')
C('C29', '100n', '+3V3', 'GND', 'usb', CL['100n'])
R('R27', '10k', 'USB_VBUS', 'USB_PRES', 'usb', RL['10k'])
R('R28', '22k', 'USB_PRES', 'GND', 'usb', RL['22k'])

# =========================================================================== MCU
P('U7', 'RF_Module:ESP32-WROOM-32E', 'ESP32-WROOM-32E-N8', 'RF_Module:ESP32-WROOM-32E',
  {'1': 'GND', '15': 'GND', '38': 'GND', '39': 'GND', '2': '+3V3', '3': 'EN',
   '4': 'V5_SENSE', '5': 'VSRV_SENSE', '6': 'VBAT_SENSE', '7': 'ISERVO', '8': 'PI_ATTN', '9': 'PI_PWR_OFF',
   '10': 'BUSB_TX', '11': 'BUSB_RX', '12': 'BUSB_DIR', '13': 'GPIO14', '14': '', '16': 'LED_USER',
   '17': '', '18': '', '19': '', '20': '', '21': '', '22': '', '32': '',
   '23': 'GPIO15', '24': 'LED_STAT', '25': 'BOOT', '26': 'BUSA_DIR', '27': 'BUSA_RX', '28': 'BUSA_TX',
   '29': 'GPIO5', '30': 'IMU_RST', '31': 'IMU_INT', '33': 'SDA', '34': 'ESP_RX0', '35': 'ESP_TX0',
   '36': 'SCL', '37': 'SERVO_EN'}, 'mcu', 'C701342', 'ESP32-WROOM-32E-N8 (8 MB)')
C('C30', '1u', 'EN', 'GND', 'mcu', CL['1u'])
R('R32', '10k', '+3V3', 'EN', 'mcu', RL['10k'])
R('R33', '10k', '+3V3', 'BOOT', 'mcu', RL['10k'])
C('C31', '22u', '+3V3', 'GND', 'mcu', 'C45783', C0805, '22 uF 25 V X5R 0805')
C('C32', '100n', '+3V3', 'GND', 'mcu', CL['100n'])
P('SW1', 'Switch:SW_Push', 'EN', 'Button_Switch_SMD:SW_Push_1P1T_XKB_TS-1187A', {'1': 'EN', '2': 'GND'}, 'mcu', 'C318884',
  'XKB TS-1187A-B-A-B')
P('SW2', 'Switch:SW_Push', 'BOOT', 'Button_Switch_SMD:SW_Push_1P1T_XKB_TS-1187A', {'1': 'BOOT', '2': 'GND'}, 'mcu', 'C318884',
  'XKB TS-1187A-B-A-B')
R('R34', '1k', 'LED_STAT', 'LED4_A', 'mcu', RL['1k'])
P('LED4', 'Device:LED', 'BLUE', LED0603, {'1': 'GND', '2': 'LED4_A'}, 'mcu', 'C72041')
R('R35', '1k', 'LED_USER', 'LED5_A', 'mcu', RL['1k'])
P('LED5', 'Device:LED', 'YELLOW', LED0603, {'1': 'GND', '2': 'LED5_A'}, 'mcu', 'C72038')

# =========================================================================== SERVO BUSES
P('JP1', 'Jumper:SolderJumper_3_Bridged12', 'VIO 3V3|5V', 'Jumper:SolderJumper-3_P1.3mm_Bridged12_RoundedPad1.0x1.5mm',
  {'1': '+3V3', '2': 'VIO', '3': '+5V'}, 'bus', '', 'solder jumper (default 3.3 V bus)')
C('C38', '1u', 'VIO', 'GND', 'bus', CL['1u'])
for bus, (tx, rx, dr, ut, ur, rs, rr, rp, rd, ct, cr) in {
        'A': ('BUSA_TX', 'BUSA_RX', 'BUSA_DIR', 'U8', 'U9', 'R39', 'R40', 'R41', 'R42', 'C34', 'C35'),
        'B': ('BUSB_TX', 'BUSB_RX', 'BUSB_DIR', 'U10', 'U11', 'R43', 'R44', 'R45', 'R46', 'C36', 'C37')}.items():
    P(ut, 'rocky_ctrl:SN74LV1T126DBV', 'SN74LV1T126', SOT235,
      {'1': dr, '2': tx, '3': 'GND', '4': 'BUS%s_Y' % bus, '5': 'VIO'}, 'bus', '', 'SN74LV1T126DBVR (TX driver, OE high)')
    P(ur, 'Logic_LevelTranslator:SN74LV1T125DBV', 'SN74LV1T125', SOT235,
      {'1': dr, '2': 'BUS' + bus, '3': 'GND', '4': rx, '5': '+3V3'}, 'bus', '', 'SN74LV1T125DBVR (RX buffer, OE low)')
    R(rs, '22', 'BUS%s_Y' % bus, 'BUS' + bus, 'bus', RL['22'])
    R(rr, '10k', rx, '+3V3', 'bus', RL['10k'])
    R(rp, '10k', 'BUS' + bus, 'VIO', 'bus', RL['10k'])
    R(rd, '10k', dr, 'GND', 'bus', RL['10k'])
    C(ct, '100n', 'VIO', 'GND', 'bus', CL['100n'])
    C(cr, '100n', '+3V3', 'GND', 'bus', CL['100n'])

LEG_BUS = {1: 'A', 2: 'A', 3: 'A', 4: 'B', 5: 'B'}
for leg in range(1, 6):
    sig = 'SIG_L%d' % leg
    P('J%d' % (4 + leg), 'Connector_Generic:Conn_01x03', 'LEG%d' % leg,
      'Connector_Molex:Molex_SPOX_5267-03A_1x03_P2.50mm_Vertical',
      {'1': 'GND', '2': 'VSERVO', '3': sig}, 'legs', '', 'Molex 5267-03A (or 2.50 mm "5264" 3P vertical header)')
    R('R%d' % (46 + leg), '33', 'BUS' + LEG_BUS[leg], sig, 'legs', RL['33'])
    P('D%d' % (4 + leg), 'Device:D_TVS', 'PESD5V0S1BB', 'Diode_SMD:D_SOD-323', {'1': sig, '2': 'GND'}, 'legs', '',
      'PESD5V0S1BB (bidirectional ESD, SOD-323)')
    C('C%d' % (38 + leg), '22u', 'VSERVO', 'GND', 'legs', '', C1206, '22 uF 25 V X5R 1206')

# =========================================================================== PI LINK
P('J4', 'Connector_Generic:Conn_02x05_Odd_Even', 'Pi GPIO 5-14', 'Connector_IDC:IDC-Header_2x05_P2.54mm_Vertical',
  {'1': '', '2': 'GND', '3': 'PI_GPIO4', '4': 'PI_TXD_RAW', '5': 'GND', '6': 'PI_RXD', '7': 'PI_RST', '8': '',
   '9': 'PI_BOOT', '10': 'GND'}, 'pi', '', '2x5 2.54 mm box header (ribbon to Pi pins 5-14)')
R('R29', '1k', 'ESP_TX0', 'PI_RXD', 'pi', RL['1k'])
R('R30', '1k', 'PI_TXD_RAW', 'PI_TXD', 'pi', RL['1k'])
R('R31', '100k', 'PI_TXD', '+3V3', 'pi', RL['100k'])
R('R38', '1k', 'PI_ATTN', 'PI_GPIO4', 'pi', RL['1k'])
P('Q7', 'Transistor_FET:2N7002', '2N7002', SOT23, {'1': 'PI_RST', '2': 'GND', '3': 'EN'}, 'pi', 'C8545')
P('Q8', 'Transistor_FET:2N7002', '2N7002', SOT23, {'1': 'PI_BOOT', '2': 'GND', '3': 'BOOT'}, 'pi', 'C8545')
R('R36', '100k', 'PI_RST', 'GND', 'pi', RL['100k'])
R('R37', '100k', 'PI_BOOT', 'GND', 'pi', RL['100k'])

# =========================================================================== IMU / IO / LEDs
P('J10', 'Connector_Generic_MountingPin:Conn_01x04_MountingPin', 'STEMMA QT', 'Connector_JST:JST_SH_SM04B-SRSS-TB_1x04-1MP_P1.00mm_Horizontal',
  {'1': 'GND', '2': '+3V3', '3': 'SDA', '4': 'SCL', 'MP': 'GND'}, 'io', 'C160390', 'JST SM04B-SRSS-TB (STEMMA QT to Adafruit BNO085)')
P('J11', 'Connector_Generic:Conn_01x02', 'IMU INT/RST', 'Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical',
  {'1': 'IMU_INT', '2': 'IMU_RST'}, 'io', '', '1x2 2.54 mm header')
R('R52', '4.7k', 'SDA', '+3V3', 'io', RL['4.7k'])
R('R53', '4.7k', 'SCL', '+3V3', 'io', RL['4.7k'])
R('R54', '10k', 'IMU_INT', '+3V3', 'io', RL['10k'])
P('J12', 'Connector_Generic:Conn_01x08', 'SPARE', 'Connector_PinHeader_2.54mm:PinHeader_1x08_P2.54mm_Vertical',
  {'1': '+3V3', '2': 'GND', '3': 'SDA', '4': 'SCL', '5': 'GPIO5', '6': 'GPIO14', '7': 'GPIO15', '8': '+5V'}, 'io', '',
  '1x8 2.54 mm header')
R('R55', '1k', '+3V3', 'LED1_A', 'io', RL['1k'])
P('LED1', 'Device:LED', 'GREEN 3V3', LED0603, {'1': 'GND', '2': 'LED1_A'}, 'io', 'C72043')
R('R56', '2.2k', '+5V', 'LED2_A', 'io', RL['2.2k'])
P('LED2', 'Device:LED', 'GREEN 5V', LED0603, {'1': 'GND', '2': 'LED2_A'}, 'io', 'C72043')
R('R8', '4.7k', 'VSERVO', 'LED3_A', 'io', RL['4.7k'])
P('LED3', 'Device:LED', 'RED SERVO', LED0603, {'1': 'GND', '2': 'LED3_A'}, 'io', 'C2286')

# =========================================================================== TEST POINTS / HOLES
for i, net in enumerate(['VBAT_F', 'VBAT_P', 'VSERVO', '+5V', '+3V3', 'GND', 'GND', 'BUSA', 'BUSB',
                         'V_LOGIC', 'ESP_TX0', 'ESP_RX0', 'ISERVO']):
    P('TP%d' % (i + 1), 'Connector:TestPoint', net.lstrip('+'), TP, {'1': net}, 'mech', '', 'test pad 1.5 mm')
# The BL bosses at (18, -63.5) and (-18, -63.5) are NOT used as screw points (XT60 body / 5 V output caps sit
# over them); the board still rests on them and no THT pin is placed over either boss.
HOLES = [('H3', 'M2.5', (-18.0, -35.5)),
         ('H4', 'M2.5', (18.0, -35.5)), ('H5', 'M2', (-30.8, 38.2)), ('H6', 'M2', (-30.8, 63.8)),
         ('H7', 'M2', (34.8, 38.2)), ('H8', 'M2', (34.8, 63.8)), ('H9', 'M2.5', (-58.46, 32.56)),
         ('H10', 'M2.5', (-58.46, 50.34)), ('H11', 'M2.5', (-38.14, 32.56)), ('H12', 'M2.5', (-38.14, 50.34))]
for ref, size, _ in HOLES:
    P(ref, 'Mechanical:MountingHole', size, 'MountingHole:MountingHole_2.7mm_M2.5' if size == 'M2.5'
      else 'MountingHole:MountingHole_2.2mm_M2', {}, 'mech', '', size + ' mounting hole')

# ESP32 GPIO map, for DESIGN.md
GPIO_MAP = [
    ('EN', 'EN', 'Reset: 10k/1u RC, EN button, CH340 auto-reset, Pi GPIO17 via 2N7002'),
    ('IO0', 'BOOT', 'Strap. BOOT button, CH340 auto-reset, Pi GPIO27 via 2N7002'),
    ('IO1 / TXD0', 'ESP_TX0', 'UART0 TX to CH340C RXD and (1k) Pi GPIO15/RXD'),
    ('IO3 / RXD0', 'ESP_RX0', 'UART0 RX from 74LVC1G3157 mux: CH340C when USB VBUS present, else Pi GPIO14/TXD'),
    ('IO2', 'LED_STAT', 'Strap (must float/low at boot; LED to GND is fine). Blue status LED'),
    ('IO4', 'BUSA_DIR', 'UART1 RTS (RS485 half-duplex mode): high = transmit on bus A'),
    ('IO16', 'BUSA_RX', 'UART1 RX (bus A: legs 1-3, servo IDs 1-9)'),
    ('IO17', 'BUSA_TX', 'UART1 TX'),
    ('IO25', 'BUSB_TX', 'UART2 TX (bus B: legs 4-5, servo IDs 10-15)'),
    ('IO26', 'BUSB_RX', 'UART2 RX'),
    ('IO27', 'BUSB_DIR', 'UART2 RTS: high = transmit on bus B'),
    ('IO23', 'SERVO_EN', 'Servo rail enable, active high, 100k pull-down on the switch side (off at reset)'),
    ('IO33', 'PI_PWR_OFF', 'High = 5 V Pi buck disabled (100k pull-down: Pi on by default)'),
    ('IO32', 'PI_ATTN', 'ESP32 -> Pi GPIO4 attention/IRQ line (1k series)'),
    ('IO21', 'SDA', 'I2C SDA (STEMMA QT IMU, spare header), 4.7k pull-up'),
    ('IO22', 'SCL', 'I2C SCL, 4.7k pull-up'),
    ('IO19', 'IMU_INT', 'BNO085 INT (10k pull-up)'),
    ('IO18', 'IMU_RST', 'BNO085 RST'),
    ('IO34', 'VBAT_SENSE', 'ADC1_CH6: battery, 100k/22k (12.6 V -> 2.27 V)'),
    ('IO35', 'ISERVO', 'ADC1_CH7: servo current, INA180A3 x 1 mOhm = 100 mV/A (0-24 A)'),
    ('IO36 / SENSOR_VP', 'V5_SENSE', 'ADC1_CH0: 5 V rail, 22k/10k (5.1 V -> 1.59 V)'),
    ('IO39 / SENSOR_VN', 'VSRV_SENSE', 'ADC1_CH3: switched servo rail, 100k/22k'),
    ('IO13', 'LED_USER', 'Yellow user LED'),
    ('IO5', 'GPIO5', 'Spare header (strap: keep high or floating at boot)'),
    ('IO14', 'GPIO14', 'Spare header'),
    ('IO15', 'GPIO15', 'Spare header (strap: boot log enable)'),
    ('IO12', '-', 'Not connected (strap MTDI must be low at boot for 3.3 V flash)'),
]
