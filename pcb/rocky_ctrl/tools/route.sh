#!/bin/bash
# Export DSN, autoroute signals with Freerouting (Plane/Manual net classes are left to pours / hand copper), import SES.
set -e
cd /home/claude/rocky/pcb/rocky_ctrl
python3 - <<'PY'
import pcbnew
b = pcbnew.LoadBoard('rocky_ctrl.kicad_pcb')
print('dsn export', pcbnew.ExportSpecctraDSN(b, '/tmp/rocky.dsn'))
PY
python3 - <<'PY'
import re
t = open('/tmp/rocky.dsn').read()
# inner layers are planes: no signal routing there
t = re.sub(r'(\(layer In[12]\.\w+\s*\(type) signal', r'\1 power', t)
# 20 um extra clearance margin for the router (KiCad DRC keeps the real rule)
t = re.sub(r'\(clearance (\d+)\)', lambda m: '(clearance %d)' % (int(m.group(1)) + 20), t)
open('/tmp/rocky.dsn', 'w').write(t)
PY
cd /tmp && rm -f rocky.ses
timeout ${TMAX:-1200} xvfb-run -a /usr/lib/jvm/java-21-openjdk-amd64/bin/java -jar /opt/fr/freerouting-1.9.0.jar -de /tmp/rocky.dsn -do /tmp/rocky.ses -mp ${PASSES:-40} -inc Plane,Manual -mt 1 -oit ${OIT:-0.2} > /tmp/fr.log 2>&1 || true
tail -5 /tmp/fr.log
ls -la /tmp/rocky.ses
