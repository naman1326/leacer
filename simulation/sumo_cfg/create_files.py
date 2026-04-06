import os

os.chdir(r"C:\Users\ghoda\Downloads\leacer\simulation\sumo_cfg")

nod = """<?xml version="1.0" encoding="UTF-8"?>
<nodes>
  <node id="N00" x="0"    y="1000" type="traffic_light"/>
  <node id="N01" x="500"  y="1000" type="traffic_light"/>
  <node id="N02" x="1000" y="1000" type="traffic_light"/>
  <node id="N10" x="0"    y="500"  type="traffic_light"/>
  <node id="N11" x="500"  y="500"  type="traffic_light"/>
  <node id="N12" x="1000" y="500"  type="traffic_light"/>
  <node id="N20" x="0"    y="0"    type="traffic_light"/>
  <node id="N21" x="500"  y="0"    type="traffic_light"/>
  <node id="N22" x="1000" y="0"    type="traffic_light"/>
  <node id="IN_W" x="-200"  y="500"  type="dead_end"/>
  <node id="IN_E" x="1200"  y="500"  type="dead_end"/>
  <node id="IN_N" x="500"   y="1200" type="dead_end"/>
  <node id="IN_S" x="500"   y="-200" type="dead_end"/>
</nodes>"""

edg = """<?xml version="1.0" encoding="UTF-8"?>
<edges>
  <edge id="E00_01" from="N00" to="N01" numLanes="2" speed="16.67" priority="2"/>
  <edge id="E01_00" from="N01" to="N00" numLanes="2" speed="16.67" priority="2"/>
  <edge id="E01_02" from="N01" to="N02" numLanes="2" speed="16.67" priority="2"/>
  <edge id="E02_01" from="N02" to="N01" numLanes="2" speed="16.67" priority="2"/>
  <edge id="E10_11" from="N10" to="N11" numLanes="3" speed="16.67" priority="3"/>
  <edge id="E11_10" from="N11" to="N10" numLanes="3" speed="16.67" priority="3"/>
  <edge id="E11_12" from="N11" to="N12" numLanes="3" speed="16.67" priority="3"/>
  <edge id="E12_11" from="N12" to="N11" numLanes="3" speed="16.67" priority="3"/>
  <edge id="E20_21" from="N20" to="N21" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E21_20" from="N21" to="N20" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E21_22" from="N21" to="N22" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E22_21" from="N22" to="N21" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E00_10" from="N00" to="N10" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E10_00" from="N10" to="N00" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E10_20" from="N10" to="N20" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E20_10" from="N20" to="N10" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E01_11" from="N01" to="N11" numLanes="3" speed="16.67" priority="3"/>
  <edge id="E11_01" from="N11" to="N01" numLanes="3" speed="16.67" priority="3"/>
  <edge id="E11_21" from="N11" to="N21" numLanes="3" speed="16.67" priority="3"/>
  <edge id="E21_11" from="N21" to="N11" numLanes="3" speed="16.67" priority="3"/>
  <edge id="E02_12" from="N02" to="N12" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E12_02" from="N12" to="N02" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E12_22" from="N12" to="N22" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E22_12" from="N22" to="N12" numLanes="2" speed="13.89" priority="2"/>
  <edge id="E_IN_W"  from="IN_W" to="N10" numLanes="2" speed="16.67" priority="1"/>
  <edge id="E_OUT_W" from="N10" to="IN_W" numLanes="2" speed="16.67" priority="1"/>
  <edge id="E_IN_E"  from="IN_E" to="N12" numLanes="2" speed="16.67" priority="1"/>
  <edge id="E_OUT_E" from="N12" to="IN_E" numLanes="2" speed="16.67" priority="1"/>
  <edge id="E_IN_N"  from="IN_N" to="N01" numLanes="2" speed="16.67" priority="1"/>
  <edge id="E_OUT_N" from="N01" to="IN_N" numLanes="2" speed="16.67" priority="1"/>
  <edge id="E_IN_S"  from="IN_S" to="N21" numLanes="2" speed="16.67" priority="1"/>
  <edge id="E_OUT_S" from="N21" to="IN_S" numLanes="2" speed="16.67" priority="1"/>
</edges>"""

cfg = """<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="leacer_network.net.xml"/>
    <route-files value="leacer_routes.rou.xml"/>
    <additional-files value="leacer_detectors.add.xml"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="3600"/>
    <step-length value="1.0"/>
  </time>
  <processing>
    <ignore-route-errors value="true"/>
    <time-to-teleport value="-1"/>
    <waiting-time-memory value="100"/>
    <max-depart-delay value="60"/>
  </processing>
  <output>
    <summary-output value="../results/sumo_summary.xml"/>
    <tripinfo-output value="../results/sumo_tripinfo.xml"/>
    <queue-output value="../results/sumo_queues.xml"/>
  </output>
  <random>
    <seed value="42"/>
  </random>
  <report>
    <no-step-log value="true"/>
    <duration-log.statistics value="true"/>
  </report>
</configuration>"""

edges = [
    "E00_01","E01_02","E10_11","E11_12","E20_21","E21_22",
    "E00_10","E10_20","E01_11","E11_21","E02_12","E12_22",
    "E01_00","E02_01","E11_10","E12_11","E21_20","E22_21",
    "E10_00","E20_10","E11_01","E21_11","E12_02","E22_12"
]
det_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<additionals>']
for e in edges:
    det_lines.append(
        f'  <inductionLoop id="det_{e}" lane="{e}_0" pos="250" '
        f'freq="60" file="../results/sumo_lanedata.xml" friendlyPos="true"/>'
    )
det_lines += [
    '  <tlLogic id="N11" type="static" programID="leacer_tl" offset="0">',
    '    <phase duration="32" state="GGrrGGrr"/>',
    '    <phase duration="5"  state="yyrryyrr"/>',
    '    <phase duration="32" state="rrGGrrGG"/>',
    '    <phase duration="5"  state="rryyrryy"/>',
    '  </tlLogic>',
    '</additionals>'
]
det = "\n".join(det_lines)

with open("leacer_network.nod.xml", "w", encoding="utf-8") as f:
    f.write(nod)
print("1. leacer_network.nod.xml  OK")

with open("leacer_network.edg.xml", "w", encoding="utf-8") as f:
    f.write(edg)
print("2. leacer_network.edg.xml  OK")

with open("leacer.sumocfg", "w", encoding="utf-8") as f:
    f.write(cfg)
print("3. leacer.sumocfg          OK")

with open("leacer_detectors.add.xml", "w", encoding="utf-8") as f:
    f.write(det)
print("4. leacer_detectors.add.xml OK")

print("\nAll 4 files created successfully!")