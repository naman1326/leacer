import os, sys, subprocess
from pathlib import Path

SIM_DIR = Path(__file__).parent.parent
MOST_DIR = SIM_DIR / "most_scenario"
IN_DIR = MOST_DIR / "in"
ROUTE_DIR = IN_DIR / "route"
ADD_DIR = IN_DIR / "add"

SUMO_TOOLS = Path(r"C:\Users\ghoda\Downloads\sumo-1.27.1\tools")
RANDOM_TRIPS = SUMO_TOOLS / "randomTrips.py"

net_file = str(IN_DIR / "most.net.xml")
vtype_file = str(ADD_DIR / "basic.vType.xml")
trips_file = str(ROUTE_DIR / "most.passenger.trips.xml")
rou_file = str(ROUTE_DIR / "most.passenger.rou.xml")

print("=" * 65)
print("MoST Scenario — Generating High-Density AM Rush-Hour Demand")
print("=" * 65)
print(f"Network : {net_file}")
print(f"Window  : 25200s - 32400s (07:00 - 09:00, 2h rush hour)")

cmd = [
    sys.executable,
    str(RANDOM_TRIPS),
    "-n", net_file,
    "-a", vtype_file,
    "-b", "25200",
    "-e", "32400",
    "-p", "0.85",
    "--min-distance", "500",
    "--fringe-factor", "4",
    "-t", 'type="passenger" departLane="best" departSpeed="max"',
    "--validate",
    "-o", trips_file,
    "-r", rou_file,
]

env = os.environ.copy()
env["SUMO_HOME"] = r"C:\Users\ghoda\Downloads\sumo-1.27.1"
env["PATH"] = rf"C:\Users\ghoda\Downloads\sumo-1.27.1\bin;{env.get('PATH', '')}"

print("\nRunning DUAROUTER / randomTrips...")
res = subprocess.run(cmd, capture_output=True, text=True, env=env)
print("STDOUT:", res.stdout)
if res.stderr:
    print("STDERR:", res.stderr[-500:])

if res.returncode == 0 and os.path.exists(rou_file):
    import xml.etree.ElementTree as ET
    tree = ET.parse(rou_file)
    root = tree.getroot()
    count = len(root.findall("vehicle"))
    print(f"\n[SUCCESS] Generated {count} validated passenger routes -> {rou_file}")
else:
    print(f"\n[ERROR] Generation failed with return code {res.returncode}")
