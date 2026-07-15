"""Decode base64 artifact tarball from a colab-run log file.

Usage: python scripts/decode_artifact.py <log_file> <output.tar.gz>
"""
import sys

log = sys.argv[1]
out = sys.argv[2]

lines = open(log, "r").read().splitlines()
start = None
for i, line in enumerate(lines):
    if line.strip() == "ARTIFACT_TAR_GZ_B64_BEGIN":
        start = i + 1
    if start is not None and line.strip() == "ARTIFACT_TAR_GZ_B64_END":
        end = i
        break
else:
    print("ERROR: artifact markers not found in log")
    sys.exit(1)

import base64
data = base64.b64decode("".join(lines[start:end]))
with open(out, "wb") as f:
    f.write(data)
print(f"Wrote {len(data)/1024**2:.1f} MB -> {out}")
