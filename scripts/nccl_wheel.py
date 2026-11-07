#!/usr/bin/env python
"""Package the experimental +nccl wheel.

Takes the plain CUDA wheel in a dist directory, adds nccl.dll next to
lib_lightgbm.dll (CPython >= 3.8 ctypes resolves a DLL's dependencies from
its own directory, not from PATH), re-tags the version with a ``+nccl``
local segment (consistent with the fork's ``+all`` tags) and regenerates
RECORD.

Usage:
    NCCL_DLL=<path>/nccl.dll python scripts/nccl_wheel.py <dist-dir>
"""
import base64
import csv
import hashlib
import io
import os
import re
import sys
import zipfile


def rvh(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def main() -> int:
    dist = sys.argv[1] if len(sys.argv) > 1 else "dist"
    nccl_dll = os.environ["NCCL_DLL"]
    whls = [f for f in os.listdir(dist) if f.endswith(".whl")]
    assert len(whls) == 1, f"expected exactly one wheel in {dist}, got {whls}"
    src = os.path.join(dist, whls[0])
    print("source wheel:", whls[0])

    zin = zipfile.ZipFile(src)
    names = zin.namelist()
    libnames = [n for n in names if n.endswith("lib_lightgbm.dll")]
    assert len(libnames) == 1, f"lib_lightgbm.dll not unique: {libnames}"
    libdir = libnames[0].rsplit("/", 1)[0]
    target_nccl = f"{libdir}/nccl.dll"
    print("lib dir:", libdir)

    old_stem = os.path.basename(src).split("-py3-")[0]
    version = old_stem.split("-", 1)[1]
    assert "+" not in version, f"source wheel already has a local version tag: {version}"
    new_version = version + "+nccl"
    old_di = [n for n in names if ".dist-info/" in n][0].split("/")[0]
    new_di = old_di.replace(f"-{version}.dist-info", f"-{new_version}.dist-info")
    new_name = os.path.basename(src).replace(f"-{version}-", f"-{new_version}-")
    print(f"retag: {version} -> {new_version}")

    nccl_bytes = open(nccl_dll, "rb").read()
    out_path = os.path.join(dist, new_name)
    zout = zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED)
    record_rows = []
    for item in zin.infolist():
        data = zin.read(item.filename)
        name = item.filename
        if name == f"{old_di}/METADATA":
            data = re.sub(rb"^Version: .*$", f"Version: {new_version}".encode(), data, count=1, flags=re.M)
            name = f"{new_di}/METADATA"
        elif name == f"{old_di}/WHEEL":
            name = f"{new_di}/WHEEL"
        elif name == f"{old_di}/RECORD":
            continue  # regenerated last
        elif name.startswith(old_di):
            name = name.replace(old_di, new_di, 1)
        item.filename = name
        zout.writestr(item, data)
        record_rows.append([name, rvh(data), str(len(data))])
    zout.writestr(target_nccl, nccl_bytes)
    record_rows.append([target_nccl, rvh(nccl_bytes), str(len(nccl_bytes))])
    record_name = f"{new_di}/RECORD"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerows(record_rows)
    w.writerow([record_name, "", ""])
    zout.writestr(record_name, buf.getvalue())
    zout.close()
    zin.close()
    os.remove(src)
    print("written:", new_name, f"({os.path.getsize(out_path)/1e6:.1f} MB, nccl.dll {len(nccl_bytes)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
