"""Read TFRecord Event/Summary simple_value scalars without external dependencies.
Validates both TFRecord CRC32C checksums. Rejects malformed/truncated records.
Tensor-valued summaries are counted separately and not treated as scalars.
"""
import csv
import json
import struct
import sys
from pathlib import Path


def varint(buf, pos):
    value = 0
    for shift in range(0, 70, 7):
        byte = buf[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError("Bad varint")


def fields(buf):
    pos = 0
    while pos < len(buf):
        key, pos = varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = varint(buf, pos)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = buf[pos : pos + size]
            pos += size
            assert len(value) == size
        elif wire == 2:
            size, pos = varint(buf, pos)
            value = buf[pos : pos + size]
            pos += size
            assert len(value) == size
        else:
            raise ValueError(f"Unsupported wire {wire}")
        yield field, wire, value


table = []
for x in range(256):
    for _ in range(8):
        x = (x >> 1) ^ (0x82F63B78 if x & 1 else 0)
    table.append(x)


def crc32c(b):
    c = 0xFFFFFFFF
    for x in b:
        c = table[(c ^ x) & 255] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def masked_crc(b):
    c = crc32c(b)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


assert crc32c(b"123456789") == 0xE3069283

src, out = map(Path, sys.argv[1:])
rows = []
records = tensor_values = 0
with src.open("rb") as f:
    while header := f.read(12):
        assert len(header) == 12
        size, crc = struct.unpack("<QI", header)
        assert masked_crc(header[:8]) == crc
        buf = f.read(size)
        footer = f.read(4)
        assert len(buf) == size and len(footer) == 4
        assert masked_crc(buf) == struct.unpack("<I", footer)[0]
        records += 1
        event = {k: v for k, w, v in fields(buf)}
        step = event.get(2, 0)
        if 5 not in event:
            continue
        for k, w, value in fields(event[5]):
            if k != 1:
                continue
            summary = {k: v for k, w, v in fields(value)}
            if 2 in summary:
                rows.append((step, summary[1].decode(), struct.unpack("<f", summary[2])[0]))
            elif 8 in summary:
                tensor_values += 1
out.mkdir(parents=True, exist_ok=True)
with (out / "scalars.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "tag", "value"])
    writer.writerows(rows)
by_tag = {}
for step, tag, value in rows:
    by_tag.setdefault(tag, []).append((step, value))
report = {"records": records, "scalar_count": len(rows), "tensor_summaries_skipped": tensor_values, "tags": {}}
for tag, vals in by_tag.items():
    if "vicregl" in tag or tag.startswith(("Loss/", "Recon/Loss", "Weighted/recon", "Perf/nan")):
        report["tags"][tag] = {"n": len(vals), "first": vals[0], "last": vals[-1], "windows": {}}
        for name, low, high in [("early", 0, 1000), ("middle", 9000, 10000), ("late", 32000, 34001)]:
            window = [v for s, v in vals if low < s <= high]
            if window:
                report["tags"][tag]["windows"][name] = {
                    "n": len(window),
                    "mean": sum(window) / len(window),
                    "min": min(window),
                    "max": max(window),
                }
(out / "summary.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
