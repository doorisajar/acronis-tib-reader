"""
partitions.py — multi-partition + incremental-chain support for sector-mode .tib.

A disk-mode `.tib` written by True Image 2019 (and likely other 2017+
builds) can hold several partitions from several disks, and a backup can
be a chain of slices (`*_full_bN_s1_v1.tib`, `*_inc_bN_s2_v1.tib`, ...).

Slice chain
-----------
Every slice ends with a 32-byte byte-reversed copy of its volume header,
preceded by a table of `slice_no` entries `{u32 slice_id, u64 slice_size}`
covering slices 1..slice_no. Offsets inside the metadata and chunk maps
are "concat" offsets: position within the concatenation of every slice's
data area (slice k's data starts at file offset 32 of its own file).

Metadata blob
-------------
The trailer body's `metaDataOffset` points at the metadata blob. The blob
is a sequence of per-slice sections; each section is a `u16`-length
header TLV followed by `u32`-length-prefixed records (length includes
itself). Each record begins with a locator

    n V[n] 01 00 m S[m]        (V = concat offset, S = size, LE)

then TLV fields `tag, sub, len, value`. The LAST section describes the
latest slice and carries one record per disk and one per partition; a
partition record's locator points at that partition's chunk map (which
lists every block of the partition, pointing into whichever slice holds
the current copy).
"""
from __future__ import annotations

import bisect
import os
import struct
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .chunkmap import transpose, decode_records
from .chunkmap_locator import VOLUME_MAGIC, TRAILER_SECTOR, UnsupportedTibFormat

HEADER_LEN = 32


@dataclass
class SliceFile:
    path: str
    slice_no: int
    slice_id: int
    concat_start: int
    concat_len: int


@dataclass
class PartitionInfo:
    number: int                 # 1-based, in metadata order
    disk_number: int            # 1-based
    disk_model: str
    label: str
    letter: str
    start_sector: int
    sector_count: int
    chunkmap_offset: int        # concat offset of the chunk-map TLV preamble
    chunkmap_size: int
    fields: Dict[Tuple[int, int], bytes] = field(default_factory=dict, repr=False)

    @property
    def size_bytes(self) -> int:
        return self.sector_count * 512


def _read_header(path: str) -> Tuple[bytes, int, int]:
    """Returns (archive_id, slice_id, slice_no) or raises UnsupportedTibFormat."""
    with open(path, "rb") as f:
        h = f.read(HEADER_LEN)
    if len(h) < HEADER_LEN or struct.unpack_from("<I", h, 0)[0] != VOLUME_MAGIC:
        raise UnsupportedTibFormat(f"{path}: not a sector-mode .tib")
    slice_id, slice_no = struct.unpack_from("<II", h, 0x10)
    return h[8:16], slice_id, slice_no


def _read_slice_table(path: str, slice_no: int) -> List[Tuple[int, int]]:
    size = os.path.getsize(path)
    n = 12 * slice_no
    with open(path, "rb") as f:
        f.seek(size - HEADER_LEN - n)
        buf = f.read(n)
    return [struct.unpack_from("<IQ", buf, 12 * i) for i in range(slice_no)]


def discover_chain(tib_path: str) -> List[SliceFile]:
    """Return the slices (oldest first) that `tib_path` depends on,
    ending with `tib_path` itself. Sibling slices are located by matching
    the archive id + slice id in each `.tib` header in the same directory."""
    tib_path = os.path.abspath(tib_path)
    archive_id, slice_id, slice_no = _read_header(tib_path)
    if slice_no < 1 or slice_no > 100000:
        raise UnsupportedTibFormat(f"{tib_path}: implausible slice number {slice_no}")
    table = _read_slice_table(tib_path, slice_no)
    if table[-1][0] != slice_id:
        raise UnsupportedTibFormat(
            f"{tib_path}: slice table does not end with this slice's id "
            f"({table[-1][0]:#x} != {slice_id:#x})"
        )

    by_id: Dict[int, str] = {slice_id: tib_path}
    if slice_no > 1:
        d = os.path.dirname(tib_path)
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if not name.lower().endswith(".tib") or p == tib_path or not os.path.isfile(p):
                continue
            try:
                aid, sid, _ = _read_header(p)
            except (UnsupportedTibFormat, OSError):
                continue
            if aid == archive_id:
                by_id.setdefault(sid, p)

    chain: List[SliceFile] = []
    start = 0
    missing = []
    for i, (sid, ssize) in enumerate(table):
        p = by_id.get(sid)
        if p is None:
            missing.append(i + 1)
        chain.append(SliceFile(p or "", i + 1, sid, start, ssize))
        start += ssize
    if missing:
        raise UnsupportedTibFormat(
            f"{os.path.basename(tib_path)} is slice {slice_no} of an incremental "
            f"chain; slice(s) {missing} were not found next to it. Put every "
            f"slice of the backup in the same directory."
        )
    return chain


class ChainReader:
    """Read bytes by concat offset across a slice chain."""

    def __init__(self, chain: List[SliceFile]):
        self.chain = chain
        self._starts = [s.concat_start for s in chain]
        self._files: Dict[int, object] = {}

    def read(self, off: int, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            k = bisect.bisect_right(self._starts, off) - 1
            if k < 0 or off >= self.chain[k].concat_start + self.chain[k].concat_len:
                raise ValueError(f"concat offset {off} outside the slice chain")
            s = self.chain[k]
            f = self._files.get(k)
            if f is None:
                f = self._files[k] = open(s.path, "rb")
            take = min(n, s.concat_start + s.concat_len - off)
            f.seek(HEADER_LEN + off - s.concat_start)
            data = f.read(take)
            if len(data) != take:
                raise ValueError(f"short read in {s.path}")
            out += data
            off += take
            n -= take
        return bytes(out)

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()


def _parse_tlv(buf: bytes) -> Dict[Tuple[int, int], bytes]:
    out: Dict[Tuple[int, int], bytes] = {}
    i = 0
    while i + 3 <= len(buf):
        tag, sub, ln = buf[i], buf[i + 1], buf[i + 2]
        i += 3
        if ln & 0x80:
            ln = ((ln & 0x7F) << 8) | buf[i]
            i += 1
        out.setdefault((tag, sub), buf[i:i + ln])
        i += ln
    return out


def _parse_locator(blob: bytes, i: int, end: int) -> Optional[Tuple[int, int, int]]:
    """Parse `n V[n] 01 00 m S[m]` at blob[i]. Returns (V, S, next_pos) or None."""
    if i >= end:
        return None
    n = blob[i]
    if not 1 <= n <= 8 or i + 1 + n + 3 > end or blob[i + 1 + n:i + 3 + n] != b"\x01\x00":
        return None
    m = blob[i + 3 + n]
    if not 1 <= m <= 8 or i + 4 + n + m > end:
        return None
    v = int.from_bytes(blob[i + 1:i + 1 + n], "little")
    s = int.from_bytes(blob[i + 4 + n:i + 4 + n + m], "little")
    return v, s, i + 4 + n + m


def _metadata_sections(reader: ChainReader, last: SliceFile):
    """Yield lists of (V, S, fields) records, one list per metadata section."""
    end = last.concat_start + last.concat_len
    tail = reader.read(end - 8, 8)
    if tail[4:] != TRAILER_SECTOR:
        raise UnsupportedTibFormat(
            f"unrecognized trailer magic {tail[4:].hex()} in {last.path}"
        )
    ts = struct.unpack_from("<I", tail, 0)[0]
    body = reader.read(end - 8 - ts, ts)
    n = body[2]
    meta_off = int.from_bytes(body[3:3 + n], "little")
    blob_len = end - 8 - ts - meta_off
    if not 0 < blob_len < (64 << 20):
        raise UnsupportedTibFormat(f"implausible metadata size {blob_len}")
    blob = reader.read(meta_off, blob_len)

    sections: List[list] = []
    i = 0
    while i < len(blob):
        if i + 4 <= len(blob):
            ln = struct.unpack_from("<I", blob, i)[0]
            if 8 <= ln and i + ln <= len(blob) and sections:
                loc = _parse_locator(blob, i + 4, i + ln)
                if loc is not None:
                    v, s, p = loc
                    sections[-1].append((v, s, _parse_tlv(blob[p:i + ln])))
                    i += ln
                    continue
        hl = struct.unpack_from("<H", blob, i)[0] if i + 2 <= len(blob) else 0
        if hl < 3 or i + hl > len(blob):
            raise UnsupportedTibFormat(f"cannot parse metadata blob at +{i}")
        sections.append([])
        i += hl
    return sections


def _utf16(b: bytes) -> str:
    if len(b) % 2:
        b += b"\x00"
    return b.decode("utf-16-le", "replace").rstrip("\x00")


def list_partitions(tib_path: str) -> Tuple[List[SliceFile], List[PartitionInfo]]:
    """Return (slice chain, partitions described by the latest slice)."""
    chain = discover_chain(tib_path)
    reader = ChainReader(chain)
    try:
        sections = _metadata_sections(reader, chain[-1])
    finally:
        reader.close()
    if not sections:
        return chain, []

    parts: List[PartitionInfo] = []
    disk_no, disk_model = 0, ""
    for v, s, f in sections[-1]:
        if (0x48, 0) in f:
            disk_no += 1
            disk_model = f.get((0x58, 0), b"").decode("ascii", "replace").strip()
            continue
        if (0x11, 0) not in f or (0x12, 0) not in f:
            continue
        parts.append(PartitionInfo(
            number=len(parts) + 1,
            disk_number=disk_no,
            disk_model=disk_model,
            label=_utf16(f.get((0xCB, 0), b"")),
            letter=f.get((0x6B, 0), b"").decode("ascii", "replace"),
            start_sector=int.from_bytes(f[(0x11, 0)], "little"),
            sector_count=int.from_bytes(f[(0x12, 0)], "little"),
            chunkmap_offset=v,
            chunkmap_size=s,
            fields=f,
        ))
    return chain, parts


def format_partition_table(chain: List[SliceFile], parts: List[PartitionInfo]) -> str:
    lines = []
    if len(chain) > 1:
        lines.append(f"backup chain: {len(chain)} slices (showing state as of the last one)")
        for s in chain:
            lines.append(f"  slice {s.slice_no}: {os.path.basename(s.path)}")
    lines.append(f"{'#':>3}  {'disk':<40} {'label':<16} {'letter':<6} {'size':>10}")
    for p in parts:
        disk = f"{p.disk_number}: {p.disk_model}"[:40]
        lines.append(
            f"{p.number:>3}  {disk:<40} {p.label or '-':<16} "
            f"{(p.letter + ':') if p.letter else '-':<6} "
            f"{p.size_bytes / 1024**3:>7.1f} GiB"
        )
    return "\n".join(lines)


def decode_partition_chunkmap(chain: List[SliceFile], part: PartitionInfo):
    """Returns (records, clusters_per_block). records[pb] = (concat_offset, length);
    length 0 = block not stored (reads as zeros)."""
    reader = ChainReader(chain)
    try:
        raw = reader.read(part.chunkmap_offset, part.chunkmap_size)
    finally:
        reader.close()
    hl = raw[0]
    hdr = _parse_tlv(raw[1:1 + hl])
    cpb = int.from_bytes(hdr.get((4, 0), b""), "little")
    spc = int.from_bytes(hdr.get((3, 0), b""), "little")
    count = int.from_bytes(hdr.get((6, 0), b""), "little")
    if cpb <= 0 or cpb % 8:
        raise UnsupportedTibFormat(f"partition {part.number}: bad clusters-per-block {cpb}")
    if spc != 8:
        raise UnsupportedTibFormat(
            f"partition {part.number}: {spc} sectors per cluster; only 4 KiB "
            f"clusters are supported"
        )
    plain = zlib.decompress(raw[1 + hl:])
    if len(plain) % 12 or len(plain) // 12 != count:
        raise ValueError(
            f"partition {part.number}: chunk map has {len(plain)} bytes, "
            f"expected {count} x 12"
        )
    records = decode_records(bytes(transpose(bytearray(plain), count, 12)), count)
    return records, cpb
