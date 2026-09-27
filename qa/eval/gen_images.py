"""Generate tiny deterministic PNGs (stdlib only) for the vision corpus."""
import struct
import zlib


def png(path, w, h, px):
    def chunk(typ, data):
        out = struct.pack(">I", len(data)) + typ + data
        return out + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"".join(bytes(px(x, y)) for x in range(w)) for y in range(h))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(data)


RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 200, 0)
WHITE = (255, 255, 255)

png("images/red-square.png", 64, 64, lambda x, y: RED if 16 <= x < 48 and 16 <= y < 48 else WHITE)
png(
    "images/shapes.png",
    96,
    64,
    lambda x, y: BLUE
    if (x - 24) ** 2 + (y - 32) ** 2 < 200
    else (GREEN if 56 <= x < 88 and 16 <= y < 48 else WHITE),
)
png("images/gradient.png", 64, 64, lambda x, y: (x * 4 % 256, y * 4 % 256, 128))
print("wrote 3 images")
