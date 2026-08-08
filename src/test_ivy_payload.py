"""Geometry check for the Canon Ivy print payload. Run: python src/test_ivy_payload.py

No hardware, no framework -- just the scale/crop/paste math, which is the part
that silently ruins a print if it's wrong.
"""

from io import BytesIO

from PIL import Image

from server import IVY_H, IVY_VIS_H, IVY_W, ivy_payload


def main():
    # A 1600x900 RGBA image, matching the real canvas backing store, with a
    # transparent border so the alpha-compositing path is exercised too.
    src = Image.new("RGBA", (1600, 900), (0, 0, 0, 0))
    src.paste((255, 0, 0, 255), (100, 100, 1500, 800))
    buf = BytesIO()
    src.save(buf, format="PNG")
    buf.seek(0)

    out = Image.open(BytesIO(ivy_payload(buf)))

    assert out.size == (IVY_W, IVY_H), f"expected {(IVY_W, IVY_H)}, got {out.size}"

    # Centre of the visible band: art landed there.
    mid = out.getpixel((IVY_W // 2, IVY_H // 2))
    assert mid != (255, 255, 255), f"centre is blank white: {mid}"

    # Well inside the ZINK margin: must stay clear, or the art is off-centre and
    # the top of the print gets eaten.
    margin_y = (IVY_H - IVY_VIS_H) // 4
    edge = out.getpixel((IVY_W // 2, margin_y))
    assert edge == (255, 255, 255), f"margin at y={margin_y} is not white: {edge}"

    print(f"ok: {out.size}, band filled, margins clear")


if __name__ == "__main__":
    main()
