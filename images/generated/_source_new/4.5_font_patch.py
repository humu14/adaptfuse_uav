"""Replace the monospace pseudo-code in Fig 4.5 with the figure's own sans font."""
from PIL import Image, ImageDraw, ImageFont

SRC = "E:/Research Projects/reserch writing/Co-Sup/images/generated/4.5.png"
OUT = "4.5_patched.png"
F = ImageFont.truetype("C:/Windows/Fonts/ARIALN.TTF", 26)
FB = ImageFont.truetype("C:/Windows/Fonts/ARIALNB.TTF", 26)
INK = (25, 25, 25)

im = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(im)


def block(x0, y0, x1, y1, title, lines, x, y, step):
    d.rectangle((x0, y0, x1, y1), fill=(255, 255, 255))
    d.text((x, y), title, font=FB, fill=INK)
    for i, s in enumerate(lines):
        d.text((x, y + step * (i + 1)), s, font=F, fill=INK)


block(450, 728, 935, 860, "Design D: fixed-weight fusion",
      ["z = concat(RGB, thermal, audio) = 576-d",
       "MLP(z) \u2192 disaster, victim",
       "No reliability gating"],
      462, 736, 31)

block(1514, 722, 1982, 870, "Design E: AdapFuse-v1",
      ["RUE \u2192 3 gates r from (q, stats)",
       "Tokens = r \u00d7 f   (3 \u00d7 192)",
       "z = GRU(mean(Attention(tokens))), 1 step",
       "Heads(z) \u2192 disaster, victim, nuisance"],
      1528, 726, 29)

im.save(OUT)
