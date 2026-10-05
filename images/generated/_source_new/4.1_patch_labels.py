from PIL import Image, ImageDraw, ImageFont

SRC = "new_src.png"
OUT = "4.1_patched.png"
F = "C:/Windows/Fonts/Roboto-Regular.ttf"
BLACK = (20, 20, 20)
WHITE = (255, 255, 255)

im = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(im)


def font(sz):
    return ImageFont.truetype(F, sz)


def erase(box):
    d.rectangle(box, fill=WHITE)


def ctext(cx, y, s, sz=23):
    f = font(sz)
    w = d.textlength(s, font=f)
    d.text((cx - w / 2, y), s, font=f, fill=BLACK)


def ltext(x, y, s, sz=23):
    d.text((x, y), s, font=font(sz), fill=BLACK)


def arrow_down(cx, y0, y1):
    d.line([(cx, y0), (cx, y1 - 9)], fill=BLACK, width=2)
    d.polygon([(cx - 6, y1 - 10), (cx + 6, y1 - 10), (cx, y1)], fill=BLACK)


def rbox(box):
    d.rounded_rectangle(box, radius=7, outline=BLACK, width=2)


# --- Fusion column: replace "+ / Stacking / Q K V ..." with the real op order
FX = 941  # centre of fusion box
erase((858, 398, 1025, 626))
ctext(FX, 400, "Q = K = V =", 20)
ctext(FX, 424, "3 tokens", 20)
arrow_down(FX, 452, 474)
rbox((872, 476, 1010, 536))
ctext(FX, 480, "Residual +", 21)
ctext(FX, 506, "LayerNorm", 21)
arrow_down(FX, 538, 560)
rbox((872, 562, 1010, 606))
ctext(FX, 573, "Mean Pooling", 21)
# old "Mean Pooling" label below the box -> fused-vector size
erase((860, 655, 1030, 700))
ctext(FX, 662, "192-d fused vector", 21)

# --- GRU column: GRU is a single-layer nn.GRU, not multi-head
erase((1118, 300, 1262, 372))
ctext(1190, 308, "GRU Cell", 24)
ctext(1190, 338, "(1 layer, 192-d)", 20)

# --- Output heads: plain 2-layer MLPs, remove duplicated loss labels
for (x0, y0, x1, y1) in [(1376, 201, 1502, 273), (1376, 449, 1501, 516)]:
    erase((x0, y0, x1, y1))
    cy = (y0 + y1) / 2
    ctext((x0 + x1) / 2, cy - 27, "2-layer", 23)
    ctext((x0 + x1) / 2, cy + 1, "MLP", 23)
erase((1395, 292, 1490, 328))
erase((1395, 537, 1480, 572))

# --- Legend: say what solid / dashed mean
erase((1140, 815, 1262, 878))
ltext(1146, 820, "Data flow", 22)
ltext(1146, 849, "Gate signal", 22)

# --- Audio input note
erase((50, 834, 252, 863))
ctext(151, 836, "(absent in ~47% rows)", 19)

im.save(OUT)
print("saved", im.size)
