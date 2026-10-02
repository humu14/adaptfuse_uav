"""Clean the supplied Fig 4.5 / 4.6 rasters so they match the code and metadata.

Fig 4.5 (Design D vs E):
  * Design D has no quality tokens (models/full_model.py, IntermediateFixed:
    project -> concat -> MLP), so the two "16-d Quality" labels are relabelled
    as the projected 192-d feature vectors they actually are.
  * The garbled pseudo-code panels are replaced with pseudo-code that matches
    the implementation.
Fig 4.6 (data provenance):
  * The 15,970 C2A/SARD bounding boxes are person boxes, not smoke/fire boxes
    (smoke/fire boxes come from the separate D-Fire benchmark).

Originals: 4.5_design_D_vs_E_original.jpg, 4.6_data_provenance_original.jpg
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
FONTS = Path("C:/Windows/Fonts")
NARROW = str(FONTS / "ARIALN.TTF")
NARROW_BOLD = str(FONTS / "ARIALNB.TTF")
MONO = str(FONTS / "consola.ttf")
INK = (20, 20, 20)
WHITE = (255, 255, 255)


def centered(draw, cx, y, text, font):
    w = draw.textlength(text, font=font)
    draw.text((cx - w / 2, y), text, font=font, fill=INK)


def fix_design_comparison():
    im = Image.open(HERE / "4.5_design_D_vs_E_original.jpg").convert("RGB")
    d = ImageDraw.Draw(im)

    # Panel (a): relabel the two stacks as projected feature vectors.
    label = ImageFont.truetype(NARROW, 30)
    d.rectangle((648, 126, 745, 228), fill=WHITE)
    for i, line in enumerate(("192-d", "RGB", "feature")):
        centered(d, 692, 132 + i * 32, line, label)
    d.rectangle((530, 552, 690, 590), fill=WHITE)
    centered(d, 610, 555, "192-d audio feature", label)

    # Code panels: replace garbled text with pseudo-code matching the code.
    code = ImageFont.truetype(MONO, 19)
    step = 29
    d.rectangle((447, 716, 937, 902), fill=WHITE)
    left = (
        "# Design D: fixed-weight fusion",
        "z = concat(f_rgb, f_th, f_au)  # 576-d",
        "y_dis, y_vic = MLP(z)",
        "# no reliability gating",
    )
    for i, line in enumerate(left):
        d.text((462, 740 + i * step), line, font=code, fill=INK)

    d.rectangle((1514, 716, 1984, 902), fill=WHITE)
    right = (
        "# Design E: AdapFuse-v1",
        "r = RUE(q, stats)       # 3 gates",
        "T = stack(r * f)        # 3 x 192",
        "z = GRU(mean(Attn(T)))  # T = 1",
        "y_dis, y_vic, y_nui = heads(z)",
    )
    for i, line in enumerate(right):
        d.text((1528, 728 + i * step), line, font=code, fill=INK)

    im.save(OUT / "4.5.png")


def fix_provenance():
    im = Image.open(HERE / "4.6_data_provenance_original.jpg").convert("RGB")
    d = ImageDraw.Draw(im)
    title = ImageFont.truetype(NARROW_BOLD, 42)
    d.rectangle((836, 870, 1114, 964), fill=WHITE)
    centered(d, 973, 872, "Person", title)
    centered(d, 973, 916, "Bounding Boxes", title)
    im.save(OUT / "4.6.png")


if __name__ == "__main__":
    fix_design_comparison()
    fix_provenance()
    print("saved", OUT / "4.5.png", OUT / "4.6.png")
