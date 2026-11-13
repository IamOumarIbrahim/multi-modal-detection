"""40-slide presentation templates for the MMSAR paper, in ten looks built from five mood boards.

  burn       55          film burn: UAV tiles shot on burnt film, glowing confidence trace
  pixel      56          pixel mosaic: tiles as dissolving colour cells, confidence as cell columns
  graffiti   57          street poster: B&W pasted prints, the person sprayed in acid green
  stipple    58          pointillism: tiles as ink dots on pink paper, the person in red dots
  collage    59          mixed media: halftone prints, tape, riso thermal, marker circles
  pburn      55+56       pixel burn: pseudo-thermal as glowing fire cells on black
  ember      55+58       ember stipple: heat as glowing ember dots
  dotmatrix  56+58       dot matrix: colour dots on a grid, LED-board feel
  street     57+59       street collage: halftone prints wheat-pasted on concrete, sprayed over
  signal     55-59       signal collage: every medium on one dark wall

Each slide -> 3840x2160 JPEG + a PPTX slide (surface picture, every image slot its own picture so
Change Picture works, decorations as transparent pictures, all text live in the custom fonts).

Run:  python -I assets.py               (once: harvest frames, boxes, traces from the repo)
      python -I deck.py                 all ten decks
      python -I deck.py graffiti 1 5    one deck, some slides (JPEG only)
      python -I deck.py calibrate       re-measure PowerPoint text offsets for every font (needs PowerPoint)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import ROOT, calibrate          # noqa: E402
from layouts import SLIDES, NAMES           # noqa: E402
from styles import STYLES                   # noqa: E402


def build(key, only=None, pptx=True):
    from pptx import Presentation
    from pptx.util import Emu
    st = STYLES[key]()
    folder = ROOT / st.folder
    folder.mkdir(exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
    for i, (fn, nm) in enumerate(zip(SLIDES, NAMES), 1):
        if only and i not in only:
            continue
        s = fn(st)
        s.render().save(folder / f"{i:02d}_{nm}.jpg", quality=92, subsampling=0)
        if pptx:
            s.to_pptx(prs)
        print(f"{key} {i}", end="  ", flush=True)
    print()
    if pptx and not only:
        out = ROOT / st.pptx
        prs.save(out)
        print("saved", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["calibrate"]:
        calibrate([f for c in STYLES.values() for f in c.fonts.values()])
        sys.exit()
    keys = [a for a in args if a in STYLES] or list(STYLES)
    only = set(int(a) for a in args if a.isdigit()) or None
    for k in keys:
        build(k, only, pptx=not only)
    print("ok")
