# Presentation templates (40 slides, 10 looks)

Every look is built from the paper's own material: real 640×640 UAV tiles (desert, forest, snow) in RGB and
pseudo-thermal, the ground-truth person box, empty tiles with a crossed-out false alert, filmstrips of
consecutive frames, the real per-frame confidence trace (raw flicker vs. moving average, alert ticks), and the
drone-footprint render. Each style shows the same story in its own medium.

| Deck | Mood board | Look | JPGs | PowerPoint |
|---|---|---|---|---|
| Film Burn | 55 | tiles shot on burnt film, glowing trace | `film_burn_40/` | `FilmBurn_Template_40.pptx` |
| Pixel Mosaic | 56 | dissolving colour cells, trace as cell columns | `pixel_mosaic_40/` | `PixelMosaic_Template_40.pptx` |
| Graffiti | 57 | B&W pasted posters, person sprayed in acid green | `graffiti_40/` | `Graffiti_Template_40.pptx` |
| Pointillism | 58 | ink stipple on pink paper, person in red dots | `pointillism_40/` | `Pointillism_Template_40.pptx` |
| Mixed Media | 59 | halftone prints, tape, riso thermal, marker | `mixed_media_40/` | `MixedMedia_Template_40.pptx` |
| Pixel Burn | 55 + 56 | pseudo-thermal as glowing fire cells | `pixel_burn_40/` | `PixelBurn_Template_40.pptx` |
| Ember Stipple | 55 + 58 | heat as glowing ember dots | `ember_stipple_40/` | `EmberStipple_Template_40.pptx` |
| Dot Matrix | 56 + 58 | colour dots on a grid, LED board | `dot_matrix_40/` | `DotMatrix_Template_40.pptx` |
| Street Collage | 57 + 59 | halftone prints wheat-pasted on concrete | `street_collage_40/` | `StreetCollage_Template_40.pptx` |
| Signal Collage | 55–59 | all five media on one dark wall | `signal_collage_40/` | `SignalCollage_Template_40.pptx` |

## The finished talk in five looks (`talk/`)

`Causal Temporal Filtering.pptx` (15 slides) rebuilt in Ember Stipple, Graffiti, Mixed Media, Pixel Burn and
Signal Collage: `talk/Causal Temporal Filtering - <Look>.pptx`, JPGs in `talk/<look>/`, source images in
`talk/source/`. Rebuild with `python -I src/talk.py` (or `python -I src/talk.py graffiti 1 4` for a preview).
Charts stay exact (printed on paper in light looks, night mode in dark looks), photos take the look's medium
with the person re-marked, RGB/thermal figures keep their colours, logos and the QR code are untouched.

**Mixed Media (current):** `talk/Causal Temporal Filtering - Mixed Media.pptx` is the talk itself restyled in
place by `python src/mm_restyle.py "<talk>.pptx" <work dir>` (needs PowerPoint for the two export passes):
every text box, equation, video and position is the talk's own; only background, heading font, paper prints,
tape and the collage cover (with the drone from `src/drone.py`) are added. Previews: `talk/mixed_media/`.

## Using the PowerPoint files

1. Install the fonts once: right-click `fonts/install_fonts.ps1` → *Run with PowerShell* (per user, no admin).
   Restart PowerPoint. Without this, PowerPoint substitutes a default font.
2. All text is live. Every image slot is its own picture: right-click → *Change Picture*.
   Tape, spray tags and other decorations that sit above pictures are separate pictures named `Decoration`.

## Rebuilding

```bash
cd Present/src
python -I assets.py              # once: frames, boxes, confidence traces from the repo
python -I deck.py                # all ten decks (about 10 min)
python -I deck.py graffiti 1 5   # one deck, a few slides, JPG only
python -I deck.py calibrate      # re-measure PowerPoint text offsets (needs PowerPoint)
```

`src/art.py` holds the media (burn, mosaic, spray, stipple, halftone, collage), `src/styles.py` the ten looks,
`src/layouts.py` the 40 layouts, `src/engine.py` the slide model and PowerPoint export.
Fonts are from Google Fonts (SIL Open Font License).
