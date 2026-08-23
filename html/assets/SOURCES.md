# MoveMaker visual identity asset sources

These assets are stored locally so the classroom demonstration does not rely
on external network requests.

## Club crests

The presentation-priority club PNGs under `clubs/` were downloaded from the
public `luukhopman/football-logos` repository:

https://github.com/luukhopman/football-logos

The files are used for identification in this educational capstone prototype.
Club names, crests, and trademarks remain the property of their respective
owners. No affiliation or endorsement is implied.

## League marks

The five files under `leagues/` are locally cached Wikimedia-hosted images:

- `GB1.png`: Premier League text logo
- `ES1.png`: LaLiga EA Sports vertical logo
- `L1.png`: Bundesliga logo
- `IT1.png`: Serie A Made in Italy logo
- `FR1.png`: Ligue 1 McDonald's logo

Source pages and individual file licenses are available through Wikimedia
Commons/Wikipedia. League marks and trademarks remain the property of their
respective owners and are used here for identification in an educational
prototype.

### 2026-08-17 processing: `IT1.png` and `FR1.png`

Both files displayed noticeably smaller than the other three league marks in
the site's league-coverage rail, and `FR1.png` carried an attached McDonald's
sponsor mark that is not part of the Ligue 1 trademark itself. Fixed as
follows (originals preserved as `IT1_original_backup.png` /
`FR1_original_backup.png` in this same folder):

- `IT1.png` (Serie A "Made in Italy" mark): the file had an opaque white
  canvas background filling the full image bounds, so `object-fit: contain`
  was sizing to that oversized white square rather than the visible mark.
  The opaque background was made transparent via border-connected flood fill
  (any pixel path-connected to the image edge and close in color to the
  border was cleared; interior pixels — e.g. inside the Serie A shield —
  were left untouched), then the canvas was tight-cropped to the mark's own
  pixel bounds plus a 6px margin. No mark pixel was redrawn, recolored, or
  distorted.
- `FR1.png` (Ligue 1 mark): the canvas was first cropped to exclude the
  McDonald's arches beneath the Ligue 1 mark and wordmark — a separate,
  attached sponsor mark, not the league's own trademark. The same
  border-connected flood fill and tight-crop steps were then applied. The
  Ligue 1 mark and wordmark pixels are unaltered.

Both files are still simple identification marks used site-wide (league
rail, page footers, and the tool's live decision-profile identity strip);
processing was cosmetic (background/crop only), not a trademark redraw.

### 2026-08-17 replacement: `GB1.png` and `L1.png`

The original `GB1.png` was a Premier League wordmark only (no lion-crest
icon) and the original `L1.png` was a Bundesliga wordmark only (no
kicking-player icon), so both looked visually thin/small next to the other
three icon-bearing marks. The user supplied replacement files (downloaded
to `~/Downloads` as `images.png` and `Bundesliga-Logo.png`) carrying the
icon + wordmark lockup for each league; originals are preserved as
`GB1_original_backup.png` / `L1_original_backup.png` in this folder.

- `GB1.png` (Premier League icon + wordmark): the supplied file's
  "transparent" background was actually a checkerboard pattern baked into
  opaque pixels (a common transparency-preview artifact from image-editing
  sites), not real alpha. Real transparency was restored via
  border-connected flood fill (any edge-connected pixel lighter than a
  fixed threshold — covering both checker tones — was cleared), then the
  canvas was tight-cropped to the mark's own pixel bounds. No mark pixel
  was redrawn or recolored.
- `L1.png` (Bundesliga icon + wordmark): the supplied file already had a
  correct transparent background; it was only tight-cropped to its pixel
  bounds. No pixel was redrawn or recolored.

Exact original publisher source pages for these two supplied files are not
known; both are official-style league marks used here for identification
in an educational capstone prototype. Trademarks remain the property of
their respective owners; no affiliation or endorsement is implied.
