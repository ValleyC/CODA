# Repeat Annotation Fixes

Pieces whose annotations in `repeat_annotations.json` were corrected after video review.
Regenerate these pieces and re-check to verify the fixes.

## BWV 511 — `BachJS__BWV511__BWV-511_synth`

**Issue**: First repeat boundary was one bar too late. The repeat barline falls one note before MSMD bar 4 ends.

**Fix**: Changed `performance_order` from `[[0,5],[0,5],[6,11],[6,11]]` to `[[0,4],[0,4],[5,11],[5,11]]`.

---

## BWV 512 — `BachJS__BWV512__BWV-512_synth`

**Issue**: Unnecessary second-half repeat. Only the first half has a repeat sign.

**Fix**: Changed `repeat_type` from `binary` to `first_half`. Changed `performance_order` from `[[0,5],[0,5],[6,11],[6,11]]` to `[[0,5],[0,5],[6,11]]`.

---

## BWV 516 — `BachJS__BWV516__BWV-516_synth`

**Issue**: Unnecessary second-half repeat. Only the first half has a repeat sign.

**Fix**: Changed `repeat_type` from `binary` to `first_half`. Changed `performance_order` from `[[0,8],[0,8],[9,17],[9,17]]` to `[[0,8],[0,8],[9,17]]`.

---

## BWVAnh113 — `BachJS__BWVAnh113__anna-magdalena-03_synth`

**Issue**: First repeat boundary was at bar 15 instead of bar 12. Missed the repeat sign at bar 12.

**Fix**: Changed `performance_order` from `[[0,15],[0,15],[16,31],[16,31]]` to `[[0,12],[0,12],[13,31],[13,31]]`.

---

## BWVAnh120 — `BachJS__BWVAnh120__BWV-120_synth`

**Issue**: First repeat boundary at bar 14, but repeat sign is between bar 12 and bar 13. Bar 13 is a single pickup note (w=42) after the repeat barline.

**Fix**: Changed `performance_order` from `[[0,14],[0,14],[15,29],[15,29]]` to `[[0,12],[0,12],[13,29],[13,29]]`. Removed `needs_review`.

---

## Bartok Romanian Folk Dance No.2 — `BartokB__rom_folk_dance_1_bartok__rom_folk_dance_1_bartok_synth`

**Issue**: No repeat signs written in the score. Was incorrectly annotated as binary repeat.

**Fix**: Changed `has_repeats` from `true` to `false`. Removed `performance_order`, `repeat_type`, `needs_review`.

---

## KV331 Variation 4 — `MozartWA__KV331__KV331_1_5_var4_synth`

**Issue**: First repeat boundary one bar too late (`[0,8]` instead of `[0,7]`).

**Fix**: Changed `performance_order` from `[[0,8],[0,8],[9,17],[9,17]]` to `[[0,7],[0,7],[8,17],[8,17]]`. Removed `needs_review`.

---

## Swedish Folk Dance 1.3 — `Traditional__traditioner_af_swenska_folk_dansar.1.3__traditioner_af_swenska_folk_dansar.1.3_synth`

**Issue**: First repeat boundary one bar too early (`[0,6]` instead of `[0,7]`).

**Fix**: Changed `performance_order` from `[[0,6],[0,6],[7,13],[7,13]]` to `[[0,7],[0,7],[8,13],[8,13]]`. Removed `needs_review`.

---

## Swedish Folk Dance 1.5 — `Traditional__traditioner_af_swenska_folk_dansar.1.5__traditioner_af_swenska_folk_dansar.1.5_synth`

**Issue**: First repeat boundary one bar too late (`[0,4]` instead of `[0,3]`).

**Fix**: Changed `performance_order` from `[[0,4],[0,4],[5,9],[5,9]]` to `[[0,3],[0,3],[4,9],[4,9]]`. Removed `needs_review`.

---

## Swedish Folk Dance 1.6 — `Traditional__traditioner_af_swenska_folk_dansar.1.6__traditioner_af_swenska_folk_dansar.1.6_synth`

**Issue**: Only has a single repeat sign at the end (full piece repeat), not a binary repeat.

**Fix**: Changed `repeat_type` from `binary` to `full_repeat`. Changed `performance_order` from `[[0,3],[0,3],[4,7],[4,7]]` to `[[0,7],[0,7]]`. Removed `needs_review`.

---

## Swedish Folk Dance 1.14 — `Traditional__traditioner_af_swenska_folk_dansar.1.14__traditioner_af_swenska_folk_dansar.1.14_synth`

**Issue**: First repeat boundary one bar too late (`[0,3]` instead of `[0,2]`).

**Fix**: Changed `performance_order` from `[[0,3],[0,3],[4,7],[4,7]]` to `[[0,2],[0,2],[3,7],[3,7]]`. Removed `needs_review`.
