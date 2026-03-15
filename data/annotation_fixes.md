# Repeat Annotation Fixes

Pieces whose annotations in `repeat_annotations.json` were corrected after video review.
Regenerate these pieces and re-check to verify the fixes.

## BWV 511 — `BachJS__BWV511__BWV-511_synth`

**Issue**: First repeat boundary was one bar too late. The repeat barline falls one note before MSMD bar 4 ends.

**Fix**: Changed `performance_order` from `[[0,5],[0,5],[6,11],[6,11]]` to `[[0,4],[0,4],[5,11],[5,11]]`.

---

## BWV 512 — `BachJS__BWV512__BWV-512_synth`

**Issue**: Unnecessary second-half repeat. Only the first half has a repeat sign. First repeat boundary also one bar too late.

**Fix**: Changed `repeat_type` from `binary` to `first_half`. Changed `performance_order` from `[[0,5],[0,5],[6,11],[6,11]]` to `[[0,4],[0,4],[5,11]]`.

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

---

## Swedish Folk Dance 1.26 — `Traditional__traditioner_af_swenska_folk_dansar.1.26__traditioner_af_swenska_folk_dansar.1.26_synth`

**Issue**: Only has a single repeat at the end, restarting from bar 1 (bar 0 is a pickup). Was incorrectly annotated as binary repeat.

**Fix**: Changed `repeat_type` from `binary` to `full_repeat`. Changed `performance_order` from `[[0,3],[0,3],[4,8],[4,8]]` to `[[0,8],[1,8]]`. Removed `needs_review`.

---

## Swedish Folk Dance 1.29 — `Traditional__traditioner_af_swenska_folk_dansar.1.29__traditioner_af_swenska_folk_dansar.1.29_synth`

**Issue**: No repeat signs in the score. Was incorrectly annotated as binary repeat.

**Fix**: Changed `has_repeats` from `true` to `false`. Removed `performance_order`, `repeat_type`, `needs_review`.

---

## Swedish Folk Dance 2.27 — `Traditional__traditioner_af_swenska_folk_dansar.2.27__traditioner_af_swenska_folk_dansar.2.27_synth`

**Issue**: First repeat boundary too late (`[0,5]` instead of `[0,3]`).

**Fix**: Changed `performance_order` from `[[0,5],[0,5],[6,11],[6,11]]` to `[[0,3],[0,3],[4,11],[4,11]]`. Removed `needs_review`.

---

## Swedish Folk Dance 3.6 — `Traditional__traditioner_af_swenska_folk_dansar.3.6__traditioner_af_swenska_folk_dansar.3.6_synth`

**Issue**: Only has a single repeat at the end (full piece repeat), not a binary repeat.

**Fix**: Changed `repeat_type` from `binary` to `full_repeat`. Changed `performance_order` from `[[0,5],[0,5],[6,12],[6,12]]` to `[[0,12],[0,12]]`. Removed `needs_review`.

---

## Swedish Folk Dance 3.8 — `Traditional__traditioner_af_swenska_folk_dansar.3.8__traditioner_af_swenska_folk_dansar.3.8_synth`

**Issue**: Has volta brackets (1st/2nd endings), not a simple binary repeat. Bar 8/9 are 1st/2nd endings of first section, bars 17/18 are 1st/2nd endings of second section. Bar 0 is pickup.

**Fix**: Changed `repeat_type` from `binary` to `volta`. Changed `performance_order` from `[[0,8],[0,8],[9,18],[9,18]]` to `[[0,8],[1,7],[9,17],[10,16],[18,18]]`. Removed `needs_review`.

---

## Swedish Folk Dance 3.9 — `Traditional__traditioner_af_swenska_folk_dansar.3.9__traditioner_af_swenska_folk_dansar.3.9_synth`

**Issue**: First repeat boundary one bar too early (`[0,2]` instead of `[0,3]`).

**Fix**: Changed `performance_order` from `[[0,2],[0,2],[3,6],[3,6]]` to `[[0,3],[0,3],[4,6],[4,6]]`. Removed `needs_review`.

---

## Swedish Folk Dance 3.12 — `Traditional__traditioner_af_swenska_folk_dansar.3.12__traditioner_af_swenska_folk_dansar.3.12_synth`

**Issue**: Has three repeated sections (ternary), not binary. Repeats at bars 5, 11, and 17.

**Fix**: Changed `repeat_type` from `binary` to `ternary`. Changed `performance_order` from `[[0,8],[0,8],[9,17],[9,17]]` to `[[0,5],[0,5],[6,11],[6,11],[12,17],[12,17]]`. Removed `needs_review`.

---

## Swedish Folk Dance 3.22 — `Traditional__traditioner_af_swenska_folk_dansar.3.22__traditioner_af_swenska_folk_dansar.3.22_synth`

**Issue**: First repeat boundary 2 bars too late (`[0,9]` instead of `[0,7]`).

**Fix**: Changed `performance_order` from `[[0,9],[0,9],[10,19],[10,19]]` to `[[0,7],[0,7],[8,19],[8,19]]`. Removed `needs_review`.

---

## Swedish Folk Dance 3.24 — `Traditional__traditioner_af_swenska_folk_dansar.3.24__traditioner_af_swenska_folk_dansar.3.24_synth`

**Issue**: First repeat boundary 2 bars too late (`[0,9]` instead of `[0,7]`).

**Fix**: Changed `performance_order` from `[[0,9],[0,9],[10,19],[10,19]]` to `[[0,7],[0,7],[8,19],[8,19]]`. Removed `needs_review`.

---

## Swedish Folk Dance 3.34 — `Traditional__traditioner_af_swenska_folk_dansar.3.34__traditioner_af_swenska_folk_dansar.3.34_synth`

**Issue**: First repeat boundary too early (`[0,6]` instead of `[0,9]`).

**Fix**: Changed `performance_order` from `[[0,6],[0,6],[7,13],[7,13]]` to `[[0,9],[0,9],[10,13],[10,13]]`. Removed `needs_review`.
