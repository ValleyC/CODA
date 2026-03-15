# Repeat Annotation Fixes

Pieces whose annotations in `repeat_annotations.json` need to be regenerated and re-verified.

## Pending fixes

### Missed repeats (had `has_repeats: false`, actually has repeats)

#### Andre Sonatine Op.34 — `AndreJ__O34__andre-sonatine_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Score has 4 repeated sections with repeat barlines at bars 7, 23, 32, and 56, followed by a coda (bars 57-61).

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `multi_section`. Set `performance_order` to `[[0,7],[0,7],[8,23],[8,23],[24,32],[24,32],[33,56],[33,56],[57,61]]`.

#### La Native — `Anonymous__lanative__lanative_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Score has D.C. al Fine — play through, then back to start, end at bar 8 (Fine).

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `da_capo`. Set `performance_order` to `[[0,16],[0,8]]`.

#### BWV 117a — `BachJS__BWV117a__BWV-117a_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Binary repeat with barlines at bars 7 and 15.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `binary`. Set `performance_order` to `[[0,7],[0,7],[8,15],[8,15]]`.

#### BWV 825 (15title-hub) — `BachJS__BWV825__15title-hub_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Has volta brackets at bars 15/16 and 38/39, plus simple repeats at 47 and 55. Bar 41 confirmed as first bar on page 2.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `volta`. Set `performance_order` to `[[0,15],[0,14],[16,38],[17,37],[39,47],[40,47],[48,55],[48,55]]`.

#### BWV 825 (16title-hub) — `BachJS__BWV825__16title-hub_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Binary repeat at bar 15.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `binary`. Set `performance_order` to `[[0,15],[0,15],[16,47],[16,47]]`.

#### BWV 829 (55title-hub) — `BachJS__BWV829__55title-hub_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Binary repeat at bar 11.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `binary`. Set `performance_order` to `[[0,11],[0,11],[12,51],[12,51]]`.

#### BWV 830-2 — `BachJS__BWV830__BWV-830-2_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Binary repeat at bar 8 (first bar of second page).

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `binary`. Set `performance_order` to `[[0,8],[0,8],[9,21],[9,21]]`.

#### BWV 1006a-5 — `BachJS__BWV1006a__bwv-1006a_5_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Binary repeat at bar 15.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `binary`. Set `performance_order` to `[[0,15],[0,15],[16,31],[16,31]]`.

#### BWVAnh131 Air — `BachJS__BWVAnh131__air_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Binary repeat at bar 7.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `binary`. Set `performance_order` to `[[0,7],[0,7],[8,16],[8,16]]`.

#### BWVAnh691 — `BachJS__BWVAnh691__BWV-691_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. First half repeat only at bar 4.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `first_half`. Set `performance_order` to `[[0,4],[0,4],[5,9]]`.

#### Beethoven Op.79 Mvt.1 — `BeethovenLv__O79__LVB_Sonate_79_1_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Volta brackets: bar 49 (1st ending, repeat to bar 1, bar 0 is pickup), bar 50 (2nd ending). Bars 170-173 (1st ending, repeat to 52), bar 174 (2nd ending).

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `volta`. Set `performance_order` to `[[0,49],[1,48],[50,173],[52,169],[174,205]]`.

#### Burgmuller Op.100 No.2 — `BurgmullerJFF__O100__25EF-02_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Volta brackets: bar 9 (1st ending, repeat to bar 2, bars 0-1 pickup), bar 10 (2nd ending). Bar 26 (1st ending, repeat to 11), bar 27 (2nd ending).

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `volta`. Set `performance_order` to `[[0,9],[2,8],[10,26],[11,25],[27,32]]`.

#### Burgmuller Op.100 No.8 — `BurgmullerJFF__O100__25EF-08_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Binary repeat at bar 7, then D.C. al Fine at bar 7.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `da_capo`. Set `performance_order` to `[[0,7],[0,7],[8,15],[0,7]]`.

#### Burgmuller Op.100 No.14 — `BurgmullerJFF__O100__25EF-14_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Repeat at 12 (back to 4), repeat at 29 (back to 13), volta at 38/39 (back to 30), then D.C. al Fine at 29.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `da_capo`. Set `performance_order` to `[[0,12],[4,12],[13,29],[13,29],[30,38],[30,37],[39,39],[0,29]]`.

#### Handel Aylesford Menuet II — `HandelGF__Aylesford__10-menuetii_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Binary repeat at bar 7.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `binary`. Set `performance_order` to `[[0,7],[0,7],[8,25],[8,25]]`.

#### Handel Aylesford Air mit Var — `HandelGF__Aylesford__16-airmitvar_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. 6 repeated sections (air + variations) with repeat barlines at bars 4, 13, 18, 27, 32, and end.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `multi_section`. Set `performance_order` to `[[0,4],[0,4],[5,13],[5,13],[14,18],[14,18],[19,27],[19,27],[28,32],[28,32],[33,42],[33,42]]`.

#### Kruetzer Lodiska — `Kruetzer__lodiska__lodiska_synth`

**Issue**: Previous annotation `[[0,8],[9,17],[0,8],[18,26],[0,8]]` was missing binary repeats in each section. Correct structure: A (bars 0-17) repeated, B (bars 18-26) repeated, then D.C. al Fine at bar 8.

**Fix**: Updated `performance_order` to `[[0,17],[0,17],[18,26],[18,26],[0,8]]`.

#### Mueller Siciliano — `MuellerAE__muller-siciliano__muller-siciliano_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. First half repeat only at bar 7.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `first_half`. Set `performance_order` to `[[0,7],[0,7],[8,21]]`.

#### Satie Gymnopedie No.1 — `SatieE__gymnopedie_1__gymnopedie_1_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Volta: repeat at bar 38 to beginning, bars 31-38 are 1st ending, bar 39 starts 2nd ending.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `volta`. Set `performance_order` to `[[0,38],[0,30],[39,46]]`.

#### Schumann Op.68 No.6 — `SchumannR__O68__schumann-op68-06-pauvre-orpheline_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Second half repeat only — repeat at end back to bar 9.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `second_half`. Set `performance_order` to `[[0,21],[9,21]]`.

#### Schumann Op.68 No.8 — `SchumannR__O68__schumann-op68-08-cavalier-sauvage_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. First half repeat only at bar 8.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `first_half`. Set `performance_order` to `[[0,8],[0,8],[9,26]]`.

#### Schumann Op.68 No.16 — `SchumannR__O68__schumann-op68-16-premier-chagrin_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Second half repeat only — repeat at end back to bar 17.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `second_half`. Set `performance_order` to `[[0,33],[17,33]]`.

#### Schumann Op.68 No.26 — `SchumannR__O68__schumann-op68-26-sans-titre_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Binary repeat at bar 8.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `binary`. Set `performance_order` to `[[0,8],[0,8],[9,23],[9,23]]`.

#### Yaniewicz Les Lanciers — `Yaniewicz__leslanciers__leslanciers_synth`

**Issue**: Was incorrectly marked as `has_repeats: false`. Volta brackets: bar 7 (1st ending, repeat to beginning), bar 8 (2nd ending). Then repeat at end back to bar 9.

**Fix**: Changed `has_repeats` to `true`. Set `repeat_type` to `volta`. Set `performance_order` to `[[0,7],[0,6],[8,17],[9,17]]`.
