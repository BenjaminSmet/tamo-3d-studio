# Phase 1 scoring sheet

Judge the finished MetaHuman, not the raw mesh, because that's what ends up on screen.
Copy this file into the actor's folder for each run you score, for example
`library/projects/tamo/actors/<actor>/generations/<run>/score.md`.

## Before you score

- [ ] Same input: the master front portrait (and profile, for later phases), unchanged since the last test
- [ ] Mesh to MetaHuman done from this run's `model.obj` or `model.glb`
- [ ] Same Unreal scene: neutral lighting, three fixed cameras (front, three-quarter, profile)
- [ ] Each render placed next to the matching reference image

## Scores (1 = poor, 5 = indistinguishable)

| What | Score | Notes |
| --- | --- | --- |
| Likeness (would someone recognise her?) | | |
| Profile shape: nose | | |
| Profile shape: jaw and chin | | |
| Profile shape: ears | | |
| Symmetry | | |
| Manual fixes needed (5 = none, 1 = lots) | | |

## Conform

| Question | Answer |
| --- | --- |
| Did Mesh to MetaHuman accept the mesh first try? | |
| Fixes needed before it conformed (scale, orientation, neck crop, holes) | |
| Time spent on fixes | |

## Verdict

- [ ] Pass: recognisable from every angle, so move on to Phase 2 (automated cleanup)
- [ ] Close: recognisable from the front only, so try the multi-view refinement from Phase 4 earlier
- [ ] Fail: generic face, so go to the fallbacks in the risks table

Run: `run.json` → mode, pipeline, seed, peak VRAM, seconds
