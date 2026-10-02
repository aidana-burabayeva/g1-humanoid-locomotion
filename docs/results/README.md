# Result files

These are machine-readable outputs from the evaluators in this repository.
Each JSON file records the task, seed, number of robots, duration, command,
per-episode outcomes, and aggregate metrics. All runs used GPU physics
(MuJoCo Warp) unless stated otherwise. GPU execution is not bit-for-bit
deterministic, so a repeated run with the same seed can give a different
count.

| Directory | Checkpoint | Contents |
| --- | --- | --- |
| [`release_route_row4/`](release_route_row4/) | `model_28996` | Release acceptance on the row-4 route, seeds 0–2, with [`verdict.json`](release_route_row4/verdict.json) (2026-09-28). |
| [`release_route_row4_rerun/`](release_route_row4_rerun/) | `model_28996` | The same protocol rerun with this repository's image: 19/18/18, PASS (2026-10-01). |
| [`isolated_surfaces_28996/`](isolated_surfaces_28996/) | `model_28996` | Random roughness, uphill slope, and downhill slope tiles at difficulty 0.44 (row 4), at 0.5 and 0.8 m/s. Episodes stop at the tile edge. |
| [`flat_protocol_28996/`](flat_protocol_28996/) | `model_28996` | Flat-ground protocol: nine required scenarios plus six diagnostic ones (finer speeds, backward). `overall` is `FAIL` only because the diagnostic backward commands exceed the 0.10 m/s limit; `plan_required_missing` is empty and all nine required scenarios pass. |
| [`flat_protocol_28996_rerun/`](flat_protocol_28996_rerun/) | `model_28996` | The nine required scenarios rerun with this repository's image: all PASS. |
| [`stairs_tiles_28996/`](stairs_tiles_28996/) | `model_28996` | Regular and inverted 5 cm stair tiles (row 9, no falls) and inverted 10 cm tiles (20/20 and 18/20 falls on seeds 0 and 1). |
| [`limits_28996/`](limits_28996/) | `model_28996` | Row-9 route (about 21.8° slope): 16/20. |
| [`onnx_parity/`](onnx_parity/) | both | torch vs ONNX Runtime parity, 2 robots × 300 steps on CPU, tolerance 1e-4, with a negative control. |
| [`sim2sim_28996/`](sim2sim_28996/) | `model_28996` | `obs_builder_check.json`: the sim-to-sim observation builder against mjlab's observation manager. `unitree_mujoco_flat_60s.json`: 60 s in Unitree's G1 MJCF on CPU; `_dt0002` and `_mjlab_joints` variants change the physics step and the joint parameters. `control_parent_18997_flat_60s.json`: the parent checkpoint `model_18997` (not distributed) with the same runner, identical to its original run. `mjlab_openloop_drift/`: the same open-loop command in mjlab, 20 robots. |
| [`stairs10_selection/`](stairs10_selection/) | `model_33800` | 10 cm stair protocol on selection seeds 0–2, including the original route and the flat protocol. PASS. |
| [`stairs10_selection_rerun/`](stairs10_selection_rerun/) | `model_33800` | The same protocol rerun with this repository's image: stairs and original route pass, fixed-10 cm route 15/16/19. FAIL. |
| [`stairs10_seed0_repeats/`](stairs10_seed0_repeats/) | `model_33800` | Fixed-10 cm route, seed 0, three repeats each in the development image and in this repository's image. Shows run-to-run spread on the GPU (17–19 and 17–19). |
| [`stairs10_heldout/`](stairs10_heldout/) | `model_33800` | Held-out seeds 4–5 and training seed 3. FAIL (seed-4 route 16/20). |

Notes:

- Checkpoint paths in older files were reduced to the file name, for
  example `model_28996.pt`. The SHA-256 fields identify the checkpoint.
- `source_sha256` and `experiment_sources` in the 2026-09-28 and 2026-10-01
  verdicts hash the source files under the file layout used when those runs
  were made. The evaluation logic is the same as in this repository; the
  current hash is in [`weights/manifest.json`](../../weights/manifest.json).
- Field names such as `stage1_meta.json` and `plan_required` are part of the
  output format of [`scripts/eval_flat_protocol.py`](../../scripts/eval_flat_protocol.py).
