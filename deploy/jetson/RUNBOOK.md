# Jetson Localization Runbook

Results and their history. For setup and commands, start with `README.md`. Paths here are as on the workstation, where this folder sits at `~/argus-localization/deploy/jetson/`.

Running AstroLoc localization on the Jetson Orin Nano: photo → retriever → search → matcher → location.

## Status (current)

**The full localization flow runs on the Jetson** — TensorRT retriever, database search and the SIFT + LightGlue matcher in one process — and reproduces the workstation.

### Retrieval only (rough location)

| Jetson Orin Nano 8GB, TensorRT FP32 | Nano v2 | Full-size |
|---|---|---|
| Top-1 / Top-10 / Top-100 | 34.5% / 50.5% / 60.5% | **68.5% / 89.0% / 98.5%** |
| Median error | 1,042 km | **79 km** |
| On the 169 photos excluded from training | 33.7%, 1,303 km | **64.5%, 86 km** |
| Time per photo | ~67 ms | ~109 ms |
| Board memory, idle → peak (of 7,620 MB) | 2,345 → 2,686 MB | 2,523 → 3,351 MB |
| Board power, peak | 6.9 W | 8.2 W |
| Fingerprint database | 62 MB | 249 MB |

Same top tile as the workstation on all 200 photos, for both models.

### Full flow: full-size retriever + matcher

The matcher checks the top 15 retrieved tiles and declares a fix when one has ≥ 30 inlier points.

| 200 photos | Jetson | Workstation |
|---|---|---|
| Correct tile in the top 15 | 91% | 91% |
| Fix rate | 54.5% (109) | 54.0% (108) |
| Fixes on a correct tile | 92.7% | 92.6% |
| Fix error: median / 90th percentile / max | **15.9 km** / 33.2 km / 272 km | 16.0 km / 33.7 km / 272 km |
| Fixes more than 50 km off | 4 | 4 |

- Same fix / no-fix decision as the workstation on 199 of 200 photos; where both fix, the locations differ by 0.02 km median (5.9 km max).
- On the 169 photos excluded from training: 53.8% fix rate, 16.1 km median error.

### Speed on the Jetson: 3.3 s → 0.73 s per photo

Each step was checked on the same 200 photos; accuracy held at every step (fix on ~55% of photos, ~16 km median error).

| Change | Time per photo | Energy per photo |
|---|---|---|
| Full flow on the Jetson, first version | 3.3 s | ~26 J |
| Tile SIFT features computed once and stored | 1.3 s | ~14 J |
| Top 10 candidate tiles instead of 15 | 0.95 s | ~10 J |
| LightGlue in half precision | 0.81 s | ~7 J |
| **Search on the GPU + tile features in one file (current)** | **0.73 s** | **~6 J** |

Tried and not adopted: see *Experiment log* below.

### Accuracy: original vs now

Same 200 held-out photos.
- **Fix rate:** % of photos the system locates confidently (≥ 30 matched points on a tile). The rest get "no fix": no answer rather than a guess.
- **Error:** km between the estimated and the true photo centre, for fixes only.

| | Original (3.3 s) | **Current default (0.73 s)** | TensorRT LightGlue (0.65 s, testing) |
|---|---|---|---|
| Setup | top 15 tiles, full precision | top 10 tiles, LightGlue half precision | as current, LightGlue as a TensorRT FP16 engine |
| Fix rate | 54.5% (109) | **55.0% (110)** | 54.0% (108) |
| Median error | 15.9 km | **15.8 km** | 15.9 km |
| 90% of fixes within | 33.2 km | 32.7 km | 32.0 km |
| Fixes > 50 km off | 4 | 4 | 3 |

Retrieval is identical in all three (right tile ranked first on 68.5% of photos).

### Setup
- **Nano v2:** `nano_train/checkpoints_v2/final.pt` (27M, 512-d).
- **Full-size:** `astroloc_train/checkpoints_faithful_lora_v2/final.pt` (106.5M, 2048-d).
- **Database:** all 7,602 zoom-9 tiles from 2021, 4 rotations each, plus their stored SIFT features (3.2 GB).
- **Test:** 200 astronaut photos near the Alps; "169 photos" = those training can't have seen.

### Takeaways
- **Full-size beats nano v2:** 2× the hit rate, 13× lower error.
- **The matcher is essential:** 79 km → 16 km, with a fix on about half the photos.
- **8 GB is enough:** 4.0 GB peak memory for the whole flow.
- **Time is the main cost:** 0.73 s per photo now (3.3 s originally), mostly LightGlue on the GPU.
- **4 of 109 fixes are wrong by > 50 km:** these need filtering before OD.

## Real two-frame pairs

OD needs at least two photos from the same pass. These are real consecutive photos, not a simulated second frame. Files are in `deploy/jetson/two_frame/`.

| Step | Result |
|---|---|
| Held-out-region photos, same mission and day, 200–1,000 km apart, both with a correct tile | 15,275 candidate pairs (3,037 photos) |
| Sample of 40 pairs, capture times from NASA's photo pages | **38 confirmed same pass**: ISS moved ~6.9 km/s between the two NASA nadir points |
| Time gap within a pass | 12–147 s, median 55 s |
| Both photos localized on the Jetson (current setup) | **8 of 38 pairs**, 6 of them with both errors under 50 km |

- **Fix rate:** 43% per photo here, against 55% on the Alps test. Needing both photos of a pair to get a fix leaves about 1 pair in 5.
- **Scaling up:** about 1 pair in 5 would give roughly 3,000 usable pairs from the 15,275 candidates.
- **One fix with 245 inliers was 90 km off**, so even a high inlier count doesn't guarantee a good fix.
- **Next:**
  - fetch times for more pairs
  - run OD on the pairs with both fixes. The astronaut photos are oblique, so the photo centre is not below the ISS; the OD step has to allow for that or use the tie points.

## Where the time goes

Current setup, 50 photos, Jetson in 15 W mode, nothing else running. "CPU" is the Jetson's ARM CPU (6 cores) and "GPU" is its GPU. Timing each step separately adds ~0.07 s, so the total reads 0.81 s against 0.73 s in a plain run. Produced by `profile_full.py --lightglue-mp --tile-pack ../tile_feats_pack --search-on-gpu`.

| Step | Runs on | ms per photo | Share |
|---|---|---|---|
| Read photo from disk | disk | 0.8 | 0.1% |
| Decode JPEG | CPU | 11.3 | 1.4% |
| Resize to 224 + normalize | CPU | 7.6 | 0.9% |
| Retriever (TensorRT), with copies in/out | GPU | 51.9 | 6.4% |
| Search 7,602 tiles | GPU | 5.2 | 0.6% |
| Photo SIFT: copy to GPU, resize to 512 | GPU | 34.8 | 4.3% |
| Photo SIFT: OpenCV SIFT | CPU, ~3 cores | 85.2 | 10.6% |
| Tile features: read from the packed file (10 tiles) | CPU | 1.2 | 0.2% |
| Tile features: copy to GPU | GPU | 11.8 | 1.5% |
| **LightGlue, 10 tiles** | **GPU** | **581.4** | **72.1%** |
| Results to CPU + MAGSAC check | CPU | 11.0 | 1.4% |
| Decide + georeference | CPU | 3.7 | 0.5% |
| **Total** | | **806** | |

- **LightGlue is the bottleneck** (72%, ~58 ms per tile). Everything before it, including the 224 px retriever, takes ~0.21 s.
- **The next-biggest step is SIFT on the photo** (85 ms on the CPU). OpenCV's SIFT has no GPU version here.
- **Inside LightGlue, the ARM CPU is the limit, not the GPU:**
  - The CPU thread launching its ~850 small GPU kernels per tile is busy 99% of the time.
  - The GPU averages 53% busy.
  - Before half precision, the GPU was the limit: busy 81% of each call (`nsys`), 59% of that in 32-bit matrix multiplies.
- **Disk is not a bottleneck:** reading the photo takes under 1 ms, and the 10 tiles' features ~1 ms from the packed file.
- **Clocks in 15 W mode:** CPU up to 1.51 GHz, GPU up to 625 MHz.

### Memory

The GPU has no separate memory; it shares the board's 7.6 GB.

| | MB |
|---|---|
| Board RAM, running (peak) | 3,890 (3,896) |
| This process, peak | 1,969 |
| … fingerprint database | 238, held twice: a CPU copy and a GPU copy for the search |
| … TensorRT engine weights + working memory | 409 + 14 |
| … PyTorch GPU tensors, peak (reserved), including the GPU database copy | 335 (366) |
| … the rest: CUDA, cuDNN and cuBLAS libraries, Python | ~1,200 |

### Power (`tegrastats` VDD_IN, the module's total input)

| | |
|---|---|
| Idle | 4.3 W |
| Running: average / peak | 8.1 W / 8.5 W |
| Energy per photo | ~6 J |
| 100 photos | ~0.16 Wh |
| Load while running: GPU / CPU | 53% / 20% |

### What would make it faster, biggest likely win first
1. **LightGlue in half precision** (`mp=True`): **done, adopted** (0.95 s → 0.81 s, same fixes; see Experiment log).
2. **Fewer points per image** (1,024 → 512): **tested, rejected** (loses 19 of 109 fixes; see Experiment log).
3. **Fewer tiles:** top 5 runs at 0.59 s and loses 2 of 109 fixes (see Experiment log).
4. **Higher power mode:** 15 W caps the GPU at 625 MHz. JetPack 6.2 adds a faster mode for this board, but needs a reflash, so it's a team decision.
5. **Smaller CPU items** (search on the GPU, tile features in one memory-mapped file): **done, adopted** (0.82 s → 0.73 s, same results; see Experiment log). Keeping all tile features in GPU memory instead gives no further gain. Photo SIFT stays on the CPU (87 ms).
6. **Fewer GPU launches in LightGlue:** batching all 10 tiles in one call was **tested and is slower** (0.83 s; see Experiment log). Converting LightGlue to TensorRT like the retriever is the remaining option.

## Experiment log

One change at a time, against a same-day baseline, on the 200 test photos, with the setup current at the time. Per-run results and logs are on the Jetson in `e2e_alps_full/` (`match_results_<tag>.json`, `match_run_<tag>.log`, `tegrastats_match_<tag>.log`).

| Change, in the order tried | Time per photo | Accuracy vs baseline | Energy | Decision |
|---|---|---|---|---|
| Tile SIFT features computed once, stored (3.2 GB) | 3.3 → 1.3 s | identical on 200/200 | 26 → 14 J | adopted |
| Top 10 tiles instead of 15 | 1.31 → 0.95 s | identical (winner never ranked below #8) | 14 → 10 J | adopted |
| Top 5 tiles | 0.59 s | loses 2 of 109 fixes | 6 J | not adopted |
| LightGlue in half precision | 0.95 → 0.81 s | +1 fix; locations move 0.02 km median | 10 → 7 J | adopted |
| 512 SIFT points instead of 1,024 | 0.78 s | loses 19 of 109 fixes | 7 J | rejected |
| Half precision + 512 points | 0.83 s | loses 19 of 109 fixes | 6 J | rejected |
| Search on the GPU + tile features in one memory-mapped file | 0.82 → 0.73 s | identical on 200/200 | 7 → 6 J | adopted |
| Tile features preloaded in GPU memory | 0.74 s | identical; +1.8 GB memory | 6 J | rejected |
| Batched LightGlue (10 tiles in one call) | 0.83 s | loses 1 fix | 8 J | rejected: no flash attention with masks, no point pruning |
| LightGlue as a TensorRT FP16 engine (`lightglue_trt.py`) | 0.74 → 0.65 s | loses 2 of 110 fixes; one fewer fix > 50 km | 6 → 5 J | testing; not default yet |
| LightGlue as a TensorRT FP32 engine | 1.18 s | +2 fixes | 13 J | rejected: slower |

The workstation (RTX 4090) runs the same flow at 0.80 s with tile SIFT per photo and 0.22 s with stored features.

## What ran on the Jetson

Everything below runs on the Jetson, in one Python process. The Jetson's chip has two processors sharing the same memory: an ARM CPU (6 cores) and a GPU. The last column says which one does each step.

| Step | What happens | Code | Runs on |
|---|---|---|---|
| 1. Preprocess | resize to 224 × 224, ImageNet-normalize | `e2e_match.py` (numpy, PIL) | CPU |
| 2. Retriever | photo → 2048-number fingerprint | `astroloc_full_fp32.trt`, TensorRT FP32 | GPU |
| 3. Search | dot product against 7,602 tiles × 4 rotations, keep the top 15 (top 10 in the current setup) | `e2e_match.py` (numpy, brute force) | CPU |
| 4. Feature points | SIFT, up to 1,024 points, image resized to 512, on the photo and each of the 15 tiles. With `--tile-feats` the tiles' points are read from disk instead | repo `SiftLightGlueMatcher`, via LightGlue's SIFT wrapper around OpenCV | CPU |
| 5. Matching | LightGlue pairs the photo's points with each tile's points | repo `SiftLightGlueMatcher`, LightGlue in PyTorch | GPU |
| 6. Geometry check | fit a homography with MAGSAC (3 px threshold), count inliers | repo `SiftLightGlueMatcher`, OpenCV | CPU |
| 7. Fix decision | tile with the most inliers wins; a fix needs ≥ 30 | `e2e_match.py`, same rule as repo `core/pipeline.py` | CPU |
| 8. Location | photo corners → tile pixels → lat/lon | repo `Georeferencer.estimate_query_footprint` | CPU |

**Output of a fix:** 30+ matched points, each a photo pixel paired with a ground lat/lon. The 16 km figure scores the centre of the photo's footprint against the true centre, the same metric as the repo's `scripts/evaluate.py`.

### Code used
- **From argus-localization, unmodified (commit `cc75c2b`):** `matchers/sift_lightglue_matcher.py`, `georeference/georeferencer.py`, `core/types.py`, `core/interfaces.py`, `data_loading/earthloc_loader.py`.
- **Ours:** `e2e_match.py` (the loop, the search, the scoring), `precompute_tile_feats.py` and the ONNX → TensorRT conversion of the retriever.
- **Differences from `core/pipeline.py`:**
  - The retriever is the TensorRT engine instead of the PyTorch model. Fingerprints agree with PyTorch to cosine 0.99997.
  - The photo's SIFT points are extracted once per photo instead of once per candidate tile. The matching logic is untouched.
  - Tile features are not kept in memory between photos, so memory reflects one photo at a time. With `--tile-feats` they are computed once by the same matcher class, stored on disk, and handed back through the matcher's own tile cache.

### Dependencies on the Jetson

| Package | Version | Source |
|---|---|---|
| JetPack / L4T | 6.0 / R36.3 | on the board |
| Python, CUDA, cuDNN | 3.10.12, 12.2, 8.9.4 | on the board |
| TensorRT | 8.6.2 | on the board |
| OpenCV | 4.8.0 | on the board |
| PIL | 9.0.1 | on the board |
| PyTorch | 2.4.0a0 (nv24.05) | NVIDIA's JetPack 6.0 wheel, in `pydeps` |
| numpy | 1.26.4 | in `pydeps` (the board's 1.21.5 is too old for this PyTorch) |
| kornia | 0.7.1 | in `pydeps` |
| LightGlue | git `eb42fee` | in `pydeps`; weights `sift_lightglue_v0-1_arxiv.pth` (45 MB) |
| networkx, jinja2, filelock, fsspec, sympy, mpmath, typing_extensions | 3.2.1, 3.1.4, 3.15.4, 2024.6.1, 1.12.1, 1.3.0, 4.12.2 | in `pydeps`, required by PyTorch |

`pydeps` is `~/Desktop/astroloc/pydeps`, installed with `pip3 install --no-deps --target` and used through `PYTHONPATH`. Nothing was installed system-wide and no sudo was used. torchvision is not installed and not needed.

### Resources

Time, memory and power per step are under *Where the time goes*.

**Disk for the flow:**
- engine 410 MB, fingerprints 249 MB, `pydeps` 2.0 GB, LightGlue weights 45 MB
- packed tile features 2.15 GB
- the tile images (720 MB) stay on disk, because their filenames carry each tile's corner coordinates

**Handoff to OD (not run yet):** `Georeferencer.make_tie_points` turns a fix into tie points, and `integration/batchopt_adapter.py::to_batchopt_measurements` converts them to the rows the FSW-Payload solver takes (timestamp, bearing in the body frame, landmark position in ECI). The adapter needs a GNC-Payload checkout and `brahe`, neither of which is on the Jetson.

## Files

**Workstation** (`~/`): `nano_v2_static.onnx`, `astroloc_full_static.onnx`, `e2e_alps_v2/`, `e2e_alps_full/` (database + test photos + workstation results). The workstation's stored tile features (3.2 GB) were deleted after the run because the home disk is 97% full; `precompute_tile_feats.py` rebuilds them in 6 min.
**Jetson** (`~/Desktop/astroloc/`): `nano_v2_fp32.trt`, `astroloc_full_fp32.trt`, `e2e_alps_v2/`, `e2e_alps_full/`, `e2e_jetson.py`, `e2e_match.py`, `run_match_full.sh`, `tiles/2021_09/` (tile images), `tile_feats_2021_09/` (stored tile SIFT features), `precompute_tile_feats.py`, `profile_match.py`, `repo/` (matcher and georeferencing code from argus-localization), `pydeps/` (PyTorch, kornia, LightGlue), `wheels/` (installers).

Superseded: `nano_fp32.trt` / `nano_static.onnx` (nano **v1**, 3% top-1), `nano_fp16.trt` (FP16 rejected), `astroloc_fp32.trt` / `astroloc_static.onnx` (oldest full-size checkpoint).

## Reproduce

Commands are for the full-size model. For nano v2, use `nano_train/checkpoints_v2/final.pt` and the `nano_v2` / `e2e_alps_v2` names.

### 1. Workstation

Build the database, export the model, run the workstation side:
```bash
cd ~/argus-localization
CKPT=/mnt/sdc1/astroloc/reference_db/astroloc_train/checkpoints_faithful_lora_v2/final.pt
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/e2e_prepare.py --checkpoint $CKPT --out ~/e2e_alps_full
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/export_onnx.py --checkpoint $CKPT \
    --output ~/astroloc_full_static.onnx --opset 17 --batch 1 --static-batch
CUDA_VISIBLE_DEVICES=0 ~/venv-export/bin/python deploy/jetson/e2e_match.py \
    --data ~/e2e_alps_full --tiles-dir /mnt/sdc1/astroloc/data/database/2021_09
```
Check `nvidia-smi` first and use a free GPU.

### 2. Jetson, retrieval only
```bash
cd ~/Desktop/astroloc
scp <you>@172.24.58.65:~/astroloc_full_static.onnx .
scp -r <you>@172.24.58.65:~/e2e_alps_full .
/usr/src/tensorrt/bin/trtexec --onnx=astroloc_full_static.onnx --saveEngine=astroloc_full_fp32.trt
python3 e2e_jetson.py --engine astroloc_full_fp32.trt --data e2e_alps_full
```
Run `tegrastats` alongside for board memory and power.

### 3. Jetson, matcher setup (once)

The Jetson has no internet, so everything is downloaded elsewhere and copied into `~/Desktop/astroloc/`:
- `wheels/`: NVIDIA's PyTorch wheel for JetPack 6.0 (`developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl`), numpy 1.26.4 for aarch64 / Python 3.10, and the pure-Python wheels in the dependency table.
- `pydeps/kornia`, `pydeps/lightglue`: copied from the workstation's `~/venv-export/lib/python3.12/site-packages/`.
- `~/.cache/torch/hub/checkpoints/sift_lightglue_v0-1_arxiv.pth`: copied from the same path on the workstation.
- `tiles/2021_09/`: copied from `/mnt/sdc1/astroloc/data/database/2021_09`.
- `repo/`: the five argus-localization files listed under *Code used*, plus the empty `__init__.py` files.
```bash
cd ~/Desktop/astroloc
pip3 install --no-deps --no-index --target pydeps wheels/*.whl
printf 'from .lightglue import LightGlue  # noqa\nfrom .sift import SIFT  # noqa\n' > pydeps/lightglue/__init__.py
```

### 4. Jetson, full flow
```bash
cd ~/Desktop/astroloc/repo
PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../e2e_match.py --data ../e2e_alps_full \
    --tiles-dir ../tiles/2021_09 --engine ../astroloc_full_fp32.trt
```
`~/Desktop/astroloc/run_match_full.sh` runs the same command with `tegrastats` logging. Results land in `e2e_alps_full/match_results.json`; the logs are `match_run.log` and `tegrastats_match.log`.

### 5. Jetson, full flow with tile features computed in advance
```bash
cd ~/Desktop/astroloc/repo
PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../precompute_tile_feats.py \
    --tiles-dir ../tiles/2021_09 --out ../tile_feats_2021_09          # once, 16 min
~/Desktop/astroloc/run_match_full.sh pre --tile-feats ../tile_feats_2021_09
```
Results land in `match_results_pre.json`, with logs `match_run_pre.log` and `tegrastats_match_pre.log`. The same two scripts and `--tile-feats` flag work on the workstation.

### 6. Top 15 / 10 / 5 comparison
```bash
cd ~/Desktop/astroloc
for k in 15 10 5; do ./run_match_full.sh k$k --tile-feats ../tile_feats_2021_09 --top-k $k; done
python3 compare_topk.py --data e2e_alps_full --ks 15 10 5
# half precision / fewer points: run_speed.sh, then
python3 compare_topk.py --data e2e_alps_full --tags base10 mp kp512 mp_kp512
```

### 7. Stage-by-stage pictures (`walkthrough/PIPELINE-WALKTHROUGH.md`)
```bash
# Jetson
cd ~/Desktop/astroloc/repo
PYTHONPATH=~/Desktop/astroloc/pydeps python3 ../pipeline_trace.py --out ../trace
# Mac, from deploy/jetson/, after copying ~/Desktop/astroloc/trace to walkthrough/pipeline_walkthrough/data (needs matplotlib)
python3 visualize_trace.py --trace walkthrough/pipeline_walkthrough/data --out walkthrough/pipeline_walkthrough
```

## Scripts (in `deploy/jetson/`)

| Script | Runs on | Purpose |
|---|---|---|
| `e2e_prepare.py` | workstation | builds the zoom-9 database, picks test photos, workstation results |
| `e2e_jetson.py` | Jetson | retrieval-only test with the TensorRT engine, compares to workstation |
| `e2e_match.py` | both | adds the matcher on top of retrieval; `--engine` runs the TensorRT retriever in the same process, `--tile-feats` uses stored tile features |
| `precompute_tile_feats.py` | both | computes and stores the SIFT features of every tile, once |
| `run_match_full.sh` | Jetson | runs `e2e_match.py` with `tegrastats` logging; the first argument tags the output files |
| `profile_match.py` | Jetson | times each matcher stage on 20 photos |
| `compare_topk.py` | Jetson | compares full-flow runs side by side (`--ks` or `--tags`) |
| `run_speed.sh` | Jetson | current vs half precision vs 512 points vs both |
| `run_speed2.sh` | Jetson | current vs small CPU fixes vs preloaded tile features vs batched LightGlue |
| `pack_tile_feats.py` | Jetson | packs the 7,602 tile feature files into one set of arrays (`--tile-pack`) |
| `test_batched_lightglue.py` | either | checks batched LightGlue against one tile at a time |
| `profile_full.py` | Jetson | time, CPU, GPU, memory and disk cost of every step (run with `tegrastats` for power) |
| `lg_bench.py` | Jetson | one LightGlue pair repeated, for `nsys profile -t cuda,nvtx` |
| `pipeline_trace.py`, `visualize_trace.py` | Jetson, Mac | record and draw every stage for one success and one failure; see `walkthrough/PIPELINE-WALKTHROUGH.md` |
| `export_onnx.py` | workstation | PyTorch → ONNX, with a parity check |
| `check_engine.py`, `compare_precision.py`, `test_images.py` | Jetson | engine sanity, FP32 vs FP16, real-image check |

## Environment

- **Workstation:** `~/venv-export` (inherits `sat_env_vision`'s CUDA torch, plus onnx, onnxruntime, onnxscript, shapely, `numpy<2.2`, kornia 0.7.1, LightGlue). No sudo.
- **Jetson:** versions are listed under *What ran on the Jetson*. `trtexec` is at `/usr/src/tensorrt/bin/trtexec`. pycuda 2022.2.2 is used by `e2e_jetson.py`. 838 GB free disk. No internet, so packages are downloaded elsewhere and copied over.
- **Mac → Jetson:** VS Code needs Local Network permission (System Settings → Privacy & Security). Or route through the workstation: `ssh -J <you>@172.24.58.65 argus@<jetson-ip>`. The Jetson's address changes (currently 172.26.104.106); run `hostname -I` on it to check.

## Known issues

1. **Pick the right checkpoint.** Nano v1 (`nano_train/checkpoints/`) scores 3% top-1; v2 scores 34.5%. There are ten full-size checkpoints, two from the latest training round (`checkpoints_faithful_lora_v2`, `checkpoints_faithful_dynamic_v2`); only `checkpoints_faithful_lora_v2` has been tested, and the repo's own OD code names it the current best model. A newer AstroLoc-tiny run, `nano_train/checkpoints_faithful_dynamic`, is untested here.
2. **LoRA checkpoints saved before commit `cc75c2b` (nano v2 among them) need their weight names remapped.** That commit renames LoRA weights (`blocks.N.` → `blocks.N.block.`). Our scripts remap the names on load, and so does the repo's own loader from `d92f5bc` on.
3. **TensorRT needs a static batch and opset 17.** Export with `--static-batch --batch 1 --opset 17`.
4. **FP16 rejected.** 2× faster but cosine 0.877 vs FP32 — enough to retrieve wrong tiles.
5. **Preprocessing is outside the engine** — resize to 224, ImageNet-normalize before calling it.
6. **Database folders are `{year}_{zoom}`** — `2021_09` is year 2021, zoom 9, not a month.
7. **The Jetson's PyTorch needs numpy ≥ 1.23**; the system has 1.21.5, so `pydeps` carries its own numpy.
8. **LightGlue's `__init__.py` imports every extractor**, and one needs torchvision. The copy in `pydeps` imports only `LightGlue` and `SIFT`.
9. **Image shape: square images.**
   - The team uses square images (Priyanka's decision), so our square test photos (1024 × 1024) match the flight input.
   - The Argus camera sensor is 4608 × 2592.
   - Priyanka's GeoTIFF evaluation (`/mnt/sdc1/astroloc/reference_db/argus_geotiff_eval`) supports the square choice. With a centre square crop, R@1 is 71.1% against 55.0% squashing the whole frame. With the matcher, the crop gives a fix on 68.8% of frames and a 5.4 km median error.

## Next steps

1. **Connect to OD** — run the tie points through the adapter and the FSW-Payload solver.
2. **Default setup: top 10 tiles + LightGlue half precision + search on the GPU + packed tile features** (0.73 s per photo). Remaining speed options: a higher power mode, the search on the GPU, tile features in one file, and shrinking the stored features (3.2 GB as 32-bit floats).
3. **Reject wrong fixes** — 4 of 109 are more than 50 km off.
4. **Argus-like square frames on the Jetson:** run square crops of the simulated Argus GeoTIFFs end to end, for accuracy and for the time at full resolution.
5. **Coordinate error for OD** — being modelled by Priyanka and Pedro.
