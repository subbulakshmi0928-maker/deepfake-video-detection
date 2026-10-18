# Sample clips

Small clips for trying the platform. Only the synthetic ones were produced by this project.

| File | What it is | Source | Licence |
| --- | --- | --- | --- |
| `sample_clip_1.mp4`, `sample_clip_2.mp4` | Synthetic cartoon-face clips from `ml-service/scripts/make_sample_video.py` (pipeline smoke tests only) | this repository | MIT |
| `real_tarun_speaking.webm` | A person speaking to camera (genuine footage) | [Wikimedia Commons: Tarun speaking 01.webm](https://commons.wikimedia.org/wiki/File:Tarun_speaking_01.webm) | CC0 1.0 |
| `real_youth_testimonial_15s.webm` | "Doing It – Youth Testimonial, 15 sec" (genuine footage) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Doing_It_-_Youth_Testimonial,_15_sec.webm) | Public domain (US federal government work) |
| `fake_kim_jong_un_representus.webm` | Face-swap deepfake PSA "Dictators – Kim Jong-Un" by RepresentUs (2020) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Dictators_-_Kim_Jong-Un_by_RepresentUs.webm) | CC BY 3.0 — © RepresentUs |
| `fake_putin_representus.webm` | Face-swap deepfake PSA "Vladimir Putin warning Americans…" by RepresentUs (2020) | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vladimir_Putin_warning_Americans_on_election_interference_and_increasing_political_divide.webm) | CC BY 3.0 — © RepresentUs |

## Verdicts with `model_87_acc_20_frames_final_data.pt` (20 frames, CPU)

| Clip | Verdict | Confidence | Faces detected / 20 |
| --- | --- | --- | --- |
| `real_tarun_speaking.webm` | REAL | 99.8 % | 20 |
| `real_youth_testimonial_15s.webm` | REAL | 96.4 % | 19 |
| `fake_kim_jong_un_representus.webm` | FAKE | 99.99 % | 20 |
| `fake_putin_representus.webm` | FAKE | 99.8 % | 20 |

The synthetic clips contain no real face, so their verdict is not meaningful; they only prove the upload → inference → storage path.
Four clips are a smoke test, not an evaluation: the checkpoint's published accuracy (87 %) was measured on its own benchmark split.
