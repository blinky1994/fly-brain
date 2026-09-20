# Third-party notices

## FLYBOARD renderer

The projection and NumPy point/glow rendering in `neuron_viewer.py` were adapted
from [NullLabTests/FLYBOARD, flyboard/render.py](https://github.com/NullLabTests/flybrain/blob/main/flyboard/render.py),
reviewed 2026-09-20. Copyright (c) 2026 NullLabTests, MIT License.
The complete license is retained in `third_party/flyboard/LICENSE`.

This adaptation uses the existing local MaleCNS worker rather than FLYBOARD's
simulator. It adds spike-count-only highlighting, source-ID matching, measured
local soma coordinates, explicit missing-position counts, visibility-aware
picking, and a GLFW/Pillow interface. It does not use FLYBOARD's stimulation
buttons, dynamics, or synthetic fallback.

## MaleCNS data

Anatomical coordinates and neuron metadata come from the existing local MaleCNS
v1.0 dataset (`data/processed/neurons.feather`). Credit: FlyEM / HHMI Janelia,
University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research.
See [the official MaleCNS project](https://male-cns.janelia.org/) for dataset
citations and terms. These data are not relicensed by the renderer's MIT license.
