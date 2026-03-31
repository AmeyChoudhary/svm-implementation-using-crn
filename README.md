# CRN-SVM

## Overview
`crn_svm.ipynb` implements a parallel-batch chemical reaction network (CRN) realization of linear SVM training driven by a dual-rail Hopf oscillator. The notebook places the Python reference model and the CRN model under the same batch schedule so that both trajectories can be compared on aligned epochs, identical minibatch structure, and the same training data.

The implementation is written for two-dimensional linear classification so that the final separating hyperplanes can be visualized directly.

## Requirements
The notebook requires:
- Python 3.10 or newer
- `numpy`
- `scipy`
- `matplotlib`
- `jupyter`

It is recommended to install the dependencies in a virtual environment. In the commands below, the environment is named `svm_crn`; this may be replaced with any preferred environment name.

```bash
python -m venv svm_crn
source svm_crn/bin/activate
pip install -r requirements.txt
```

If a virtual environment is already active, install the dependencies with `requirements.txt`:

```bash
pip install -r requirements.txt
```

For reference, the environment only requires `numpy`, `scipy`, `matplotlib`, and `notebook`.

Launch the notebook with:

```bash
jupyter notebook crn_svm.ipynb
```

## Notebook Layout (`crn_svm.ipynb`)
The notebook follows the same order as its internal section headings:
- `Setup`
- `Data And Parallel Reference Utilities`
- `CRN Engine`
- `Hopf Clock And Oscillator Readout`
- `Optional Standalone Hopf Demo`
- `Parallel SVM Reaction Modules`
- `Continuous Training Driver`
- `End-To-End Experiment`
- `Results`

## Batch Schedule
The notebook accepts a requested `batch_size` and constructs a single `BatchSchedule` object that is reused by both the Python reference and the CRN. For `N` training samples and a requested batch size `B`, the effective batch size is chosen as

- `B_eff = max { d : 1 <= d <= B and N % d == 0 }`.

If `B` does not divide `N`, the notebook falls back to the largest divisor not exceeding `B`. This guarantees that:
- every block has the same size,
- every active lane is occupied in every block,
- the Python and CRN paths traverse the same block structure.

The resulting schedule stores:
- `requested_batch_size`,
- `effective_batch_size`, 
- `active_lanes` (equals to the `effective_batch_size`),
- `n_blocks`,
- `lane_lists`, the strided sample allocation for each lane.

## Assignment Module
The parallel chemical loader follows the assignment-module structure used in the manuscript. Sample reservoirs are represented by dual-rail species of the form
- `I1p_i, I1n_i`,
- `I2p_i, I2n_i`,
- `I3p_i, I3n_i`,

where `i` is the sample index.

For each active lane `lane`, the lane-local input species are
- `X1p_l{lane}, X1n_l{lane}`,
- `X2p_l{lane}, X2n_l{lane}`,
- `Yp_l{lane}, Yn_l{lane}`.

The order species are represented as
- `C{i}_l{lane}`,
- `Ctil{i}_l{lane}`.

The assignment module proceeds in three oscillator slots:
1. `O0`: the active order species loads one sample reservoir into the lane-local input rails,
2. `O1`: the active order species is handed off to the auxiliary order species,
3. `O2`: the auxiliary order species advances to the next sample assigned to that lane.

The implementation is sparse: for a fixed lane, only those sample indices that can become active in that lane are instantiated. This matches the manuscript convention that inactive `(lane, i)` pairs may be omitted from the Python realization without changing the intended chemical logic.

## Parallel Reaction Program
The SVM network is organized as a 12-slot oscillator program:

1. assignment/load (`O0`),
2. handoff (`O1`),
3. shift (`O2`),
4. feedforward multiplication,
5. feedforward accumulation,
6. label-conditioned projection to `Qp` and `Qn`,
7. subtraction `Q = max(Qp - Qn, 0)`,
8. comparison of `Q` against `K`,
9. approximate-majority sharpening,
10. storage of update ingredients,
11. commit of the stored update ingredients into the persistent parameters,
12. reset of transient species.

Slots 4 through 9 are lane-local: each active lane carries its own feedforward intermediates, subtraction species, and comparison species. The parameter rails are shared across lanes.

## Dual-Rail Representation
Signed quantities are encoded in dual rail throughout the notebook. For a scalar quantity `v`, the encoded rails are
- `v_p = max(v, 0)`,
- `v_n = max(-v, 0)`.

The decoded value is always `v_p - v_n`.

This convention is used consistently for:
- the training samples,
- the trainable parameters,
- the Hopf oscillator state,
- intermediate comparison species,
- intermediate update species.

## Update and Commit Chemistry
The update stage follows the paper-style decomposition implemented in the notebook.

The shared update species are:
- `R1p, R1n, R2p, R2n` for the regularized carry-forward of the weight rails,
- `XY1p, XY1n, XY2p, XY2n` for the minibatch-averaged weight-gradient contribution,
- `Zp, Zn` for the stored bias rails,
- `Sbp, Sbn` for the bias-gradient contribution.

During the update slot, the network stores the quantities consumed in the next slot. During the commit slot, the persistent parameter rails
- `W1p, W1n, W2p, W2n, Bp, Bn`

are rebuilt from the stored update species. During the reset slot, lane-local transient species and shared update species are cleared, and `KgQ` and `KlQ` are restored from lane-local seed species.

## Mean Minibatch Scaling
The CRN performs one parameter update per block. Lane-wise contributions to the update species are scaled by
- `1 / effective_batch_size`.

Consequently, the notebook implements a mean minibatch update rather than a summed minibatch update. The same scaling is used in the Python reference so the two trajectories remain directly comparable.

## Hopf Oscillator Module
The notebook constructs a dual-rail Hopf clock CRN and merges it with the SVM CRN into one integrated mass-action system. The dual-rail Hopf state is represented by
- `H_Xp, H_Xn, H_Zp, H_Zn`,

which decode to
- `x = H_Xp - H_Xn`,
- `z = H_Zp - H_Zn`.

The decoded dynamics follow the Hopf normal form
- `x_dot = (mu - x^2 - z^2) x - omega z`,
- `z_dot = (mu - x^2 - z^2) z + omega x`.

The phase `theta = atan2(z, x) mod 2*pi` is partitioned into `n_clock` bins. At each time point, the active bin defines a one-hot oscillator vector whose peak value is exactly `1.0`.

The Hopf reactions themselves are always on: in the compiled CRN they are stored with oscillator index `-1`, meaning they evolve continuously while driving the oscillator-indexed SVM reactions.

## Timing
Let:
- `tau` denote the slot duration,
- `B_eff` denote the effective batch size,
- `n_blocks = N / B_eff`.

Then:
- one oscillator cycle lasts `12 * tau`,
- one epoch lasts `n_blocks * 12 * tau`.

The notebook solves the CRN as one continuous ODE system over the full requested training horizon. Epoch-level parameter histories are obtained by sampling that trajectory at evenly spaced epoch markers.

## Outputs
The notebook produces:
- Python and CRN parameter histories across epochs,
- final training and test accuracy,
- block-level diagnostics for the comparison module,
- result figures for `w1`, `w2`, and `b`,
- final hyperplane comparisons on the training and test sets,
- a CSV export of the parameter histories when the export cell is executed.

When `SAVE_RESULT_FIGURES=True`, the `Results` section writes the figures to the configured results directory at the requested DPI.
