# Multi-Horizon Trajectory Prediction for Mixed Road Users at Urban Intersections Using Graphormer

Code for the master's thesis of **Mehran Ommani**, Chair of Data Science, University of Passau (2026).

Archived snapshot (code and trained models): [doi:10.5281/zenodo.23145476](https://doi.org/10.5281/zenodo.23145476)

The thesis compares three graph-based trajectory prediction models on the
[inD dataset](https://levelxdata.com/ind-dataset/) (drone recordings of four
unsignalised intersections in Aachen, Germany):

| Model | Role | Upstream code |
|---|---|---|
| Social-STGCNN | class-agnostic baseline | https://github.com/abduallahmohamed/Social-STGCNN |
| Multiclass-SGCN | class-aware baseline | https://github.com/Carrotsniper/Multiclass-SGCN |
| Graphormer | proposed model | https://github.com/leffff/graphormer-pyg |

All three are trained and evaluated on the same samples at 5 Hz, in three
settings: 3 s observed / 1 s predicted, 4 s / 2 s and 5 s / 3 s.

## Contents

```
pipeline/
  aggregate_ind_recordings.py    step 0: combine the 33 inD recordings into 3 CSV files
  downsample_to_5hz.py           step 1: 25 Hz -> 5 Hz
  merge_and_analyze_csvs_5hz.py  step 2: merge tables, label manoeuvres (scenarios)
  sampling_data_V6_5hz.py        step 3: trajectory-level 70/15/15 split, stratified sampling
  baking_pipeline_v28_5hz.py     step 4: build model inputs (windows, features, graphs)
notebooks/
  model_V29_final.ipynb          step 5: training, evaluation and figures for all three models
patches/
  multiclass_sgcn_model.patch    changes made to Multiclass-SGCN's model.py
results/
  all_results.json               ADE / FDE per model and setting
  eval_focal_aligned.json        final focal-aligned evaluation (990 / 984 / 995 samples)
  all_histories.json             training and validation loss histories
  training_curves.png            training curves
```

The trained model checkpoints are archived on [Zenodo](https://doi.org/10.5281/zenodo.23145476).

## Data

The inD data is **not** included. Its licence allows free non-commercial
research use but not redistribution. Request access at
https://levelxdata.com/ind-dataset/ and place the original CSV files in
`data/inD-dataset-v1/data/`.

## Setup

All scripts are run from the repository root and expect this layout:

```
data/inD-dataset-v1/data/        original inD CSV files
script/graphormer-pyg/           upstream repository
script/Multiclass-SGCN/          upstream repository, with patches/multiclass_sgcn_model.patch applied
script/Social-STGCNN/            upstream repository
lane_graphs/                     optional, see the note on lane graphs below
```

```bash
mkdir script
git clone https://github.com/leffff/graphormer-pyg            script/graphormer-pyg
git clone https://github.com/Carrotsniper/Multiclass-SGCN     script/Multiclass-SGCN
git clone https://github.com/abduallahmohamed/Social-STGCNN   script/Social-STGCNN
git -C script/graphormer-pyg  checkout cb644583802e1e76240247ba95735dc8fc6678e1
git -C script/Multiclass-SGCN checkout 68b669af67499514dd747848624a2a367a567bc7
git -C script/Social-STGCNN   checkout 333d3a57b4d2705e129b21aefefa09c79b2b9ae1
git -C script/Multiclass-SGCN apply ../../patches/multiclass_sgcn_model.patch

pip install -r requirements.txt
```

The commit hashes are the upstream versions used in the thesis.

## Running

```bash
python pipeline/aggregate_ind_recordings.py
python pipeline/downsample_to_5hz.py
python pipeline/merge_and_analyze_csvs_5hz.py
python pipeline/sampling_data_V6_5hz.py
python pipeline/baking_pipeline_v28_5hz.py
```

Then run `notebooks/model_V29_final.ipynb` from the repository root to train
and evaluate all three models. Training uses a fixed seed of 42.

## Note on lane graphs

`baking_pipeline_v28_5hz.py` can add lane nodes (agent-lane and lane-lane
edges) to the Graphormer graphs when a module `lane_graph_builder` and the files
`lane_graphs/<intersection>_lane_graph.pkl` are available. They were not
available when the graphs for the reported results were built, so those graphs
contain road-user nodes only. Running the pipeline as released reproduces this
setting.

## Citation

If you use this code, please cite the thesis:

> Ommani, M. (2026). *Multi-Horizon Trajectory Prediction for Mixed Road Users at Urban Intersections Using Graphormer.* Master's thesis, University of Passau.

and the inD dataset:

> Bock, J. et al. (2019). The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections. arXiv:1911.07602.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`).
The upstream model repositories and the inD dataset keep their own licences.
