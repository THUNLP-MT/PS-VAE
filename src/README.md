# Instructions

## Quick Links

- [Requirements](#requirements)
- [Data and Checkpoints](#data-and-checkpoints)
- [Usage](#usage)
  - [Principal Subgraph Extraction](#principal-subgraph-extraction)
  - [Training](#training)
  - [Property Optimization](#property-optimization)
  - [Constrained Property Optimization](#constrained-property-optimization)
  - [Guacamol Benchmarks](#guacamol-benchmarks)
    - [Distribution Learning](#distribution-learning)
    - [Goal-directed Benchmarks](#goal-directed-benchmarks)

## Requirements

```
python>=3.8
tqdm>=4.64.1
joblib>=1.2.0
torch>=1.8.1
pytorch-geometric  # Please refer to its documentation for installation
pytorch-lightning>=1.5.7
rdkit
networkx>=2.5
```

You can use conda to install the rdkit package: `conda install -c conda-forge rdkit`. For pytorch-geometric, please refer to its [documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to select the version that fits your OS/PyTorch/CUDA combination. For pytorch-lightning, it will **unexpectedly** upgrade your torch to fit its own latest version, so we recommand specifying its version when installing it. For example, with *torch==1.8.1*, you can specify *pytorch_lightning==1.5.7* to avoid this problem.

## Data and Checkpoints
We have provided the ZINC250K and QM9 dataset in *PS-VAE/data* as well as the train/valid/test splits used in our experiments. We also uploaded the checkpoints in our experiments to the [Google Drive](https://drive.google.com/drive/folders/1FeKZFJAM-mS_Rj4LD9biMTxKLmSVsG2V?usp=sharing). If you want to use them or run the bash scripts of our experiments, please download and extract them to *PS-VAE/ckpts*.

## Usage

Please add the root directory of our codes to the environment variable PYTHONPATH before running any python scripts below.

For example, if the path to our codes is *~/PS-VAE*, then the following command is needed:

```bash
export PYTHONPATH=~/PS-VAE/src:$PYTHONPATH
```

We have also provided bash scripts of the experiments in the following sections at *PS-VAE/scripts*. 


### Principal Subgraph Extraction
```bash
python data/mol_bpe.py \
    --data /path/to/your/dataset \
    --output /path/to/vocabfile \
    --vocab_size size_of_the_vocabulary
```

We have provided the bash script for constructing a vocabulary of 300 principal subgraphs from the training set of ZINC250K at *scripts/ps_extract.sh* as an example.

We have also provided a polished version of the principal subgraph extraction algorithm in the directory **ps**, which we recommend you to use if you are only interested in the subgraph-level decomposition of molecules.
> The vocabulary file produced by the codes **ps** has slight differences, therefore may not be compatible with the codes under the current folder.


### Training

You can train your model as follows. The data will be automatically preprocessed on the first access and produce a processed_data.pkl in the same directory of the data. Afterwards the processed data will be automatically loaded instead of reprocessed. Here we use ZINC250K as the target dataset as an example. A predictor of QED and PlogP is trained jointly with the model. The bash script is provided at *scripts/train.sh*. You can also directly run `python train.py` for detailed description of each argument.

```bash
python train.py \
	--train_set ../data/zinc250k/train.txt \
	--valid_set ../data/zinc250k/valid.txt \
	--test_set ../data/zinc250k/test.txt \
	--vocab ../ckpts/zinc250k/prop_opt/zinc_bpe_300.txt \
	--batch_size 32 \
	--shuffle \
	--alpha 0.1 \
	--beta 0 \
	--max_beta 0.01 \
	--step_beta 0.002 \
	--kl_anneal_iter 1000 \
	--kl_warmup 0 \
	--lr 1e-3 \
	--save_dir zinc_exps/ckpt/yours \
	--grad_clip 10.0 \
	--epochs 6 \
	--gpus 0 \
	--props qed logp \
	--latent_dim 56 \
	--node_hidden_dim 300 \
	--graph_embedding_dim 400 \
	--patience 3
```

### Property Optimization
You can generate molecules with optimized properties as follows. If you want to add multi-objective constraints, you can use a comma to split the properties (e.g. qed,logp). The checkpoint in ckpts/zinc250k/prop_opt/epoch5.ckpt is trained for qed and logp, which can be directly used for optimization of these two properties. The bash scripts for running the QED and PlogP optimization in our paper is located at *scripts/qed_opt.sh* and *scripts/plogp_opt.sh*, respectively.
```bash
python generate.py --eval \
  --ckpt /path/to/checkpoint \
  --props qed \
  --n_samples 10000 \
  --output_path qed.smi \
  --lr 0.1 \
  --max_iter 100 \
  --patience 3 \
  --target 2 \
  --cpus 8 \
```

### Constrained Property Optimization
This task requires optimizing molecular properties under a similarity constraints to the starting molecule. We copy the 800 molecules with the lowest Penalized logP in the test set from the offical codes of JTVAE, just as the way GCPN does. We recommend the checkpoint in ckpts/zinc250k/constraint_prop_opt/epoch5.ckpt for this task. The bash script for running this experiment in our paper is located at *scripts/constraint_prop_opt.sh*.
```bash
python generate.py \
  --ckpt /path/to/checkpoint \
  --props logp \
  --n_samples 800 \
  --output_path cons_res \
  --lr 0.1 \
  --max_iter 80 \
  --constraint_optim \
  --zinc800_logp ../data/zinc250k/jtvae_zinc800_logp.smi \
  --cpus 8 \
  --gpus 0
```


### Guacamol Benchmarks

To run the guacamol benchmarks, you need to install **guacamol** with pip:
```bash
pip install guacamol
```
Before running these benchmarks, please prepare a pretrained checkpoint on the target dataset. We have provided the checkpoints used in our paper in the *ckpts* folder.

#### Distribution Learning

You can run the distribution learning benchmarks as follows:

```bash
python guacamol_exps/distribution_learning.py \
  --ckpt /path/to/checkpoint \
  --gpu 0 \
  --output_dir results \
  --dist_file /path/to/train/set \
```

The corresponding bash script for test distribution learning on ZINC250K is located at *scripts/guaca_dist.sh*.

#### Goal-directed Benchmarks

To run the goal-directed benchmarks, please first train a predictor of these properties from the embeded molecular latent variable as follows:

```bash
python guacamol_exps/goal_directed/train.py \
  --train_set /path/to/train.txt \
  --valid_set /path/to/valid.txt \
  --save_dir /path/to/save/the/predictor \
  --shuffle \
  --batch_size 32 \
  --gpu 0 \
  --ckpt /path/to/pretrained/checkpoint
```

We have also provided the checkpoints of predictors used in our paper at *ckpts/DATASET/DATASET_guaca_goal/predictor_epoch9.ckpt* where DATASET=qm9 or zinc250k.

Then you can run the following command to test the performance on the goal-directed benchmarks:

```bash
python guacamol_exps/goal_directed/goal_directed_grad.py \
  --ckpt /path/to/pretrained/checkpoint \
  --pred_ckpt /path/to/predictor \
  --gpu 0 \
  --output_dir /path/to/save/results
```

The bash script of running the goal-directed benchmarks is provided at *scripts/guaca_goal.sh*.