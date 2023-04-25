# Motif_Backdoor
This is a PyThorh implementation of Motif-Backdoor:Rethinking the Backdoor Attack on Graph Neural Networks from Motifs, as described in our paper:


## Step -1: Requirement

The code requires Python >=3.6 and is built on PyTorch. Note that PyTorch may need to be [installed manually](https://pytorch.org/get-started/locally/) depending on different platforms and CUDA drivers.

## Step 0: Datasets

We provide the datasets used in our paper:

```
[ "PROTEINS","AIDS" ,"NCI1","DBLP_v1"]
```

## Step 1: Preparation

Training the benign model
```
python main.py
```

## Step 2: Attack

Training the backdoored model
```
python attack_main.py
```
