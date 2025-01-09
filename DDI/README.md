# MAIL

### Molecule AI Lab

Gyoung Jin Park,  Youngbin Cho, Minji Suh, Sunyoung Kwon

## Installation

```sh
conda create -n mail python=3.12 -y
conda activate mail

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

pip install tensorflow transformers rdkit deepchem py3Dmol  
pip install matplotlib tqdm tabulate jupyter
```

---

[//]: # (## Datasets  <a name="datasets"></a>)

[//]: # ()
[//]: # (The files in `data` contain the splits used for the various datasets. Below instructions for how to download each of the different datasets used for training and evaluation:)

[//]: # ()
[//]: # ( - **PDBBind:** download the processed complexes from [zenodo]&#40;https://zenodo.org/record/6408497&#41;, unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`.)

[//]: # ( - **BindingMOAD:** download the processed complexes from [zenodo]&#40;https://zenodo.org/records/10656052&#41; under `BindingMOAD_2020_processed.tar`, unzip the directory and place it into `data` such that you have the path `data/BindingMOAD_2020_processed`.)

[//]: # ( - **DockGen:** to evaluate the performance of `DiffDock-L` with this repository you should use directly the data from BindingMOAD above. For other purposes you can download exclusively the complexes of the DockGen benchmark already processed &#40;e.g. chain cutoff&#41; from [zenodo]&#40;https://zenodo.org/records/10656052&#41; downloading the `DockGen.tar` file.)

[//]: # ( - **PoseBusters:** download the processed complexes from [zenodo]&#40;https://zenodo.org/records/8278563&#41;.)

[//]: # ( - **van der Mers:** the protein structures used for the van der Mers data augmentation strategy were downloaded [here]&#40;https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz&#41;.)



## Contact (Questions/Bugs/Requests)
Please submit a GitHub issue or contact me [rudwls2717@pusan.ac.kr](rudwls2717@pusan.ac.kr)

## Acknowledgements
Thank you for our [Laboratory](https://www.k-medai.com/).

If you find this code useful, please consider citing our work.