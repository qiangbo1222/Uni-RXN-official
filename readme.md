# Uni-RXN: An Unified Framework that Bridge the Gap between Chemical Reaction Pretraining and Conditional Molecule Generation

Chemical reactions are fundamental to drug design and organic chemistry research. Therefore there is an urgent need for a large-scale pre-training computational framework that could efficiently capture the basic rules of chemical reactions.
 In addition, applying chemical reactions to molecule generation tasks is often limited to a small number of reaction templates. In this paper, we propose a unified framework that addresses both the representation learning and generative tasks, which allows for a more holistic approach. Inspired by the organic chemistry mechanism, we developed a novel pretraining framework that enables us to incorporate inductive biases into the model. Our framework achieves state-of-the-art results on challenging downstream tasks, for example, reaction classification. In the generative tasks, such as virtual drug-like chemical library design, our model demonstrates the great potential and is able to generate more synthesizable structure analogs.

## Installation

To install the required dependencies, run:

`conda env create -f environment.yml`


## dataset construction

First download the USTPO_MIT and USTPO_STERO dataset from the ORD (https://docs.open-reaction-database.org/en/latest/) and change the path to your directory

run the following to create your own dataset
```
python dataset/syn_dags_from_qb/ create_dataset.py
python clean_all_reaction_for_pretrain.py
```

## training

```
python trainer_pretrainig_graph_pl.py
#change the path to your saved ckpt
python trainer_set_vae.py
```

## fingerprint generation and molecule generation

Before generation, you need to deploy your own MolTransformer as in (https://github.com/pschwllr/MolecularTransformer), and run `predictor.sh` to start the server. Also remember to set up your own reactant/reagents library, we provide the code to build from zinc fragment subset.

after you change the ckpt path, input molecule path, input reaction path in the code, you are able to generate your own fingerprint or analogues!! Hooray!

```
#generate Uni-RXN FP
python generation/featurize.py

#build library
python generation/build_react_lib.py

#generate structure analogues
python generation/generate_paths.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.