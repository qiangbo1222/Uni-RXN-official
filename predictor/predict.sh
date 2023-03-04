cd /root/jupyter/DAG_Transformer/mol_transformer/MolecularTransformer;
conda activate mtransformer;
export CUDA_VISIBLE_DEVICES="1" ;
python server.py --config available_models/mtransformer_example_server.conf.json;