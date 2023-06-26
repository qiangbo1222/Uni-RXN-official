import os
import pickle
from collections import Counter

import rdkit
import rdkit.Chem as Chem
import tqdm

zinc_reactant_path = 'dataset/raw/zinc'
ustpo_reagent_path = 'data/pretrain_reaction_dataset_add_change_idx.pkl'
output_dir = 'dataset/syn_dag_from_qb/data/'

#read smi files  
zinc_reactant_smi = []
smi_paths = [os.path.join(zinc_reactant_path, f) for f in os.listdir(zinc_reactant_path) if f.endswith('.smi')]
for smi_path in tqdm.tqdm(smi_paths):
    suppl = Chem.SmilesMolSupplier(smi_path)
    for mol in suppl:
        if mol is not None:
            zinc_reactant_smi.append(Chem.MolToSmiles(mol))

ustpo_reagent_smi = []
with open(ustpo_reagent_path, 'rb') as f:
    ustpo_data = pickle.load(f)

for reaction in tqdm.tqdm(ustpo_data):
    for reagent in reaction[1].split('.'):
        ustpo_reagent_smi.append(reagent)

#remove rare reagents
ustpo_reagent_count = Counter(ustpo_reagent_smi)
ustpo_reagent_smi = list(set(ustpo_reagent_smi))
ustpo_reagent_smi = [reagent for reagent in ustpo_reagent_smi if ustpo_reagent_count[reagent] > 20]
ustpo_reagent_smi = [smi for smi in ustpo_reagent_smi if smi != '']

with open(os.path.join(output_dir, 'react_lib_smi.pkl'), 'wb') as f:
    pickle.dump({'reactant': zinc_reactant_smi, 'reagent': ustpo_reagent_smi}, f)

print(f'reactant: {len(zinc_reactant_smi)}, reagent: {len(ustpo_reagent_smi)}')