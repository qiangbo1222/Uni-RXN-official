import re
import time

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def can_smiles(smiles: str):
    final_mol = Chem.MolFromSmiles(smiles)
    if final_mol is None:
        final_mol = Chem.MolFromSmarts(smiles)
    if final_mol is None:
        return ''
    for atom in final_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        final_mol = Chem.RemoveHs(Chem.MolFromSmiles(Chem.MolToSmiles(final_mol)))
        Chem.SanitizeMol(final_mol)
    except:
        return ''
    return Chem.MolToSmiles(final_mol)

dictionary = [
    "$pad$",
    '.',
    "(",
    "=",
    ")",
    "#",
    "[",
    "]",
    "+",
    "-",
    "@",
    "/",
    "\\",
    "%",
    "I",
    "B",
    "H",
    "P",
    "Br",
    "S",
    "s",
    "F",
    "Cl",
    "N",
    "n",
    "O",
    "o",
    "C",
    "c",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "$bos$",
    "$eos$",
    "Ce",
    "Os",
    "Co",
    "Se",
    "Rh",
    "Ru",
    "Ag",
    "Ca",
    "Cr",
    "Ti",
    "Mn",
    "Pt",
    "Ni",
    "Sn",
    "Zn",
    "Fe",
    "Cu",
    "Mg",
    "Al",
    "Cs",
    "Si",
    "Li",
    "Pd",
    "K",
    "Na",
    "Rare",
    "$mak$",
    ]
concat_dictionary = [
    '[C]',
    '[Mn]',
    '[P]',
    '[Br-]',
    '[Al+3]',
    '[Pd+2]',
    '[Mg+2]',
    '[Cl-]',
    '[Ag+]',
    '[O+]',
    '[BH-]',
    '[NH+]',
    '[C@@H]',
    '[AlH]',
    '[Cl+]',
    '[Cu+2]',
    '[Li+]',
    '[Zn+2]',
    '[LiH]',
    '[Mg]',
    '[Rh]',
    '[NH4+]',
    '[BH4-]',
    '[nH]',
    '[P-]',
    '[Ti+4]',
    '[Ru]',
    '[N+]',
    '[SiH]',
    '[Cs+]',
    '[Os]',
    '[Ti]',
    '[Cr]',
    '[O-]',
    '[KH]',
    '[CH]',
    '[I-]',
    '[I+3]',
    '[BH3-]',
    '[Al]',
    '[C@H]',
    '[Al+]',
    '[S+]',
    '[Si]',
    '[Cu]',
    '[Fe]',
    '[Sn]',
    '[NaH]',
    '[K+]',
    '[C-]',
    '[P+]',
    '[Na+]',
    '[Zn]',
    '[HH2-]',
    '[H]',
    '[MgH2]',
    '[S-]',
    '[C@]',
    '[C@@]',
    '[Pt]',
    '[n+]',
    '[Li]',
    '[Ni]',
    '[Mn+4]',
    '[Ca+2]',
    '[Pd]',
    '[Fe+2]',
    '[B-]',
    '[Zn+]',
    '[nH+]',
    '[c-]',
    '[Na]',
    '[O-2]',
    '[F-]',
    '[N-]',
    '[OH-]'
]
abandon_dictionary = [
    "Sr",
    "Nb",
    "Pr",
    "Be",
    "Dy",
    "Ar",
    "Xe",
    "Ga",
    "La",
    "Re",
    "Y",
    "Ge",
    "Cd",
    "Au",
    "Sb",
    "Sc",
    "Sm",
    "Ta",
    "Te",
    "Tl",
    "Ir",
    "In",
    "Bi",
    "As",
    "Mo",
    "Zr",
    "V",
    "W",
    "Hg",
    "Yb",
    "Pb",
    "Ba",
    "He"
]
compact_dictionary = [
    [3, 25],
    [5, 23],
    [6, 27],
    [8, 7],
    [9, 28],
    [10, 16],
    [11, 27],
    [12, 27],
    [14, 4],
    [15, 2],
    [16, 7],
    [17, 2],
    [18, 4],
    [19, 2],
    [20, 28],
    [21, 4],
    [22, 4],
    [25, 4],
    [26, 28],
    [66, 30],
]
reverse_dictionary = dictionary + concat_dictionary
reverse_dictionary = reverse_dictionary + [reverse_dictionary[c[0]] + reverse_dictionary[c[1]] for c in compact_dictionary]
dictionary = {key: value for value, key in enumerate(dictionary)}
concat_dictionary = {key: value + len(dictionary) for value, key in enumerate(concat_dictionary)}


def smiles_to_vec(smiles: str):
    start_time = time.time()
    # return numpy array of dictionary index encoding
    vec = np.zeros([len(smiles), len(dictionary) + len(concat_dictionary)])
    #print(vec.shape)
    ind = 0
    p1 = re.compile(r'[\[](.*?)[\]]', re.S)
    while ind < len(smiles):
        letter = smiles[ind]
        try:
            if letter == "$":
                vec[ind, dictionary[smiles[ind : ind + 5]]] = 1
                ind += 5
            
            elif letter =='[':
                tmp = smiles[ind: ind + 6]
                find_word = re.findall(p1, tmp)[0]
                vec[ind, concat_dictionary['[' + find_word + ']']] = 1
                ind += (len(find_word) + 2)

            elif letter.isupper():
                two_atom = smiles[ind : ind + 2]
                if two_atom in dictionary.keys():
                    vec[ind, dictionary[two_atom]] = 1
                    ind += 2

                elif two_atom in abandon_dictionary:
                    vec[ind, dictionary['Rare']] = 1
                    ind += 2

                elif letter in dictionary.keys():
                    vec[ind, dictionary[letter]] = 1
                    ind += 1

                elif letter in abandon_dictionary:
                    vec[ind, dictionary['Rare']] = 1
                    ind += 1
                else:
                    ind += 1

            elif letter == "*":
                vec[ind, dictionary["Rare"]] = 1
                ind += 1
            
            else:
                vec[ind, dictionary[letter]] = 1
                ind += 1
        except:
            ind += 1

    # delete the columns that have all zeros
    idx = np.argwhere(np.all(vec[:, ...] == 0, axis=1))
    vec = np.delete(vec, idx, axis=0)
    #convert one-hot into array
    if time.time() - start_time > 5:
        print(f'time out at {smiles}')
    return [np.argmax(line)for line in vec]
'''
test_line = smiles_to_vec('Eu+3]')
print(test_line)
'''

def smiles_to_compact_vec(smiles: str):#using bpe to parse for reactants
    compact_vec = []
    sparse_vec = smiles_to_vec(smiles)
    i = 0
    while i < len(sparse_vec) - 1:
        if sparse_vec[i : i + 2] in compact_dictionary:
            compact_vec.append(compact_dictionary.index(sparse_vec[i : i + 2]) + len(dictionary) + len(concat_dictionary))
            i += 2
        else:
            compact_vec.append(sparse_vec[i])
            i += 1
        if i == len(sparse_vec) - 1:
            compact_vec.append(sparse_vec[i])
    return compact_vec

def vec_to_smiles(vec, can=True, sanitize=True):
    
    try:
        if int(vec[0]) == 39:
            vec = vec[1: ]
        while int(vec[-1]) == 40 or int(vec[-1]) == 0:
            vec = vec[: -1]
    except:
        return 'unvalid smiles'
    if can == True:
        raw_smiles = ''.join([reverse_dictionary[i] for i in vec if i != 67])
        #can_smiles = '.'.join([Chem.MolToSmiles(Chem.MolFromSmiles(r, sanitize=True)) for r in raw_smiles.split('.')])
        can_outputs = []
        #print(raw_smiles)
        for r in raw_smiles.split('.'):
            if r != '':
                try:
                    can_outputs.append(Chem.MolToSmiles(Chem.MolFromSmiles(r, sanitize=sanitize)))
                except:
                    continue
        if len(can_outputs) == 0:
            return 'unvalid smiles'
        else:
            can_outputs = '.'.join(can_outputs)
            return can_outputs
    else:
        raw_smiles = ''.join([reverse_dictionary[i] for i in vec if i != 67])
        return raw_smiles


#debug function
#print(smiles_to_compact_vec('.[Mg+2]cc.C=O.[Li]CCCC[BH4-].C=O'))
#print(vec_to_smiles(smiles_to_compact_vec('.[Mg+2]cc.C=O.[Li]CCCC[BH4-].C=O'), sanitize=False))
#full length 144
#full length 164 for compact
#smallest length 68
