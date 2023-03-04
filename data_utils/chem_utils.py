import time

import numpy as np
from rdkit import Chem


def can_smiles(smiles: str):
    final_mol = Chem.MolFromSmarts(smiles)
    for atom in final_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(final_mol)


def smiles_to_vec(smiles: str):
    # return numpy array of dictionary index encoding
    dictionary = [
        "$pad$",
        ".",
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
        "He",
    ]
    dictionary = {key: value for value, key in enumerate(dictionary)}
    start_time = time.time()
    vec = np.zeros([len(smiles), len(dictionary)])
    ind = 0
    while ind < len(smiles):
        letter = smiles[ind]
        try:
            if letter == "$":
                vec[ind, dictionary[smiles[ind : ind + 5]]] = 1
                ind += 5

            elif letter.isupper():
                two_atom = smiles[ind : ind + 2]
                if two_atom in dictionary.keys():
                    vec[ind, dictionary[two_atom]] = 1
                    ind += 2

                elif two_atom in abandon_dictionary:
                    vec[ind, dictionary["Rare"]] = 1
                    ind += 2

                elif letter in dictionary.keys():
                    vec[ind, dictionary[letter]] = 1
                    ind += 1

                elif letter in abandon_dictionary:
                    vec[ind, dictionary["Rare"]] = 1
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
    # convert one-hot into array
    if time.time() - start_time > 5:
        print(f"time out at {smiles}")
    return [np.argmax(line) for line in vec]


"""
test_line = smiles_to_vec('Eu+3]')
print(test_line)
"""


def smiles_to_compact_vec(smiles: str):
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
    compact_vec = []
    sparse_vec = smiles_to_vec(smiles)
    i = 0
    while i < len(sparse_vec) - 1:
        if sparse_vec[i : i + 2] in compact_dictionary:
            compact_vec.append(compact_dictionary.index(sparse_vec[i : i + 2]) + 68)
            i += 2
        else:
            compact_vec.append(sparse_vec[i])
            i += 1
    return compact_vec


def vec_to_smiles(vec, can=True):
    dictionary = [
        "$pad$",
        ".",
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
    try:
        if int(vec[0]) == 39:
            vec = vec[1:]
        while int(vec[-1]) == 40 or int(vec[-1]) == 0:
            vec = vec[:-1]
    except:
        return "unvalid smiles"
    if can == True:
        try:
            raw_smiles = "".join(
                [dictionary[i] for i in vec if i in list(range(1, 65))]
            )
            can_smiles = ".".join(
                [
                    Chem.MolToSmiles(Chem.MolFromSmiles(r, sanitize=True))
                    for r in raw_smiles.split(".")
                ]
            )
            return can_smiles

        except:
            return "unvalid smiles"
    else:
        raw_smiles = "".join([dictionary[i] for i in vec if i in list(range(1, 66))])
        return raw_smiles
