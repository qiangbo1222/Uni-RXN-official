
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def can_smiles(smiles: str):
    """
    Canonicalize smiles string
    """
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