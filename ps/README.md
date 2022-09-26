# Instructions

## Requirements
```
tqdm
networkx >= 2.5
rdkit
```

You can use conda to install the rdkit package: `conda install -c conda-forge rdkit`

## Vocabulary Construction
You can use the following command to set up your principal subgraph vocabulary:

```bash
python mol_bpe.py \
	   --data /path/to/dataset/of/smiles \
	   --vocab_size 500 \
	   --output /path/to/save/the/vocabulary
```

where vocab_size can be changed according to your need. For the ZINC250K dataset of 250,000 molecules, we recommand a vocab_size between about 300 and 500. We have provided a toy dataset in ps/data/toy_set.txt which consists of 100 molecules. If you want to kekulize the dataset (i.e. replace aromatic bonds with alternating single and double bonds), you need to further add the argument `--kekulize`.

After running the script, you will get a text file of vocabulary where the first line records whether to kekulize the molecules and the other lines record the extracted principal subgraphs. Each line is composed of three items: the smiles of the principal subgraph, the number of atoms in it, the count of appearance of it in the dataset. Here is a example:
```
CCNC(=O)c1ccccc1	11	6
```


## Subgraph-level Decomposition

The Tokenizer defined in the mol_bpe.py will decompose a molecule into a Molecule object (defined in molecule.py) of subgraph level. To decompose a molecule, you can run the following commands in python:

```python
from mol_bpe import Tokenizer

smiles = 'COc1cc(C=NNC(=O)c2ccc(O)cc2O)ccc1OCc1ccc(Cl)cc1'
# construct a tokenizer from a given vocabulary
tokenizer = Tokenizer('path/to/the/vocabulary')
# piece-level decomposition
mol = tokenizer(smiles)
print('piece level decomposition:')
print(mol)
```
Here the variable `mol` is a Molecule object defined in molecule.py, and `print(mol)` will produce a subgraph-level description of the molecule. In this case, the output might be like:
```
nodes: 
0:
                    smiles: N,
                    position: 0,
                    atom map: {6: 0}
                
1:
                    smiles: O,
                    position: 1,
                    atom map: {14: 0}
                
2:
                    smiles: O,
                    position: 2,
                    atom map: {17: 0}
                
3:
                    smiles: O,
                    position: 3,
                    atom map: {21: 0}
                
4:
                    smiles: Cl,
                    position: 4,
                    atom map: {27: 0}
                
5:
                    smiles: Cc1ccccc1,
                    position: 5,
                    atom map: {5: 0, 20: 4, 19: 5, 18: 6, 4: 1, 3: 2, 2: 3}
                
6:
                    smiles: Cc1ccccc1,
                    position: 6,
                    atom map: {29: 6, 28: 5, 26: 4, 25: 3, 24: 2, 23: 1, 22: 0}
                
7:
                    smiles: CO,
                    position: 7,
                    atom map: {1: 1, 0: 0}
                
8:
                    smiles: NC(=O)c1ccccc1,
                    position: 8,
                    atom map: {9: 2, 16: 8, 15: 7, 13: 6, 12: 5, 11: 4, 10: 3, 8: 1, 7: 0}
                
edges: 
0-5:
                    src subgraph: 0, dst subgraph: 5,
                    atom bonds: [(0, 0, rdkit.Chem.rdchem.BondType.DOUBLE)]
                
0-8:
                    src subgraph: 0, dst subgraph: 8,
                    atom bonds: [(0, 0, rdkit.Chem.rdchem.BondType.SINGLE)]
                
1-8:
                    src subgraph: 1, dst subgraph: 8,
                    atom bonds: [(0, 6, rdkit.Chem.rdchem.BondType.SINGLE)]
                
2-8:
                    src subgraph: 2, dst subgraph: 8,
                    atom bonds: [(0, 8, rdkit.Chem.rdchem.BondType.SINGLE)]
                
3-5:
                    src subgraph: 3, dst subgraph: 5,
                    atom bonds: [(0, 4, rdkit.Chem.rdchem.BondType.SINGLE)]
                
3-6:
                    src subgraph: 3, dst subgraph: 6,
                    atom bonds: [(0, 0, rdkit.Chem.rdchem.BondType.SINGLE)]
                
4-6:
                    src subgraph: 4, dst subgraph: 6,
                    atom bonds: [(0, 4, rdkit.Chem.rdchem.BondType.SINGLE)]
                
5-7:
                    src subgraph: 5, dst subgraph: 7,
                    atom bonds: [(3, 1, rdkit.Chem.rdchem.BondType.SINGLE)]
```
The results contain the details of subgraph-level nodes and edges. The first part describes the indexes of the nodes and the smiles of the subgraphs they represent. The "atom map" attribute maps the index of the atom in the molecule to the index of the atom within the subgrah. The second part tells us how these subgraph-level nodes are connected with each other. For example, the first subgraph-level edge (0-5) means that there are bonds between node 0 and node 5. The bonds only contain a double bond connecting the atom 0 in node 0 and the atom 0 in node 5. An SVG figure of subgraph-level decomposition will also be exported as **example.svg**. Below is an SVG figure for the molecule in the above example:

<img src="../images/example.svg" width="300">