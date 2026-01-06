import os
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import get_surface, residue_depth


def calc_depth(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('PDB', pdb_file)
    model = struct[0]
    chain = model[chain_id]

    # Pre-fill zeros so missing MSMS still yields depth values for all residues.
    depth_dict = {}
    for res in chain:
        res_id = res.get_id()
        pos = str(res_id[1]).strip() + str(res_id[2]).strip()
        depth_dict[pos] = 0.0

    try:
        surface = get_surface(chain)  # requires msms
    except Exception as e:
        print(f"[calc_depth] WARNING: depth unavailable (MSMS) for {os.path.basename(pdb_file)} chain={chain_id}: {e}", flush=True)
        return depth_dict

    for res in chain:
        res_id = res.get_id()
        pos = str(res_id[1]).strip() + str(res_id[2]).strip()
        try:
            depth_dict[pos] = float(residue_depth(res, surface))
        except Exception:
            depth_dict[pos] = 0.0

    return depth_dict