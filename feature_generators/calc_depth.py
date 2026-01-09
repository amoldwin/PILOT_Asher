import os
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import get_surface, residue_depth


def _single_chain_id_from_file(pdb_file: str):
    """
    Return the only chain id present in model 0 if there is exactly one chain; else None.
    """
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("PDB", pdb_file)
    m = s[0]
    chains = [c.id for c in m.get_chains()]
    if len(chains) == 1:
        return chains[0]
    return None


def calc_depth(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('PDB', pdb_file)
    model = struct[0]

    # ESMFold/single-chain fallback: if requested chain missing but only one chain exists, use it.
    if chain_id not in model.child_dict:
        single = _single_chain_id_from_file(pdb_file)
        if single is not None:
            print(
                f"[calc_depth] WARNING: chain '{chain_id}' not found in {os.path.basename(pdb_file)}; "
                f"falling back to only chain '{single}'.",
                flush=True
            )
            chain_id = single
        else:
            available = [c.id for c in model.get_chains()]
            raise KeyError(f"Chain '{chain_id}' not found in {pdb_file}. Available chains: {available}")

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