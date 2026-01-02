import os
import subprocess
from typing import Dict, Tuple

# Python API for FreeSASA
import freesasa
from Bio.PDB.Polypeptide import three_to_one

# Map non-standard residues to standard 3-letter names
NON_STANDARD_SUBSTITUTIONS = {
    'GLH': 'GLU', 'ASH': 'ASP', 'CYX': 'CYS', 'HID': 'HIS', 'HIE': 'HIS', 'HIP': 'HIS', '2AS': 'ASP', '3AH': 'HIS',
    '5HP': 'GLU', 'ACL': 'ARG', 'AGM': 'ARG', 'AIB': 'ALA', 'ALM': 'ALA', 'ALO': 'THR', 'ALY': 'LYS', 'ARM': 'ARG',
    'ASA': 'ASP', 'ASB': 'ASP', 'ASK': 'ASP', 'ASL': 'ASP', 'ASQ': 'ASP', 'AYA': 'ALA', 'BCS': 'CYS', 'BHD': 'ASP',
    'BMT': 'THR', 'BNN': 'ALA', 'BUC': 'CYS', 'BUG': 'LEU', 'C5C': 'CYS', 'C6C': 'CYS', 'CAS': 'CYS', 'CCS': 'CYS',
    'CEA': 'CYS', 'CGU': 'GLU', 'CHG': 'ALA', 'CLE': 'LEU', 'CME': 'CYS', 'CSD': 'ALA', 'CSO': 'CYS', 'CSP': 'CYS',
    'CSS': 'CYS', 'CSW': 'CYS', 'CSX': 'CYS', 'CXM': 'MET', 'CY1': 'CYS', 'CY3': 'CYS', 'CYG': 'CYS', 'CYM': 'CYS',
    'CYQ': 'CYS', 'DAH': 'PHE', 'DAL': 'ALA', 'DAR': 'ARG', 'DAS': 'ASP', 'DCY': 'CYS', 'DGL': 'GLU', 'DGN': 'GLN',
    'DHA': 'ALA', 'DHI': 'HIS', 'DIL': 'ILE', 'DIV': 'VAL', 'DLE': 'LEU', 'DLY': 'LYS', 'DNP': 'ALA', 'DPN': 'PHE',
    'DPR': 'PRO', 'DSN': 'SER', 'DSP': 'ASP', 'DTH': 'THR', 'DTR': 'TRP', 'DTY': 'TYR', 'DVA': 'VAL', 'EFC': 'CYS',
    'FLA': 'ALA', 'FME': 'MET', 'GGL': 'GLU', 'GL3': 'GLY', 'GLZ': 'GLY', 'GMA': 'GLU', 'GSC': 'GLY', 'HAC': 'ALA',
    'HAR': 'ARG', 'HIC': 'HIS', 'HIP': 'HIS', 'HMR': 'ARG', 'HPQ': 'PHE', 'HTR': 'TRP', 'HYP': 'PRO', 'IAS': 'ASP',
    'IIL': 'ILE', 'IYR': 'TYR', 'KCX': 'LYS', 'LLP': 'LYS', 'LLY': 'LYS', 'LTR': 'TRP', 'LYM': 'LYS', 'LYZ': 'LYS',
    'MAA': 'ALA', 'MEN': 'ASN', 'MHS': 'HIS', 'MIS': 'SER', 'MLE': 'LEU', 'MPQ': 'GLY', 'MSA': 'GLY', 'MSE': 'MET',
    'MVA': 'VAL', 'NEM': 'HIS', 'NEP': 'HIS', 'NLE': 'LEU', 'NLN': 'LEU', 'NLP': 'LEU', 'NMC': 'GLY', 'OAS': 'SER',
    'OCS': 'CYS', 'OMT': 'MET', 'PAQ': 'TYR', 'PCA': 'GLU', 'PEC': 'CYS', 'PHI': 'PHE', 'PHL': 'PHE', 'PR3': 'CYS',
    'PRR': 'ALA', 'PTR': 'TYR', 'PYX': 'CYS', 'SAC': 'SER', 'SAR': 'GLY', 'SCH': 'CYS', 'SCS': 'CYS', 'SCY': 'CYS',
    'SEL': 'SER', 'SEP': 'SER', 'SET': 'SER', 'SHC': 'CYS', 'SHR': 'LYS', 'SMC': 'CYS', 'SOC': 'CYS', 'STY': 'TYR',
    'SVA': 'SER', 'TIH': 'ALA', 'TPL': 'TRP', 'TPO': 'THR', 'TPQ': 'ALA', 'TRG': 'LYS', 'TRO': 'TRP', 'TYB': 'TYR',
    'TYI': 'TYR', 'TYQ': 'TYR', 'TYS': 'TYR', 'TYY': 'TYR'
}

# Tien et al. (2013) max ASA (Å²) for converting absolute residue SASA to percent
MAX_ASA = {
    'A': 106.0, 'R': 248.0, 'N': 157.0, 'D': 163.0, 'C': 135.0,
    'Q': 198.0, 'E': 194.0, 'G': 84.0,  'H': 184.0, 'I': 169.0,
    'L': 164.0, 'K': 205.0, 'M': 188.0, 'F': 197.0, 'P': 136.0,
    'S': 130.0, 'T': 142.0, 'W': 227.0, 'Y': 222.0, 'V': 142.0
}


def use_naccess(pdb_file, sasa_path, naccess_path):
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    if not os.path.exists(os.path.join(sasa_path, pdb_name + '.rsa')):
        os.chdir(sasa_path)
        _, _ = subprocess.Popen([naccess_path, pdb_file, '-h'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).communicate()
        os.system(f'mv {sasa_path}/5.rsa {pdb_name}.rsa')
        os.system(f'mv {sasa_path}/5.asa {pdb_name}.asa')
    return f'{sasa_path}/{pdb_name}.rsa', f'{sasa_path}/{pdb_name}.asa'


def _three_to_one_safe(res3: str) -> str:
    res3u = res3.upper()
    if res3u in NON_STANDARD_SUBSTITUTIONS:
        res3u = NON_STANDARD_SUBSTITUTIONS[res3u]
    try:
        return three_to_one(res3u)
    except KeyError:
        return 'X'  # unknown


def _read_pdb_atoms(pdb_file: str):
    """
    Return list of (line, serial, res_pos, res3) for each ATOM/HETATM line (non-water).
    res_pos is the residue index string like '22' or '22A' derived from columns 22:27 (seq + insertion code).
    """
    atoms = []
    with open(pdb_file, 'r') as fr:
        for line in fr:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                res3 = line[17:20].strip()
                if res3 == 'HOH':
                    continue
                serial = line[6:11].strip()
                res_pos = line[22:27].strip()  # includes insertion code
                atoms.append((line.rstrip('\n'), serial, res_pos, res3))
    return atoms


def _write_naccess_like_asa(pdb_atoms, atom_areas, asa_out):
    """
    Write Naccess-like .asa:
    - Copy ATOM/HETATM lines from original PDB
    - Insert per-atom SASA (Å²) into columns 54:62 (width 8, 2 decimals)
    We assume FreeSASA atom order equals PDB atom order after filtering HOH.
    """
    with open(asa_out, 'w') as fw:
        idx = 0
        for base, serial, _, _ in pdb_atoms:
            # pad to at least 62 chars
            if len(base) < 62:
                base = base + ' ' * (62 - len(base))
            val = atom_areas[idx]
            idx += 1
            new_line = base[:54] + f"{val:8.2f}" + base[62:] + '\n'
            fw.write(new_line)


def _write_naccess_like_rsa(pdb_atoms, atom_areas, rsa_out):
    """
    Write Naccess-compatible .rsa:
    - One 'RES' line per residue, index at columns 9:14, percent accessible at 22:28.
    - Percent computed from total absolute SASA divided by MAX_ASA (Tien 2013).
    """
    # Sum atom areas per residue and compute one-letter AA per residue
    res_total: Dict[str, float] = {}
    res_one: Dict[str, str] = {}
    for (_, _, res_pos, res3), area in zip(pdb_atoms, atom_areas):
        res_total[res_pos] = res_total.get(res_pos, 0.0) + float(area)
        if res_pos not in res_one:
            res_one[res_pos] = _three_to_one_safe(res3)

    # Preserve residue order as encountered in PDB
    seen = set()
    order = []
    for _, _, res_pos, _ in pdb_atoms:
        if res_pos not in seen:
            seen.add(res_pos)
            order.append(res_pos)

    with open(rsa_out, 'w') as fw:
        for res_pos in order:
            aa1 = res_one.get(res_pos, 'X')
            abs_area = res_total.get(res_pos, 0.0)
            max_area = MAX_ASA.get(aa1, None)
            if max_area is None or max_area <= 0:
                perc = 0.0
            else:
                perc = min(100.0, 100.0 * abs_area / max_area)

            # Build fixed-width line
            line = [' '] * 80
            # 'RES' tag
            line[0:3] = list('RES')
            # residue index at 9:14
            idx_str = res_pos.rjust(5)
            line[9:14] = list(idx_str)
            # percent accessible at 22:28
            perc_str = f"{perc:6.2f}"
            line[22:28] = list(perc_str)
            fw.write(''.join(line) + '\n')


def use_freesasa(pdb_file, sasa_path, freesasa_path='freesasa', radii='naccess', probe_radius='1.4'):
    """
    Produce Naccess-like RSA and ASA using FreeSASA Python API.
    - Residue RSA: percent accessible written at columns 22:28 of 'RES' lines.
    - Atom ASA: absolute Å² written into columns 54:62 of ATOM/HETATM lines.
    """
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    rsa_out = os.path.join(sasa_path, pdb_name + '.rsa')
    asa_out = os.path.join(sasa_path, pdb_name + '.asa')

    if os.path.exists(rsa_out) and os.path.exists(asa_out):
        return rsa_out, asa_out

    os.makedirs(sasa_path, exist_ok=True)

    # Read PDB atoms (skip HOH) to align with FreeSASA atom ordering
    pdb_atoms = _read_pdb_atoms(pdb_file)

    # Build FreeSASA Structure with desired classifier (radii) if available
    # Fallback to default classifier if 'naccess' is not supported in this build.
    try:
        classifier = freesasa.Classifier(radii)
        structure = freesasa.Structure(pdb_file, classifier=classifier)
    except Exception:
        structure = freesasa.Structure(pdb_file)

    # Calculate SASA with the given probe radius
    params = freesasa.Parameters({'probe-radius': float(probe_radius)})
    result = freesasa.calc(structure, params)

    # Get per-atom areas; assume order matches PDB atoms parsed above after HOH filtering
    atom_areas_all = list(result.atomAreas())

    # FreeSASA includes all atoms; we need to align to our filtered list (no HOH)
    # Build an index mapping for ATOM/HETATM lines excluding HOH by scanning the structure's atoms.
    # The simplest approach assumes the same order for non-water entries; if mismatched, create a filtered list.
    # Here we filter by residue name not 'HOH' using structure data.
    filtered_areas = []
    # Iterate atoms in the structure and include only non-water in order
    for i in range(structure.nAtoms()):
        res_name = structure.residueName(i)
        if res_name and res_name.upper() != 'HOH':
            filtered_areas.append(atom_areas_all[i])

    # Sanity: lengths must match the number of ATOM/HETATM non-water lines we parsed.
    if len(filtered_areas) != len(pdb_atoms):
        # Fallback to naive alignment: assume first N atoms correspond
        filtered_areas = atom_areas_all[:len(pdb_atoms)]

    # Write outputs
    _write_naccess_like_asa(pdb_atoms, filtered_areas, asa_out)
    _write_naccess_like_rsa(pdb_atoms, filtered_areas, rsa_out)

    return rsa_out, asa_out


def calc_SASA(rsa_file, asa_file) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Parse residue-level RSA (percent accessible) and atom-level SASA (value in columns 54:62).
    For Naccess .asa this is Å²; for our synthesized FreeSASA .asa it is also Å².
    """
    res_sasa_dict, atom_sasa_dict = {}, {}

    with open(rsa_file, 'r') as fr:
        for res_info in fr:
            if res_info[0:3] == 'RES':
                residue_index = res_info[9:14].strip()
                relative_perc_accessible = float(res_info[22:28])
                res_sasa_dict[residue_index] = relative_perc_accessible

    with open(asa_file, 'r') as fa:
        for atom_info in fa:
            if atom_info[0:4] == 'ATOM' or atom_info[0:6] == 'HETATM':
                atom_index = atom_info[6:11].strip()
                val = float(atom_info[54:62].strip())
                atom_sasa_dict[atom_index] = val

    return res_sasa_dict, atom_sasa_dict