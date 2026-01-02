import os
import subprocess
from typing import Dict, Tuple, List

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
        return 'X'


def _read_pdb_atom_lines(pdb_file: str):
    """
    Return a list of dicts for each ATOM/HETATM line (including HOH) in file order.
    We keep HOH so we can align 1:1 with FreeSASA's atom indexing, then skip HOH at write-time.
    """
    atoms = []
    with open(pdb_file, 'r') as fr:
        for line in fr:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                res3 = line[17:20].strip()
                serial = line[6:11].strip()
                res_pos = line[22:27].strip()  # includes insertion code
                atoms.append({
                    'line': line.rstrip('\n'),
                    'serial': serial,
                    'res3': res3,
                    'res_pos': res_pos,
                    'is_hoh': (res3.upper() == 'HOH'),
                })
    return atoms


def _get_atom_areas_compatible(structure, result) -> List[float]:
    """
    FreeSASA Python API compatibility:
    - Newer: result.atomAreas() -> iterable/list
    - Older: result.atomArea(i) per atom
    """
    if hasattr(result, 'atomAreas'):
        return list(result.atomAreas())
    if hasattr(result, 'atomArea'):
        return [result.atomArea(i) for i in range(structure.nAtoms())]
    raise AttributeError("FreeSASA Result has neither atomAreas() nor atomArea(i)")


def _write_naccess_like_asa(pdb_atoms_all, area_by_serial: Dict[str, float], asa_out: str):
    """
    Write Naccess-like .asa:
    - ATOM/HETATM lines copied from PDB (non-water only)
    - SASA value (Å²) inserted into columns 54:62 (8 chars, 2 decimals)
    """
    with open(asa_out, 'w') as fw:
        for a in pdb_atoms_all:
            if a['is_hoh']:
                continue
            base = a['line']
            serial = a['serial']
            if serial not in area_by_serial:
                continue
            val = area_by_serial[serial]
            if len(base) < 62:
                base = base + ' ' * (62 - len(base))
            fw.write(base[:54] + f"{val:8.2f}" + base[62:] + '\n')


def _write_naccess_like_rsa(pdb_atoms_all, area_by_serial: Dict[str, float], rsa_out: str):
    """
    Write Naccess-compatible .rsa:
    - One 'RES' line per residue, residue index at cols 9:14, percent accessible at 22:28.
    - Percent computed using Tien 2013 MAX_ASA.
    Uses non-water atoms only.
    """
    res_total: Dict[str, float] = {}
    res_one: Dict[str, str] = {}
    order: List[str] = []
    seen = set()

    for a in pdb_atoms_all:
        if a['is_hoh']:
            continue
        serial = a['serial']
        if serial not in area_by_serial:
            continue
        res_pos = a['res_pos']
        res3 = a['res3']

        res_total[res_pos] = res_total.get(res_pos, 0.0) + float(area_by_serial[serial])
        if res_pos not in res_one:
            res_one[res_pos] = _three_to_one_safe(res3)
        if res_pos not in seen:
            seen.add(res_pos)
            order.append(res_pos)

    with open(rsa_out, 'w') as fw:
        for res_pos in order:
            aa1 = res_one.get(res_pos, 'X')
            abs_area = res_total.get(res_pos, 0.0)
            max_area = MAX_ASA.get(aa1)
            if max_area is None or max_area <= 0:
                perc = 0.0
            else:
                perc = min(100.0, 100.0 * abs_area / max_area)

            line = [' '] * 80
            line[0:3] = list('RES')
            line[9:14] = list(res_pos.rjust(5))
            line[22:28] = list(f"{perc:6.2f}")
            fw.write(''.join(line) + '\n')


def use_freesasa(pdb_file, sasa_path, freesasa_path='freesasa', radii='naccess', probe_radius='1.4'):
    """
    Produce Naccess-like RSA and ASA using FreeSASA Python API.
    - RSA: 'RES' lines with percent accessible in columns 22:28
    - ASA: ATOM/HETATM lines with absolute SASA (Å²) in columns 54:62
    """
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    rsa_out = os.path.join(sasa_path, pdb_name + '.rsa')
    asa_out = os.path.join(sasa_path, pdb_name + '.asa')

    if os.path.exists(rsa_out) and os.path.exists(asa_out):
        return rsa_out, asa_out

    os.makedirs(sasa_path, exist_ok=True)

    pdb_atoms_all = _read_pdb_atom_lines(pdb_file)

    try:
        classifier = freesasa.Classifier(radii)
        structure = freesasa.Structure(pdb_file, classifier=classifier)
    except Exception:
        structure = freesasa.Structure(pdb_file)

    params = freesasa.Parameters({'probe-radius': float(probe_radius)})
    result = freesasa.calc(structure, params)

    atom_areas = _get_atom_areas_compatible(structure, result)

    # Map PDB serials to atom areas by aligning in file order.
    # We include HOH in the alignment to keep indices consistent with FreeSASA's atom list.
    if len(atom_areas) < len(pdb_atoms_all):
        # This is unusual; fall back to best-effort truncation
        n = len(atom_areas)
        pdb_atoms_all = pdb_atoms_all[:n]
    elif len(atom_areas) > len(pdb_atoms_all):
        atom_areas = atom_areas[:len(pdb_atoms_all)]

    area_by_serial: Dict[str, float] = {}
    for a, area in zip(pdb_atoms_all, atom_areas):
        area_by_serial[a['serial']] = float(area)

    _write_naccess_like_asa(pdb_atoms_all, area_by_serial, asa_out)
    _write_naccess_like_rsa(pdb_atoms_all, area_by_serial, rsa_out)

    return rsa_out, asa_out


def calc_SASA(rsa_file, asa_file) -> Tuple[Dict[str, float], Dict[str, float]]:
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