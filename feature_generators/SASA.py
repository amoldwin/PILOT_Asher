import os
import subprocess


def use_naccess(pdb_file, sasa_path, naccess_path):
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    if not os.path.exists(os.path.join(sasa_path, pdb_name + '.rsa')):
        os.chdir(sasa_path)
        _, _ = subprocess.Popen([naccess_path, pdb_file, '-h'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).communicate()
        os.system(f'mv {sasa_path}/5.rsa {pdb_name}.rsa')
        os.system(f'mv {sasa_path}/5.asa {pdb_name}.asa')

    return f'{sasa_path}/{pdb_name}.rsa', f'{sasa_path}/{pdb_name}.asa'


def _parse_freesasa_asasa(asasa_path):
    """
    Parse FreeSASA --print-asasa output into a {serial: area} dict.
    Lines start with ATOM/HETATM and end with an area number (Å²).
    """
    asa_map = {}
    with open(asasa_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                serial = line[6:11].strip()
                try:
                    area = float(line.split()[-1])
                except Exception:
                    continue
                asa_map[serial] = area
    return asa_map


def _write_naccess_like_asa(pdb_path, asa_map, asa_out):
    """
    Write a Naccess-like .asa file:
    - Copy ATOM/HETATM lines from the original PDB
    - Insert the SASA value (Å²) into fixed-width columns 54:62
    """
    with open(pdb_path, 'r') as fr, open(asa_out, 'w') as fw:
        for line in fr:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                serial = line[6:11].strip()
                val = asa_map.get(serial)
                if val is None:
                    continue
                base = line.rstrip('\n')
                if len(base) < 62:
                    base = base + ' ' * (62 - len(base))
                new_line = base[:54] + f"{val:8.2f}" + base[62:] + '\n'
                fw.write(new_line)


def use_freesasa(pdb_file, sasa_path, freesasa_path='freesasa', radii='naccess', probe_radius='1.4'):
    """
    Use FreeSASA to produce Naccess-like RSA and ASA files:
    - RSA: freesasa --radii=<radii> --probe-radius=<probe_radius> --format=rsa -o <pdb>.rsa
    - ASA: freesasa --radii=<radii> --probe-radius=<probe_radius> --print-asasa=<tmp>, then rewrite to <pdb>.asa
      by injecting the per-atom SASA (Å²) into columns 54:62 of the original PDB ATOM/HETATM lines.
    """
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    rsa_out = os.path.join(sasa_path, pdb_name + '.rsa')
    asa_out = os.path.join(sasa_path, pdb_name + '.asa')
    tmp_asa = os.path.join(sasa_path, pdb_name + '.asasa.tmp')

    if not (os.path.exists(rsa_out) and os.path.exists(asa_out)):
        os.makedirs(sasa_path, exist_ok=True)
        # Residue RSA (Naccess-compatible)
        if not os.path.exists(rsa_out):
            subprocess.run(
                [freesasa_path, pdb_file, f'--radii={radii}', f'--probe-radius={probe_radius}', '--format=rsa', '-o', rsa_out],
                check=True
            )
        # Atom absolute SASA
        if not os.path.exists(asa_out):
            subprocess.run(
                [freesasa_path, pdb_file, f'--radii={radii}', f'--probe-radius={probe_radius}', f'--print-asasa={tmp_asa}'],
                check=True
            )
            asa_map = _parse_freesasa_asasa(tmp_asa)
            _write_naccess_like_asa(pdb_file, asa_map, asa_out)
            try:
                os.remove(tmp_asa)
            except OSError:
                pass

    return rsa_out, asa_out


def calc_SASA(rsa_file, asa_file):
    """
    Parse residue-level RSA (percent accessible) and atom-level SASA (value in columns 54:62).
    For Naccess .asa this is Å²; for our synthesized FreeSASA .asa it is also Å².
    """
    res_naccess_output, atom_naccess_output = [], []
    res_sasa_dict, atom_sasa_dict = {}, {}

    res_naccess_output += open(rsa_file, 'r').readlines()
    atom_naccess_output += open(asa_file, 'r').readlines()

    for res_info in res_naccess_output:
        if res_info[0:3] == 'RES':
            residue_index = res_info[9:14].strip()
            relative_perc_accessible = float(res_info[22:28])
            res_sasa_dict[residue_index] = relative_perc_accessible

    for atom_info in atom_naccess_output:
        if atom_info[0:4] == 'ATOM' or atom_info[0:6] == 'HETATM':
            atom_index = atom_info[6:11].strip()
            val = float(atom_info[54:62].strip())
            atom_sasa_dict[atom_index] = val

    return res_sasa_dict, atom_sasa_dict