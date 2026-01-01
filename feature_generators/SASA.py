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


def use_freesasa(pdb_file, sasa_path, freesasa_path='freesasa'):
    """
    Use FreeSASA CLI to generate Naccess-compatible RSA/ASA outputs.
    This is a drop-in replacement for use_naccess(), returning the same two file paths.
    """
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    rsa_out = os.path.join(sasa_path, pdb_name + '.rsa')
    asa_out = os.path.join(sasa_path, pdb_name + '.asa')

    if not os.path.exists(rsa_out) or not os.path.exists(asa_out):
        os.makedirs(sasa_path, exist_ok=True)
        # Emit residue-level RSA (Naccess-like)
        subprocess.run([freesasa_path, pdb_file, '-o', rsa_out, '--format=rsa'], check=True)
        # Emit atom-level ASA (Naccess-like)
        subprocess.run([freesasa_path, pdb_file, '-o', asa_out, '--format=asa'], check=True)

    return rsa_out, asa_out


def calc_SASA(rsa_file, asa_file):
    res_naccess_output, atom_naccess_output = [], []
    res_sasa_dict, atom_sasa_dict = {}, {}

    res_naccess_output += open(rsa_file, 'r').readlines()
    atom_naccess_output += open(asa_file, 'r').readlines()

    for res_info in res_naccess_output:
        if res_info[0:3] == 'RES':
            residue_index = res_info[9:14].strip()
            relative_perc_accessible = float(res_info[22:28])
            res_sasa_dict[residue_index] = relative_perc_accessible

    # 注意原子的位置也要对应起来
    for atom_info in atom_naccess_output:
        if atom_info[0:4] == 'ATOM':
            atom_index = atom_info[6:11].strip()
            # Ensure numeric type for downstream normalization
            relative_perc_accessible = float(atom_info[54:62].strip())
            atom_sasa_dict[atom_index] = relative_perc_accessible

    return res_sasa_dict, atom_sasa_dict