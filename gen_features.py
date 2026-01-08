import os
import argparse
import numpy as np
import torch
import shutil
from feature_generators.gen_pdb import gen_all_pdb
from feature_generators.gen_fasta import gen_all_fasta
from feature_generators.SASA import use_naccess, use_freesasa, calc_SASA
from feature_generators.evol_info import use_psiblast, use_hhblits, gen_msa, get_pssm, process_hhm, calc_res_freq
from feature_generators.use_esm2 import use_esm2
from feature_generators.calc_ss import calc_ss
from feature_generators.calc_depth import calc_depth
from feature_generators.calc_cs import calculate_conservation_score, blosum_background_distr
from feature_generators.res_atom_selection import get_nearest_resindex, get_res_atom_dict
from feature_generators.feature_alignment import feature_alignment
from feature_generators.get_edge import get_edge

foldx_path = './software/foldx'
rosetta_scripts_path = shutil.which('rosetta_scripts.static.linuxgccrelease') or 'rosetta_scripts.static.linuxgccrelease'

dssp_path = shutil.which('mkdssp') or 'mkdssp'
psi_path = shutil.which('psiblast') or 'psiblast'
blastdbcmd_path = shutil.which('blastdbcmd') or 'blastdbcmd'
clustalo_path = shutil.which('clustalo') or 'clustalo'
hhblits_path = shutil.which('hhblits') or 'hhblits'
freesasa_path = shutil.which('freesasa') or 'freesasa'

uniref90_path = '/scratch/amoldwin/datasets/uniref90/uniref90'
uniRef30_path = os.environ.get('HHBLITS_DB', './software/database/uniref30/UniRef30_2022_02')
naccess_path = './software/naccess2.1.1/naccess'


def _nonempty(p: str) -> bool:
    try:
        return os.path.exists(p) and os.path.getsize(p) > 0
    except OSError:
        return False


def get_new_pdb_array(pdb_array, res_pdbpos, atom_indexpos):
    new_pdb_array = []
    res_index_pos_dict, atom_index_pos = {}, {}
    res_pos = -10
    res_index, atom_index = 0, 0
    for atom_info in pdb_array:
        if (atom_info[6] in res_pdbpos) and (atom_info[1] in atom_indexpos):
            new_pdb_array.append(atom_info)
            atom_index_pos[atom_index] = atom_info[1]
            atom_index += 1
            if atom_info[6] != res_pos:
                res_index_pos_dict[res_index] = atom_info[6]
                res_index += 1
                res_pos = atom_info[6]
    return np.array(new_pdb_array, dtype='str'), res_index_pos_dict, atom_index_pos


def generate_node_feature(feature_dict, index_pos_dict, feature_num):
    residue_node_feature = np.empty((len(index_pos_dict), feature_num), dtype=np.float64)
    all_pdbpos = index_pos_dict.values()
    for i, pdbpos in enumerate(all_pdbpos):
        residue_node_feature[i] = [float(feature) for feature in feature_dict[pdbpos]]
    return residue_node_feature


def get_esm2(esm_name, index_pos_dict, pdb2uniprot_posdict, esm_dir):
    extra_feature = np.empty((len(index_pos_dict), 1280), dtype=np.float64)
    esm2_dir = f'{esm_dir}/{esm_name}.pt'
    esm2 = torch.load(esm2_dir)
    all_pdbpos = index_pos_dict.values()
    for i, pdbpos in enumerate(all_pdbpos):
        uniprot_pos = pdb2uniprot_posdict[pdbpos]
        extra_feature[i] = [float(feature) for feature in esm2[int(uniprot_pos)-1]]
    return extra_feature


def _ensure_dirs(dir):
    row_pdb_dir = f'{dir}/row_pdb'
    cleaned_pdb_dir = f'{dir}/cleaned_pdb'
    fasta_dir = f'{dir}/fasta'
    pssm_dir = f'{dir}/rawmsa_pssm'
    msa_dir = f'{dir}/msa'
    hhm_dir = f'{dir}/hhm'
    sasa_dir = f'{dir}/sasa'
    esm_dir = f'{dir}/esm'
    input_dir = f'{dir}/input'
    for d in [row_pdb_dir, cleaned_pdb_dir, fasta_dir, pssm_dir, msa_dir, hhm_dir, sasa_dir, esm_dir, input_dir]:
        if not os.path.exists(d):
            os.mkdir(d)
    return row_pdb_dir, cleaned_pdb_dir, fasta_dir, pssm_dir, msa_dir, hhm_dir, sasa_dir, esm_dir, input_dir


def gen_features(
    pdb_id,
    chain_id,
    mut_pos,
    wild_type,
    mutant,
    dir,
    sasa_backend='naccess',
    step='all',
    mutator_backend='foldx',
    skip_pdb_download: bool = False,
    row_pdb_name_mode: str = "pdb",
    row_pdb_suffix: str = "",
):
    row_pdb_dir, cleaned_pdb_dir, fasta_dir, pssm_dir, msa_dir, hhm_dir, sasa_dir, esm_dir, input_dir = _ensure_dirs(dir)

    pdb_chain = pdb_id + '_' + chain_id
    seq_id = pdb_id + '_' + chain_id + '_' + wild_type + mut_pos + mutant
    struct_id = seq_id + '__' + mutator_backend

    print('--------------------------------------')
    print(f'Pipeline step "{step}" for mutation {struct_id}.')
    print('--------------------------------------')

    print('Ensuring PDB and FASTA ...')
    wild_pdb, mut_pdb, mutated_by_structure, mut_id_from_pdb = gen_all_pdb(
        pdb_id, chain_id, mut_pos, wild_type, mutant,
        row_pdb_dir, cleaned_pdb_dir, foldx_path,
        mutator_backend=mutator_backend,
        rosetta_scripts_path=rosetta_scripts_path,
        download_pdb=(not skip_pdb_download),
        row_pdb_name_mode=row_pdb_name_mode,
        row_pdb_suffix=row_pdb_suffix,
    )
    struct_id = mut_id_from_pdb

    # IMPORTANT: make FASTA generation read the same (possibly suffixed) cleaned WT PDB
    wild_fasta, mut_fasta, wild_seq, mut_seq, pdb_positions, pdbpos2uniprotpos_dict = gen_all_fasta(
        pdb_id, chain_id, mut_pos, wild_type, mutant,
        cleaned_pdb_dir, fasta_dir,
        mutated_by_structure=mutated_by_structure,
        mut_id=seq_id,
        cleaned_pdb_suffix=row_pdb_suffix,
        fasta_suffix=row_pdb_suffix,
    )

    if step == 'structures':
        return

    do_sasa = step in ['all', 'precompute', 'precompute_psiblast_msa']
    do_psiblast = step in ['all', 'precompute', 'precompute_psiblast_msa']
    do_msa = step in ['all', 'precompute', 'precompute_psiblast_msa']
    do_hhblits = step in ['all', 'precompute', 'precompute_hhblits']

    if step in ['all', 'precompute', 'precompute_psiblast_msa', 'precompute_hhblits']:

        if do_sasa:
            if sasa_backend.lower() == 'freesasa':
                print('Using FreeSASA ...')
                wild_frsa, wild_fasa = use_freesasa(wild_pdb, sasa_dir, freesasa_path)
                if mutated_by_structure:
                    _ = use_freesasa(mut_pdb, sasa_dir, freesasa_path)
                else:
                    mut_frsa = os.path.join(sasa_dir, f'{struct_id}.rsa')
                    mut_fasa = os.path.join(sasa_dir, f'{struct_id}.asa')
                    if not _nonempty(mut_frsa):
                        shutil.copyfile(wild_frsa, mut_frsa)
                    if not _nonempty(mut_fasa):
                        shutil.copyfile(wild_fasa, mut_fasa)
            else:
                print('Using Naccess ...')
                wild_frsa, wild_fasa = use_naccess(wild_pdb, sasa_dir, naccess_path)
                if mutated_by_structure:
                    _ = use_naccess(mut_pdb, sasa_dir, naccess_path)
                else:
                    mut_frsa = os.path.join(sasa_dir, f'{struct_id}.rsa')
                    mut_fasa = os.path.join(sasa_dir, f'{struct_id}.asa')
                    if not _nonempty(mut_frsa):
                        shutil.copyfile(wild_frsa, mut_frsa)
                    if not _nonempty(mut_fasa):
                        shutil.copyfile(wild_fasa, mut_fasa)

        if do_psiblast:
            print('Using psiblast ...')
            wild_frawmsa, wild_fpssm = use_psiblast(wild_fasta, pssm_dir, psi_path, uniref90_path)
            mut_frawmsa, mut_fpssm = use_psiblast(mut_fasta, pssm_dir, psi_path, uniref90_path)

        if do_msa:
            print('Generating MSA files ...')
            wild_frawmsa = os.path.join(pssm_dir, f'{os.path.basename(wild_fasta).split(".")[0]}.rawmsa')
            mut_frawmsa = os.path.join(pssm_dir, f'{os.path.basename(mut_fasta).split(".")[0]}.rawmsa')
            # prot_id for msa/hhm/esm should match fasta_name to stay consistent
            wild_prot_id = os.path.basename(wild_fasta).split('.')[0]
            mut_prot_id = os.path.basename(mut_fasta).split('.')[0]
            _ = gen_msa(wild_prot_id, wild_seq, wild_frawmsa, msa_dir, clustalo_path, blastdbcmd_path, uniref90_path)
            _ = gen_msa(mut_prot_id, mut_seq, mut_frawmsa, msa_dir, clustalo_path, blastdbcmd_path, uniref90_path)

        if do_hhblits:
            print('Using hhblits ...')
            wild_prot_id = os.path.basename(wild_fasta).split('.')[0]
            mut_prot_id = os.path.basename(mut_fasta).split('.')[0]
            _ = use_hhblits(wild_prot_id, wild_fasta, hhblits_path, uniRef30_path, hhm_dir)
            _ = use_hhblits(mut_prot_id, mut_fasta, hhblits_path, uniRef30_path, hhm_dir)

        if step in ['precompute', 'precompute_psiblast_msa', 'precompute_hhblits']:
            return

    if step in ['all', 'esm2']:
        print('Using esm2 ...')
        wild_prot_id = os.path.basename(wild_fasta).split('.')[0]
        mut_prot_id = os.path.basename(mut_fasta).split('.')[0]
        use_esm2(wild_fasta, wild_prot_id, esm_dir)
        use_esm2(mut_fasta, mut_prot_id, esm_dir)
        if step == 'esm2':
            return

    # assemble section unchanged below ...
    # (rest of file identical to your attached version)
    print('Assembling features ...')
    # NOTE: omitted here for brevity in this snippet; keep your existing assemble code block.
    raise NotImplementedError("Apply the same prot_id naming changes in assemble, or keep current if you don't suffix fasta outputs.")