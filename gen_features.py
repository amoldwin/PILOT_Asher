import os
import argparse
import numpy as np
import torch
import shutil
import sys
import traceback

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
    """
    step: one of
      - all
      - structures
      - precompute
      - precompute_psiblast_msa   (psiblast + msa + sasa, no hhblits)
      - precompute_hhblits        (hhblits only)
      - esm2
      - assemble

    NOTE on suffixing (mode B):
      If row_pdb_suffix is set (e.g. "_esmfold"), then this run will also suffix FASTA/pssm/rawmsa/msa/hhm/esm
      by deriving all sequence-derived IDs from FASTA basenames.
    """
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

    wild_fasta, mut_fasta, wild_seq, mut_seq, pdb_positions, pdbpos2uniprotpos_dict = gen_all_fasta(
        pdb_id, chain_id, mut_pos, wild_type, mutant,
        cleaned_pdb_dir, fasta_dir,
        mutated_by_structure=mutated_by_structure,
        mut_id=seq_id,
        cleaned_pdb_suffix=row_pdb_suffix,
        fasta_suffix=row_pdb_suffix,
    )

    wild_prot_id = os.path.basename(wild_fasta).split('.')[0]
    mut_prot_id = os.path.basename(mut_fasta).split('.')[0]

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
            wild_frawmsa = os.path.join(pssm_dir, f'{wild_prot_id}.rawmsa')
            mut_frawmsa = os.path.join(pssm_dir, f'{mut_prot_id}.rawmsa')
            _ = gen_msa(wild_prot_id, wild_seq, wild_frawmsa, msa_dir, clustalo_path, blastdbcmd_path, uniref90_path)
            _ = gen_msa(mut_prot_id, mut_seq, mut_frawmsa, msa_dir, clustalo_path, blastdbcmd_path, uniref90_path)

        if do_hhblits:
            print('Using hhblits ...')
            _ = use_hhblits(wild_prot_id, wild_fasta, hhblits_path, uniRef30_path, hhm_dir)
            _ = use_hhblits(mut_prot_id, mut_fasta, hhblits_path, uniRef30_path, hhm_dir)

        if step in ['precompute', 'precompute_psiblast_msa', 'precompute_hhblits']:
            return

    if step in ['all', 'esm2']:
        print('Using esm2 ...')
        use_esm2(wild_fasta, wild_prot_id, esm_dir)
        use_esm2(mut_fasta, mut_prot_id, esm_dir)
        if step == 'esm2':
            return

    print('Assembling features ...')

    wild_pdb_name = os.path.basename(wild_pdb).split('.')[0]
    wild_rsa = os.path.join(sasa_dir, f'{wild_pdb_name}.rsa')
    wild_asa = os.path.join(sasa_dir, f'{wild_pdb_name}.asa')

    mut_rsa = os.path.join(sasa_dir, f'{struct_id}.rsa')
    mut_asa = os.path.join(sasa_dir, f'{struct_id}.asa')

    def _ensure_sasa(pdb_file, rsa_path, asa_path):
        if not (_nonempty(rsa_path) and _nonempty(asa_path)):
            if sasa_backend.lower() == 'freesasa':
                use_freesasa(pdb_file, sasa_dir, freesasa_path)
            else:
                use_naccess(pdb_file, sasa_dir, naccess_path)

    _ensure_sasa(wild_pdb, wild_rsa, wild_asa)
    if mutated_by_structure:
        _ensure_sasa(mut_pdb, mut_rsa, mut_asa)
    else:
        if not _nonempty(mut_rsa):
            shutil.copyfile(wild_rsa, mut_rsa)
        if not _nonempty(mut_asa):
            shutil.copyfile(wild_asa, mut_asa)

    wild_rsasa, wild_asasa = calc_SASA(wild_rsa, wild_asa)
    mut_rsasa, mut_asasa = calc_SASA(mut_rsa, mut_asa)

    wild_ss = calc_ss(wild_pdb, dssp_path)
    mut_ss = calc_ss(mut_pdb if mutated_by_structure else wild_pdb, dssp_path)

    wild_depth = calc_depth(wild_pdb, chain_id)
    mut_depth = calc_depth(mut_pdb if mutated_by_structure else wild_pdb, chain_id)

    wild_fpssm = os.path.join(pssm_dir, f'{wild_prot_id}.pssm')
    mut_fpssm = os.path.join(pssm_dir, f'{mut_prot_id}.pssm')
    if not os.path.exists(wild_fpssm) or not os.path.exists(mut_fpssm):
        print('PSSM missing; running psiblast ...')
        wild_frawmsa, wild_fpssm = use_psiblast(wild_fasta, pssm_dir, psi_path, uniref90_path)
        mut_frawmsa, mut_fpssm = use_psiblast(mut_fasta, pssm_dir, psi_path, uniref90_path)
    wild_pssm, wild_res_dict = get_pssm(wild_fpssm)
    mut_pssm, mut_res_dict = get_pssm(mut_fpssm)

    wild_fhhm = os.path.join(hhm_dir, f'{wild_prot_id}.hhm')
    mut_fhhm = os.path.join(hhm_dir, f'{mut_prot_id}.hhm')
    if not os.path.exists(wild_fhhm) or not os.path.exists(mut_fhhm):
        print('HHM missing; running hhblits ...')
        wild_fhhm = use_hhblits(wild_prot_id, wild_fasta, hhblits_path, uniRef30_path, hhm_dir)
        mut_fhhm = use_hhblits(mut_prot_id, mut_fasta, hhblits_path, uniRef30_path, hhm_dir)
        wild_hhm = process_hhm(wild_fhhm, expected_len=len(wild_seq))
        mut_hhm = process_hhm(mut_fhhm, expected_len=len(mut_seq))

    wild_fmsa = os.path.join(msa_dir, f'{wild_prot_id}.msa')
    mut_fmsa = os.path.join(msa_dir, f'{mut_prot_id}.msa')
    if not os.path.exists(wild_fmsa) or not os.path.exists(mut_fmsa):
        print('MSA missing; generating ...')
        wild_frawmsa = os.path.join(pssm_dir, f'{wild_prot_id}.rawmsa')
        mut_frawmsa = os.path.join(pssm_dir, f'{mut_prot_id}.rawmsa')
        wild_fmsa = gen_msa(wild_prot_id, wild_seq, wild_frawmsa, msa_dir, clustalo_path, blastdbcmd_path, uniref90_path)
        mut_fmsa = gen_msa(mut_prot_id, mut_seq, mut_frawmsa, msa_dir, clustalo_path, blastdbcmd_path, uniref90_path)

    wild_res_freq = calc_res_freq(wild_fmsa)
    mut_res_freq = calc_res_freq(mut_fmsa)

    wild_cs = calculate_conservation_score().calculate_js_div_from_msa(wild_fmsa, blosum_background_distr)
    mut_cs = calculate_conservation_score().calculate_js_div_from_msa(mut_fmsa, blosum_background_distr)

    wild_data = {
        'sasa_res': wild_rsasa,
        'sasa_atom': wild_asasa,
        'ss_dict': wild_ss,
        'depth_dict': wild_depth,
        'pssm_dict': wild_pssm,
        'res_dict': wild_res_dict,
        'hhm_score': wild_hhm,
        'conservation_dict': wild_res_freq,
        'conservation_score': wild_cs
    }

    mut_data = {
        'sasa_res': mut_rsasa,
        'sasa_atom': mut_asasa,
        'ss_dict': mut_ss,
        'depth_dict': mut_depth,
        'pssm_dict': mut_pssm,
        'res_dict': mut_res_dict,
        'hhm_score': mut_hhm,
        'conservation_dict': mut_res_freq,
        'conservation_score': mut_cs
    }

    wild_selected_res, wild_selected_atom, wild_pdb_array = get_nearest_resindex(wild_pdb, chain_id, mut_pos, aa_num=16)
    mut_selected_res, mut_selected_atom, mut_pdb_array = get_nearest_resindex(
        mut_pdb if mutated_by_structure else wild_pdb, chain_id, mut_pos, aa_num=16
    )
    wild_res_dict2, wild_atom_dict = get_res_atom_dict(wild_pdb_array)
    mut_res_dict2, mut_atom_dict = get_res_atom_dict(mut_pdb_array)

    wild_res_feat_dict, wild_atom_feat_dict = feature_alignment(
        wild_selected_res, wild_selected_atom, wild_data, wild_res_dict2, wild_atom_dict, pdbpos2uniprotpos_dict, mut_pos
    )
    mut_res_feat_dict, mut_atom_feat_dict = feature_alignment(
        mut_selected_res, mut_selected_atom, mut_data, mut_res_dict2, mut_atom_dict, pdbpos2uniprotpos_dict, mut_pos
    )

    wild_pdb_array2, wild_res_index_pos_dict, wild_atom_index_pos = get_new_pdb_array(
        wild_pdb_array, wild_res_feat_dict.keys(), wild_atom_feat_dict.keys()
    )
    mut_pdb_array2, mut_res_index_pos_dict, mut_atom_index_pos = get_new_pdb_array(
        mut_pdb_array, mut_res_feat_dict.keys(), mut_atom_feat_dict.keys()
    )

    wild_res_edge_index, wild_res_edge_feature, wild_atom_res_index = get_edge().generate_res_edge_feature_postmut(
        mut_pos, wild_pdb_array2)
    mut_res_edge_index, mut_res_edge_feature, mut_atom_res_index = get_edge().generate_res_edge_feature_postmut(
        mut_pos, mut_pdb_array2)

    wild_atom_edge_index, wild_atom_edge_feature = get_edge().generate_atom_edge_feature(wild_pdb_array2)
    mut_atom_edge_index, mut_atom_edge_feature = get_edge().generate_atom_edge_feature(mut_pdb_array2)

    wild_res_node_feat = generate_node_feature(wild_res_feat_dict, wild_res_index_pos_dict, 105)
    mut_res_node_feat = generate_node_feature(mut_res_feat_dict, mut_res_index_pos_dict, 105)

    wild_atom_node_feat = generate_node_feature(wild_atom_feat_dict, wild_atom_index_pos, 5)
    mut_atom_node_feat = generate_node_feature(mut_atom_feat_dict, mut_atom_index_pos, 5)

    wild_mb_data = get_esm2(wild_prot_id, wild_res_index_pos_dict, pdbpos2uniprotpos_dict, esm_dir)
    mut_mb_data = get_esm2(mut_prot_id, mut_res_index_pos_dict, pdbpos2uniprotpos_dict, esm_dir)

    res_node_wt_path = f'{input_dir}/{struct_id}_RN_wt.npy'
    res_edge_wt_path = f'{input_dir}/{struct_id}_RE_wt.npy'
    res_edge_index_wt_path = f'{input_dir}/{struct_id}_REI_wt.npy'
    atom_node_wt_path = f'{input_dir}/{struct_id}_AN_wt.npy'
    atom_edge_wt_path = f'{input_dir}/{struct_id}_AE_wt.npy'
    atoms_edge_index_wt_path = f'{input_dir}/{struct_id}_AEI_wt.npy'
    atom2res_index_wt_path = f'{input_dir}/{struct_id}_I_wt.npy'
    extra_feat_wt_path = f'{input_dir}/{struct_id}_EF_wt.npy'

    res_node_mt_path = f'{input_dir}/{struct_id}_RN_mt.npy'
    res_edge_mt_path = f'{input_dir}/{struct_id}_RE_mt.npy'
    res_edge_index_mt_path = f'{input_dir}/{struct_id}_REI_mt.npy'
    atom_node_mt_path = f'{input_dir}/{struct_id}_AN_mt.npy'
    atom_edge_mt_path = f'{input_dir}/{struct_id}_AE_mt.npy'
    atom_edge_index_mt_path = f'{input_dir}/{struct_id}_AEI_mt.npy'
    atom2res_index_mt_path = f'{input_dir}/{struct_id}_I_mt.npy'
    extra_feat_mt_path = f'{input_dir}/{struct_id}_EF_mt.npy'

    np.save(res_node_wt_path, wild_res_node_feat)
    np.save(res_edge_wt_path, wild_res_edge_feature)
    np.save(res_edge_index_wt_path, wild_res_edge_index)
    np.save(atom_node_wt_path, wild_atom_node_feat)
    np.save(atom_edge_wt_path, wild_atom_edge_feature)
    np.save(atoms_edge_index_wt_path, wild_atom_edge_index)
    np.save(atom2res_index_wt_path, wild_atom_res_index)
    np.save(extra_feat_wt_path, wild_mb_data)

    np.save(res_node_mt_path, mut_res_node_feat)
    np.save(res_edge_mt_path, mut_res_edge_feature)
    np.save(res_edge_index_mt_path, mut_res_edge_index)
    np.save(atom_node_mt_path, mut_atom_node_feat)
    np.save(atom_edge_mt_path, mut_atom_edge_feature)
    np.save(atom_edge_index_mt_path, mut_atom_edge_index)
    np.save(atom2res_index_mt_path, mut_atom_res_index)
    np.save(extra_feat_mt_path, mut_mb_data)


def main():
    parser = argparse.ArgumentParser(description='Step-aware feature generation for PILOT')
    parser.add_argument('-i', '--mutant-list', dest='mutant_list', type=str, required=True,
                        help='Input file: each line "pdb_id chain_id mut_pos amino_acid", e.g. 1A23 A 32 H/S')
    parser.add_argument('-d', '--feature-dir', dest='dir', type=str, required=True,
                        help='Root path to store intermediate features and model inputs.')
    parser.add_argument('-s', '--step', dest='step', type=str, default='all',
                        choices=['all', 'structures', 'precompute', 'precompute_psiblast_msa', 'precompute_hhblits',
                                 'esm2', 'assemble'],
                        help='Which pipeline step to run.')
    parser.add_argument('--sasa-backend', dest='sasa_backend', type=str, default='naccess',
                        choices=['naccess', 'freesasa'], help='SASA backend to use.')
    parser.add_argument('--freesasa-path', dest='freesasa_path', type=str, default='freesasa',
                        help='Path to freesasa binary if not on PATH.')

    parser.add_argument('--mutator-backend', dest='mutator_backend', type=str, default='foldx',
                        choices=['foldx', 'proxy', 'rosetta'],
                        help='Mutant structure builder: foldx, rosetta, or proxy (no structural change).')

    parser.add_argument('--rosetta-scripts-path', dest='rosetta_scripts_path', type=str,
                        default=(shutil.which('rosetta_scripts.static.linuxgccrelease') or 'rosetta_scripts.static.linuxgccrelease'),
                        help='Path to rosetta_scripts.static.linuxgccrelease (or mpi binary) if not on PATH.')

    parser.add_argument('--skip-pdb-download', dest='skip_pdb_download', action='store_true',
                        help='Do not download PDBs from RCSB. Require FEATURE_DIR/row_pdb files to already exist.')

    parser.add_argument('--row-pdb-name-mode', dest='row_pdb_name_mode', default='pdb',
                        choices=['pdb', 'pdb_chain'],
                        help='How to name row_pdb inputs: {pdb_id}.pdb or {pdb_id}_{chain_id}.pdb.')
    parser.add_argument('--row-pdb-suffix', dest='row_pdb_suffix', default='',
                        help="Optional suffix appended before .pdb for row_pdb and cleaned_pdb WT files (e.g. '_esmfold').")

    # NEW: keep going on per-mutation failures
    parser.add_argument('--continue-on-error', dest='continue_on_error', action='store_true',
                        help='If set, failures for one mutation are printed to stderr and the script continues.')

    args = parser.parse_args()

    global freesasa_path
    freesasa_path = args.freesasa_path

    global rosetta_scripts_path
    rosetta_scripts_path = args.rosetta_scripts_path

    with open(args.mutant_list, 'r') as f_r:
        for lnum, line in enumerate(f_r, 1):
            raw = line.rstrip('\n')
            if not raw or raw.lstrip().startswith('#'):
                continue
            mut_info = raw.split()
            if len(mut_info) != 4:
                raise ValueError(f'Malformed mutation line at {args.mutant_list}:{lnum}: '
                                 f'expected 4 fields "pdb_id chain_id mut_pos amino_acid", got {len(mut_info)}: {raw!r}')
            pdb_id, chain_id, mut_pos, aa_field = mut_info
            if '/' not in aa_field or len(aa_field) < 3:
                raise ValueError(f'Malformed amino_acid field at {args.mutant_list}:{lnum}: expected like H/S, got {aa_field!r}')
            wild_type = aa_field[0]
            mutant = aa_field[-1]

            if not args.continue_on_error:
                gen_features(
                    pdb_id, chain_id, mut_pos, wild_type, mutant, args.dir,
                    sasa_backend=args.sasa_backend,
                    step=args.step,
                    mutator_backend=args.mutator_backend,
                    skip_pdb_download=args.skip_pdb_download,
                    row_pdb_name_mode=args.row_pdb_name_mode,
                    row_pdb_suffix=args.row_pdb_suffix,
                )
            else:
                try:
                    gen_features(
                        pdb_id, chain_id, mut_pos, wild_type, mutant, args.dir,
                        sasa_backend=args.sasa_backend,
                        step=args.step,
                        mutator_backend=args.mutator_backend,
                        skip_pdb_download=args.skip_pdb_download,
                        row_pdb_name_mode=args.row_pdb_name_mode,
                        row_pdb_suffix=args.row_pdb_suffix,
                    )
                except Exception as e:
                    header = (
                        f"[ERROR] line={lnum} mut='{pdb_id} {chain_id} {mut_pos} {wild_type}/{mutant}' "
                        f"step={args.step} backend={args.mutator_backend} suffix={args.row_pdb_suffix!r} name_mode={args.row_pdb_name_mode}"
                    )
                    print(header, file=sys.stderr, flush=True)
                    print(str(e), file=sys.stderr, flush=True)
                    print(traceback.format_exc(), file=sys.stderr, flush=True)
                    # continue to next line


if __name__ == "__main__":
    main()