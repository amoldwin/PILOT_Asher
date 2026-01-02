import os
import argparse
import numpy as np
import torch
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
naccess_path = './software/naccess2.1.1/naccess'
dssp_path = '../anaconda3/envs/pilot/bin/mkdssp'
psi_path = './software/ncbi-blast-2.13.0+/bin/psiblast'
clustalo_path = './software/clustalo'
hhblits_path = './software/hh-suite/build/bin/hhblits'
uniref90_path = '/scratch/amoldwin/datasets/uniref90'
uniRef30_path = './software/database/uniref30/UniRef30_2022_02'
# Set to your FreeSASA binary, or leave as 'freesasa' if it's on PATH
freesasa_path = 'freesasa'


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


def gen_features(pdb_id, chain_id, mut_pos, wild_type, mutant, dir, seq_dict,
                 sasa_backend='naccess', step='all'):
    """
    step: one of ['all', 'structures', 'precompute', 'esm2', 'assemble']
    - structures: download/clean chain PDB; build mutant; write FASTA
    - precompute: SASA (Naccess/FreeSASA), PSI-BLAST+PSSM, CLUSTALO MSA, HHblits HHM
    - esm2: ESM2 embeddings (.pt) [GPU recommended]
    - assemble: parse features and save final .npy inputs (will compute DSSP/depth on the fly)
    - all: run everything end-to-end
    """
    row_pdb_dir, cleaned_pdb_dir, fasta_dir, pssm_dir, msa_dir, hhm_dir, sasa_dir, esm_dir, input_dir = _ensure_dirs(dir)

    pdb_chain = pdb_id + '_' + chain_id
    mut_id = pdb_id + '_' + chain_id + '_' + wild_type + mut_pos + mutant
    print('--------------------------------------')
    print(f'Pipeline step "{step}" for mutation {mut_id}.')
    print('--------------------------------------')

    # Structures & FASTA (always ensured; cached if present)
    print('Ensuring PDB and FASTA ...')
    wild_pdb, mut_pdb = gen_all_pdb(pdb_id, chain_id, mut_pos, wild_type, mutant, row_pdb_dir, cleaned_pdb_dir, foldx_path)
    wild_fasta, mut_fasta, wild_seq, mut_seq, pdb_positions, pdbpos2uniprotpos_dict = \
        gen_all_fasta(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, fasta_dir)

    if step == 'structures':
        return

    # Precompute (CPU-heavy): SASA, PSI-BLAST+PSSM, CLUSTALO MSA, HHblits HHM
    if step in ['all', 'precompute']:
        if sasa_backend.lower() == 'freesasa':
            print('Using FreeSASA ...')
            _ = use_freesasa(wild_pdb, sasa_dir, freesasa_path)
            _ = use_freesasa(mut_pdb, sasa_dir, freesasa_path)
        else:
            print('Using Naccess ...')
            _ = use_naccess(wild_pdb, sasa_dir, naccess_path)
            _ = use_naccess(mut_pdb, sasa_dir, naccess_path)

        print('Using psiblast ...')
        wild_frawmsa, wild_fpssm = use_psiblast(wild_fasta, pssm_dir, psi_path, uniref90_path)
        mut_frawmsa, mut_fpssm = use_psiblast(mut_fasta, pssm_dir, psi_path, uniref90_path)

        print('Generating MSA files ...')
        _ = gen_msa(pdb_chain, wild_seq, wild_frawmsa, seq_dict, msa_dir, clustalo_path)
        _ = gen_msa(mut_id, mut_seq, mut_frawmsa, seq_dict, msa_dir, clustalo_path)

        print('Using hhblits ...')
        _ = use_hhblits(pdb_chain, wild_fasta, hhblits_path, uniRef30_path, hhm_dir)
        _ = use_hhblits(mut_id, mut_fasta, hhblits_path, uniRef30_path, hhm_dir)

        if step == 'precompute':
            return

    # ESM2 (GPU-heavy)
    if step in ['all', 'esm2']:
        print('Using esm2 ...')
        use_esm2(wild_fasta, pdb_chain, esm_dir)
        use_esm2(mut_fasta, mut_id, esm_dir)
        if step == 'esm2':
            return

    # Assemble final features and save .npy (CPU; expects ESM2 .pt to exist)
    print('Assembling features ...')
    # Resolve SASA file paths and ensure they exist (generate if missing)
    wild_rsa = os.path.join(sasa_dir, f'{pdb_id}_{chain_id}.rsa')
    wild_asa = os.path.join(sasa_dir, f'{pdb_id}_{chain_id}.asa')
    mut_rsa = os.path.join(sasa_dir, f'{mut_id}.rsa')
    mut_asa = os.path.join(sasa_dir, f'{mut_id}.asa')

    def _ensure_sasa(pdb_file, rsa_path, asa_path):
        if not (os.path.exists(rsa_path) and os.path.exists(asa_path)):
            if sasa_backend.lower() == 'freesasa':
                use_freesasa(pdb_file, sasa_dir, freesasa_path)
            else:
                use_naccess(pdb_file, sasa_dir, naccess_path)

    _ensure_sasa(wild_pdb, wild_rsa, wild_asa)
    _ensure_sasa(mut_pdb, mut_rsa, mut_asa)

    wild_rsasa, wild_asasa = calc_SASA(wild_rsa, wild_asa)
    mut_rsasa, mut_asasa = calc_SASA(mut_rsa, mut_asa)

    # DSSP and Depth (quick CPU)
    wild_ss = calc_ss(wild_pdb, dssp_path)
    mut_ss = calc_ss(mut_pdb, dssp_path)

    wild_depth = calc_depth(wild_pdb, chain_id)
    mut_depth = calc_depth(mut_pdb, chain_id)

    # PSSM (must exist or will be generated here)
    wild_fpssm = os.path.join(pssm_dir, f'{pdb_chain}.pssm')
    mut_fpssm = os.path.join(pssm_dir, f'{mut_id}.pssm')
    if not os.path.exists(wild_fpssm) or not os.path.exists(mut_fpssm):
        print('PSSM missing; running psiblast ...')
        wild_frawmsa, wild_fpssm = use_psiblast(wild_fasta, pssm_dir, psi_path, uniref90_path)
        mut_frawmsa, mut_fpssm = use_psiblast(mut_fasta, pssm_dir, psi_path, uniref90_path)
    wild_pssm, wild_res_dict = get_pssm(wild_fpssm)
    mut_pssm, mut_res_dict = get_pssm(mut_fpssm)

    # HHM (must exist or will be generated here)
    wild_fhhm = os.path.join(hhm_dir, f'{pdb_chain}.hhm')
    mut_fhhm = os.path.join(hhm_dir, f'{mut_id}.hhm')
    if not os.path.exists(wild_fhhm) or not os.path.exists(mut_fhhm):
        print('HHM missing; running hhblits ...')
        wild_fhhm = use_hhblits(pdb_chain, wild_fasta, hhblits_path, uniRef30_path, hhm_dir)
        mut_fhhm = use_hhblits(mut_id, mut_fasta, hhblits_path, uniRef30_path, hhm_dir)
    wild_hhm = process_hhm(wild_fhhm)
    mut_hhm = process_hhm(mut_fhhm)

    # MSA frequency + conservation (will generate .msa if missing)
    wild_fmsa = os.path.join(msa_dir, f'{pdb_chain}.msa')
    mut_fmsa = os.path.join(msa_dir, f'{mut_id}.msa')
    if not os.path.exists(wild_fmsa) or not os.path.exists(mut_fmsa):
        print('MSA missing; generating ...')
        wild_frawmsa = os.path.join(pssm_dir, f'{pdb_chain}.rawmsa')
        mut_frawmsa = os.path.join(pssm_dir, f'{mut_id}.rawmsa')
        # gen_msa regenerates both if needed
        wild_fmsa = gen_msa(pdb_chain, wild_seq, wild_frawmsa, seq_dict, msa_dir, clustalo_path)
        mut_fmsa = gen_msa(mut_id, mut_seq, mut_frawmsa, seq_dict, msa_dir, clustalo_path)
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

    # Select residues/atoms nearest the mutation
    wild_selected_res, wild_selected_atom, wild_pdb_array = get_nearest_resindex(wild_pdb, chain_id, mut_pos, aa_num=16)
    mut_selected_res, mut_selected_atom, mut_pdb_array = get_nearest_resindex(mut_pdb, chain_id, mut_pos, aa_num=16)
    wild_res_dict2, wild_atom_dict = get_res_atom_dict(wild_pdb_array)
    mut_res_dict2, mut_atom_dict = get_res_atom_dict(mut_pdb_array)

    # Align features
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

    # Edge features
    wild_res_edge_index, wild_res_edge_feature, wild_atom_res_index = get_edge().generate_res_edge_feature_postmut(
        mut_pos, wild_pdb_array2)
    mut_res_edge_index, mut_res_edge_feature, mut_atom_res_index = get_edge().generate_res_edge_feature_postmut(
        mut_pos, mut_pdb_array2)

    wild_atom_edge_index, wild_atom_edge_feature = get_edge().generate_atom_edge_feature(wild_pdb_array2)
    mut_atom_edge_index, mut_atom_edge_feature = get_edge().generate_atom_edge_feature(mut_pdb_array2)

    # Node features
    wild_res_node_feat = generate_node_feature(wild_res_feat_dict, wild_res_index_pos_dict, 105)
    mut_res_node_feat = generate_node_feature(mut_res_feat_dict, mut_res_index_pos_dict, 105)

    wild_atom_node_feat = generate_node_feature(wild_atom_feat_dict, wild_atom_index_pos, 5)
    mut_atom_node_feat = generate_node_feature(mut_atom_feat_dict, mut_atom_index_pos, 5)

    # ESM2 embeddings must exist
    wild_mb_data = get_esm2(pdb_chain, wild_res_index_pos_dict, pdbpos2uniprotpos_dict, esm_dir)
    mut_mb_data = get_esm2(mut_id, mut_res_index_pos_dict, pdbpos2uniprotpos_dict, esm_dir)

    # Save inputs
    res_node_wt_path = f'{input_dir}/{mut_id}_RN_wt.npy'
    res_edge_wt_path = f'{input_dir}/{mut_id}_RE_wt.npy'
    res_edge_index_wt_path = f'{input_dir}/{mut_id}_REI_wt.npy'
    atom_node_wt_path = f'{input_dir}/{mut_id}_AN_wt.npy'
    atom_edge_wt_path = f'{input_dir}/{mut_id}_AE_wt.npy'
    atoms_edge_index_wt_path = f'{input_dir}/{mut_id}_AEI_wt.npy'
    atom2res_index_wt_path = f'{input_dir}/{mut_id}_I_wt.npy'
    extra_feat_wt_path = f'{input_dir}/{mut_id}_EF_wt.npy'

    res_node_mt_path = f'{input_dir}/{mut_id}_RN_mt.npy'
    res_edge_mt_path = f'{input_dir}/{mut_id}_RE_mt.npy'
    res_edge_index_mt_path = f'{input_dir}/{mut_id}_REI_mt.npy'
    atom_node_mt_path = f'{input_dir}/{mut_id}_AN_mt.npy'
    atom_edge_mt_path = f'{input_dir}/{mut_id}_AE_mt.npy'
    atom_edge_index_mt_path = f'{input_dir}/{mut_id}_AEI_mt.npy'
    atom2res_index_mt_path = f'{input_dir}/{mut_id}_I_mt.npy'
    extra_feat_mt_path = f'{input_dir}/{mut_id}_EF_mt.npy'

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


def gen_seq_dict(uniref90_fasta_path= '/scratch/amoldwin/datasets/uniref90.fasta'):
    seq_dict = {}
    print('Reading uniref90.fasta ...')
    with open(uniref90_fasta_path, 'r') as f_r:
        seq = ''
        for line in f_r:
            if line.startswith('>'):
                if seq:
                    seq_dict[identifier] = seq
                identifier = line.strip().split()[0].split('_')[1]
                seq = ''
            else:
                seq += line.strip()
        seq_dict[identifier] = seq
    print('finished reading!')
    return seq_dict


def main():
    parser = argparse.ArgumentParser(description='Step-aware feature generation for PILOT')
    parser.add_argument('-i', '--mutant-list', dest='mutant_list', type=str, required=True,
                        help='Input file: each line "pdb_id chain_id mut_pos amino_acid", e.g. 1A23 A 32 H/S')
    parser.add_argument('-d', '--feature-dir', dest='dir', type=str, required=True,
                        help='Root path to store intermediate features and model inputs.')
    parser.add_argument('-s', '--step', dest='step', type=str, default='all',
                        choices=['all', 'structures', 'precompute', 'esm2', 'assemble'],
                        help='Which pipeline step to run.')
    parser.add_argument('--sasa-backend', dest='sasa_backend', type=str, default='naccess',
                        choices=['naccess', 'freesasa'], help='SASA backend to use.')
    parser.add_argument('--freesasa-path', dest='freesasa_path', type=str, default='freesasa',
                        help='Path to freesasa binary if not on PATH.')
    parser.add_argument('--uniref90-fasta', dest='uniref90_fasta', type=str,
                        default='/scratch/amoldwin/datasets/uniref90.fasta', help='Path to uniref90.fasta')
    args = parser.parse_args()

    global freesasa_path
    freesasa_path = args.freesasa_path

    seq_dict = gen_seq_dict(args.uniref90_fasta)

    with open(args.mutant_list, 'r') as f_r:
        for line in f_r:
            mut_info = line.strip().split()
            pdb_id = mut_info[0]
            chain_id = mut_info[1]
            mut_pos = mut_info[2]
            wild_type = mut_info[3][0]
            mutant = mut_info[3][-1]
            gen_features(pdb_id, chain_id, mut_pos, wild_type, mutant, args.dir, seq_dict,
                         sasa_backend=args.sasa_backend, step=args.step)


if __name__ == "__main__":
    main()