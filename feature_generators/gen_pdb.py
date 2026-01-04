import os
import subprocess


def download_row_pdb(pdb_id, row_pdb_dir):
    pdb_file = os.path.join(row_pdb_dir, pdb_id + '.pdb')
    os.makedirs(row_pdb_dir, exist_ok=True)

    # If a previous run left a zero-byte file, remove it so wget/curl won't create .1/.2/.3 files
    if os.path.exists(pdb_file) and os.path.getsize(pdb_file) == 0:
        os.remove(pdb_file)

    # If file exists and is non-empty, keep it
    if os.path.exists(pdb_file) and os.path.getsize(pdb_file) > 0:
        return pdb_file

    url = f"https://files.rcsb.org/view/{pdb_id}.pdb"

    # Download directly to the target name (prevents 1OTR.pdb.4 behavior)
    try:
        subprocess.run(
            ["wget", "-q", "-O", pdb_file, url],
            check=True
        )
    except Exception as e:
        # Clean up partial file
        try:
            if os.path.exists(pdb_file):
                os.remove(pdb_file)
        except OSError:
            pass
        raise RuntimeError(f"Failed to download PDB {pdb_id} from {url}: {e}")

    if not os.path.exists(pdb_file) or os.path.getsize(pdb_file) == 0:
        raise RuntimeError(f"Downloaded PDB file is missing/empty: {pdb_file}")

    return pdb_file


def cleaned_row_pdb(pdb_id, chain_id, row_pdb_dir, cleaned_pdb_dir):
    row_pdb = os.path.join(row_pdb_dir, pdb_id + '.pdb')
    cleaned_pdb = os.path.join(cleaned_pdb_dir, pdb_id + '_' + chain_id + '.pdb')
    with open(row_pdb, 'r') as f_r, open(cleaned_pdb, 'w') as f_w:
        for line in f_r:
            info = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27], line[30:38],
                    line[38:46], line[46:54]]
            info = [i.strip() for i in info]
            if info[5] == chain_id and (info[0] == 'ATOM' or info[0] == 'HETAT'):
                f_w.write(line)
            if 'ENDMDL' in line:
                break
    return cleaned_pdb


def gen_mut_pdb(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, foldx_path):
    mut_info = wild_type + chain_id + mut_pos + mutant
    os.chdir(foldx_path)
    os.system(f'cp {cleaned_pdb_dir}/{pdb_id}_{chain_id}.pdb ./')
    pdbfile = './' + pdb_id + '_' + chain_id + '.pdb'
    mut_id = pdb_id + '_' + chain_id + '_' + wild_type + mut_pos + mutant

    workdir = os.path.join(cleaned_pdb_dir, 'foldx_work', mut_id)
    if not os.path.exists(workdir):
        os.makedirs(workdir, exist_ok=True)

    if not os.path.exists(f'{cleaned_pdb_dir}/{mut_id}.pdb'):
        print(f'{cleaned_pdb_dir}/{mut_id}.pdb')
        with open(f'{foldx_path}/individual_list.txt', 'w') as f:
            f.write(mut_info + ';')

        cmd = './foldx --command=BuildModel --pdb={}  --mutant-file={}  --output-dir={} --pdb-dir={} >{}/foldx.log'.format(
            pdbfile, 'individual_list.txt', workdir, './', workdir)

        os.system(cmd)
        os.system(f'cp {workdir}/{pdb_id}_{chain_id}_1.pdb  {cleaned_pdb_dir}/{mut_id}.pdb')
    return f'{cleaned_pdb_dir}/{mut_id}.pdb'


def gen_all_pdb(pdb_id, chain_id, mut_pos, wild_type, mutant,
                row_pdb_dir, cleaned_pdb_dir, foldx_path, mutator_backend='foldx'):
    download_row_pdb(pdb_id, row_pdb_dir)
    wild_pdb = cleaned_row_pdb(pdb_id, chain_id, row_pdb_dir, cleaned_pdb_dir)

    mutated_by_structure = False
    if mutator_backend == 'foldx':
        try:
            mut_pdb = gen_mut_pdb(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, foldx_path)
            if os.path.exists(mut_pdb):
                mutated_by_structure = True
            else:
                mut_pdb = wild_pdb
        except Exception:
            mut_pdb = wild_pdb
    else:
        mut_pdb = wild_pdb

    return wild_pdb, mut_pdb, mutated_by_structure