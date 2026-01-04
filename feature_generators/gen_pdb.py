import os
import time
import subprocess


def _acquire_lock(lock_path: str, timeout_sec: int = 1800, poll_sec: float = 1.0):
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(poll_sec)


def _release_lock(lock_path: str):
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def download_row_pdb(pdb_id, row_pdb_dir):
    """
    Download the full PDB file to row_pdb_dir/<pdb_id>.pdb.
    Safe for SLURM arrays: uses a lock + writes directly to target with wget -O.
    """
    os.makedirs(row_pdb_dir, exist_ok=True)
    pdb_file = os.path.join(row_pdb_dir, pdb_id + ".pdb")
    lock_path = os.path.join(row_pdb_dir, f".{pdb_id}.download.lock")

    # Fast path
    if os.path.exists(pdb_file) and os.path.getsize(pdb_file) > 0:
        return pdb_file

    _acquire_lock(lock_path)
    try:
        # Re-check after waiting
        if os.path.exists(pdb_file) and os.path.getsize(pdb_file) > 0:
            return pdb_file

        # Remove stale zero-byte file
        if os.path.exists(pdb_file) and os.path.getsize(pdb_file) == 0:
            os.remove(pdb_file)

        url = f"https://files.rcsb.org/view/{pdb_id}.pdb"

        try:
            subprocess.run(["wget", "-q", "-O", pdb_file, url], check=True)
        except Exception as e:
            try:
                if os.path.exists(pdb_file):
                    os.remove(pdb_file)
            except OSError:
                pass
            raise RuntimeError(f"Failed to download PDB {pdb_id} from {url}: {e}")

        if not os.path.exists(pdb_file) or os.path.getsize(pdb_file) == 0:
            raise RuntimeError(f"Downloaded PDB file is missing/empty: {pdb_file}")

        return pdb_file
    finally:
        _release_lock(lock_path)


def cleaned_row_pdb(pdb_id, chain_id, row_pdb_dir, cleaned_pdb_dir):
    """
    Extract ATOM/HETATM records for the given chain and write cleaned_pdb_dir/<pdb_id>_<chain>.pdb.
    Safe for SLURM arrays: lock + atomic write (tmp file then os.replace).
    """
    os.makedirs(cleaned_pdb_dir, exist_ok=True)
    row_pdb = os.path.join(row_pdb_dir, pdb_id + ".pdb")
    cleaned_pdb = os.path.join(cleaned_pdb_dir, f"{pdb_id}_{chain_id}.pdb")
    lock_path = os.path.join(cleaned_pdb_dir, f".{pdb_id}_{chain_id}.clean.lock")

    # Fast path
    if os.path.exists(cleaned_pdb) and os.path.getsize(cleaned_pdb) > 0:
        return cleaned_pdb

    _acquire_lock(lock_path)
    try:
        # Re-check after waiting
        if os.path.exists(cleaned_pdb) and os.path.getsize(cleaned_pdb) > 0:
            return cleaned_pdb

        # Ensure source exists
        if not os.path.exists(row_pdb) or os.path.getsize(row_pdb) == 0:
            raise FileNotFoundError(f"row_pdb is missing/empty: {row_pdb}")

        tmp_path = cleaned_pdb + f".tmp_{os.getpid()}"
        wrote_any = False

        with open(row_pdb, "r") as f_r, open(tmp_path, "w") as f_w:
            for line in f_r:
                # Only parse lines long enough
                if len(line) < 54:
                    continue
                rec = line[0:6].strip()
                chain = line[21].strip()
                if chain == chain_id and (rec == "ATOM" or rec == "HETATM"):
                    f_w.write(line)
                    wrote_any = True
                if "ENDMDL" in line:
                    break

        # Atomic replace
        os.replace(tmp_path, cleaned_pdb)

        if (not wrote_any) or os.path.getsize(cleaned_pdb) == 0:
            # Leave a clear failure for downstream parsing
            raise RuntimeError(
                f"Cleaned PDB is empty for {pdb_id}_{chain_id}. "
                f"Chain may not exist in the PDB or extraction failed: {cleaned_pdb}"
            )

        return cleaned_pdb
    finally:
        # Cleanup tmp if present (in case of crash before replace)
        try:
            tmp_path = cleaned_pdb + f".tmp_{os.getpid()}"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        _release_lock(lock_path)


def gen_mut_pdb(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, foldx_path):
    mut_info = wild_type + chain_id + mut_pos + mutant
    os.chdir(foldx_path)
    os.system(f"cp {cleaned_pdb_dir}/{pdb_id}_{chain_id}.pdb ./")
    pdbfile = "./" + pdb_id + "_" + chain_id + ".pdb"
    mut_id = pdb_id + "_" + chain_id + "_" + wild_type + mut_pos + mutant

    workdir = os.path.join(cleaned_pdb_dir, "foldx_work", mut_id)
    if not os.path.exists(workdir):
        os.makedirs(workdir, exist_ok=True)

    if not os.path.exists(f"{cleaned_pdb_dir}/{mut_id}.pdb"):
        with open(f"{foldx_path}/individual_list.txt", "w") as f:
            f.write(mut_info + ";")

        cmd = "./foldx --command=BuildModel --pdb={}  --mutant-file={}  --output-dir={} --pdb-dir={} >{}/foldx.log".format(
            pdbfile, "individual_list.txt", workdir, "./", workdir
        )
        os.system(cmd)
        os.system(f"cp {workdir}/{pdb_id}_{chain_id}_1.pdb  {cleaned_pdb_dir}/{mut_id}.pdb")

    return f"{cleaned_pdb_dir}/{mut_id}.pdb"


def gen_all_pdb(pdb_id, chain_id, mut_pos, wild_type, mutant,
                row_pdb_dir, cleaned_pdb_dir, foldx_path, mutator_backend="foldx"):
    download_row_pdb(pdb_id, row_pdb_dir)
    wild_pdb = cleaned_row_pdb(pdb_id, chain_id, row_pdb_dir, cleaned_pdb_dir)

    mutated_by_structure = False
    if mutator_backend == "foldx":
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