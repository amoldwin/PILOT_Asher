import os
import time
import subprocess
import shutil
from typing import Optional, Literal


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


def _nonempty(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


RowPdbNameMode = Literal["pdb", "pdb_chain"]


def row_pdb_path(row_pdb_dir: str, pdb_id: str, chain_id: str, name_mode: RowPdbNameMode, suffix: str = "") -> str:
    """
    Construct the expected row_pdb path for a given naming mode.

    name_mode:
      - "pdb":       {pdb_id}{suffix}.pdb
      - "pdb_chain": {pdb_id}_{chain_id}{suffix}.pdb

    suffix should include leading underscore if desired, e.g. "_esmfold".
    """
    if suffix and not suffix.startswith("_"):
        # allow "esmfold" but normalize to "_esmfold"
        suffix = "_" + suffix

    if name_mode == "pdb":
        base = f"{pdb_id}{suffix}"
    elif name_mode == "pdb_chain":
        base = f"{pdb_id}_{chain_id}{suffix}"
    else:
        raise ValueError(f"Unknown row_pdb_name_mode: {name_mode}")
    return os.path.join(row_pdb_dir, base + ".pdb")


def cleaned_pdb_path(cleaned_pdb_dir: str, pdb_id: str, chain_id: str, suffix: str = "") -> str:
    """
    Construct WT cleaned PDB output path. suffix should include leading underscore if desired.
    """
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    return os.path.join(cleaned_pdb_dir, f"{pdb_id}_{chain_id}{suffix}.pdb")


def download_row_pdb(pdb_id, row_pdb_dir, out_path: Optional[str] = None):
    """
    Download the full PDB file from RCSB.

    By default writes row_pdb_dir/{pdb_id}.pdb, but if out_path is given, writes there instead.
    Safe for SLURM arrays: uses a lock + writes directly to target with wget -O.
    """
    os.makedirs(row_pdb_dir, exist_ok=True)
    pdb_file = out_path or os.path.join(row_pdb_dir, pdb_id + ".pdb")
    lock_key = os.path.basename(pdb_file)
    lock_path = os.path.join(row_pdb_dir, f".{lock_key}.download.lock")

    # Fast path
    if _nonempty(pdb_file):
        return pdb_file

    _acquire_lock(lock_path)
    try:
        # Re-check after waiting
        if _nonempty(pdb_file):
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

        if not _nonempty(pdb_file):
            raise RuntimeError(f"Downloaded PDB file is missing/empty: {pdb_file}")

        return pdb_file
    finally:
        _release_lock(lock_path)


def cleaned_row_pdb(
    pdb_id: str,
    chain_id: str,
    row_pdb_dir: str,
    cleaned_pdb_dir: str,
    row_pdb_file: Optional[str] = None,
    cleaned_out_path: Optional[str] = None,
):
    """
    Extract ATOM/HETATM records for the given chain and write cleaned_pdb_dir/{pdb_id}_{chain_id}.pdb
    (or cleaned_out_path if given).

    row_pdb_file:
      - if provided, read from that file instead of row_pdb_dir/{pdb_id}.pdb
    cleaned_out_path:
      - if provided, write to that exact path (still in cleaned_pdb_dir typically)

    Safe for SLURM arrays: lock + atomic write (tmp file then os.replace).
    """
    os.makedirs(cleaned_pdb_dir, exist_ok=True)

    row_pdb = row_pdb_file or os.path.join(row_pdb_dir, pdb_id + ".pdb")
    cleaned_pdb = cleaned_out_path or os.path.join(cleaned_pdb_dir, f"{pdb_id}_{chain_id}.pdb")
    lock_key = os.path.basename(cleaned_pdb)
    lock_path = os.path.join(cleaned_pdb_dir, f".{lock_key}.clean.lock")

    # Fast path
    if _nonempty(cleaned_pdb):
        return cleaned_pdb

    _acquire_lock(lock_path)
    try:
        # Re-check after waiting
        if _nonempty(cleaned_pdb):
            return cleaned_pdb

        # Ensure source exists
        if not _nonempty(row_pdb):
            raise FileNotFoundError(f"row_pdb is missing/empty: {row_pdb}")

        tmp_path = cleaned_pdb + f".tmp_{os.getpid()}"
        wrote_any = False

        with open(row_pdb, "r") as f_r, open(tmp_path, "w") as f_w:
            for line in f_r:
                if len(line) < 54:
                    continue
                rec = line[0:6].strip()
                chain = line[21].strip()
                if chain == chain_id and (rec == "ATOM" or rec == "HETATM"):
                    f_w.write(line)
                    wrote_any = True
                if "ENDMDL" in line:
                    break

        os.replace(tmp_path, cleaned_pdb)

        if (not wrote_any) or (not _nonempty(cleaned_pdb)):
            raise RuntimeError(
                f"Cleaned PDB is empty for {pdb_id}_{chain_id}. "
                f"Chain may not exist in the PDB or extraction failed: {cleaned_pdb}"
            )

        return cleaned_pdb
    finally:
        try:
            tmp_path = cleaned_pdb + f".tmp_{os.getpid()}"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        _release_lock(lock_path)


def backend_tagged_mut_id(pdb_id: str, chain_id: str, mut_pos: str, wild_type: str, mutant: str, mutator_backend: str) -> str:
    base = f"{pdb_id}_{chain_id}_{wild_type}{mut_pos}{mutant}"
    return f"{base}__{mutator_backend}"


def gen_mut_pdb_foldx(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, foldx_path, mut_id: str):
    mut_info = wild_type + chain_id + mut_pos + mutant
    os.chdir(foldx_path)
    os.system(f"cp {cleaned_pdb_dir}/{pdb_id}_{chain_id}.pdb ./")
    pdbfile = "./" + pdb_id + "_" + chain_id + ".pdb"

    workdir = os.path.join(cleaned_pdb_dir, "foldx_work", mut_id)
    if not os.path.exists(workdir):
        os.makedirs(workdir, exist_ok=True)

    out_pdb = f"{cleaned_pdb_dir}/{mut_id}.pdb"

    if not os.path.exists(out_pdb):
        with open(f"{foldx_path}/individual_list.txt", "w") as f:
            f.write(mut_info + ";")

        cmd = "./foldx --command=BuildModel --pdb={}  --mutant-file={}  --output-dir={} --pdb-dir={} >{}/foldx.log".format(
            pdbfile, "individual_list.txt", workdir, "./", workdir
        )
        os.system(cmd)
        os.system(f"cp {workdir}/{pdb_id}_{chain_id}_1.pdb  {out_pdb}")

    return out_pdb


def _parse_mut_pos(mut_pos: str):
    s = mut_pos.strip()
    if not s:
        raise ValueError("Empty mut_pos")
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i == 0:
        raise ValueError(f"mut_pos does not start with digits: {mut_pos!r}")
    resnum = int(s[:i])
    icode = s[i:].strip()
    return resnum, icode


def _write_rosettascripts_xml(
    xml_path: str,
    chain_id: str,
    resnum: int,
    icode: str,
    mutant_aa: str,
    pack_radius: float = 8.0,
    resfile_path: str = None
):
    """
    Write a powerful RosettaScripts XML using FastRelax, backbone/sidechain movement,
    and resfile-driven mutation, suitable for ddG prediction (Rosetta 3.13+).
    """
    if resfile_path is None:
        resfile_path = os.path.join(os.path.dirname(xml_path), "mutate.resfile")
    resfile_path = os.path.abspath(resfile_path).replace("\\", "/")
    xml = f"""<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="ref2015" weights="ref2015"/>
  </SCOREFXNS>
  <TASKOPERATIONS>
    <ReadResfile name="rrf" filename="{resfile_path}"/>
    <InitializeFromCommandline name="init"/>
  </TASKOPERATIONS>
  <MOVERS>
    <FastRelax name="relax" scorefxn="ref2015" repeats="5" task_operations="rrf,init"/>
  </MOVERS>
  <PROTOCOLS>
    <Add mover="relax"/>
  </PROTOCOLS>
  <OUTPUT scorefxn="ref2015"/>
</ROSETTASCRIPTS>
"""
    with open(xml_path, "w") as f:
        f.write(xml)


def gen_mut_pdb_rosetta(
    pdb_id: str,
    chain_id: str,
    mut_pos: str,
    wild_type: str,
    mutant: str,
    cleaned_pdb_dir: str,
    mut_id: str,
    rosetta_scripts_path: str = "rosetta_scripts.static.linuxgccrelease",
    pack_radius: float = 8.0,
    nstruct: int = 5  # Number of decoys/outputs for robustness
):
    import glob
    in_pdb = os.path.join(cleaned_pdb_dir, f"{pdb_id}_{chain_id}.pdb")
    out_pdb = os.path.join(cleaned_pdb_dir, f"{mut_id}.pdb")

    if _nonempty(out_pdb):
        return out_pdb

    workdir = os.path.join(cleaned_pdb_dir, "rosetta_work", mut_id)
    os.makedirs(workdir, exist_ok=True)

    lock = os.path.join(workdir, ".rosetta.lock")
    _acquire_lock(lock)
    try:
        if _nonempty(out_pdb):
            return out_pdb

        if not _nonempty(in_pdb):
            raise FileNotFoundError(f"Missing WT cleaned pdb: {in_pdb}")

        resnum, icode = _parse_mut_pos(mut_pos)

        xml_path = os.path.join(workdir, "mutate_repack_minimize.xml")
        resfile_path = os.path.join(workdir, "mutate.resfile")

        # Always use chain A for Rosetta resfile mutation
        chain_for_resfile = "A"

        with open(resfile_path, "w") as f:
            f.write("NATRO\n")
            f.write("start\n")
            f.write(f"{resnum}{icode if icode else ''} {chain_for_resfile} PIKAA {mutant}\n")

        _write_rosettascripts_xml(
            xml_path,
            chain_id=chain_for_resfile,
            resnum=resnum,
            icode=icode,
            mutant_aa=mutant,
            pack_radius=pack_radius,
            resfile_path=resfile_path,
        )

        # Run Rosetta with multiple decoys for robustness (-nstruct N)
        cmd = [
            rosetta_scripts_path,
            "-s", in_pdb,
            "-parser:protocol", xml_path,
            "-out:path:all", workdir,
            "-overwrite",
            "-mute", "all",
            "-nstruct", str(nstruct)
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        log_path = os.path.join(workdir, "rosetta.log")
        with open(log_path, "w") as f:
            f.write("CMD: " + " ".join(cmd) + "\n\n")
            f.write("STDOUT:\n" + proc.stdout + "\n\n")
            f.write("STDERR:\n" + proc.stderr + "\n")

        if proc.returncode != 0:
            raise RuntimeError(f"RosettaScripts failed (exit={proc.returncode}). See {log_path}")

        # Find all output PDBs (one per nstruct)
        pdb_candidates = sorted(
            glob.glob(os.path.join(workdir, "*.pdb")),
            key=lambda p: os.path.getmtime(p)
        )
        pdb_candidates = [p for p in pdb_candidates if os.path.abspath(p) != os.path.abspath(in_pdb)]

        if not pdb_candidates:
            raise RuntimeError(f"RosettaScripts did not produce a PDB in {workdir}. See {log_path}")

        # (Advanced: If score file is present, select best scored PDB)
        scorefile = os.path.join(workdir, "score.sc")
        best_pdb = None
        if os.path.exists(scorefile):
            # Parse the score file for the lowest total_score
            scores = {}
            with open(scorefile) as f:
                header = None
                for line in f:
                    if line.startswith("SCORE:"):
                        cols = line.strip().split()
                        if header is None:
                            header = cols
                        else:
                            cols_dict = dict(zip(header, cols))
                            if "total_score" in cols_dict and "description" in cols_dict:
                                try:
                                    score = float(cols_dict["total_score"])
                                    pdb_name = cols_dict["description"]
                                    scores[pdb_name] = score
                                except Exception:
                                    continue
            # Find PDB basename with min score
            if scores:
                min_pdb = min(scores.keys(), key=lambda k: scores[k])
                # Append .pdb if not present
                min_pdb = min_pdb if min_pdb.endswith(".pdb") else min_pdb + ".pdb"
                min_pdb_path = os.path.join(workdir, min_pdb)
                if os.path.exists(min_pdb_path):
                    best_pdb = min_pdb_path
        if not best_pdb:
            # Default: most recent PDB (should work for small nstruct)
            best_pdb = max(pdb_candidates, key=os.path.getmtime)

        shutil.copyfile(best_pdb, out_pdb)

        if not _nonempty(out_pdb):
            raise RuntimeError(f"Rosetta output PDB missing/empty: {out_pdb}")

        return out_pdb
    finally:
        _release_lock(lock)


def gen_all_pdb(
    pdb_id,
    chain_id,
    mut_pos,
    wild_type,
    mutant,
    row_pdb_dir,
    cleaned_pdb_dir,
    foldx_path,
    mutator_backend="foldx",
    rosetta_scripts_path: Optional[str] = None,
    download_pdb: bool = True,
    row_pdb_name_mode: RowPdbNameMode = "pdb",
    row_pdb_suffix: str = "",
):
    """
    row_pdb_name_mode:
      - "pdb":       use row_pdb/{pdb_id}{suffix}.pdb
      - "pdb_chain": use row_pdb/{pdb_id}_{chain_id}{suffix}.pdb   (ESMFold convention)

    row_pdb_suffix:
      append to both row_pdb and cleaned_pdb WT filenames for this run, e.g. "_esmfold".

    If download_pdb=False, do not fetch from RCSB; require the expected row_pdb file exists.
    """
    rp = row_pdb_path(row_pdb_dir, pdb_id, chain_id, row_pdb_name_mode, suffix=row_pdb_suffix)
    wp = cleaned_pdb_path(cleaned_pdb_dir, pdb_id, chain_id, suffix=row_pdb_suffix)

    if download_pdb:
        # Download only makes sense for real PDB IDs; still allow it if requested.
        download_row_pdb(pdb_id, row_pdb_dir, out_path=rp)
    else:
        if not _nonempty(rp):
            raise FileNotFoundError(
                f"--skip-pdb-download was set, but required file is missing/empty: {rp}\n"
                f"Provide it (copy/symlink) or run without --skip-pdb-download.\n"
                f"(hint: you set row_pdb_name_mode={row_pdb_name_mode} row_pdb_suffix={row_pdb_suffix!r})"
            )

    wild_pdb = cleaned_row_pdb(
        pdb_id, chain_id, row_pdb_dir, cleaned_pdb_dir,
        row_pdb_file=rp,
        cleaned_out_path=wp,
    )

    mut_id = backend_tagged_mut_id(pdb_id, chain_id, mut_pos, wild_type, mutant, mutator_backend)

    mutated_by_structure = False
    mut_pdb = wild_pdb

    if mutator_backend == "foldx":
        try:
            mut_pdb = gen_mut_pdb_foldx(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, foldx_path, mut_id=mut_id)
            if _nonempty(mut_pdb):
                mutated_by_structure = True
            else:
                mut_pdb = wild_pdb
        except Exception:
            mut_pdb = wild_pdb

    elif mutator_backend == "rosetta":
        try:
            rsp = rosetta_scripts_path or "rosetta_scripts.static.linuxgccrelease"
            mut_pdb = gen_mut_pdb_rosetta(
                pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir,
                mut_id=mut_id,
                rosetta_scripts_path=rsp,
            )
            if _nonempty(mut_pdb):
                mutated_by_structure = True
            else:
                mut_pdb = wild_pdb
        except Exception:
            mut_pdb = wild_pdb

    else:
        mut_pdb = wild_pdb

    return wild_pdb, mut_pdb, mutated_by_structure, mut_id