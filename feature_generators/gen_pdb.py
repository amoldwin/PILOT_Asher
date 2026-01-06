import os
import time
import subprocess
import shutil
from typing import Optional


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
    Download the full PDB file to row_pdb_dir/.pdb.
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
    Extract ATOM/HETATM records for the given chain and write cleaned_pdb_dir/_.pdb.
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


def backend_tagged_mut_id(pdb_id: str, chain_id: str, mut_pos: str, wild_type: str, mutant: str, mutator_backend: str) -> str:
    """
    Return a backend-tagged mutant identifier to avoid collisions between foldx/proxy/rosetta.
    """
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

        # FoldX writes {pdb_id}_{chain_id}_1.pdb
        os.system(f"cp {workdir}/{pdb_id}_{chain_id}_1.pdb  {out_pdb}")

    return out_pdb


def _write_rosettascripts_xml(xml_path: str, chain_id: str, resnum: int, icode: str, mutant_aa: str, pack_radius: float = 8.0):
    """
    Minimal RosettaScripts: mutate residue, repack neighborhood, minimize.
    Uses PDB numbering (chain+resnum+icode).
    """
    icode_str = icode if icode else ""
    # Rosetta PDB residue selector supports insertion codes via "resnum+icode" in some builds;
    # safest is to use explicit PDB numbering in a selector string.
    # We'll use ResidueIndexSelector with "resnum+icode" and chain restriction via ChainSelector intersection.
    res_spec = f"{resnum}{icode_str}"

    xml = f"""<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="ref2015" weights="ref2015"/>
  </SCOREFXNS>

  <RESIDUE_SELECTORS>
    <Chain name="chain" chains="{chain_id}"/>
    <ResidueIndex name="mutpos" resnums="{res_spec}"/>
    <Neighborhood name="nbr" selector="mutpos" distance="{pack_radius}" include_focus_in_subset="true"/>
    <And name="mut_in_chain" selectors="chain,mutpos"/>
    <And name="nbr_in_chain" selectors="chain,nbr"/>
  </RESIDUE_SELECTORS>

  <TASKOPERATIONS>
    <InitializeFromCommandline name="init"/>
    <IncludeCurrent name="ic"/>
    <RestrictToRepacking name="repack_only"/>
    <OperateOnResidueSubset name="prevent_repack_outside" selector="nbr_in_chain">
      <RestrictToRepackingRLT/>
    </OperateOnResidueSubset>
    <OperateOnResidueSubset name="prevent_repack_elsewhere" selector="chain">
      <RestrictToRepackingRLT/>
    </OperateOnResidueSubset>
  </TASKOPERATIONS>

  <MOVERS>
    <MutateResidue name="mutate" target="{res_spec}" new_res="{mutant_aa}"/>
    <PackRotamersMover name="pack" scorefxn="ref2015" task_operations="init,ic"/>
    <MinMover name="min" scorefxn="ref2015" type="lbfgs_armijo_nonmonotone" tolerance="0.0001" max_iter="200" bb="0" chi="1"/>
  </MOVERS>

  <PROTOCOLS>
    <Add mover="mutate"/>
    <Add mover="pack"/>
    <Add mover="min"/>
  </PROTOCOLS>

  <OUTPUT scorefxn="ref2015"/>
</ROSETTASCRIPTS>
"""
    with open(xml_path, "w") as f:
        f.write(xml)


def _parse_mut_pos(mut_pos: str):
    """
    mut_pos is like "32" or "32A" (insertion codes).
    Returns (resnum:int, icode:str).
    """
    s = mut_pos.strip()
    if not s:
        raise ValueError("Empty mut_pos")
    # split numeric prefix
    i = 0
    while i < len(s) and s[i].isdigit():
        i += 1
    if i == 0:
        raise ValueError(f"mut_pos does not start with digits: {mut_pos!r}")
    resnum = int(s[:i])
    icode = s[i:].strip()
    return resnum, icode


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
):
    """
    Build a mutant PDB using RosettaScripts:
      - mutate
      - repack
      - minimize (chi only)
    Writes cleaned_pdb_dir/{mut_id}.pdb
    """
    in_pdb = os.path.join(cleaned_pdb_dir, f"{pdb_id}_{chain_id}.pdb")
    out_pdb = os.path.join(cleaned_pdb_dir, f"{mut_id}.pdb")

    if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0:
        return out_pdb

    workdir = os.path.join(cleaned_pdb_dir, "rosetta_work", mut_id)
    os.makedirs(workdir, exist_ok=True)

    lock = os.path.join(workdir, ".rosetta.lock")
    _acquire_lock(lock)
    try:
        if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0:
            return out_pdb

        if not os.path.exists(in_pdb) or os.path.getsize(in_pdb) == 0:
            raise FileNotFoundError(f"Missing WT cleaned pdb: {in_pdb}")

        resnum, icode = _parse_mut_pos(mut_pos)

        xml_path = os.path.join(workdir, "mutate_repack_minimize.xml")
        _write_rosettascripts_xml(xml_path, chain_id=chain_id, resnum=resnum, icode=icode, mutant_aa=mutant, pack_radius=pack_radius)

        # Output handling: Rosetta writes into -out:path:all, usually as {input}_0001.pdb (or similar).
        # We'll capture any .pdb written and copy it to out_pdb.
        cmd = [
            rosetta_scripts_path,
            "-s", in_pdb,
            "-parser:protocol", xml_path,
            "-out:path:all", workdir,
            "-overwrite",
            "-mute", "all",  # reduce log spam; errors still bubble
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        log_path = os.path.join(workdir, "rosetta.log")
        with open(log_path, "w") as f:
            f.write("CMD: " + " ".join(cmd) + "\n\n")
            f.write("STDOUT:\n" + proc.stdout + "\n\n")
            f.write("STDERR:\n" + proc.stderr + "\n")

        if proc.returncode != 0:
            raise RuntimeError(f"RosettaScripts failed (exit={proc.returncode}). See {log_path}")

        # Find produced pdb (newest .pdb in workdir excluding the input)
        candidates = []
        for fn in os.listdir(workdir):
            if not fn.lower().endswith(".pdb"):
                continue
            full = os.path.join(workdir, fn)
            if os.path.abspath(full) == os.path.abspath(in_pdb):
                continue
            candidates.append(full)

        if not candidates:
            raise RuntimeError(f"RosettaScripts did not produce a PDB in {workdir}. See {log_path}")

        newest = max(candidates, key=lambda p: os.path.getmtime(p))
        shutil.copyfile(newest, out_pdb)

        if not os.path.exists(out_pdb) or os.path.getsize(out_pdb) == 0:
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
):
    download_row_pdb(pdb_id, row_pdb_dir)
    wild_pdb = cleaned_row_pdb(pdb_id, chain_id, row_pdb_dir, cleaned_pdb_dir)

    mut_id = backend_tagged_mut_id(pdb_id, chain_id, mut_pos, wild_type, mutant, mutator_backend)

    mutated_by_structure = False
    mut_pdb = wild_pdb

    if mutator_backend == "foldx":
        try:
            mut_pdb = gen_mut_pdb_foldx(pdb_id, chain_id, mut_pos, wild_type, mutant, cleaned_pdb_dir, foldx_path, mut_id=mut_id)
            if os.path.exists(mut_pdb) and os.path.getsize(mut_pdb) > 0:
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
            if os.path.exists(mut_pdb) and os.path.getsize(mut_pdb) > 0:
                mutated_by_structure = True
            else:
                mut_pdb = wild_pdb
        except Exception:
            mut_pdb = wild_pdb

    else:
        # proxy (or any unknown): no structural mutation
        mut_pdb = wild_pdb

    return wild_pdb, mut_pdb, mutated_by_structure, mut_id