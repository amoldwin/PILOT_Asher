import os
import io
import time
import subprocess
import numpy as np
from Bio import SeqIO


def _slurm_cpus(default: int) -> int:
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if not v:
        return default
    try:
        return max(1, int(v))
    except ValueError:
        return default


# -------------------------
# PSI-BLAST
# -------------------------
def use_psiblast(fasta_file, rawmsa_dir, psi_path, uniref90_path, num_threads: int = 16):
    """
    Run PSI-BLAST and generate:
      - {rawmsa_dir}/{fasta_name}.rawmsa (pairwise output)
      - {rawmsa_dir}/{fasta_name}.pssm   (ASCII PSSM)

    Raises a RuntimeError with stdout/stderr if BLAST fails.
    """
    fasta_name = os.path.basename(fasta_file).split('.')[0]
    rawmsa_file = os.path.join(rawmsa_dir, fasta_name + '.rawmsa')
    pssm_file = os.path.join(rawmsa_dir, fasta_name + '.pssm')

    if os.path.exists(pssm_file) and os.path.exists(rawmsa_file):
        return rawmsa_file, pssm_file

    os.makedirs(rawmsa_dir, exist_ok=True)

    threads = _slurm_cpus(num_threads)

    cmd = [
        psi_path,
        "-query", fasta_file,
        "-db", uniref90_path,
        "-out", rawmsa_file,
        "-evalue", "0.001",
        "-matrix", "BLOSUM62",
        "-num_iterations", "3",
        "-num_threads", str(threads),
        "-out_ascii_pssm", pssm_file,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"psiblast failed for {fasta_file} with db={uniref90_path}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n\nstderr:\n{e.stderr}\n"
        ) from e

    if not os.path.exists(rawmsa_file):
        raise FileNotFoundError(f"psiblast did not create rawmsa file: {rawmsa_file}")
    if not os.path.exists(pssm_file):
        raise FileNotFoundError(f"psiblast did not create pssm file: {pssm_file}")

    return rawmsa_file, pssm_file


# -------------------------
# MSA build via blastdbcmd (low-memory)
# -------------------------
def _extract_hit_ids_from_rawmsa(prot_id: str, rawmsa_file: str):
    """
    Parse PSI-BLAST pairwise output (.rawmsa) and extract hit identifiers from lines starting with '>'.

    Returns IDs in order (deduplicated) suitable for blastdbcmd -entry_batch.
    """
    ids = []
    with open(rawmsa_file, "r") as infile:
        for line in infile:
            if not line.startswith(">"):
                continue
            header = line.strip().split()[0].lstrip(">")

            # Skip query if it appears (conservative)
            if prot_id in header:
                continue

            ids.append(header)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _blastdbcmd_fetch_fasta(entry_ids, blastdbcmd_path: str, uniref90_path: str) -> str:
    """
    Fetch sequences for entry IDs using blastdbcmd -entry_batch; return FASTA text.
    """
    if not entry_ids:
        return ""

    tmpdir = os.environ.get("SLURM_TMPDIR", "/tmp")
    os.makedirs(tmpdir, exist_ok=True)
    batch_file = os.path.join(tmpdir, f"blastdbcmd_ids_{os.getpid()}.txt")

    try:
        with open(batch_file, "w") as f:
            for eid in entry_ids:
                f.write(eid + "\n")

        cmd = [
            blastdbcmd_path,
            "-db", uniref90_path,
            "-dbtype", "prot",
            "-entry_batch", batch_file,
            "-outfmt", "%f",
        ]
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"blastdbcmd failed (exit={proc.returncode})\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr}\n"
            )
        return proc.stdout
    finally:
        try:
            os.remove(batch_file)
        except OSError:
            pass


def format_rawmsa_via_blastdbcmd(
    prot_id: str,
    rawmsa_file: str,
    formatted_output_file: str,
    blastdbcmd_path: str,
    uniref90_path: str,
    max_hits: int = 500,
) -> int:
    """
    Convert PSI-BLAST pairwise output to a FASTA containing hit sequences fetched from the BLAST DB.

    Returns number of hit sequences written (does not include the query sequence).
    """
    if not os.path.exists(rawmsa_file):
        return 0

    hit_ids = _extract_hit_ids_from_rawmsa(prot_id, rawmsa_file)
    if not hit_ids:
        return 0

    hit_ids = hit_ids[:max_hits]

    # Try direct header IDs first
    fasta_text = _blastdbcmd_fetch_fasta(hit_ids, blastdbcmd_path, uniref90_path)

    # Fallback: if headers look like "..._<ID>" use split('_')[1]
    if not fasta_text.strip():
        fallback = []
        for h in hit_ids:
            parts = h.split("_")
            if len(parts) >= 2:
                fallback.append(parts[1])
        fallback = fallback[:max_hits]
        fasta_text = _blastdbcmd_fetch_fasta(fallback, blastdbcmd_path, uniref90_path)

    if not fasta_text.strip():
        return 0

    wrote = 0
    with open(formatted_output_file, "w") as out:
        handle = io.StringIO(fasta_text)
        for rec in SeqIO.parse(handle, "fasta"):
            SeqIO.write(rec, out, "fasta")
            wrote += 1

    return wrote


def run_clustal(clustal_input_file, clustal_output_file, clustalo_path, num_threads=6):
    with open(clustal_input_file, "r") as f:
        # FASTA expected: 2 lines per record at minimum, but sequences can wrap.
        # This heuristic is imperfect but used in the original code.
        numseqs = len(f.readlines()) / 2

    if numseqs > 1:
        cmd = [
            clustalo_path,
            "-i", clustal_input_file,
            "-o", clustal_output_file,
            "--force",
            "--threads", str(num_threads)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"clustalo failed for {clustal_input_file}\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{e.stdout}\n\nstderr:\n{e.stderr}\n"
            ) from e
    else:
        subprocess.run(["cp", clustal_input_file, clustal_output_file], check=True)


def format_clustal(clustal_output_file, formatted_output_file):
    msa_info = []
    with open(clustal_output_file, "r") as f:
        seq_name = ""
        seq = ""
        for line in f:
            if line.startswith(">"):
                if seq_name:
                    msa_info.append(seq_name)
                    msa_info.append(seq)
                seq_name = line.strip()
                seq = ""
            else:
                seq += line.strip()
        msa_info.append(seq_name)
        msa_info.append(seq.replace("U", "-"))

    outtxt = ""
    gaps = []
    for idx, line in enumerate(msa_info):
        if idx % 2 == 0:
            outtxt += line + "\n"
        elif idx == 1:
            for i in range(len(line)):
                gaps.append(line[i] == "-")

        if idx % 2 == 1:
            newseq = ""
            for i in range(len(gaps)):
                if not gaps[i]:
                    if i < len(line):
                        newseq += line[i]
                    else:
                        newseq += "-"
            outtxt += newseq + "\n"

    with open(formatted_output_file, "w") as f:
        f.write(outtxt)


def _acquire_lock(lock_path: str, timeout_sec: int = 1800, poll_sec: float = 2.0):
    """
    Cross-process lock using atomic file create (O_EXCL).
    Prevents concurrent array tasks from clobbering the same MSA intermediates.
    """
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


def gen_msa(prot_id, prot_seq, rawmsa_file, output_dir, clustalo_path, blastdbcmd_path, uniref90_path):
    """
    Generate .msa file for a protein:
      - Parse PSI-BLAST output (.rawmsa)
      - Fetch hit sequences with blastdbcmd (no uniref90.fasta in RAM)
      - Run clustalo and post-process

    Safe for SLURM arrays: uses a per-prot lock to avoid file collisions.
    """
    formatted_fasta_file = os.path.join(output_dir, prot_id + "_rawmsa.fasta")
    clustal_input_file = os.path.join(output_dir, prot_id + ".clustal_input")
    clustal_output_file = os.path.join(output_dir, prot_id + ".clustal")
    formatted_clustal_file = os.path.join(output_dir, prot_id + ".msa")

    os.makedirs(output_dir, exist_ok=True)

    # Fast path: final MSA already exists
    if os.path.exists(formatted_clustal_file) and os.path.getsize(formatted_clustal_file) > 0:
        return formatted_clustal_file

    lock_path = os.path.join(output_dir, f".{prot_id}.msa.lock")
    _acquire_lock(lock_path)
    try:
        # Re-check after waiting
        if os.path.exists(formatted_clustal_file) and os.path.getsize(formatted_clustal_file) > 0:
            return formatted_clustal_file

        # Build hit FASTA (if missing)
        if not os.path.exists(formatted_fasta_file):
            n_hits_written = format_rawmsa_via_blastdbcmd(
                prot_id, rawmsa_file, formatted_fasta_file, blastdbcmd_path, uniref90_path
            )
        else:
            # If it exists, assume it has content; if empty we will fall back below
            n_hits_written = 1 if os.path.getsize(formatted_fasta_file) > 0 else 0

        # No hits -> trivial MSA with only query
        if n_hits_written == 0:
            with open(formatted_clustal_file, "w") as out:
                out.write(f">{prot_id}\n{prot_seq}\n")
            return formatted_clustal_file

        # Create clustal input that includes query at top
        if not os.path.exists(clustal_input_file):
            with open(formatted_fasta_file, "r") as infile:
                hits = infile.read().strip()
            with open(clustal_input_file, "w") as outfile:
                outfile.write(f">{prot_id}\n{prot_seq}\n")
                if hits:
                    outfile.write(hits + "\n")

        # Run clustalo if needed
        if not os.path.exists(clustal_output_file):
            threads = _slurm_cpus(6)
            run_clustal(clustal_input_file, clustal_output_file, clustalo_path, num_threads=threads)

        # Hard check: clustalo must have produced output
        if not os.path.exists(clustal_output_file):
            raise FileNotFoundError(
                f"clustalo did not create expected output: {clustal_output_file}\n"
                f"Input: {clustal_input_file}\n"
            )

        if not os.path.exists(formatted_clustal_file):
            format_clustal(clustal_output_file, formatted_clustal_file)

        return formatted_clustal_file
    finally:
        _release_lock(lock_path)


# -------------------------
# HHblits
# -------------------------
def use_hhblits(seq_name, fasta_file, hhblits_path, uniRef30_path, hhm_dir, cpu: int = 16):
    """
    Run hhblits and generate {hhm_dir}/{seq_name}.hhm
    """
    os.makedirs(hhm_dir, exist_ok=True)
    out_hhm = os.path.join(hhm_dir, seq_name + ".hhm")
    if os.path.exists(out_hhm):
        return out_hhm

    threads = _slurm_cpus(cpu)

    cmd = [hhblits_path, "-cpu", str(threads), "-i", fasta_file, "-d", uniRef30_path, "-ohhm", out_hhm]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"hhblits failed for {fasta_file} with db={uniRef30_path}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n\nstderr:\n{e.stderr}\n"
        ) from e

    if not os.path.exists(out_hhm):
        raise FileNotFoundError(f"hhblits did not create hhm file: {out_hhm}")

    return out_hhm


# -------------------------
# PSSM/HHM parsing + MSA stats
# -------------------------
def get_pssm(pssm_path):
    pssm_dict, new_pssm_dict, res_dict = {}, {}, {}
    with open(pssm_path, "r") as f_r:
        next(f_r)
        next(f_r)
        next(f_r)
        for line in f_r:
            line = line.split()
            if len(line) > 20:
                pos = line[0]
                aa = line[1]
                pssm = line[2:22]
                pssm_dict[pos] = [float(i) for i in pssm]
                res_dict[pos] = aa
        for key in pssm_dict.keys():
            pssm = np.array(pssm_dict[key])
            pssm = 1 / (np.exp(-pssm) + 1)
            new_pssm_dict[key] = list(pssm)
    return new_pssm_dict, res_dict


def process_hhm(path):
    with open(path, "r") as fin:
        fin_data = fin.readlines()
        hhm_begin_line = 0
        hhm_end_line = 0
        for i in range(len(fin_data)):
            if "#" in fin_data[i]:
                hhm_begin_line = i + 5
            elif "//" in fin_data[i]:
                hhm_end_line = i
        feature = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])
        axis_x = 0
        for i in range(hhm_begin_line, hhm_end_line, 3):
            line1 = fin_data[i].split()[2:-1]
            line2 = fin_data[i + 1].split()
            axis_y = 0
            for j in line1:
                feature[axis_x][axis_y] = (9999 if j == "*" else float(j)) / 10000.0
                axis_y += 1
            for j in line2:
                feature[axis_x][axis_y] = (9999 if j == "*" else float(j)) / 10000.0
                axis_y += 1
            axis_x += 1
        feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
        return feature


def loadAASeq(infile):
    seqs = []
    for i in SeqIO.parse(infile, "fasta"):
        seqs.append(i.seq)
    return seqs, len(seqs[0])


def calc_res_freq(infile):
    aa_name = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
    seqs, length = loadAASeq(infile)
    conservation_dict = {}
    for res_pos in range(1, length + 1):
        conservation_dict[res_pos] = np.zeros((21))
    for seq in seqs:
        for res_pos in range(1, length + 1):
            res = seq[int(res_pos) - 1]
            try:
                index = aa_name.index(res)
            except ValueError:
                continue
            conservation_dict[res_pos][index] += 1
    for res_pos in range(1, length + 1):
        conservation_dict[res_pos] = conservation_dict[res_pos] / len(seqs)
    return conservation_dict