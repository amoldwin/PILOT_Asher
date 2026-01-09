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
# Locking helpers (stale-aware)
# -------------------------
def _pid_is_alive(pid: int) -> bool:
    """Best-effort: check whether a PID is alive on this node."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # exists but not ours
        return True


def _acquire_lock(lock_path: str, timeout_sec: int = 1800, poll_sec: float = 2.0, stale_sec: int = 7200):
    """
    Cross-process lock using atomic file create (O_EXCL), with stale lock eviction.

    - timeout_sec: how long to wait before giving up
    - stale_sec: if lock file older than this, consider it stale and remove
    """
    start = time.time()
    last_notice = 0.0

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
            os.close(fd)
            return
        except FileExistsError:
            # Stat lock (may disappear between checks)
            try:
                st = os.stat(lock_path)
                age = time.time() - st.st_mtime
            except FileNotFoundError:
                continue

            # Read PID from lock file
            lock_pid = None
            try:
                with open(lock_path, "r") as f:
                    first = f.read().strip().splitlines()[0]
                    lock_pid = int(first)
            except Exception:
                lock_pid = None

            pid_alive = _pid_is_alive(lock_pid)

            # If process is definitely gone, evict stale lock
            if (lock_pid is not None) and (not pid_alive) and age >= 10:
                try:
                    os.remove(lock_path)
                    continue
                except OSError:
                    pass

            # Age-based eviction backstop
            if age > stale_sec:
                try:
                    os.remove(lock_path)
                    continue
                except OSError:
                    pass

            now = time.time()
            if now - start > timeout_sec:
                raise TimeoutError(
                    f"Timed out waiting for lock: {lock_path} "
                    f"(age={age:.0f}s pid={lock_pid} alive={pid_alive})"
                )

            # (optional) emit occasional note; keep disabled to reduce log spam
            if now - last_notice > 300:
                last_notice = now

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


# -------------------------
# PSI-BLAST
# -------------------------
def use_psiblast(fasta_file, rawmsa_dir, psi_path, uniref90_path, num_threads: int = 16):
    """
    Run PSI-BLAST and generate:
      - {rawmsa_dir}/{fasta_name}.rawmsa (pairwise output)
      - {rawmsa_dir}/{fasta_name}.pssm   (ASCII PSSM)

    Safe for SLURM arrays: uses a per-fasta lock to avoid file collisions and partial files.
    Writes to temp files and os.replace() atomically.
    """
    fasta_name = os.path.basename(fasta_file).split('.')[0]
    rawmsa_file = os.path.join(rawmsa_dir, fasta_name + '.rawmsa')
    pssm_file = os.path.join(rawmsa_dir, fasta_name + '.pssm')

    # Fast path
    if _nonempty(pssm_file) and _nonempty(rawmsa_file):
        return rawmsa_file, pssm_file

    os.makedirs(rawmsa_dir, exist_ok=True)

    lock_path = os.path.join(rawmsa_dir, f".{fasta_name}.psiblast.lock")
    _acquire_lock(lock_path)
    try:
        # Re-check after waiting
        if _nonempty(pssm_file) and _nonempty(rawmsa_file):
            return rawmsa_file, pssm_file

        # Remove stale/partial outputs (common after preemption)
        for p in (rawmsa_file, pssm_file):
            if os.path.exists(p) and os.path.getsize(p) == 0:
                try:
                    os.remove(p)
                except OSError:
                    pass

        # Temp outputs (same dir so os.replace is atomic)
        tmp_rawmsa = rawmsa_file + f".tmp_{os.getpid()}"
        tmp_pssm = pssm_file + f".tmp_{os.getpid()}"

        # Clean any previous temp files for this pid (unlikely but harmless)
        for p in (tmp_rawmsa, tmp_pssm):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

        threads = _slurm_cpus(num_threads)

        cmd = [
            psi_path,
            "-query", fasta_file,
            "-db", uniref90_path,
            "-out", tmp_rawmsa,
            "-evalue", "0.001",
            "-matrix", "BLOSUM62",
            "-num_iterations", "3",
            "-num_threads", str(threads),
            "-out_ascii_pssm", tmp_pssm,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            # Remove temp outputs on failure to avoid confusing future runs
            for p in (tmp_rawmsa, tmp_pssm):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            raise RuntimeError(
                f"psiblast failed for {fasta_file} with db={uniref90_path}\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{e.stdout}\n\nstderr:\n{e.stderr}\n"
            ) from e

        if not _nonempty(tmp_rawmsa):
            raise FileNotFoundError(f"psiblast did not create rawmsa file: {tmp_rawmsa}")
        if not _nonempty(tmp_pssm):
            raise FileNotFoundError(f"psiblast did not create pssm file: {tmp_pssm}")

        # Atomic finalize
        os.replace(tmp_rawmsa, rawmsa_file)
        os.replace(tmp_pssm, pssm_file)

        return rawmsa_file, pssm_file
    finally:
        _release_lock(lock_path)


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
    max_hits: int = 100,
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

    # Fallback: if headers look like "..._" use split('_')[1]
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


def process_hhm(path, expected_len: int | None = None):
    """
    Parse HHblits .hhm profile into an (L,30) float array in [0,1].

    Robustness:
    - If the file is malformed/truncated/unexpected, return zeros of shape (expected_len,30) if provided,
      else attempt to infer L, else return (0,30).
    - This prevents single bad HHM files from killing large SLURM arrays.
    """
    def _fallback():
        L = 0
        if expected_len is not None and expected_len > 0:
            L = int(expected_len)
        return np.zeros((L, 30), dtype=np.float32)

    try:
        with open(path, "r") as fin:
            fin_data = fin.readlines()

        hhm_begin_line = None
        hhm_end_line = None
        for i in range(len(fin_data)):
            if "#" in fin_data[i] and hhm_begin_line is None:
                hhm_begin_line = i + 5
            if "//" in fin_data[i]:
                hhm_end_line = i
                break

        if hhm_begin_line is None or hhm_end_line is None or hhm_end_line <= hhm_begin_line:
            print(f"[process_hhm] WARNING: could not find HHM table region in {path}; using zeros.", flush=True)
            return _fallback()

        n_rows = int((hhm_end_line - hhm_begin_line) / 3)
        if n_rows <= 0:
            return _fallback()

        feature = np.zeros([n_rows, 30], dtype=np.float32)
        axis_x = 0

        for i in range(hhm_begin_line, hhm_end_line, 3):
            if axis_x >= n_rows:
                break
            line1 = fin_data[i].split()[2:-1]
            line2 = fin_data[i + 1].split() if (i + 1) < len(fin_data) else []

            axis_y = 0
            for j in line1:
                if axis_y >= 30:
                    break
                try:
                    v = 9999.0 if j == "*" else float(j)
                except ValueError:
                    # malformed token like 'F'
                    print(f"[process_hhm] WARNING: non-numeric token {j!r} in {path}; using zeros.", flush=True)
                    return _fallback()
                feature[axis_x][axis_y] = v / 10000.0
                axis_y += 1

            for j in line2:
                if axis_y >= 30:
                    break
                try:
                    v = 9999.0 if j == "*" else float(j)
                except ValueError:
                    print(f"[process_hhm] WARNING: non-numeric token {j!r} in {path}; using zeros.", flush=True)
                    return _fallback()
                feature[axis_x][axis_y] = v / 10000.0
                axis_y += 1

            axis_x += 1

        den = (np.max(feature) - np.min(feature))
        feature = (feature - np.min(feature)) / (den + 1e-8)
        return feature
    except Exception as e:
        print(f"[process_hhm] WARNING: failed to parse {path}: {e}; using zeros.", flush=True)
        return _fallback()


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