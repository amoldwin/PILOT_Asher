import os
import subprocess
import numpy as np
from Bio import SeqIO
import io
import time


# ... keep everything above unchanged ...


def _acquire_lock(lock_path: str, timeout_sec: int = 1800, poll_sec: float = 2.0):
    """
    Simple cross-process lock using O_EXCL file creation.
    Prevents multiple array tasks from writing the same MSA intermediates concurrently.
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
    Generate .msa file for a protein.
    NOTE: This function is called by many array jobs; we must lock by prot_id to avoid collisions.
    """
    formatted_fasta_file = os.path.join(output_dir, prot_id + '_rawmsa.fasta')
    clustal_input_file = os.path.join(output_dir, prot_id + '.clustal_input')
    clustal_output_file = os.path.join(output_dir, prot_id + '.clustal')
    formatted_clustal_file = os.path.join(output_dir, prot_id + '.msa')

    os.makedirs(output_dir, exist_ok=True)

    # Fast path if final exists
    if os.path.exists(formatted_clustal_file) and os.path.getsize(formatted_clustal_file) > 0:
        return formatted_clustal_file

    lock_path = os.path.join(output_dir, f".{prot_id}.msa.lock")
    _acquire_lock(lock_path)
    try:
        # Another fast path after waiting
        if os.path.exists(formatted_clustal_file) and os.path.getsize(formatted_clustal_file) > 0:
            return formatted_clustal_file

        if not os.path.exists(formatted_fasta_file):
            n_hits_written = format_rawmsa_via_blastdbcmd(
                prot_id, rawmsa_file, formatted_fasta_file, blastdbcmd_path, uniref90_path
            )
        else:
            n_hits_written = 1

        if n_hits_written == 0:
            with open(formatted_clustal_file, "w") as out:
                out.write(f">{prot_id}\n{prot_seq}\n")
            return formatted_clustal_file

        if not os.path.exists(clustal_input_file):
            with open(formatted_fasta_file, 'r') as infile:
                lines = infile.readlines()

            with open(clustal_input_file, 'w') as outfile:
                outfile.write('>' + prot_id + '\n' + prot_seq + '\n')
                for line in lines:
                    outfile.write(line)

        if not os.path.exists(clustal_output_file):
            threads = _slurm_cpus(6)
            run_clustal(clustal_input_file, clustal_output_file, clustalo_path, num_threads=threads)

        # Hard check: clustalo must have produced the file
        if not os.path.exists(clustal_output_file):
            raise FileNotFoundError(
                f"clustalo did not create expected output: {clustal_output_file}\n"
                f"Input was: {clustal_input_file}\n"
                f"Formatted hits: {formatted_fasta_file}\n"
            )

        if not os.path.exists(formatted_clustal_file):
            format_clustal(clustal_output_file, formatted_clustal_file)

        return formatted_clustal_file
    finally:
        _release_lock(lock_path)