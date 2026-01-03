import os
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


def use_psiblast(fasta_file, rawmsa_dir, psi_path, uniref90_path, num_threads: int = 16):
    """
    Run PSI-BLAST and generate:
      - {rawmsa_dir}/{fasta_name}.rawmsa (pairwise alignment output)
      - {rawmsa_dir}/{fasta_name}.pssm   (ASCII PSSM)
    Raises a RuntimeError with a clear message if BLAST fails.
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


def format_rawmsa(prot_id, rawmsa_file, seq_dict, formatted_output_file):
    identifiers_to_align = set()
    if not os.path.exists(rawmsa_file):
        # If PSI-BLAST failed, caller should have already errored out.
        # If PSI-BLAST produced nothing (rare), treat as no hits.
        return 0

    with open(rawmsa_file, 'r') as infile:
        for line in infile:
            if line.startswith('>'):
                identifier = line.strip().split()[0]
                if identifier.split('_')[1] != prot_id:
                    identifiers_to_align.add(identifier)

    if len(identifiers_to_align) > 0:
        with open(formatted_output_file, 'w') as outfile:
            wrote = 0
            for identifier in sorted(identifiers_to_align):
                key = identifier.split('_')[1]
                if key in seq_dict:
                    outfile.write(identifier + '\n' + seq_dict[key] + '\n')
                    wrote += 1
            return wrote
    return 0


def run_clustal(clustal_input_file, clustal_output_file, clustalo_path, num_threads=6):
    with open(clustal_input_file, 'r') as f:
        numseqs = len(f.readlines()) / 2

    if numseqs > 1:
        cmd = [clustalo_path, "-i", clustal_input_file, "-o", clustal_output_file, "--force", "--threads", str(num_threads)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"clustalo failed for {clustal_input_file}\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{e.stdout}\n\nstderr:\n{e.stderr}\n"
            ) from e
    else:
        # Only the query sequence
        subprocess.run(["cp", clustal_input_file, clustal_output_file], check=True)


def format_clustal(clustal_output_file, formatted_output_file):
    msa_info = []
    with open(clustal_output_file, 'r') as f:
        seq_name = ''
        seq = ''
        for line in f:
            if line.startswith('>'):
                if seq_name:
                    msa_info.append(seq_name)
                    msa_info.append(seq)
                seq_name = line.strip()
                seq = ''
            else:
                seq += line.strip()
        msa_info.append(seq_name)
        msa_info.append(seq.replace('U', '-'))

    outtxt = ''
    gaps = []
    for idx, line in enumerate(msa_info):
        if idx % 2 == 0:
            outtxt += line + '\n'
        elif idx == 1:
            for i in range(len(line)):
                gaps.append(line[i] == '-')

        if idx % 2 == 1:
            newseq = ''
            for i in range(len(gaps)):
                if not gaps[i]:
                    if i < len(line):
                        newseq += line[i]
                    else:
                        newseq += '-'
            outtxt += newseq + '\n'

    with open(formatted_output_file, 'w') as f:
        f.write(outtxt)


def gen_msa(prot_id, prot_seq, rawmsa_file, seq_dict, output_dir, clustalo_path):
    formatted_fasta_file = os.path.join(output_dir, prot_id + '_rawmsa.fasta')
    clustal_input_file = os.path.join(output_dir, prot_id + '.clustal_input')
    clustal_output_file = os.path.join(output_dir, prot_id + '.clustal')
    formatted_clustal_file = os.path.join(output_dir, prot_id + '.msa')

    os.makedirs(output_dir, exist_ok=True)

    n_hits_written = format_rawmsa(prot_id, rawmsa_file, seq_dict, formatted_fasta_file)

    # If no hits were written, still create a trivial "MSA" with only the query.
    if n_hits_written == 0:
        if not os.path.exists(formatted_clustal_file):
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

        threads = _slurm_cpus(6)
        run_clustal(clustal_input_file, clustal_output_file, clustalo_path, num_threads=threads)

    if not os.path.exists(formatted_clustal_file):
        format_clustal(clustal_output_file, formatted_clustal_file)

    return formatted_clustal_file


def use_hhblits(seq_name, fasta_file, hhblits_path, uniRef30_path, hhm_dir, cpu: int = 16):
    """
    Run hhblits and generate {hhm_dir}/{seq_name}.hhm.
    Raises a RuntimeError with a clear message if hhblits fails.
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


def get_pssm(pssm_path):
    pssm_dict, new_pssm_dict, res_dict = {}, {}, {}
    with open(pssm_path, 'r') as f_r:
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
    with open(path, 'r') as fin:
        fin_data = fin.readlines()
        hhm_begin_line = 0
        hhm_end_line = 0
        for i in range(len(fin_data)):
            if '#' in fin_data[i]:
                hhm_begin_line = i + 5
            elif '//' in fin_data[i]:
                hhm_end_line = i
        feature = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])
        axis_x = 0
        for i in range(hhm_begin_line, hhm_end_line, 3):
            line1 = fin_data[i].split()[2:-1]
            line2 = fin_data[i + 1].split()
            axis_y = 0
            for j in line1:
                if j == '*':
                    feature[axis_x][axis_y] = 9999 / 10000.0
                else:
                    feature[axis_x][axis_y] = float(j) / 10000.0
                axis_y += 1
            for j in line2:
                if j == '*':
                    feature[axis_x][axis_y] = 9999 / 10000.0
                else:
                    feature[axis_x][axis_y] = float(j) / 10000.0
                axis_y += 1
            axis_x += 1
        feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
        return feature


def loadAASeq(infile):
    seqs = []
    for i in SeqIO.parse(infile, 'fasta'):
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