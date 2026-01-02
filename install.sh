conda create -n pilot python==3.10 -y
source activate pilot

# Core ML stack
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install torch-geometric==2.3.1

# Project dependencies
pip install numpy scipy
pip install scikit-learn
pip install biopython==1.81
pip install networkx

# Optional: install available external tools into the conda env
# (adjust paths in gen_features.py accordingly)
# conda install -c ostrokach dssp -y              # mkdssp
# conda install -c bioconda blast -y              # provides psiblast
# conda install -c bioconda hhsuite -y            # hhblits
# conda install -c bioconda clustalo -y           # clustal omega
# conda install -c bioconda freesasa -y           # FreeSASA CLI (alternative to Naccess)

# Tools not on conda: install separately and set paths in gen_features.py
# - FoldX
# - Naccess
# - MSMS (needed by Bio.PDB.ResidueDepth)
# Also ensure 'wget' is installed on your system (or modify gen_pdb.py to use curl/requests).