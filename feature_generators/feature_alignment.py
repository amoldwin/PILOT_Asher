aa2property = {'A':[1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
               'G':[0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
               'V':[3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
               'L':[2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
               'I':[4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
               'F':[2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
               'Y':[2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
               'W':[3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
               'T':[3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
               'S':[1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
               'R':[2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
               'K':[1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
               'H':[2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
               'D':[1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
               'E':[1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
               'N':[1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
               'Q':[1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
               'M':[2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
               'P':[2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
               'C':[1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41]}

atom2code_dist = {'C':[1, 0, 0, 0], 'N':[0, 1, 0, 0], 'O':[0, 0, 1, 0], 'S':[0, 0, 0, 1]}


def aa2code():
    aa2code = {}
    aa_name = ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H']
    for i in range(20):
        code = []
        for j in range(20):
            code.append(1 if i == j else 0)
        aa2code[aa_name[i]] = code
    return aa2code


def feature_alignment(selected_res, selected_atom, feature_data, res_dict, atom_dict, pdb2uniprot_posdict, mut_pos):
    res_feature_dict, atom_feature_dict = {}, {}

    sasa_res = feature_data['sasa_res']
    sasa_atom = feature_data['sasa_atom']
    ss_dict = feature_data['ss_dict'] or {}
    depth_dict = feature_data['depth_dict'] or {}
    pssm_dict = feature_data['pssm_dict']
    hhm_score = feature_data['hhm_score']
    conservation_dict = feature_data['conservation_dict']
    conservation_score = feature_data['conservation_score']

    aa2code_dict = aa2code()

    # Default fillers (shape-critical)
    default_ss = [0, 0, 0]
    default_depth = [0.0]
    default_sasa = [0.0]
    default_pssm = [0.0] * 20
    default_hhm = [0.0] * 30
    default_cs1 = [0.0] * 21
    default_cs2 = 0.0

    # ---------------- res features ----------------
    for pos in selected_res:
        # If we don't even know the residue identity, we can't build the 20-AA onehot/properties reliably.
        res_name = res_dict.get(pos)
        if res_name is None:
            continue
        if res_name not in aa2code_dict or res_name not in aa2property:
            continue

        aa_code = aa2code_dict[res_name]
        properties = aa2property[res_name]

        ss = ss_dict.get(pos, default_ss)
        depth = [float(depth_dict.get(pos, 0.0))]
        sasa = [float(sasa_res.get(pos, 0.0))]

        # Uniprot pos mapping; if missing, we can't index pssm/hhm/cs reliably.
        up = pdb2uniprot_posdict.get(pos)
        if up is None:
            pssm = default_pssm
            hhm = default_hhm
            cs1 = default_cs1
            cs2 = default_cs2
        else:
            uniprot_pos = str(up)
            pssm = pssm_dict.get(uniprot_pos, default_pssm)

            # hhm_score indexed by (pos-1)
            i = int(up) - 1
            if 0 <= i < len(hhm_score):
                hhm = list(hhm_score[i])
            else:
                hhm = default_hhm

            cs1 = list(conservation_dict.get(int(up), default_cs1))

            if 0 <= i < len(conservation_score):
                cs2 = float(conservation_score[i])
            else:
                cs2 = default_cs2

        is_mut = [1] if pos == mut_pos else [0]

        res_feature_dict[pos] = aa_code + ss + depth + properties + sasa + list(pssm) + list(hhm) + list(cs1) + [cs2] + is_mut

    # Ensure mutation residue is present if possible
    if mut_pos in selected_res and mut_pos not in res_feature_dict and mut_pos in res_dict:
        # try again but with reduced requirements: allow unknown ss/depth/sasa and no evo
        res_name = res_dict.get(mut_pos)
        if res_name in aa2code_dict and res_name in aa2property:
            aa_code = aa2code_dict[res_name]
            properties = aa2property[res_name]
            res_feature_dict[mut_pos] = aa_code + default_ss + default_depth + properties + default_sasa + default_pssm + default_hhm + default_cs1 + [default_cs2] + [1]

    # ---------------- atom features ----------------
    for atom_index in selected_atom.keys():
        at = atom_dict.get(atom_index)
        if at is None:
            continue
        if len(at) == 0:
            continue
        elem = at[0]
        if elem not in atom2code_dist:
            continue
        sasa = [float(sasa_atom.get(atom_index, 0.0))]
        atom_feature_dict[atom_index] = atom2code_dist[elem] + sasa

    return res_feature_dict, atom_feature_dict