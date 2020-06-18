from numba import jit
import numpy as np
import scipy.stats
import threading
import time
import sys

ref_dict = {}
eqc_flag = False
output_file = ""
input_file = ""
sr_pos = 0

mu = 200
sd = 25
d = scipy.stats.norm(mu, sd)
small_eps = np.exp(-20)
big_eps = np.exp(-8)
lookup = [d.cdf(i) for i in range(1000)]
effective_length_dict = {}

''' Read transcripts in input_file '''
def read_transcripts():
    try:
        with open(input_file, "r") as file:
            global sr_pos
            sr_pos = int(file.readline())
            ref_dict = {}
            for i in range(sr_pos):
                l = file.readline().split("\t")
                ref_dict[l[0]] = (int(l[1].strip("\n")), i)
            file.close()
            return ref_dict
    except FileNotFoundError:
        print("File not found.")

''' Return location to first record in input_file '''
def first_record_p():
    if sr_pos > 0:
        try:
            file = open(input_file, "r")
            for i in range(sr_pos + 1):
                next(file)
            return file
        except FileNotFoundError:
            print("File not found")

''' Writes results to output_file '''
def write_to_file(est_num_reads):
    global d
    cmeans = correction_factors_from_mass(d)
    names = list(ref_dict.keys())
    vals = list(ref_dict.values())
    write_file = open(output_file, "w")
    write_file.write("name\teffective_length\test_frag\n")
    for i in range(0, len(est_num_reads)):
        write_file.write(str(names[i]) + "\t" + str(eff_len(vals[i][0], cmeans)) + "\t"+str(est_num_reads[i]) + "\n")
    write_file.close()

''' Correction factor used by eff_len '''
def correction_factors_from_mass(mass):
    maxLen = 1000
    correctionFactors = np.zeros(maxLen)
    vals = np.zeros(maxLen)
    multiplicities = np.zeros(maxLen)
    multiplicities[0] = mass.pdf(0)
    v = 0.0
    for i in range(1, maxLen):
        v = mass.pdf(i)
        vals[i] = v * i + vals[i - 1]
        multiplicities[i] = v + multiplicities[i - 1]
        if multiplicities[i] > 0 :
            correctionFactors[i] = vals[i] / multiplicities[i]
    return correctionFactors

''' Returns effective length for length '''
def eff_len(length, cmeans):
    global d 
    global effective_length_dict
    if length in effective_length_dict:
        return effective_length_dict[length]
    else:        
        if length >= 1000:
            return round(length - cmeans[-1])
        else:
            return round(length - cmeans[length])

''' Calculate P3 value used by calc_p_vals '''
def calc_p3(ori, tlen, pos):
    global d
    k = 0
    if ori == "f":
        k = tlen - pos
    else:
        k = pos + 100

    if k >= 1000:
        return 1.0
    else:
        return lookup[k]

''' Calculates P values '''
def calc_p_vals():
    global ref_dict
    ref_dict = read_transcripts()
    file = first_record_p()
    label_list = []
    prob_list = []
    ind_list = [0]
    cmeans = correction_factors_from_mass(d)

    while True:
        aln_line = file.readline()
        if aln_line == "":
            break
        na = int(aln_line)
        for i in range(na):
            toks = file.readline().strip().split("\t")
            txp = toks[0]
            ori = toks[1]
            pos = int(toks[2])
            tlen, tid = ref_dict[txp]
            label_list.append(tid)
            P_2 = 1 / eff_len(tlen, cmeans)
            P_3 = calc_p3(ori, tlen, pos)
            P_4 = float(toks[3])
            prob_list.append(P_2 * P_3 * P_4)
        ind_list.append(len(prob_list))
    label_list = np.array(label_list)
    prob_list = np.array(prob_list)
    ind_list = np.array(ind_list)
    return (label_list, prob_list, ind_list)

''' Implements full EM Algorithm '''
@jit(nopython=True)
def full_em(label_list, prob_list, ind_list):
    eta = np.ones(sr_pos) / float(sr_pos)
    eta_p = np.zeros(sr_pos)
    converged = False
    est_num_reads = np.zeros(sr_pos)
    it = 0
    ni = len(ind_list) - 1
    while not converged:
        it += 1
        for i in range(ni):
            norm = 0.0
            for j in range(ind_list[i], ind_list[i+1]):
                # Calculate normalizer.
                P_1 = eta[label_list[j]]
                un_norm_contr = P_1 * prob_list[j]    
                norm += un_norm_contr
            for j in range(ind_list[i], ind_list[i+1]):
            # Proportionally allocate fragments contribution to each transcript to which it aligns.
                P_1 = eta[label_list[j]]
                norm_contr = 0
                if norm != 0:
                    un_norm_contr = float(P_1 * prob_list[j])
                    norm_contr = un_norm_contr / norm
                est_num_reads[label_list[j]] += norm_contr
        # Calculate new eta in eta_p
        for i in range(0, sr_pos):
            new_eta = (est_num_reads[i]) / float(sr_pos)    
            if new_eta < small_eps:
                eta_p[i] = 0
            else:
                eta_p[i] = (est_num_reads[i]) / float(sr_pos)
        converged = True
        for i in range(0, len(eta)):
            if abs(eta_p[i] - eta[i]) > small_eps:
                converged = False
                break
        if converged:
            break
        else:
            eta = eta_p
            eta_p = np.zeros(sr_pos)
            est_num_reads = np.zeros(sr_pos)
    return (est_num_reads)

''' Create EQ Classes for EQC EM Algorithm '''
def create_eq_classes():
    global ref_dict
    ref_dict = read_transcripts()
    cmeans = correction_factors_from_mass(d)
    file = first_record_p()
    eq_classes = {}
    eq_p2 = {}
    while True:
        aln_line = file.readline()
        if aln_line == "":
            break
        na = int(aln_line)
        tup = ()
        p2_tup = ()
        for i in range(na):
            toks = file.readline().strip().split("\t")
            txp = toks[0]
            tlen, tid = ref_dict[txp]
            tup = tup + (txp,)
            p2_tup = p2_tup + (1/eff_len(tlen, cmeans),)
        if tup in eq_classes:
            eq_classes[tup] += 1
        else:
            eq_classes[tup] = 1
            eq_p2[tup] = p2_tup
    return (eq_classes, eq_p2)

''' Implement EQC EM Algorithm '''
def eqc_em(eq_classes_keys, eq_classes_vals, p2_vals):
    eta = np.ones(sr_pos) / float(sr_pos)
    eta_p = np.zeros(sr_pos)
    converged = False
    est_num_reads = np.zeros(sr_pos)
    it = 0
    ni = len(eq_classes_keys)
    while not converged:
        print("Iter: " + str(it))
        it += 1
        for i in range(ni):
            weight = eq_classes_vals[i]
            # Calculate norm.
            norm = 0
            for j in range(0, len(eq_classes_keys[i])):
                key = eq_classes_keys[i][j]
                ind = ref_dict[key][1]
                first = (eta[ind])
                second = (p2_vals[i][j])
                norm += (first * second)
            for j in range(0, len(eq_classes_keys[i])):
                first = eq_classes_vals[i]
                key = eq_classes_keys[i][j]
                ind = ref_dict[key][1]
                f = eta[ind]
                s = p2_vals[i][j]
                second = f * s
                top = first * second
                contr = 0
                if norm != 0:
                    contr = (top) / (norm)
                est_num_reads[ind] += contr
        # Calculate new eta in eta_p
        for i in range(0, sr_pos):
            new_eta = (est_num_reads[i]) / float(sr_pos)    
            if new_eta < small_eps:
                eta_p[i] = 0
            else:
                eta_p[i] = (est_num_reads[i]) / float(sr_pos)
        converged = True
        for i in range(0, len(eta)):
            if abs(eta_p[i] - eta[i]) > big_eps:
                converged = False
                break
        if converged:
            break
        else:
            eta = eta_p
            eta_p = np.zeros(sr_pos)
            est_num_reads = np.zeros(sr_pos)
    return est_num_reads

''' Parses user input and calls EM Algorithm '''
def main(argv):
    for i in range(0, len(argv)):
        if argv[i] == "--in":
            global input_file
            input_file = argv[i + 1]
            i += 1
        elif argv[i] == "--out":
            global output_file
            output_file = argv[i + 1]
            i += 1
        elif argv[i] == "--eqc":
            global eqc_flag
            eqc_flag = True

    if (input_file != "") and (output_file != ""):
        if not eqc_flag:
            label_list, prob_list, ind_list = calc_p_vals()
            (est_num_reads) = full_em(label_list, prob_list, ind_list)
            write_to_file(est_num_reads)
        else:
            (eq_classes, eq_p2) = create_eq_classes()
            eq_classes_keys = list(eq_classes.keys())
            eq_classes_vals = list(eq_classes.values())
            p2_vals = list(eq_p2.values())

            est_num_reads = eqc_em(eq_classes_keys, eq_classes_vals, p2_vals)
            write_to_file(est_num_reads)

''' Calls main function '''
if __name__ == "__main__":
    main(sys.argv[1:])