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
lookup = [d.cdf(i) for i in range(1000)]
effective_length_dict = {}


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

def first_record_p():
    if sr_pos > 0:
        try:
            file = open(input_file, "r")
            for i in range(sr_pos + 1):
                next(file)
            return file
        except FileNotFoundError:
            print("File not found")

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

def eff_len(length, cmeans):
    global d 
    global effective_length_dict
    
    if length in effective_length_dict:
        return effective_length_dict[length]
    else:
        # p = np.array([d.pdf(i) for i in range(length + 1)])
        # p /= p.sum()
        # cond_mean = np.sum([i * p[i] for i in range(len(p))])
        # eff_len = length - cond_mean
        
        if length >= 1000:
            return round(length - cmeans[-1])
        else:
            return round(length - cmeans[length])

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

def calc_p_vals():
    global ref_dict
    ref_dict = read_transcripts()
    file = first_record_p()

    label_list = []
    prob_list = []
    ind_list = [0]
    
    cmeans = correction_factors_from_mass(d)

    while True:
        # Testing output.
        print("Calculating p values for " + str(len(ind_list)) + ".")

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
            print("Calculating the effective lenth for P2")

            P_2 = 1 / eff_len(tlen, cmeans)
            
            print("Calculating P3")
            P_3 = calc_p3(ori, tlen, pos)
            P_4 = float(toks[3])
            prob_list.append(P_2 * P_3 * P_4)
        ind_list.append(len(prob_list))
        
    label_list = np.array(label_list)
    prob_list = np.array(prob_list)
    ind_list = np.array(ind_list)
    
    return (label_list, prob_list, ind_list)

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



@jit(nopython=True)
def full_em(label_list, prob_list, ind_list):
    eta = np.ones(sr_pos) / float(sr_pos)
    eta_p = np.zeros(sr_pos)
    converged = False

    est_num_reads = np.zeros(sr_pos)

    it = 0
    ni = len(ind_list) - 1

    while not converged:
        
        print(it)

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

        
        # converged_res = np.array([False, False])
        # converged_res = np.isclose(eta, eta_p, small_eps)

        converged = True
        
        # for i in range(0, len(converged_res)):
            # if not converged_res[i]:
                # converged = False

        for i in range(0, len(eta)):
            if abs(eta_p[i] - eta[i]) > small_eps:
                converged = False
                break

        #################




        if converged:
            break
        else:
            eta = eta_p
            eta_p = np.zeros(sr_pos)
            est_num_reads = np.zeros(sr_pos)
    
    return (est_num_reads)

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
        label_list, prob_list, ind_list = calc_p_vals()

        if not eqc_flag:
            (est_num_reads) = full_em(label_list, prob_list, ind_list)
            write_to_file(est_num_reads)
        else:
            print("EQC Flag enabled.")

if __name__ == "__main__":
    main(sys.argv[1:])







