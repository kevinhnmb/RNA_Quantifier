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
big_eps = np.exp(-10)
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

def eqc_em(eq_classes_keys, eq_classes_vals, p2_vals):
    eta = np.ones(sr_pos) / float(sr_pos)
    eta_p = np.zeros(sr_pos)
    converged = False

    est_num_reads = np.zeros(sr_pos)

    it = 0
    ni = len(eq_classes_keys)

    while not converged:
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
            print("Full EM.")
        else:
            (eq_classes, eq_p2) = create_eq_classes()
            eq_classes_keys = list(eq_classes.keys())
            eq_classes_vals = list(eq_classes.values())
            p2_vals = list(eq_p2.values())

            est_num_reads = eqc_em(eq_classes_keys, eq_classes_vals, p2_vals)
            write_to_file(est_num_reads)

if __name__ == "__main__":
    main(sys.argv[1:])