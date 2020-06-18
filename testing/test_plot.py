import matplotlib.pyplot as plt
import numpy as np

actual = "../data/true_counts.tsv"
output1 = "../quants.tsv"
output2 = "../quants_eqc.tsv"

x1 = []
x2 = []
y = []

# Read actual data
with open(actual, "r") as file:
    file.readline()
    while True:
        cl = file.readline()
        if cl == "":
            break
        cl_d = cl.strip("\n").split("\t")
        y.append(float(cl_d[1]))
    file.close()

# Read output data
with open(output1, "r") as file:
    file.readline()
    while True:
        cl = file.readline()
        if cl == "":
            break
        cl_d = cl.strip("\n").split("\t")
        x1.append(float(cl_d[2]))
    file.close()

# Read output data
with open(output2, "r") as file:
    file.readline()
    while True:
        cl = file.readline()
        if cl == "":
            break
        cl_d = cl.strip("\n").split("\t")
        x2.append(float(cl_d[2]))
    file.close()


area = 15  # 0 to 15 point radii


# plt.scatter(x1, y, s=area, alpha=0.5)

plt.subplot(1, 2, 1)
plt.scatter(x1, y, s=area, alpha=0.5)
plt.title('Full EM Algorithm')
plt.grid(True)
plt.xlabel('Estimated Abundance')
plt.ylabel('Actual Abundance')

plt.subplot(1, 2, 2)
plt.scatter(x2, y, s=area, alpha=0.5)
plt.title("EQC EM Algorithm")
plt.grid(True)
plt.xlabel('Estimated Abundance')
plt.ylabel('Actual Abundance')

plt.savefig("test.png")
plt.show()



