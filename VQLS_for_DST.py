from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile, assemble
import math
import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time


def apply_fixed_ansatz(qubits, parameters):

    for iz in range(0, len(qubits)):
        circ.ry(parameters[0][iz], qubits[iz])

    circ.cz(qubits[0], qubits[1])
    #     circ.cz(qubits[1], qubits[0])

    for iz in range(0, len(qubits)):
        circ.ry(parameters[1][iz], qubits[iz])

    circ.cz(qubits[1], qubits[0])
    #     circ.cz(qubits[0], qubits[1])

    for iz in range(0, len(qubits)):
        circ.ry(parameters[2][iz], qubits[iz])


def had_test(gate_type, qubits, auxiliary_index, parameters, mode):
    # mode: 0 for real value, 1 for imag value

    circ.h(auxiliary_index)

    if (mode == 1):
        circ.sdg(auxiliary_index)

    #     parameters = [parameters[2],parameters[1],parameters[0]]
    apply_fixed_ansatz(qubits, parameters)

    for ie in range(0, len(gate_type[0])):
        if (gate_type[0][ie] == 1):
            circ.cx(auxiliary_index, qubits[ie])
        if (gate_type[0][ie] == 2):
            circ.cy(auxiliary_index, qubits[ie])
        if (gate_type[0][ie] == 3):
            circ.cz(auxiliary_index, qubits[ie])

    for ie in range(0, len(gate_type[1])):
        if (gate_type[1][ie] == 1):
            circ.cx(auxiliary_index, qubits[ie])
        if (gate_type[1][ie] == 2):
            circ.cy(auxiliary_index, qubits[ie])
        if (gate_type[1][ie] == 3):
            circ.cz(auxiliary_index, qubits[ie])

    circ.h(auxiliary_index)


def control_fixed_ansatz(qubits, parameters, auxiliary):
    for i in range(0, len(qubits)):
        circ.cry(parameters[0][i], auxiliary, qubits[i])

    circ.ccx(auxiliary, qubits[0], 3)
    circ.cz(qubits[1], 3)
    circ.ccx(auxiliary, qubits[0], 3)

    for i in range(0, len(qubits)):
        circ.cry(parameters[1][i], auxiliary, qubits[i])

    circ.ccx(auxiliary, qubits[1], 3)
    circ.cz(qubits[0], 3)
    circ.ccx(auxiliary, qubits[1], 3)

    for i in range(0, len(qubits)):
        circ.cry(parameters[2][i], auxiliary, qubits[i])

def reverse_para_array(array):
    return [[-l for l in array[0]], [-l for l in array[1]],[-l for l in array[2]]]
def make_b_paras(m):
    theta1 = math.atan(math.sqrt((math.pow(m[2], 2) + math.pow(m[3], 2)) / (
                math.pow(m[1], 2) + math.pow(m[0], 2))))
    theta2 = math.atan(math.sqrt(math.pow(m[1], 2) / math.pow(m[0], 2)))
    theta3 = math.atan(math.sqrt(math.pow(m[3], 2) / math.pow(m[2], 2)))
    return[theta1, theta2, theta3]
def reverse_b_paras(array):
    return [-l for l in array]
def control_b(auxiliary, qubits, m_paras):
    circ.cry(2 * m_paras[0], auxiliary, qubits[0])
    circ.cx(auxiliary, qubits[0])
    circ.ccx(auxiliary, qubits[0], 3)
    circ.cry(2 * m_paras[1], 3, qubits[1])
    circ.ccx(auxiliary, qubits[0], 3)
    circ.cx(auxiliary, qubits[0])
    circ.ccx(auxiliary, qubits[0], 3)
    circ.cry(2 * m_paras[2], 3, qubits[1])
    circ.ccx(auxiliary, qubits[0], 3)


def special_had_test(gate_type, qubits, auxiliary_index, parameters, b_paras, mode):
    # U/V => 1
    # V/U => 2

    circ.h(auxiliary_index)

    if (mode == 1):
        circ.sdg(auxiliary_index)

    control_b(auxiliary_index, qubits, reverse_b_paras(b_paras))
    #     control_b(auxiliary_index, qubits, (b_paras))

    for ie in range(0, len(gate_type)):
        if (gate_type[ie] == 1):
            circ.cx(auxiliary_index, qubits[ie])
        if (gate_type[ie] == 2):
            circ.cy(auxiliary_index, qubits[ie])
        if (gate_type[ie] == 3):
            circ.cz(auxiliary_index, qubits[ie])

            # [parameters[2],parameters[1],parameters[0]]
    control_fixed_ansatz(qubits, [parameters[2], parameters[1], parameters[0]], auxiliary_index)

    circ.h(auxiliary_index)


def calculate_cost_function(parameters):
    time_start = time.time()

    global opt

    overall_sum_1 = 0;

    parameters = [parameters[0:2], parameters[2:4], parameters[4:6]]

    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):

            global circ

            multiply = coefficient_set[i] * complex(coefficient_set[j].real, -coefficient_set[j].imag)

            re = 0
            im = 0

            # real part
            qctl = QuantumRegister(4)
            qc = ClassicalRegister(4)

            circ = QuantumCircuit(qctl, qc)

            backend = Aer.get_backend('aer_simulator')

            had_test([gate_set[j], gate_set[i]], [1, 2], 0, parameters, 0)

            circ.save_statevector()
            t_circ = transpile(circ, backend)
            # qobj = assemble(t_circ)
            job = backend.run(circ)

            result = job.result()
            outputstate = np.array(result.get_statevector(circ, decimals=decimals_num))
            o = outputstate

            m_sum = 0
            for l in range(0, len(o)):
                #     print(o[l]**2)
                if (l % 2 == 1):
                    n = o[l] * complex(o[l].real, -o[l].imag)
                    n = n.real
                    m_sum += n

            re = 1 - (2 * m_sum)

            # imag part
            qctl = QuantumRegister(4)
            qc = ClassicalRegister(4)

            circ = QuantumCircuit(qctl, qc)

            backend = Aer.get_backend('aer_simulator')

            had_test([gate_set[j], gate_set[i]], [1, 2], 0, parameters, 1)

            circ.save_statevector()
            t_circ = transpile(circ, backend)
            # qobj = assemble(t_circ)
            job = backend.run(t_circ)

            result = job.result()
            outputstate = np.array(result.get_statevector(circ, decimals=decimals_num))
            o = outputstate

            m_sum = 0
            for l in range(0, len(o)):
                #     print(o[l]**2)
                if (l % 2 == 1):
                    n = o[l] * complex(o[l].real, -o[l].imag)
                    n = n.real
                    m_sum += n

            im = 1 - (2 * m_sum)

            im = -im

            overall_sum_1 += (multiply * complex(re, im)).real

    overall_sum_2 = 0

    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):

            multiply = coefficient_set[i] * complex(coefficient_set[j].real, -coefficient_set[j].imag)
            mult = 1

            for extra in range(0, 2):

                if (extra == 0):
                    test_gate = gate_set[i]
                if (extra == 1):
                    test_gate = gate_set[j]

                re = 0
                im = 0

                # real part
                qctl = QuantumRegister(4)
                qc = ClassicalRegister(4)

                circ = QuantumCircuit(qctl, qc)

                backend = Aer.get_backend('aer_simulator')

                special_had_test(test_gate, [1, 2], 0, parameters, b_paras, 0)

                circ.save_statevector()
                t_circ = transpile(circ, backend)
                # qobj = assemble(t_circ)
                job = backend.run(t_circ)

                result = job.result()
                outputstate = np.array(result.get_statevector(circ, decimals=decimals_num))
                o = outputstate

                m_sum = 0
                for l in range(0, len(o)):
                    #     print(o[l]**2)
                    if (l % 2 == 1):
                        n = o[l] * complex(o[l].real, -o[l].imag)
                        n = n.real
                        m_sum += n

                re = 1 - (2 * m_sum)

                # imag part
                qctl = QuantumRegister(4)
                qc = ClassicalRegister(4)

                circ = QuantumCircuit(qctl, qc)

                backend = Aer.get_backend('aer_simulator')

                special_had_test(test_gate, [1, 2], 0, parameters, b_paras, 1)

                circ.save_statevector()
                t_circ = transpile(circ, backend)
                # qobj = assemble(t_circ)
                job = backend.run(t_circ)

                result = job.result()
                outputstate = np.array(result.get_statevector(circ, decimals=decimals_num))
                o = outputstate

                m_sum = 0
                for l in range(0, len(o)):
                    #     print(o[l]**2)
                    if (l % 2 == 1):
                        n = o[l] * complex(o[l].real, -o[l].imag)
                        n = n.real
                        m_sum += n

                im = 1 - (2 * m_sum)


                if 1 in test_gate and not test_gate == [1, 1]:
                    re = -re
                    im = -im

                if extra == 1:
                    im = -im

                mult = (mult * complex(re, im))

            #             print(multiply*mult)
            overall_sum_2 += (multiply * mult).real
    #             print(overall_sum_2)

    time_end = time.time()
    # print('time cost', time_end - time_start, 's')

    print("Current Cost: ", 1 - overall_sum_2 / overall_sum_1, "    <\psi|\psi>: ", overall_sum_1, "    |<b|\psi>|^2: ", overall_sum_2)
    # print([overall_sum_1, overall_sum_2])
    mem.append(abs(1 - float(overall_sum_2 / (overall_sum_1))))
    calc_1.append(overall_sum_1)
    calc_2.append(overall_sum_2)

    return abs(1 - float(overall_sum_2 / (overall_sum_1)))


# Set the parameters for linear system: Ax = b

# Set the matrix A
# Here is an example of A is a transformation matrix MfrB
coefficient_set = [1, -0.5, complex(0,0.5), -0.5, 0.25, complex(0, -0.25), complex(0, 0.5), complex(0, -0.25), -0.25]
gate_set = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]

# Set the vector b
mass_array = [0.05,0.3,0.1,0.55]

# Calculate the parameters of preparing |b>
b_paras = make_b_paras(mass_array)
decimals_num = 100

total_mem = []
total_calc_1 = []
total_calc_2 = []

# Iteration Number of Independent Experiments
iter_num = 20

for i in range(iter_num):
    print("%%%%%%%%%%   Current Iter Num: ",i, "    %%%%%%%%%%")
    mem = []
    calc_1 = []
    calc_2 = []
    out = minimize(calculate_cost_function, x0=[float(random.randint(0,3000))/1000 for i in range(0, 6)], method="COBYLA", options={'maxiter':100,'catol':1e-9})
    print(out)
    total_mem.append(mem)
    total_calc_1.append(calc_1)
    total_calc_2.append(calc_2)


total_mem_true = total_mem[:]
total_calc_1_true = total_calc_1[:]
total_calc_2_true = total_calc_2[:]

mem_array_true = np.mat(total_mem_true)
calc_1_array_true = np.mat(total_calc_1_true)
calc_2_array_true = np.mat(total_calc_2_true)

mem_avg_true = np.mean(mem_array_true, axis = 0)
calc_1_avg_true = np.mean(calc_1_array_true, axis = 0)
calc_2_avg_true = np.mean(calc_2_array_true, axis = 0)

mem_avg_true = np.mean(mem_array_true, axis = 0)
calc_1_avg_true = np.mean(calc_1_array_true, axis = 0)
calc_2_avg_true = np.mean(calc_2_array_true, axis = 0)


mem_avg_true = np.asarray(mem_avg_true)[0]
calc_1_avg_true = np.asarray(calc_1_avg_true)[0]
calc_2_avg_true = np.asarray(calc_2_avg_true)[0]

x = np.arange(len(mem_avg_true))

fig = plt.figure()

ax1 = fig.add_subplot(111)

lns1 = ax1.plot(x, mem_avg_true, '-r', linewidth=2.0, label=r"$C(\alpha)$")
ax2 = ax1.twinx()
lns2 = ax2.plot(x, calc_1_avg_true,'g',linewidth=1.0, label=r'$ \langle \psi|\psi\rangle $')
lns3 = ax2.plot(x, calc_2_avg_true,'b',linewidth=1.0, label=r'$ |\langle b| \psi \rangle|^2$')

lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="upper right")


ax1.set_yscale("log")
ax1.set_adjustable("datalim")
ax1.set_ylim(1e-1, 1.2)
ax1.set_ylabel("Cost function")
ax1.set_xlabel("Iteration")
ax2.set_ylim([0, 3])
ax2.set_ylabel(r"Value of $ \langle \psi|\psi\rangle $ and $ |\langle b| \psi \rangle|^2$")

# plt.savefig('convergence.pdf',dpi=600, bbox_inches='tight')
plt.show()

