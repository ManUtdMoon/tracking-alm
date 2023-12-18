import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('exp.pkl', 'rb') as f:
    exp = pickle.load(f)

# load arial font and make it for default font for matplotlib
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "Arial"

# plot the optimization results of the first step
alm_process = exp[0]["alm_process"] # obj_aug_term obj_primal max_violation_dyanmics max_violation_lb max_violation_ub
iterations = np.arange(1, 1+len(alm_process["max_violation_dyanmics"]))
print(alm_process["max_violation_dyanmics"])
print(alm_process["max_violation_lb"])
print(alm_process["max_violation_ub"])

print(alm_process["cost"])
print(alm_process["obj_primal"])
print(alm_process["obj_aug_term"])

# # Objective function
# fig, ax1 = plt.subplots(figsize=(5, 2))

# ax1.set_xlabel('Iterations')
# ax1.set_ylabel('Tracking cost', color="r")
# ax1.plot(iterations, alm_process["obj_primal"], "r-")
# ax1.tick_params(axis='y', labelcolor='r')

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# ax2.set_ylabel('Augmented term', color="b")  # we already handled the x-label with ax1
# ax2.semilogy(iterations, alm_process["obj_aug_term"], "b--")
# ax2.tick_params(axis='y', labelcolor="b")

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig("./plot/obj.pdf")
# plt.show()


# # violation
# plt.figure(figsize=(5, 2))
# plt.plot(iterations, alm_process["max_violation_lb"], 'r-', label="Action lb violation")
# plt.plot(iterations, alm_process["max_violation_ub"], 'b--', label="Action ub violation")
# plt.xlabel("Iterations")
# plt.ylabel("Violation")
# plt.legend()
# plt.tight_layout()
# plt.savefig("./plot/act_violation.pdf")


# # Plot dynamics violation on the right subplot with log-y axis
# plt.figure(figsize=(5, 2))
# plt.semilogy(iterations, alm_process["max_violation_dyanmics"], 'k-', label="Dynamics violation")
# plt.xlabel("Iterations")
# plt.ylabel("Violation")
# plt.legend()

# plt.tight_layout()
# plt.savefig("./plot/dyn_violation.pdf")
# plt.show()


# # plot the trajectory of ego car and reference trajectory
# plt.figure(figsize=(5, 2))
# ref_x = np.array([d["x_ref"][0] for d in exp])
# ref_y = np.array([d["x_ref"][1] for d in exp])

# ego_x = np.array([d["x"][0] for d in exp])
# ego_y = np.array([d["x"][4] for d in exp])
# plt.plot(ref_x, ref_y, 'r--', label="reference trajectory")
# plt.plot(ego_x, ego_y, 'k-', label="ego trajectory")

# plt.legend()
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.ylim([-12, 10])
# plt.tight_layout()
# plt.savefig("./plot/traj.pdf")
# plt.show()

# # plot the action
# delta = np.array([d["u"][0] for d in exp])
# a = np.array([d["u"][1] for d in exp])
# fig, ax1 = plt.subplots(figsize=(5, 2))

# steps = np.arange(1, len(delta)+1)

# ax1.set_xlabel('Time step')
# ax1.set_ylabel('Steering angle [rad]', color="r")
# ax1.plot(steps, delta, "r-")
# ax1.tick_params(axis='y', labelcolor='r')

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# ax2.set_ylabel(r'Acceleration [m/s$^2$]', color="b")  # we already handled the x-label with ax1
# ax2.plot(steps, a, "b--")
# ax2.tick_params(axis='y', labelcolor="b")

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig("./plot/action.pdf")
# plt.show()


# plot the iterations of each step
plt.figure(figsize=(5, 2))
steps = np.arange(1, len(exp)+1)
plt.plot(steps, [d["num_iteration"] for d in exp], "k-")
plt.xlabel("Time step")
plt.ylabel("Number of iterations")
plt.tight_layout()
plt.savefig("./plot/iterations.pdf")
plt.show()