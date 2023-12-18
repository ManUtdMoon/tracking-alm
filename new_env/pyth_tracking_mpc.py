import random
import math
import numpy as np
import torch
from copy import deepcopy
import casadi

np.set_printoptions(precision=4)

class ModelPredictiveController(object):
    def __init__(self, env, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.Np = env.Np
        self.step_T = env.step_T
        self.action_space = env.action_space
        self.action_lb = torch.tensor(env.action_space.low)
        self.action_ub = torch.tensor(env.action_space.high)
        self.u_target = env.u_target
        self.path = env.path
        # Tunable parameters (19):
        # model - Cf, Cr, a, b, m, Iz
        # stage cost - dx_w, dy_w, dphi_w, v_w, yaw_w, str_w, acc_w (du_w is set as 0.01) - log space
        # terminal cost - dx_w, dy_w, dphi_w, v_w, yaw_w, du_w - log space
        self.tunable_para_unmapped = torch.tensor([-128916, -85944, 1.06, 1.85, 1412, 1536.7], dtype=torch.float32)
        self.Q_mat = torch.diag(torch.tensor([1.0, 1e-5, 1., 10., 30., 50.], dtype=torch.float32))
        self.R_mat = torch.diag(torch.tensor([60., 1.], dtype=torch.float32))
        self.Q_vec = torch.tensor([1.0, 1e-5, 1., 10., 30., 50.], dtype=torch.float32)
        self.R_vec = torch.tensor([60., 1.], dtype=torch.float32)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x0 = torch.rand(8 * self.Np, dtype=torch.float32, requires_grad=True)
        self.dim = len(self.x0)
        self.ref_p = None
        self.gamma = 0.99

    def batch_step_forward(self, states, actions):
        assert states.shape[0] == actions.shape[0]
        assert states.shape[1] == 6
        assert actions.shape[1] == 2
        x, v_x, v_y, r, y, phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f, C_r, a, b, mass, I_z = self.tunable_para_unmapped[:6].tolist()
        tau = self.step_T

        next_x = x + tau * (v_x * torch.cos(phi) - v_y * torch.sin(phi))
        next_u = v_x + tau * a_x
        next_v = (mass * v_y * v_x + tau * (a * C_f - b * C_r) * r -
                 tau * C_f * steer * v_x - tau * mass * (v_x ** 2) * r) / (mass * v_x - tau * (C_f + C_r))
        next_w = (I_z * r * v_x + tau * (a * C_f - b * C_r) * v_y
                 - tau * a * C_f * steer * v_x) / (I_z * v_x - tau * ((a ** 2) * C_f + (b ** 2) * C_r))
        next_y = y + tau * (v_x * torch.sin(phi) + v_y * torch.cos(phi))
        next_phi = phi + tau * r

        next_states = torch.stack([next_x, next_u, next_v, next_w, next_y, next_phi], dim=1)

        return next_states

    def step_forward(self, state, action):
        x, v_x, v_y, r, y, phi = state[0], state[1], state[2], state[3], state[4], state[5]
        steer, a_x = action[0], action[1]
        C_f, C_r, a, b, mass, I_z = self.tunable_para_unmapped[:6].tolist()
        tau = self.step_T
        next_state = torch.tensor([x + tau * (v_x * torch.cos(phi) - v_y * torch.sin(phi)),
                      v_x + tau * a_x,
                      (mass * v_y * v_x + tau * (a * C_f - b * C_r) * r -
                       tau * C_f * steer * v_x - tau * mass * (v_x ** 2) * r)
                      / (mass * v_x - tau * (C_f + C_r)),
                      (I_z * r * v_x + tau * (a * C_f - b * C_r) * v_y
                       - tau * a * C_f * steer * v_x) /
                      (I_z * v_x - tau * ((a ** 2) * C_f + (b ** 2) * C_r)),
                      y + tau * (v_x * torch.sin(phi) + v_y * torch.cos(phi)),
                      phi + tau * r])
        return next_state
    
    def compute_cost_batch(self, x):
        cost = torch.tensor(0., dtype=torch.float32)
        x_init = torch.tensor(self.initial_state[:6], dtype=torch.float32)
        x = x.reshape(-1, 8)
        actions = x[:, 0:2]
        next_states_variable = x[:, 2:] # (Np, 6) x_1~x_Np
        discount = torch.tensor([self.gamma**i for i in range(1, 1 + self.Np)], dtype=torch.float32)

        # tracking cost
        ref_x = torch.as_tensor(self.initial_state[6::3], dtype=torch.float32) # x
        ref_y = torch.as_tensor(self.initial_state[7::3], dtype=torch.float32) # y
        ref_phi = torch.as_tensor(self.initial_state[8::3], dtype=torch.float32) # phi
        ref = torch.stack(
            [
                ref_x, # x
                self.u_target * torch.ones_like(ref_x, dtype=torch.float32), # u
                torch.zeros_like(ref_x, dtype=torch.float32), # v
                torch.zeros_like(ref_x, dtype=torch.float32), # w
                ref_y, # y
                ref_phi, # phi
            ],
            dim=1
        )
        assert ref.shape == next_states_variable.shape
        tracking_error = next_states_variable - ref # (Np, 6)
        tracking_cost = torch.sum(torch.sum(tracking_error * tracking_error * self.Q_vec, dim=1) * discount) # scalar

        # action cost
        action_cost = torch.sum(torch.sum(actions * actions * self.R_vec, dim=1) * discount) # scalar
        cost += tracking_cost + action_cost

        # dynamics constraints
        current_states = torch.cat((x_init.reshape(1, -1), x[:-1, 2:]), dim=0) # (Np, 6) x_0~x_(Np-1)
        next_states_rollout = self.batch_step_forward(current_states, actions) # (Np, 6) x_1~x_Np
        state_deviation = (next_states_rollout - next_states_variable).reshape(-1) # (Np * 6,)
        cost += torch.dot(self.lam, state_deviation) + self.rho / 2. * torch.linalg.norm(state_deviation, 1) ** 2

        # action bound constraints
        violation_lb = torch.max(self.action_lb - actions, torch.zeros_like(actions, dtype=torch.float32)).reshape(-1)
        violation_ub = torch.max(actions - self.action_ub, torch.zeros_like(actions, dtype=torch.float32)).reshape(-1)
        cost += torch.dot(self.lam_lb, violation_lb) + self.rho / 2. * torch.linalg.norm(violation_lb, 1) ** 2
        cost += torch.dot(self.lam_ub, violation_ub) + self.rho / 2. * torch.linalg.norm(violation_ub, 1) ** 2

        return cost

    def compute_cost(self, x):
        cost = torch.tensor(0., dtype=torch.float32)
        # optimized vector: u0, x1, u1, ..., xn-1, un-1, xn
        last_state = torch.tensor(self.initial_state[:6], dtype=torch.float32)
        for k in range(1, self.Np + 1):
            # dynamic_state: x, u, v, yaw, y, phi
            act = x[(k-1)*8: (k-1)*8+2]
            act_cost = torch.matmul(act, torch.matmul(self.R_mat, act)) * (self.gamma ** (k-1))
            cost += act_cost
            ref = torch.tensor([
                self.initial_state[6 + (k - 1) * 3], # x
                self.u_target, # u
                0., # v
                0., # w
                self.initial_state[6 + (k - 1) * 3 + 1], # y
                self.initial_state[6 + (k - 1) * 3 + 2], # phi
            ],
            dtype=torch.float32).detach()
            state = x[(k-1)*8+2: (k-1)*8+8]
            state_cost = torch.matmul(state - ref, torch.matmul(self.Q_mat, state - ref)) * (self.gamma ** (k))
            cost += state_cost
            lam_dynamics_now = self.lam[(k-1)*6 : k*6]
            lam_lb_now = self.lam_lb[(k-1)*2 : k*2]
            lam_ub_now = self.lam_ub[(k-1)*2 : k*2]

            violation_dynamics = self.step_forward(last_state, act) - state
            violation_lb = torch.max(self.action_lb - act, torch.zeros(2, dtype=torch.float32))
            violation_ub = torch.max(act - self.action_ub, torch.zeros(2, dtype=torch.float32))

            cost += torch.dot(lam_dynamics_now, violation_dynamics) + self.rho / 2. * torch.linalg.norm(violation_dynamics) ** 2
            last_state = state

            cost += torch.dot(lam_lb_now, self.action_lb - act) + self.rho / 2. * torch.linalg.norm(violation_lb) ** 2
            cost += torch.dot(lam_ub_now, act - self.action_ub) + self.rho / 2. * torch.linalg.norm(violation_ub) ** 2
        return cost


    def get_action(self, initial_state):
        self.ref_p = []
        x = casadi.SX.sym('x', 6)
        u = casadi.SX.sym('u', 2)
        # Create empty NLP
        w = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        ref_list = []
        G = []
        J = 0
        Xk = casadi.MX.sym('X0', 6)
        w += [Xk]
        lbw += [initial_state[0], initial_state[1], initial_state[2], initial_state[3], initial_state[4],
                initial_state[5]]
        ubw += [initial_state[0], initial_state[1], initial_state[2], initial_state[3], initial_state[4],
                initial_state[5]]
        for k in range(1, self.Np + 1):
            f = casadi.vertcat(*self.step_forward(x, u))
            F = casadi.Function("F", [x, u], [f])
            Uname = 'U' + str(k - 1)
            Uk = casadi.MX.sym(Uname, 2)
            w += [Uk]
            lbw += [self.action_space.low[0], self.action_space.low[1]]
            ubw += [self.action_space.high[0], self.action_space.high[1]]
            Fk = F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = casadi.MX.sym(Xname, 6)
            w += [Xk]
            ubw += [casadi.inf, casadi.inf, casadi.inf, casadi.inf, casadi.inf, casadi.inf]
            lbw += [-casadi.inf, -casadi.inf, -casadi.inf, -casadi.inf, -casadi.inf, -casadi.inf]
            # dynamic_state: x, u, v, yaw, y, phi
            G += [Fk - Xk]
            ubg += [0., 0., 0., 0., 0., 0.]
            lbg += [0., 0., 0., 0., 0., 0.]
            REFname = 'REF' + str(k)
            REFk = casadi.MX.sym(REFname, 3)
            ref_list += [REFk]
            self.ref_p += [initial_state[6 + (k - 1) * 3], initial_state[6 + (k - 1) * 3 + 1],
                           initial_state[6 + (k - 1) * 3 + 2]]
        
            ref_cost = 1.0 * casadi.power(w[k * 2][1] - self.u_target, 2)  # u
            ref_cost += self.tunable_para_unmapped[6] * casadi.power(w[k * 2][0] - ref_list[k - 1][0], 2)  # x
            ref_cost += self.tunable_para_unmapped[7] * casadi.power(w[k * 2][4] - ref_list[k - 1][1], 2)  # y
            ref_cost += self.tunable_para_unmapped[8] * casadi.power(w[k * 2][5] - ref_list[k - 1][2], 2)  # phi
            ref_cost += self.tunable_para_unmapped[9] * casadi.power(w[k * 2][2], 2)  # v
            ref_cost += self.tunable_para_unmapped[10] * casadi.power(w[k * 2][3], 2)  # yaw
            ref_cost *= casadi.power(self.gamma, k)
        
            act_cost = self.tunable_para_unmapped[11] * casadi.power(w[k * 2 - 1][0], 2)  # steer
            act_cost += self.tunable_para_unmapped[12] * casadi.power(w[k * 2 - 1][1], 2)  # ax
            act_cost *= casadi.power(self.gamma, k-1)
            J += (ref_cost + act_cost)
        nlp = dict(f=J, g=casadi.vertcat(*G), x=casadi.vertcat(*w), p=casadi.vertcat(*ref_list))
        S = casadi.nlpsol('S', 'ipopt', nlp,
                          {'ipopt.max_iter': 200, 'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0})
        r = S(lbx=lbw, ubx=ubw, x0=self.x0, lbg=lbg, ubg=ubg, p=self.ref_p)
        X = np.array(r['x']).tolist()
        action = np.array([X[6][0], X[7][0]])
        self.x0 = casadi.DM(
            X[8:] + X[-8] + X[-7] + X[-6] + X[-5] + X[-4] + X[-3] + X[-2] + X[-1])  # warm start for faster optimization
        return action, np.array(X[8:14])
    
    def get_action_alm(self, initial_state):
        # ALM-related parameters
        rho = 10.0 # penalty parameter
        rho_amplifier = 10.0 # rho amplification factor
        lam = np.zeros(self.Np * 6, dtype=np.float32).reshape(-1, 1) # Lagrangian multipliers for dynamics
        lam_lb = np.zeros(self.Np * 2, dtype=np.float32).reshape(-1, 1) # for action lb: lb - u <0
        lam_ub = np.zeros(self.Np * 2, dtype=np.float32).reshape(-1, 1) # for action lb: u - ub <0

        max_lam_iter = 100
        
        last_us = np.zeros(self.Np * 2, dtype=np.float32).reshape(-1, 1)
        us_error = np.inf
        us_error_threshold = 1e-4

        max_violation = np.inf
        max_violation_threshold = 1e-5

        alm_per_step = {
            "obj_aug_term": [],
            "obj_primal": [],
            "max_violation_dyanmics": [],
            "max_violation_lb": [],
            "max_violation_ub": [],
            "cost": [],
        }
        
        # outer loop for ALM
        for alm_iter in range(max_lam_iter):
            if us_error < us_error_threshold and max_violation < max_violation_threshold:
                print("----------")
                break

            self.ref_p = []
            x = casadi.SX.sym('x', 6)
            u = casadi.SX.sym('u', 2)
            # Create empty NLP
            w = []
            lbw = []
            ubw = []
            lbg = []
            ubg = []
            ref_list = []
            G = []
            J = 0
            Xk = casadi.MX.sym('X0', 6)
            w += [Xk]
            lbw += [initial_state[0], initial_state[1], initial_state[2], initial_state[3], initial_state[4],
                    initial_state[5]]
            ubw += [initial_state[0], initial_state[1], initial_state[2], initial_state[3], initial_state[4],
                    initial_state[5]]
            for k in range(1, self.Np + 1):
                f = casadi.vertcat(*self.step_forward_casadi(x, u))
                F = casadi.Function("F", [x, u], [f])
                Uname = 'U' + str(k - 1)
                Uk = casadi.MX.sym(Uname, 2)
                w += [Uk]
                lbw += [-casadi.inf, -casadi.inf]
                ubw += [casadi.inf, casadi.inf]
                Fk = F(Xk, Uk)
                Xname = 'X' + str(k)
                Xk = casadi.MX.sym(Xname, 6)
                w += [Xk]
                ubw += [casadi.inf, casadi.inf, casadi.inf, casadi.inf, casadi.inf, casadi.inf]
                lbw += [-casadi.inf, -casadi.inf, -casadi.inf, -casadi.inf, -casadi.inf, -casadi.inf]
                # dynamic_state: x, u, v, yaw, y, phi
                G += [Fk - Xk]
                ubg += [0., 0., 0., 0., 0., 0.]
                lbg += [0., 0., 0., 0., 0., 0.]
                REFname = 'REF' + str(k)
                REFk = casadi.MX.sym(REFname, 3)
                ref_list += [REFk]
                self.ref_p += [initial_state[6 + (k - 1) * 3], initial_state[6 + (k - 1) * 3 + 1],
                            initial_state[6 + (k - 1) * 3 + 2]]
            
                ref_cost = 1.0 * casadi.power(w[k * 2][1] - self.u_target, 2)  # u
                ref_cost += self.tunable_para_unmapped[6] * casadi.power(w[k * 2][0] - ref_list[k - 1][0], 2)  # x
                ref_cost += self.tunable_para_unmapped[7] * casadi.power(w[k * 2][4] - ref_list[k - 1][1], 2)  # y
                ref_cost += self.tunable_para_unmapped[8] * casadi.power(w[k * 2][5] - ref_list[k - 1][2], 2)  # phi
                ref_cost += self.tunable_para_unmapped[9] * casadi.power(w[k * 2][2], 2)  # v
                ref_cost += self.tunable_para_unmapped[10] * casadi.power(w[k * 2][3], 2)  # yaw
                ref_cost *= casadi.power(self.gamma, k)
            
                act_cost = self.tunable_para_unmapped[11] * casadi.power(w[k * 2 - 1][0], 2)  # steer
                act_cost += self.tunable_para_unmapped[12] * casadi.power(w[k * 2 - 1][1], 2)  # ax
                act_cost *= casadi.power(self.gamma, k-1)
                J += (ref_cost + act_cost)

                # put dynamics constraints into lagrangian
                lam_dynamics_now = lam[(k-1)*6 : k*6, :]
                lam_lb_now = lam_lb[(k-1)*2 : k*2, :]
                lam_ub_now = lam_ub[(k-1)*2 : k*2, :]

                violation_dynamics = Fk - Xk
                violation_lb = casadi.fmax(self.action_lb - Uk, np.zeros((2,1), dtype=np.float32))
                violation_ub = casadi.fmax(Uk - self.action_ub, np.zeros((2,1), dtype=np.float32))

                J += casadi.dot(lam_dynamics_now, violation_dynamics) + rho / 2. * casadi.sum1(casadi.power(violation_dynamics, 2))
                J += casadi.dot(lam_lb_now, self.action_lb - Uk) + rho / 2. * casadi.sum1(casadi.power(violation_lb, 2))
                J += casadi.dot(lam_ub_now, Uk - self.action_ub) + rho / 2. * casadi.sum1(casadi.power(violation_ub, 2))
        
            nlp = dict(f=J, x=casadi.vertcat(*w), p=casadi.vertcat(*ref_list))
            S = casadi.nlpsol('S', 'ipopt', nlp,
                            {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0})
            r = S(lbx=lbw, ubx=ubw, x0=self.x0, p=self.ref_p)
            X = np.array(r['x']).tolist()
            action = np.array([X[6][0], X[7][0]])
            self.x0 = casadi.DM(
                X[8:] + X[-8] + X[-7] + X[-6] + X[-5] + X[-4] + X[-3] + X[-2] + X[-1]
            )  # warm start for faster optimization

            # update Lagrangian multipliers
            ## current constraint violation
            sol = np.array(X).reshape(-1)
            xs, us, x_gts = [], [], []

            for k in range(1, self.Np + 1):
                u_sol = sol[8*k-2 : 8*k]
                x_sol = sol[8*k : 8*k+6]
                x_last = sol[8*(k-1) : 8*k-2]
                x_gt = self.step_forward(x_last, u_sol)

                us.append(u_sol) # (2,)
                xs.append(x_sol) # (6,)
                x_gts.append(x_gt) # (6,)
            
            xs = np.array(xs).reshape(-1, 1)
            x_gts = np.array(x_gts).reshape(-1, 1)
            us = np.array(us).reshape(-1, 1)

            ulbs = np.array([self.action_lb] * self.Np).reshape(-1, 1)
            uubs = np.array([self.action_ub] * self.Np).reshape(-1, 1)

            violation = x_gts - xs

            violation_lb = ulbs - us
            violation_ub = us - uubs

            us_error = np.abs(us - last_us).max()
            last_us = us

            max_violation = np.max(np.array([np.abs(violation).max(), violation_lb.max(), violation_ub.max()]))

            assert us.shape == (2*self.Np, 1)
            assert violation.shape == (6*self.Np, 1)
            assert lam.shape == (6*self.Np, 1)

            alm_per_step["obj_aug_term"].append(np.dot(lam.reshape(-1), violation.reshape(-1))
                + np.dot(lam_lb.reshape(-1), violation_lb.reshape(-1))
                + np.dot(lam_ub.reshape(-1), violation_ub.reshape(-1))
                + rho / 2. * (np.sum(violation**2) 
                              + np.sum(np.maximum(violation_lb, 0)**2)
                              + np.sum(np.maximum(violation_ub, 0)**2))
            )
            alm_per_step["obj_primal"].append(np.array(r['f']).item() - alm_per_step["obj_aug_term"][-1])
            alm_per_step["max_violation_dyanmics"].append(np.abs(violation).max())
            alm_per_step["max_violation_lb"].append(violation_lb.max())
            alm_per_step["max_violation_ub"].append(violation_ub.max())

            cost = 0
            for i in range(self.Np):
                xx = xs[i*6 : (i+1)*6, :].reshape(-1)
                uu = us[i*2 : (i+1)*2, :].reshape(-1)

                act_cost = self.tunable_para_unmapped[11] * uu[0]**2  # steer
                act_cost += self.tunable_para_unmapped[12] * uu[1]**2
                cost += act_cost * self.gamma**(i+1)

                ref_point = np.array([self.ref_p[i*3], self.ref_p[i*3+1], self.ref_p[i*3+2]])
                x, y, phi = xx[0], xx[4], xx[5]
                ref_x, ref_y, ref_phi = ref_point[0], ref_point[1], ref_point[2]
                ref_cost = self.tunable_para_unmapped[6] * (x - ref_x)**2
                ref_cost += self.tunable_para_unmapped[7] * (y - ref_y)**2
                ref_cost += self.tunable_para_unmapped[8] * (phi - ref_phi)**2
                ref_cost += self.tunable_para_unmapped[9] * xx[2]**2
                ref_cost += self.tunable_para_unmapped[10] * xx[3]**2
                ref_cost += 1.0 * (xx[1]-self.u_target)**2
                cost += ref_cost * self.gamma**i
            alm_per_step["cost"].append(cost)
                

            lam = lam + rho * violation
            lam_lb = np.maximum(lam_lb + rho * violation_lb, 0.)
            lam_ub = np.maximum(lam_ub + rho * violation_ub, 0.)
            rho *= rho_amplifier

            print(f"ALM iteration {alm_iter}: us_error {us_error:.6f}, max_violation {max_violation:.6f}")
            print(f"lam: {lam.reshape(-1)[-5:]}, lam_lb: {lam_lb.reshape(-1)[:5]}, lam_ub: {lam_ub.reshape(-1)[:5]}")

        else:
            print(f"reach max iteration {max_lam_iter} for ALM")

        # store per-step alm result
        opt_result = {}
        opt_result["num_iteration"] = alm_iter + 1
        opt_result["u"] = action
        opt_result["x"] = np.array(initial_state[0:6]).reshape(-1)
        opt_result["x_ref"] = np.array(initial_state[6:9]).reshape(-1)
        opt_result["x_next"] = np.array(X[8:14]).reshape(-1)
        opt_result["alm_process"] = alm_per_step

        return action, opt_result

    def get_action_alm_gd(self, initial_state):
        # ALM-related parameters
        self.rho = 10.0 # penalty parameter
        self.rho_amplifier = 10.0 # rho amplification factor
        self.lam = torch.zeros(self.Np * 6, dtype=torch.float32) # Lagrangian multipliers for dynamics
        self.lam_lb = torch.zeros(self.Np * 2, dtype=torch.float32) # for action lb: lb - u <0
        self.lam_ub = torch.zeros(self.Np * 2, dtype=torch.float32) # for action lb: u - ub <0
        self.initial_state = initial_state

        max_lam_iter = 100
        
        last_us = np.zeros(self.Np * 2, dtype=np.float32)
        us_error = np.inf
        us_error_threshold = 1e-4

        max_violation = np.inf
        max_violation_threshold = 1e-5

        alm_per_step = {
            "obj_aug_term": [],
            "obj_primal": [],
            "max_violation_dyanmics": [],
            "max_violation_lb": [],
            "max_violation_ub": [],
            "cost": [],
        }

        
        # outer loop for ALM
        for alm_iter in range(max_lam_iter):
            if us_error < us_error_threshold and max_violation < max_violation_threshold:
                print("----------")
                break
            # inner loop for Newton
            max_iteration = 500
            for iter in range(max_iteration):
                # compute newton step and decrement
                cost = self.compute_cost_batch(self.x0)
                g = torch.autograd.grad(cost, self.x0, retain_graph=True, create_graph=True)[0]
                H = torch.autograd.functional.hessian(self.compute_cost_batch, self.x0)
                inv_H = torch.linalg.inv(H)
                with torch.no_grad():
                    delta = - torch.matmul(inv_H, g)
                    lam_sq = - torch.matmul(g, delta)
                    # delta = -g[0]
                    # norm = torch.linalg.norm(delta)
                    print(lam_sq)
                if lam_sq < 1e-5:
                    break
                # backtracking line search
                alpha, beta = 0.25, 0.8
                t = 1
                with torch.no_grad():
                    while self.compute_cost_batch(self.x0 + t * delta) > cost+alpha*t*torch.dot(g, delta):
                        t *= beta
                self.x0 = self.x0 + t * delta
                print(self.x0[:2])
            else:
                print("Reach maximum inner loop.")


            # update Lagrangian multipliers
            ## current constraint violation
            with torch.no_grad():
                sol = torch.cat((torch.as_tensor(initial_state[:6], dtype=torch.float32), self.x0.detach().reshape(-1)), 0)
                xs, us, x_gts = [], [], []

                for k in range(1, self.Np + 1):
                    u_sol = sol[8*k-2 : 8*k]
                    x_sol = sol[8*k : 8*k+6]
                    x_last = sol[8*(k-1) : 8*k-2]
                    x_gt = self.step_forward(x_last, u_sol)

                    us.append(u_sol) # (2,)
                    xs.append(x_sol) # (6,)
                    x_gts.append(x_gt) # (6,)
                
                xs = torch.cat(xs, 0)
                x_gts = torch.cat(x_gts, 0)
                us = torch.cat(us, 0)

                ulbs = self.action_lb.repeat(self.Np).reshape(-1)
                uubs = self.action_ub.repeat(self.Np).reshape(-1)

                violation = x_gts - xs

                violation_lb = ulbs - us
                violation_ub = us - uubs

                us_error = torch.abs(us - last_us).max()
                last_us = us

                max_violation = torch.max(torch.tensor([torch.abs(violation).max(), violation_lb.max(), violation_ub.max()]))

                assert us.shape == (2*self.Np,)
                assert violation.shape == (6*self.Np,)
                assert self.lam.shape == (6*self.Np,)

                alm_per_step["obj_aug_term"].append(torch.dot(self.lam, violation)
                    + torch.dot(self.lam_lb, violation_lb)
                    + torch.dot(self.lam_ub, violation_ub)
                    + self.rho / 2. * (torch.sum(violation**2) 
                                + torch.sum(torch.maximum(violation_lb, torch.zeros_like(violation_lb))**2)
                                + torch.sum(torch.maximum(violation_ub, torch.zeros_like(violation_ub))**2)
                    )
                )
                alm_per_step["obj_primal"].append(cost - alm_per_step["obj_aug_term"][-1])
                alm_per_step["max_violation_dyanmics"].append(torch.abs(violation).max())
                alm_per_step["max_violation_lb"].append(violation_lb.max())
                alm_per_step["max_violation_ub"].append(violation_ub.max())

                self.lam = self.lam + self.rho * violation
                self.lam_lb = torch.maximum(self.lam_lb + self.rho * violation_lb, torch.zeros_like(self.lam_lb))
                self.lam_ub = torch.maximum(self.lam_ub + self.rho * violation_ub, torch.zeros_like(self.lam_ub))
                self.rho *= self.rho_amplifier

                print(f"ALM iteration {alm_iter}: us_error {us_error:.6f}, max_violation {max_violation:.6f}")
                print(f"lam: {self.lam.reshape(-1)[-5:]}, lam_lb: {self.lam_lb.reshape(-1)[:5]}, lam_ub: {self.lam_ub.reshape(-1)[:5]}")
        else:
            print(f"reach max iteration {max_lam_iter} for ALM")
            
        # store per-step alm result
        opt_result = {}
        opt_result["num_iteration"] = alm_iter + 1
        opt_result["u"] = sol[6:8]
        opt_result["x"] = np.array(initial_state[0:6]).reshape(-1)
        opt_result["x_ref"] = np.array(initial_state[6:9]).reshape(-1)
        opt_result["x_next"] = np.array(x_sol[8:14]).reshape(-1)
        opt_result["alm_process"] = alm_per_step

        return sol[6:8], opt_result

