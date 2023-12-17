import random
import math
import numpy as np
import casadi
np.set_printoptions(precision=4)

class ModelPredictiveController(object):
    def __init__(self, env, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.Np = env.Np
        self.step_T = env.step_T
        self.action_space = env.action_space
        self.action_lb = env.action_space.low.reshape(-1, 1)
        self.action_ub = env.action_space.high.reshape(-1, 1)
        self.u_target = env.u_target
        self.path = env.path
        # Tunable parameters (19):
        # model - Cf, Cr, a, b, m, Iz
        # stage cost - dx_w, dy_w, dphi_w, v_w, yaw_w, str_w, acc_w (du_w is set as 0.01) - log space
        # terminal cost - dx_w, dy_w, dphi_w, v_w, yaw_w, du_w - log space
        self.tunable_para_high = np.array([-8e4, -8e4, 2.2, 2.2, 2000, 2000,
                                           1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2,
                                           1e2, 1e2, 1e2, 1e2, 1e2, 1e2])
        self.tunable_para_low = np.array([-16e4, -16e4, 0.8, 0.8, 1000, 1000,
                                          1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6,
                                          1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        self.x0 = ([0, 0, 0, 0, 0, 0, 0, 0] * (self.Np + 1))
        self.x0.pop(-1)
        self.x0.pop(-1)
        self.ref_p = None
        self.lin_para_gain = 0.5 * (self.tunable_para_high[:6] - self.tunable_para_low[:6])
        self.lin_para_bias = 0.5 * (self.tunable_para_high[:6] + self.tunable_para_low[:6])
        self.log_para_gain = 0.5 * (np.log10(self.tunable_para_high[6:]) - np.log10(self.tunable_para_low[6:]))
        self.log_para_bias = 0.5 * (np.log10(self.tunable_para_high[6:]) + np.log10(self.tunable_para_low[6:]))
        self.tunable_para_mapped = np.random.uniform(-1, 1, 19)
        self.tunable_para_unmapped = self.tunable_para_transform(self.tunable_para_mapped, after_map=True)
        self.tunable_para_sigma = 0.01 * np.ones(19)
        self.para_num = 19
        self.model_para_num = 6
        self.gamma = 0.99

    def tunable_para_transform(self, para_in, after_map):
        if after_map:
            lin_para_mapped = para_in[:6]
            log_para_mapped = para_in[6:]
            lin_para_unmapped = self.lin_para_gain * lin_para_mapped + self.lin_para_bias
            log_para_unmapped = np.power(10, self.log_para_gain * log_para_mapped + self.log_para_bias)
            para_out = np.concatenate((lin_para_unmapped, log_para_unmapped))
        else:
            lin_para_unmapped = para_in[:6]
            log_para_unmapped = para_in[6:]
            lin_para_mapped = (lin_para_unmapped - self.lin_para_bias) / self.lin_para_gain
            log_para_mapped = (np.log10(log_para_unmapped) - self.log_para_bias) / self.log_para_gain
            para_out = np.concatenate((lin_para_mapped, log_para_mapped))
        return para_out

    def get_flat_param(self, after_map=True):
        if after_map:
            return self.tunable_para_mapped
        else:
            return self.tunable_para_unmapped

    def set_flat_param(self, para, after_map=True):
        if after_map:
            para = np.clip(para, -1., 1.)
            self.tunable_para_mapped = para
            para_unmapped = self.tunable_para_transform(para, after_map)
            self.tunable_para_unmapped = para_unmapped
        else:
            para = np.clip(para, self.tunable_para_low, self.tunable_para_high)
            self.tunable_para_unmapped = para
            para_mapped = self.tunable_para_transform(para, after_map)
            self.tunable_para_mapped = para_mapped

    def get_flat_sigma(self):
        return self.tunable_para_sigma

    def set_flat_sigma(self, para):
        self.tunable_para_sigma = np.clip(para, 1e-2, 0.2)

    def step_forward(self, state, action):
        x, v_x, v_y, r, y, phi = state[0], state[1], state[2], state[3], state[4], state[5]
        steer, a_x = action[0], action[1]
        C_f, C_r, a, b, mass, I_z = self.tunable_para_unmapped[:6].tolist()
        tau = self.step_T
        next_state = [x + tau * (v_x * casadi.cos(phi) - v_y * casadi.sin(phi)),
                      v_x + tau * a_x,
                      (mass * v_y * v_x + tau * (a * C_f - b * C_r) * r -
                       tau * C_f * steer * v_x - tau * mass * (v_x ** 2) * r)
                      / (mass * v_x - tau * (C_f + C_r)),
                      (I_z * r * v_x + tau * (a * C_f - b * C_r) * v_y
                       - tau * a * C_f * steer * v_x) /
                      (I_z * v_x - tau * ((a ** 2) * C_f + (b ** 2) * C_r)),
                      y + tau * (v_x * casadi.sin(phi) + v_y * casadi.cos(phi)),
                      phi + tau * r]
        return next_state

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
                f = casadi.vertcat(*self.step_forward(x, u))
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
            