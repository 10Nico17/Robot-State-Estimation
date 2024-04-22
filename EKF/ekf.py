import scipy.io
import numpy as np


class EKF:

    def __init__(self, range_max):

        self.x_true = None
        self.y_true = None
        self.th_true = None
        self.l_true = None
        self.true_valid = None
        self.valid_ids = None

        self.timesteps = None
        self.d = None

        self.T = 1 / 10

        self.v = None
        self.v_var = None

        self.omega = None
        self.omega_var = None

        self.range_meas = None
        self.r_var = None

        self.bear_meas = None
        self.b_var = None

        self.range_max = range_max

    def parse_data(self):

        mat = scipy.io.loadmat('dataset2.mat')

        self.x_true = mat["x_true"][:, 0]
        self.y_true = mat["y_true"][:, 0]
        self.th_true = mat["th_true"][:, 0]
        self.l_true = mat["l"]
        self.true_valid = mat["true_valid"][:, 0]
        self.valid_ids = np.nonzero(self.true_valid)

        self.timesteps = mat["t"][:, 0]
        self.d = mat["d"][0][0]

        self.T = 1 / 10

        self.v = mat["v"][:, 0]
        self.v_var = mat["v_var"][0][0]

        self.omega = mat["om"][:, 0]
        self.omega_var = mat["om_var"][0][0]

        self.range_meas = mat["r"]
        self.r_var = mat["r_var"][0][0]

        self.bear_meas = mat["b"]
        self.b_var = mat["b_var"][0][0]



    def calc_f(self, x_k_1, y_k_1, theta_k_1, v_k, omega_k):

        xytheta = np.array([x_k_1, y_k_1, theta_k_1])
        u_k = np.array([v_k, omega_k])

        middle_matrix = np.zeros((3, 2))
        middle_matrix[0][0] = self.T * np.cos(theta_k_1)
        middle_matrix[1][0] = self.T * np.sin(theta_k_1)
        middle_matrix[2][1] = self.T

        return xytheta + middle_matrix @ u_k

    def calc_g(self, k, x_check):

        range_ids = self.slice_valid_measurements(k)

        g = []

        for range_id in range_ids:
            landmark = [self.l_true[range_id[0]][0], self.l_true[range_id[0]][1]]
            a = self.substitute_a(landmark[0], x_check[0], x_check[2])
            b = self.substitute_b(landmark[1], x_check[1], x_check[2])

            g.append(np.sqrt(a ** 2 + b ** 2))
            g.append(self.normalize_angle(np.arctan2(b, a) - x_check[2]))

        g = np.array(g)

        return g

    # Calculate Q prime
    def calc_Q_prime(self, theta_k_1):

        Q_original = np.diag([self.v_var,self.omega_var])

        jacob_w = np.zeros((3,2))
        jacob_w[0][0] = self.T*np.cos(theta_k_1)
        jacob_w[1][0] = self.T*np.sin(theta_k_1)
        jacob_w[2][1] = self.T

        Q_prime = jacob_w @ Q_original @ jacob_w.T


        return Q_prime

    def calc_F_k_1(self, v_k, theta_k_1):
        F_k_1 = np.zeros((3, 3))
        F_k_1[0][0] = 1
        F_k_1[1][1] = 1
        F_k_1[2][2] = 1

        F_k_1[0][2] = -self.T * v_k * np.sin(theta_k_1)
        F_k_1[1][2] = self.T * v_k * np.cos(theta_k_1)

        return F_k_1


    ''' Calculate substitutions here (as in the PDF) to simplify things'''
    def substitute_a(self, x_l, x_k, theta_k):
        return x_l - x_k - self.d * np.cos(theta_k)

    def substitute_b(self, y_l, y_k, theta_k):
        return y_l - y_k - self.d * np.sin(theta_k)


    def get_partial_G(self, landmark, x_k_1, y_k_1, theta_k_1):
        G = np.zeros((2, 3))

        a = self.substitute_a(landmark[0], x_k_1, theta_k_1)
        b = self.substitute_b(landmark[1], y_k_1, theta_k_1)

        ab_2 = (a ** 2) + (b ** 2)
        sqrt_ab = np.sqrt(ab_2)

        G[0][0] = -a / sqrt_ab
        G[0][1] = -b / sqrt_ab
        G[0][2] = (self.d * (np.sin(theta_k_1) * a - np.cos(theta_k_1) * b)) / sqrt_ab

        G[1][0] = b / ab_2
        G[1][1] = -a / ab_2
        G[1][2] = ((-self.d * (np.cos(theta_k_1) * a + b * np.sin(theta_k_1))) / ab_2) - 1

        return G


    def calc_G(self, x_check, k):

        G_n = np.array([], dtype=np.float64).reshape(0, 3)

        range_ids = self.slice_valid_measurements(k)

        y_k = []

        x_k_1 = x_check[0]
        y_k_1 = x_check[1]
        theta_k_1 = x_check[2]

        for range_id in range_ids: # stack measurements into G_n, depending on how many measurements were in sight = within rage

            y_k.append(self.range_meas[k][range_id[0]])
            y_k.append(self.normalize_angle(self.bear_meas[k][range_id[0]]))

            landmark = [self.l_true[range_id[0]][0], self.l_true[range_id[0]][1]]

            G = self.get_partial_G(landmark,x_k_1,y_k_1,theta_k_1)

            G_n = np.vstack((G_n, G))

        y_k = np.array(y_k)

        return y_k, G_n

    def calc_R_prime(self, k):

        range_ids = self.slice_valid_measurements(k)
        R_prime = np.zeros((2 * len(range_ids), 2 * len(range_ids)))

        for i in range(2 * len(range_ids)): # again stack the values based on the size of used measurements at that time
            if i % 2 == 0:
                R_prime[i][i] = self.r_var
            else:
                R_prime[i][i] = self.b_var

        return R_prime

    def predictor(self, P_post_k_1, x_check, k):

        x_k_1 = x_check[0]
        y_k_1 = x_check[1]
        theta_k_1 = x_check[2]

        F_k_1 = self.calc_F_k_1(self.v[k], theta_k_1)
        Q_prime = self.calc_Q_prime(theta_k_1)
        x_check = self.calc_f(x_k_1, y_k_1, theta_k_1, self.v[k], self.omega[k])

        P_check = F_k_1 @ P_post_k_1 @ F_k_1.T + Q_prime

        return P_check, x_check

    def kalman_gain(self, P_check, x_check, k):

        y_k, G = self.calc_G(x_check, k)
        R_prime = self.calc_R_prime(k)
        Kalman_gain = P_check @ G.T @ np.linalg.inv(G @ P_check @ G.T + R_prime)

        return Kalman_gain, G, y_k

    def corrector(self, P_check, Kalman_gain, G, x_check, y_k, k):

        identity = np.identity(Kalman_gain.shape[0])
        P_post = (identity - Kalman_gain @ G) @ P_check
        x_post = x_check + Kalman_gain @ (y_k - self.calc_g(k, x_check)) # innovation part within parenthesis

        return P_post, x_post

    def slice_valid_measurements(self, k): # get only range measurements that are within the "allowed/visible" distance
        return np.argwhere((self.range_meas[k] != 0) & (self.range_meas[k] <= self.range_max))

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def run_EKF(self, x_0, P_0, task_no):
        estimated_states = []
        actual_states = []

        self.parse_data()
        x_post = x_0
        P_post = P_0

        covariances = []
        laser_readings = []

        print("Starting EKF algorithm!, Inputs : range_max : ", str(self.range_max), ", task_no: ", str(task_no))
        for k in range(1, self.timesteps.shape[0]):

            if task_no == 3: # CRLB - set the x_post-1 as the ground truth
                x_post = [self.x_true[k-1], self.y_true[k-1], self.th_true[k-1]]

            P_check, x_check = self.predictor(P_post, x_post, k)  # 1. prediction step

            Kalman_gain, G, y_k = self.kalman_gain(P_check, x_check, k) # 2. calc. kalman gain
            P_post, x_post = self.corrector(P_check, Kalman_gain, G, x_check, y_k, k) # 3. innovation and correction step

            ''' Collecting results here'''
            covariances.append(P_post)
            actual_states.append([self.x_true[k], self.y_true[k], self.th_true[k]])
            estimated_states.append(x_post)

            # for animation
            valid_landmarks = self.l_true[(self.range_meas[k] != 0) & (self.range_meas[k] <= self.range_max)]
            laser_readings.append(valid_landmarks)

        print("Finished EKF algorithm!")

        return estimated_states, actual_states, covariances, laser_readings
