import numpy as np
import ekf
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, FuncAnimation


def plot_error_results(difs, covariances, range_max, task_no):
    labels = ["x", "y", "theta"]

    for plot_var in range(3):

        print("Plotting results...")
        plt.figure(figsize=(16, 8))

        covs = [x[plot_var][plot_var] for x in covariances]
        sigma3 = 3 * np.sqrt(np.array(covs))

        title = "Error between the true " + labels[plot_var] + " and estimated " + \
                labels[plot_var] + " value with " + str(range_max) + "m maximum range"

        title += ", task " + str(task_no)

        plt.title(title, size=18)

        plt.plot(difs[plot_var], label=labels[plot_var] + " values residual error")
        plt.plot(sigma3, color="orange", label="3-sigma bound, upper", linestyle="dotted")
        plt.plot(-sigma3, color="orange", label="3-sigma bound, lower", linestyle="dotted")

        plt.xlabel("Timesteps", size=14)
        plt.ylabel("Error value", size=14)
        plt.grid()
        plt.legend()

        filename = "plots/"+str(task_no) + "_" + str(range_max) + "m_" + labels[plot_var]
        plt.savefig(filename)
        plt.close()

        print("Finished plotting results...")


def plot_trajectories(estimated_states, actual_states, range_max, landmarks, task_no):
    labels = ["x", "y", "theta"]

    print("Plotting trajectory...")

    plt.figure(figsize=(20, 8))

    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]


    title = "Trajectory of the true robot position and estimated robot position with " +\
            str(range_max) + "m maximum range"

    title += ", task " + str(task_no)

    plt.title(title, size=18)

    estimated_x = [x[0] for x in estimated_states]
    estimated_y = [x[1] for x in estimated_states]

    actual_x = [x[0] for x in actual_states]
    actual_y = [x[1] for x in actual_states]

    plt.plot(estimated_x, estimated_y, label="Estimated robot position", color="red", linestyle="dashed", alpha=1, lw=1)
    plt.plot(actual_x,actual_y, label="Ground truth robot position", color="blue", linestyle="dotted", alpha=1, lw=1)

    plt.xlabel("x position", size=14)
    plt.ylabel("y position", size=14)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    filename = "plots/"+str(task_no) + "_" + str(range_max) + "m_trajectory"
    plt.savefig(filename)
    # plt.show()
    # plt.close()

    print("Finished plotting trajectory...")



    # plt.show()



def init_EKF(x_0, P_post_0, range_max, task_no):
    ekf_ = ekf.EKF(range_max)
    estimated_states, actual_states, covariances, laser_readings = ekf_.run_EKF(x_0, P_post_0, task_no=task_no)

    dif_x = []
    dif_y = []
    dif_theta = []

    for i in range(len(estimated_states)):
        dif_x.append(actual_states[i][0] - estimated_states[i][0])
        dif_y.append(actual_states[i][1] - estimated_states[i][1])
        dif_th = (actual_states[i][2] - estimated_states[i][2] + np.pi) % (2 * np.pi) - np.pi
        dif_theta.append(dif_th)

    difs = [dif_x, dif_y, dif_theta]


    if task_no==1 and range_max==1:
        from plot_animation import plot_animation
        plot_animation(covariances,estimated_states, actual_states, laser_readings, range_max)

    plot_error_results(difs, covariances, range_max, task_no)
    plot_trajectories(estimated_states, actual_states, range_max, ekf_.l_true, task_no)


# Run main to run the script.
if __name__ == '__main__':

    x_0 = np.array([3.019756132877692, 0.0708990475403322, -2.910157363570845]) # this is x_0 ground truth
    P_post_0 = np.identity(3)
    P_post_0[2][2] = 0.1

    ranges = [5, 3, 1]


    # plot_animation(None, None)
    for task_no in range(1,4):
        if task_no == 2:
            x_0 = np.array([1,1,0.1])
        for range_ in ranges:
            init_EKF(x_0, P_post_0,range_, task_no)


