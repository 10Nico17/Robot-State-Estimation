import numpy as np
import ekf
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, FuncAnimation

fig, axs = plt.subplots(2, figsize=(18, 10))  # for animation
fig.suptitle('Estimated and ground truth position of the robot over time', size=14)



def get_updated_ellipsis_vals(cov_matrix):
    p_cov = np.array(
        [
            [cov_matrix[0][0], cov_matrix[0][1]],
            [cov_matrix[1][0], cov_matrix[1][1]]
        ])
    eigenvalues, eigenvectors = np.linalg.eig(p_cov)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width = 3 * np.sqrt(eigenvalues[0])
    height = 3 * np.sqrt(eigenvalues[1])
    return angle, width, height


def update(k, estimates, groundtruth, laser_readings, covariances, estimate_plots, ground_truth_plots, plot_lasers,
           ellipses, axs1, txt):
    for i in range(2):
        angle_, width_, height_ = get_updated_ellipsis_vals(covariances[k])
        ellipses[i].set_angle(angle_)
        ellipses[i].set_width(width_)
        ellipses[i].set_height(height_)
        ellipses[i].set_center([estimates[k][0], estimates[k][1]])

        if laser_readings != [] and i != 1:
            laser_lines_x = []
            laser_lines_y = []

            for landmark in laser_readings[k]:
                laser_lines_x.append([groundtruth[k][0], landmark[0]])
                laser_lines_y.append([groundtruth[k][1], landmark[1]])

            plot_lasers.set_data(laser_lines_x, laser_lines_y)

        estimate_plots[i].set_offsets(np.c_[estimates[k][0], estimates[k][1]])
        ground_truth_plots[i].set_offsets(np.c_[groundtruth[k][0], groundtruth[k][1]])

    txt.set_text(f"Timestep: {k}")
    axs1.set_xlim(estimates[k][0] - 0.5, estimates[k][0] + 0.5)
    axs1.set_ylim(estimates[k][1] - 0.5, estimates[k][1] + 0.5)


def plot_animation(covariances, estimated_states, actual_states, laser_readings, range_max):
    ekf_ = ekf.EKF(range_max)

    plt.ioff()
    ekf_.parse_data()

    print("Starting the animation function!")

    landmarks_x = ekf_.l_true[:, 0]
    landmarks_y = ekf_.l_true[:, 1]

    p_cov = np.array(
        [
            [covariances[0][0][0], covariances[0][0][1]],
            [covariances[0][1][0], covariances[0][1][1]]
        ])

    eigenvalues, eigenvectors = np.linalg.eig(p_cov)
    angle_ = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width_ = 3 * np.sqrt(eigenvalues[0])
    height_ = 3 * np.sqrt(eigenvalues[1])

    ellipses = []
    estimate_plots = []
    ground_truth_plots = []

    for index, ax in enumerate(axs):
        if index == 0:
            plot_lasers, = ax.plot([], [], color="green", alpha=0.8, linewidth=0.8, label="Laser measurements")

        ellips = ax.add_patch(
            Ellipse(xy=(estimated_states[0][0], estimated_states[0][1]), width=width_, height=height_, angle=angle_,
                    facecolor="None", edgecolor="r", lw=2, alpha=0.8))
        ellipses.append(ellips)

        scatter1 = ax.scatter([], [], color="red", label="Estimated robot position")
        estimate_plots.append(scatter1)

        scatter2 = ax.scatter([], [], color="blue", label="Ground truth robot position")
        ground_truth_plots.append(scatter2)

        ax.scatter(landmarks_x, landmarks_y, color="black")
        ax.grid(alpha=0.5)

        if index == 0:
            ax.set_xlim([-2, 10])
            ax.set_ylim([-4, 4])

        ax.set_xlabel("x position", size=12)
        ax.set_ylabel("y position", size=12)
        ax.legend(loc="lower right")

    txt = axs[0].annotate("", xy=(8.2, 3.5), xytext=(8.2, 3.5), fontsize=16)

    plt.tight_layout()

    ani = FuncAnimation(fig, update, frames=len(covariances),
                        fargs=[estimated_states, actual_states, laser_readings, covariances,
                               estimate_plots, ground_truth_plots, plot_lasers, ellipses,
                               axs[1], txt], repeat=False, interval=1000 / 30) # 30 FPS Animation
    ani.save("animation.mp4")


    print("Successfully generated the animation and saved it in animation.mp4!")
    return
