import sys

import matplotlib
import numpy as np

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
elif sys.platform == 'linux':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import skimage
from habitat_sim.utils import viz_utils as vut
import cv2

def visualize(fig, ax, img, grid, pos, gt_pos, dump_dir, rank, ep_no, t,
              visualize, print_images, vis_style):
    for i in range(2):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    ax[0].imshow(img)
    ax[0].set_title("Observation", family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    if vis_style == 1:
        title = "Predicted Map and Pose"
    else:
        title = "Ground-Truth Map and Pose"

    ax[1].imshow(grid)
    ax[1].set_title(title, family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    # Draw GT agent pose
    agent_size = 8
    x, y, o = gt_pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Grey'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * (agent_size * 0.5),
                head_width=agent_size*0.5, head_length=agent_size * 0.5,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.9)

    # Draw predicted agent pose
    x, y, o = pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0
    dx = 0
    dy = 0
    fc = 'Red'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * agent_size * 0.5,
                head_width=agent_size*0.5, head_length=agent_size * 0.5,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.6)

    for _ in range(5):
        plt.tight_layout()

    if visualize:
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    if print_images:
        fn = '{}/episodes/{}/{}/{}-{}-Vis-{}.png'.format(
            dump_dir, (rank + 1), ep_no, rank, ep_no, t)
        plt.savefig(fn)
        #plt.imsave(fn,grid)
       


def insert_circle(mat, x, y, value):
    mat[x - 2: x + 3, y - 2:y + 3] = value
    mat[x - 3:x + 4, y - 1:y + 2] = value
    mat[x - 1:x + 2, y - 3:y + 4] = value
    return mat


def fill_color(colored, mat, color):
    for i in range(3):
        colored[:, :, 2 - i] *= (1 - mat)
        colored[:, :, 2 - i] += (1 - color[i]) * mat
    return colored


def get_colored_map(mat, collision_map, visited, visited_gt, goal, stg,
                    explored, gt_map, gt_map_explored):
    # m, n = mat.shape
    # colored = np.ones((m, n, 3))
    # #White for explored map
    # current_palette = [(1,1,1)]
    # colored = fill_color(colored, explored, current_palette[0])  
    # #Balck for th obstacle map
    # current_palette = [(0,0,0)]
    # colored = fill_color(colored, mat, current_palette[0])      
    # colored = 1 - colored
    # colored *= 255
    # colored = colored.astype(np.uint8)
    
    
    m, n = mat.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    current_palette = [(0.9, 0.9, 0.9)]
    colored = fill_color(colored, gt_map, current_palette[0])

    current_palette = [(235. / 255., 243. / 255., 1.)]
    colored = fill_color(colored, explored, current_palette[0])

    green_palette = sns.light_palette("green")
    colored = fill_color(colored, mat, pal[2])

    current_palette = [(0.6, 0.6, 0.6)]
    colored = fill_color(colored, gt_map_explored, current_palette[0])

    colored = fill_color(colored, mat * gt_map_explored, pal[3])

    red_palette = sns.light_palette("red")

    colored = fill_color(colored, visited_gt, current_palette[0])
    colored = fill_color(colored, visited, pal[4])
    colored = fill_color(colored, visited * visited_gt, pal[5])

    colored = fill_color(colored, collision_map, pal[2])

    current_palette = sns.color_palette()

    selem = skimage.morphology.disk(4)
    goal_mat = np.zeros((m, n))
    goal_mat[goal[0], goal[1]] = 1
    goal_mat = 1 - skimage.morphology.binary_dilation(
        goal_mat, selem) != True

    colored = fill_color(colored, goal_mat, current_palette[0])

    selem = skimage.morphology.disk(3)
    goal_mat = np.zeros((m, n))
    goal_mat[int(stg[0]), int(stg[1])] = 1
    goal_mat = 1 - skimage.morphology.binary_dilation(
        goal_mat, selem) != True

    colored = fill_color(colored, goal_mat, current_palette[2])

    current_palette = sns.color_palette("Paired")

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)

    return colored



def make_video_cv2(
    output_path,observations, cross_hair=None, prefix="", open_vid=True, fps=10
):
    #sensor_keys = list(observations[0])
    videodims = observations[0].shape
    videodims = (videodims[3], videodims[2])  # flip to w,h order
    print(videodims)
    
    video_file = output_path + prefix + ".mp4"
    print("Encoding the video: %s " % video_file)
    writer = vut.get_fast_video_writer(video_file, fps=fps)
    for ob in observations:
        # If in RGB/RGBA format, remove the alpha channel
        ob = np.array(ob, dtype=np.uint8)
        rgb_im_1st_person = cv2.cvtColor(ob[0].transpose(1,2,0), cv2.COLOR_RGBA2RGB)
        if cross_hair is not None:
            rgb_im_1st_person[
                cross_hair[0] - 2 : cross_hair[0] + 2,
                cross_hair[1] - 2 : cross_hair[1] + 2,
            ] = [255, 0, 0]

        if rgb_im_1st_person.shape[:2] != videodims:
            rgb_im_1st_person = cv2.resize(
                rgb_im_1st_person, videodims, interpolation=cv2.INTER_AREA
            )
        # write the 1st person observation to video
        writer.append_data(rgb_im_1st_person)
    writer.close()

    if open_vid:
        print("Displaying video")
        vut.display_video(video_file)