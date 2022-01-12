import matplotlib.pyplot as plt
import uuid
import numpy as np
from matplotlib import cm
from tensorflow import concat, constant, int32
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d


def create_figure_and_axes(size_pixels):
  """Initializes a unique figure and axes for plotting."""
  fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

  # Sets output image to pixel resolution.
  dpi = 100
  size_inches = size_pixels / dpi
  fig.set_size_inches([size_inches, size_inches])
  fig.set_dpi(dpi)
  fig.set_facecolor('white')
  ax.set_facecolor('white')
#   ax.xaxis.label.set_color('black')
#   ax.tick_params(axis='x', colors='black')
#   ax.yaxis.label.set_color('black')
#   ax.tick_params(axis='y', colors='black')
  #fig.set_tight_layout(True)
  ax.grid(False)
  return fig, ax

def fig_canvas_image(fig):
  """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
  # Just enough margin in the figure to display xticks and yticks.
  fig.subplots_adjust(
      left=0, bottom=0, right=1, top=1, wspace=0.0, hspace=0.0)
  fig.canvas.draw()
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))



def get_colormap(num_agents):
  """Compute a color map array of shape [num_agents, 4]."""
  colors = cm.get_cmap('jet', num_agents)
  colors = colors(range(num_agents))
  np.random.shuffle(colors)
  return colors


def get_viewport(all_states, all_states_mask):
  """Gets the region containing the data.

  Args:
    all_states: states of agents as an array of shape [num_agents, num_steps,
      2].
    all_states_mask: binary mask of shape [num_agents, num_steps] for
      `all_states`.

  Returns:
    center_y: float. y coordinate for center of data.
    center_x: float. x coordinate for center of data.
    width: float. Width of data.
  """
  valid_states = all_states[all_states_mask]
  all_y = valid_states[..., 1]
  all_x = valid_states[..., 0]

  center_y = (np.max(all_y) + np.min(all_y)) / 2
  center_x = (np.max(all_x) + np.min(all_x)) / 2

  range_y = np.ptp(all_y)
  range_x = np.ptp(all_x)

  width = max(range_y, range_x)

  return center_y, center_x, width

def visualize_scenario(past_states, cur_state, future_state, gt_target, road,road_type, track_to_predict,
    size_pixels=1000,):
    past_states = past_states.numpy()

    # [num_agents, 1, 2] float32.
    current_states = cur_state.numpy()
    gt_target = gt_target.numpy()
    # [num_agents, num_future_steps, 2] float32.
    future_states = future_state.numpy()
    

    # [num_points, 3] float32.
    roadgraph_xyz = road.numpy()
    mask = constant(1, int32, (past_states.shape[0],91))>0
    color_map = get_colormap(128)

    # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
    all_states = np.concatenate([past_states, current_states, future_states], 1)

    # [num_agens, num_past_steps + 1 + num_future_steps] float32.
 

    center_y, center_x, width = get_viewport(all_states, mask)
    print(center_x)
    rg_pts = roadgraph_xyz[:, :2].T
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)
    # ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)  # prints roadgrpahs 
    #print(np.shape(all_states))
    count = 0
    maxy = 0
    miny = 0
    maxx = 0
    minx = 0
    yavg = np.average(all_states[:,:,1])
    xavg = np.average(all_states[:,:,0])
    print_at_most = 0
    for i in range(128):
        if not track_to_predict[i]:
          continue
        if print_at_most >= 1:
          continue
        print_at_most += 1
        # print input
        # ax.scatter(
        # all_states[i,:11,0],
        #     all_states[i,:11,1],
        #     marker='.',
        #     linewidths=1,
        #     color='blue',
        # )
        
        xold = all_states[i,11:,0]
        yold = all_states[i,11:,1]
        # xsort = np.sort(xold)
        # ysort = np.sort(yold)
        # if xold[-1] - xold[0]  < 0:
        #   xsort = xsort[::-1]

        # if yold[-1] - yold[0] < 0:
        #   ysort = ysort[::-1]
        # print(xsort)
        # print(ysort)
        # xnew = np.linspace(np.min(xsort), np.max(xsort), num=80, endpoint=True)
        # ynew = interp1d(xsort, ysort, kind='cubic')
        ysmoothed = gaussian_filter1d(yold, sigma=2)
        print(yold)
        print(ysmoothed)
        quit()
        ax.scatter(
        all_states[i,11:,0],
            all_states[i,11:,1],
            marker='.',
            linewidths=0,
            color='black',
        )
        ax.scatter(
        xold,
            ysmoothed,
            marker='.',
            linewidths=0,
            color='red',
        )
        ax.scatter(
        gt_target[i,:,0],
            gt_target[i,:,1],
            marker='.',
            linewidths=0,
            color='blue',
        )

        tmaxy = np.max(all_states[i,:,1])
        tminy = np.min(all_states[i,:,1])
        tmaxx = np.max(all_states[i,:,0])
        tminx = np.min(all_states[i,:,0])
        if tmaxx == -1 or tminy == -1 or tmaxy == -1 or tminx == -1:
            continue
        # if np.abs(tmaxx - xavg) > 100 or np.abs(tminx - xavg) > 100 or np.abs(tmaxy - yavg) > 100 or np.abs(tminy - yavg) > 100:
        #   continue
        # print(tmaxx)
        # print(tmaxy)
        # print(tminx)
        # print(tminy)
        maxy = tmaxy if tmaxy != -1 else maxy
        miny = tminy if tminy != -1 else miny
        maxx = tmaxx if tmaxx != -1 else maxx
        minx = tminx if tminx != -1 else minx
    print(maxx)
    print(maxy)
    print(minx)
    print(miny)

    max1= int(maxx - minx)
    max2= int(maxy - miny)
    truemax = max1 if max1 > max2 else max2
    size = max(50, truemax * 1.0 + 200)
    ax.axis([
        -size / 2 + int((maxx - minx)/2 + minx), size / 2 + int((maxx - minx)/2 + minx), -size / 2 + int((maxy - miny)/2 + miny),
        size / 2 + int((maxy - miny)/2 + miny)
    ])
    ax.set_aspect('equal')
    image = fig_canvas_image(fig)
    plt.close(fig)
    return image
    # plt.imsave('final.jpeg', image)
    # quit()
        # need to find proper widths
        # maxy = np.max(all_states[i,:,1])
        # miny = np.min(all_states[i,:,1])
        # maxx = np.max(all_states[i,:,0])
        # minx = np.min(all_states[i,:,0])
        # max1= int(maxx - minx)
        # max2= int(maxy - miny)
        # truemax = max1 if max1 > max2 else max2
        # size = max(50, truemax * 1.0 + 40)
        # ax.axis([
        #     -size / 2 + int((maxx - minx)/2 + minx), size / 2 + int((maxx - minx)/2 + minx), -size / 2 + int((maxy - miny)/2 + miny),
        #     size / 2 + int((maxy - miny)/2 + miny)
        # ])
        # ax.set_aspect('equal')
        # image = fig_canvas_image(fig)
        # plt.imsave('final.jpeg', image)
        # quit()


def visualize_one_step(states,
                       mask,
                       roadgraph,
                       title,
                       center_y,
                       center_x,
                       width,
                       color_map, road_type,
                       size_pixels=1000, marker_size=6, radius=15):
  """Generate visualization for a single step."""

  # Create figure and axes.
  fig, ax = create_figure_and_axes(size_pixels=size_pixels)
    # [0, 19]. LaneCenter-Freeway = 1, LaneCenter-SurfaceStreet = 2, 
    # LaneCenter-BikeLane = 3, RoadLine-BrokenSingleWhite = 6, 
    # RoadLine-SolidSingleWhite = 7, RoadLine-SolidDoubleWhite = 8, 
    # RoadLine-BrokenSingleYellow = 9, RoadLine-BrokenDoubleYellow = 10, 
    # Roadline-SolidSingleYellow = 11, Roadline-SolidDoubleYellow=12, 
    # RoadLine-PassingDoubleYellow = 13, RoadEdgeBoundary = 15, 
    # RoadEdgeMedian = 16, StopSign = 17, Crosswalk = 18, SpeedBump = 19, 
    # other values are unknown types and should not be present.
  # Plot roadgraph.
  rg_pts = roadgraph[:, :2].T
  road_type = np.reshape(road_type, (np.shape(road_type)[0]))

  rt_par = {'-1': ['.', 0, 0, 'black'],'0': ['.', 0, 0, 'black'],'1': ['s', 1, 2, 'black'],'2': ['s', 1, 2, 'black'],
  '3': ['s', 1, 2, 'orange'],'4': ['.', 0, 0, 'black'],'5': ['.', 0, 0, 'black'],'6': ['.', 1, 2, 'black'],
  '7': ['.', 1, 2, 'black'],'8': ['x', 1, 2, 'black'],'9': ['.', 1, 2, 'yellow'],'10': ['x', 1, 2, 'yellow'],
  '11': ['s', 1, 2, 'yellow'],'12': ['x', 1, 2, 'yellow'],'13': ['x', 1, 2, 'yellow'],'14': ['.', 0, 0, 'black'],
  '15': ['s', 1, 2, 'red'],'16': ['.', 1, 2, 'red'],'17': ['h', 1, 15, 'red'],'18': ['*', 1, 2, 'green'], '19': ['o', 1, 2, 'red']}
  for i in range(len(road_type)):
      ax.plot(rg_pts[0, i], rg_pts[1, i], rt_par[str(road_type[i])][0], alpha=rt_par[str(road_type[i])][1], ms=rt_par[str(road_type[i])][2], color=rt_par[str(road_type[i])][3])
  #ax.plot(rg_pts[0, :], rg_pts[1, :], 'k|', alpha=1, ms=marker_size)
  masked_x = states[:, 0][mask]
  masked_y = states[:, 1][mask]
  colors = color_map[mask]

  # Plot agent current position.
  ax.scatter(
      masked_x,
      masked_y,
      marker='s',
      linewidths=4,
      color='blue',
  )

  # Title.
  #ax.set_title(title)

  # Set axes.  Should be at least 10m on a side and cover 160% of agents.
#   size = max(radius, 10 * 1.0)
#   ax.axis([
#       -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
#       size / 2 + center_y
#   ])
  ax.set_aspect('equal')

  image = fig_canvas_image(fig)
  plt.close(fig)
  return image


def visualize_all_agents_smooth(
    past_states, past_states_mask, cur_state, current_states_mask, future_state, future_states_mask, road,road_type,
    size_pixels=1000,
):
  """Visualizes all agent predicted trajectories in a serie of images.

  Args:
    decoded_example: Dictionary containing agent info about all modeled agents.
    size_pixels: The size in pixels of the output image.

  Returns:
    T of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas.
  """
  # [num_agents, num_past_steps, 2] float32.
  # only
  past_states = past_states.numpy()
  past_states_mask = past_states_mask.numpy()

  # [num_agents, 1, 2] float32.
  current_states = cur_state.numpy()
  current_states_mask = current_states_mask.numpy()

  # [num_agents, num_future_steps, 2] float32.
  future_states = future_state.numpy()
  future_states_mask = future_states_mask.numpy()

  # [num_points, 3] float32.
  roadgraph_xyz = road.numpy()

  num_agents, num_past_steps, _ = past_states.shape
  num_future_steps = future_states.shape[1]

  color_map = get_colormap(num_agents)

  # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
  all_states = np.concatenate([past_states, current_states, future_states], 1)

  # [num_agens, num_past_steps + 1 + num_future_steps] float32.
  all_states_mask = np.concatenate(
      [past_states_mask, current_states_mask, future_states_mask], 1)

  center_y, center_x, width = get_viewport(all_states, all_states_mask)

  images = []

  # Generate images from past time steps.
  for i, (s, m) in enumerate(
      zip(
          np.split(past_states, num_past_steps, 1),
          np.split(past_states_mask, num_past_steps, 1))):
    im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                            'past: %d' % (num_past_steps - i), center_y,
                            center_x, width, color_map,road_type.numpy(), size_pixels)
    plt.imsave('noshift.jpeg', im)
    quit()
    images.append(np.reshape(im,(1,size_pixels,size_pixels, 3)))

  # Generate one image for the current time step.
  s = current_states
  m = current_states_mask

  im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz, 'current', center_y,
                          center_x, width, color_map,road_type.numpy(), size_pixels)
  images.append(np.reshape(im,(1,size_pixels,size_pixels, 3)))

  # Generate images from future time steps.
#   for i, (s, m) in enumerate(
#       zip(
#           np.split(future_states, num_future_steps, 1),
#           np.split(future_states_mask, num_future_steps, 1))):
#     im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
#                             'future: %d' % (i + 1), center_y, center_x, width,
#                             color_map,road_type.numpy(), size_pixels)
#     images.append(np.reshape(im,(1,size_pixels,size_pixels, 3)))
#   #imgplot = plt.imshow(images[0])
#   #quit()
#   plt.imsave('noshift.jpeg', images[10][0])
  return concat(images,0)

