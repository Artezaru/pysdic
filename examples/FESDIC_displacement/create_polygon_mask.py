
import numpy

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.widgets as widgets

def create_polygon_mask(image: numpy.ndarray) -> numpy.ndarray:
    """
    Click to create a polygon and returns the corresponding mask.
    Left click: add point
    Right click: remove last point
    Use "Erase" button to clear all points
    Use "Done" button or close the window to finish
    """
    points = []
    done = [False]  # mutable flag to track if user finished

    # Figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # space for buttons
    ax.imshow(image, cmap='gray')
    ax.set_title("Instructions:\nLeft click: add point\nRight click: remove last point\nUse buttons below to finish or erase points", fontsize=10)

    # Line and scatter for real-time display
    line, = ax.plot([], [], 'r-')
    scatter, = ax.plot([], [], 'ro')

    # Button axes
    ax_done = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_erase = plt.axes([0.81, 0.05, 0.1, 0.075])
    btn_done = widgets.Button(ax_done, 'Done')
    btn_erase = widgets.Button(ax_erase, 'Erase')

    # Button callbacks
    def done_callback(event):
        done[0] = True
        plt.close(fig)

    def erase_callback(event):
        points.clear()
        line.set_data([], [])
        scatter.set_data([], [])
        fig.canvas.draw()

    btn_done.on_clicked(done_callback)
    btn_erase.on_clicked(erase_callback)

    # Mouse click callback
    def onclick(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # left click -> add
            points.append((event.xdata, event.ydata))
        elif event.button == 3:  # right click -> remove last
            if points:
                points.pop()

        # Update display
        if points:
            xs, ys = zip(*points)
            line.set_data(xs + (xs[0],) if len(points) > 2 else xs,
                          ys + (ys[0],) if len(points) > 2 else ys)
            scatter.set_data(xs, ys)
        else:
            line.set_data([], [])
            scatter.set_data([], [])
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Show figure and wait until user clicks "Done" or closes the window
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    # Create mask
    mask = numpy.zeros(image.shape[:2], dtype=numpy.uint8)
    if points:
        path = mpath.Path(points)
        xx, yy = numpy.meshgrid(numpy.arange(image.shape[1]), numpy.arange(image.shape[0]))
        coords = numpy.vstack((xx.ravel(), yy.ravel())).T
        mask_flat = path.contains_points(coords)
        mask = mask_flat.reshape(image.shape[:2]).astype(numpy.uint8)

    return mask


