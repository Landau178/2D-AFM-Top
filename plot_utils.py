def set_ticks_on_ax(ax, ticks, fs=15, axis="x"):
    """
    Set the ticks on y-axis.
    Parameters:
    -----------
        ax: (matplotlib.axes.Axes)
            Axes of plot.
        ticks: (list of floats)
            List with the positions
            of the desired ticks.
        fs: (float, default is 15)
            Fontsize of ticks.
    Returns:
    --------
        None

    """
    set_ticks = {"x": ax.set_xticks,
                 "y": ax.set_yticks}[axis]
    set_ticklabels = {"x": ax.set_xticklabels,
                      "y": ax.set_yticklabels}[axis]

    labels = list(map(str, ticks))
    set_ticks(ticks)
    ax_kwargs = {"fontsize": fs}
    set_ticklabels(labels, **ax_kwargs)
