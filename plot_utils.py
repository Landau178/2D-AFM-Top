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


def int2cart(number_components):
    """
    This function takes a tuple representing a 3-rank
    tensor component. It returns the associated strings
    in cartesian coordinates. 
    Example:  (0, 1, 0) --> ("x", "y", "z")
    Parameters:
    -----------
        number_components: (tuple of 3 ints)
                Integers representing cartesian components.
    Returns:
    --------
        x,y,z: (str,zstr,str)
                String representing the component
            in cartesian coordinates.
    """
    comp_dict = {0: "x", 1: "y", 2: "z"}
    cart_comp = []
    for n in number_components:
        cart_comp.append(comp_dict[n])
    return cart_comp


def sigma_s_label(number_components):
    """
    This function creates a label for spin(charge) conductivities.
    Parameters:
    -----------
        number_components: (tuple of 3(2) ints)
            Integers representing cartesian components.
    Returns:
    --------
        label: (str,str,str)
            Label representing the component of the spin
            conductivity tensor.
    """
    cart_comp = int2cart(number_components)
    n = len(cart_comp)
    if n == 3:
        str_x1 = "{" + cart_comp[0] + "}"
        str_x2x3 = "{" + cart_comp[1] + cart_comp[2] + "}"
        label = fr"$\sigma^{str_x1}_{str_x2x3}$"
    elif n == 2:
        str_x1x2 = "{" + cart_comp[0] + cart_comp[1] + "}"
        label = fr"$\sigma_{str_x1x2}$"
    else:
        raise Exception("number_components should have lenght 3 or 2")
    return label
