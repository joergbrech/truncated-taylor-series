import streamlit as st

import matplotlib.pyplot as plt
import numpy as np

from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import x


# Define some helpful functions

def evaluate_shifted_polynomial(coeffs, x0, degree, xs):
    """
    Given an array coeffs of symbolic expressions in x, where len(coeffs) >= degree, evaluate the shifted
    polynomial

      f(x) = sum( coeffs[k](x)*(x-x0)**k)

    where the sum goes over k from 0 to degree. This is used to evaluate the Taylor polynomial, where the coefficients
    coeffs[k] = df^k/dx^k(x0)/k! are precomputed.
    """
    ys = np.ones(np.shape(xs)) * coeffs[0].subs(x, x0)
    xs_shift = xs - np.ones(np.shape(xs)) * x0
    for k in range(1, degree + 1):
        ys = ys + coeffs[k].subs(x, x0) * xs_shift ** k
    return ys


# we need helper functions to interactively update horizontal and vertical lines in a plot
# https://stackoverflow.com/questions/29331401/updating-pyplot-vlines-in-interactive-plot

def update_vlines(*, h, x, ymin=None, ymax=None):
    """
    If h is a handle to a vline object in a matplotlib plot, this function can be used to update x, ymin, ymax
    """
    seg_old = h.get_segments()
    if ymin is None:
        ymin = seg_old[0][0, 1]
    if ymax is None:
        ymax = seg_old[0][1, 1]

    seg_new = [np.array([[x, ymin],
                         [x, ymax]]), ]

    h.set_segments(seg_new)


def update_hlines(*, h, y, xmin=None, xmax=None):
    """
    If h is a handle to a hline object in a matplotlib plot, this function can be used to update y, xmin, xmax
    """
    seg_old = h.get_segments()
    if xmin is None:
        xmin = seg_old[0][0, 0]
    if xmax is None:
        xmax = seg_old[0][1, 0]

    seg_new = [np.array([[xmin, y],
                         [xmax, y]]), ]

    h.set_segments(seg_new)


#############################################
# Define the function that updates the plot #
#############################################

#To Do: Why does caching update_plot hang?
#@st.cache(suppress_st_warning=True)
def update_plot(visible,
                x0,
                degree,
                xmin,
                xmax,
                ymin,
                ymax,
                xresolution):
    """
    Updates the plot. visible, function, x0, degree, xmin, xmax, ymin, ymax are all controlled by widgets. xresolution, coeffients
    and handles are fixed.

    coefficients is the array containing the precomputed Taylor polynomial coefficients (in symbolic representation)
    handles is a dictionary of the plots that are to be updated.

    Note: The input "function" is actually not needed, because coefficients[0] already contains the parsed function expression.
    """

    coefficients = st.session_state.coefficients
    handles = st.session_state.handles

    # parse symbolic representation of function
    f = coefficients[0]

    # evaluate function at x0
    fx0 = f.subs(x, x0)

    # update the x values for plotting
    xs = np.linspace(xmin, xmax, int(xresolution))

    ax = st.session_state.fig.axes[0]

    # if the dictionary of plot handles is empty, the plot does not exist yet. We create it. Otherwise the plot exists,
    # and we can update the plot handles in fs, without having to redraw everything (better performance).
    if not handles:
        #######################
        # Initialize the plot #
        #######################

        # plot f and append the plot handle
        handles["func"] = ax.plot(xs, lambdify(x, f, 'numpy')(xs), label="function f")[0]

        # plot the Taylor polynomial
        handles["taylor"] = ax.plot(xs, evaluate_shifted_polynomial(coefficients, x0, degree, xs),
                                   color='g',
                                   label='Taylor polynomial at x0'.format(degree))[0]

        handles["taylor"].set_visible(visible)

        ###############################
        # Beautify the plot some more #
        ###############################

        plt.title('Taylor approximation of a function f at x0')
        plt.xlabel('x', horizontalalignment='right', x=1)
        plt.ylabel('y', horizontalalignment='right', x=0, y=1)

        # set the z order of the axes spines
        for k, spine in ax.spines.items():
            spine.set_zorder(0)

        # set the axes locations and style
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['right'].set_color('none')

        # draw lines for (x0, f(x0))
        handles["vline"] = plt.vlines(x=x0, ymin=float(min(0, fx0)), ymax=float(max(0, fx0)), colors='black', ls=':', lw=2)
        handles["hline"] = plt.hlines(y=float(fx0), xmin=xmin, xmax=x0, colors='black', ls=':', lw=2)

        # show legend
        legend_handles = [handles["func"], ]
        if visible:
            legend_handles.append(handles["taylor"])
        ax.legend(handles=legend_handles,
                  loc='lower center',
                  bbox_to_anchor=(0.5, -0.15),
                  ncol=2)

    else:
        ###################
        # Update the plot #
        ###################

        # Update the function plot
        handles["func"].set_xdata(xs)
        handles["func"].set_ydata(lambdify(x, f, 'numpy')(xs))

        # update the taylor polynomial plot
        handles["taylor"].set_xdata(xs)
        handles["taylor"].set_ydata(evaluate_shifted_polynomial(coefficients, x0, degree, xs))

        # update the visibility of the Taylor expansion
        handles["taylor"].set_visible(visible)

        update_vlines(h=handles["vline"], x=x0, ymin=float(min(0, fx0)), ymax=float(max(0, fx0)))
        update_hlines(h=handles["hline"], y=float(fx0), xmin=xmin, xmax=x0)

    # set x and y ticks, labels and limits respectively
    xticks = []
    xticklabels = []
    if xmin <= 0 <= xmax:
        xticks.append(0)
        xticklabels.append("0")
    if xmin <= x0 <= xmax:
        xticks.append(x0)
        xticklabels.append("x0")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    yticks = []
    yticklabels = []
    if ymin <= 0 <= ymax:
        yticks.append(0)
        yticklabels.append("0")
    if ymin <= fx0 <= ymax:
        yticks.append(fx0)
        yticklabels.append("f(x0)")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # set the x and y limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # make all changes visible
    st.session_state.fig.canvas.draw()

@st.cache(suppress_st_warning=False)
def update_coefficients(function_string):
    """
    stores symbolic representations of the coefficients k! * df/dk(x0) of the Taylor polynomial in an array from the
    outer scope.

    This is used to cache the derivative calculation for better performance, when evaluating the Taylor polynomial
    """

    coeffs = [0 for k in range(0, st.session_state.degree_max + 1)]
    coeffs[0] = parse_expr(function_string)
    fac = 1
    for k in range(1, st.session_state.degree_max + 1):
        coeffs[k] = diff(coeffs[k - 1] * fac, x) / (fac * k)
        fac = fac * k

    st.session_state.coefficients = coeffs
    return coeffs


if __name__ == '__main__':

    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.title('Truncated Taylor series')

    if 'fig' not in st.session_state:
        # initialize the figure and initialize an empty dict of plot handles
        plt.xkcd()  # <-- beautiful xkcd style
        st.session_state.fig = plt.figure(figsize=(8, 3))
        st.session_state.fig.add_axes([0., 0., 1., 1.])

    if 'handles' not in st.session_state:
        st.session_state.handles = {}

    if 'degree_max' not in st.session_state:
        st.session_state.degree_max = 10

    st.sidebar.title("Advanced settings")

    func_str = st.sidebar.text_input(label="function",
                                     value='25 + exp(x)*sin(x**2) - 10*x')

    visible = st.sidebar.checkbox(label='Show Taylor Polynomial', value=True)

    st.sidebar.markdown("Visualization Options")
    xcol1, xcol2 = st.sidebar.columns(2)
    with xcol1:
        xmin = st.number_input(label='xmin', value=1.)
        ymin = st.number_input(label='ymin', value=-50.)
    with xcol2:
        xmax = st.number_input(label='xmax', value=4.)
        ymax = st.number_input(label='ymax', value=50.)

    res = st.sidebar.number_input(label='resolution', value=50)

    update_coefficients(func_str)

    col1, col2 = st.columns(2)
    with col1:
        x0 = st.slider(
            'x0',
            min_value=xmin,
            max_value=xmax,
            value=2.3
        )

    with col2:
        degree = st.slider(
            'degree',
            min_value=0,
            max_value=st.session_state.degree_max,
            value=int(0)
        )

    update_plot(visible, x0, degree, xmin, xmax, ymin, ymax, res)

    st.pyplot(st.session_state.fig)
