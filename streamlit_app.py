import streamlit as st

from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import x

import numpy as np
import pandas as pd

import altair as alt

import matplotlib.pyplot as plt
import matplotlib.font_manager


# make sure the humor sans font is found. This only needs to be done once
# on a system, but it is done here at start up for usage on share.streamlit.io.
matplotlib.font_manager.findfont('Humor Sans', rebuild_if_missing=True)

# Define some helpful functions

def evaluate_shifted_polynomial(coeffs, x0, degree, xs):
    """
    Given an array coeffs of symbolic expressions in sympy.abc.x, where len(coeffs) >= degree, evaluate the shifted
    polynomial

      f(x) = sum( coeffs[k](x0)*(x-x0)**k)

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

@st.cache(suppress_st_warning=True)
def update_data(coefficients, degree, x0, xmin, xmax, xresolution):
    """
    Calculates the np-arrays needed to plot the function and Taylor polyonmial

    :param coefficients: symbolic representation of the coefficients of the Taylor polynomial
    :param degree: degree of the Taylor polynomial
    :param x0: The evaluation point of the Taylor polynomial
    :param xmin: minimum value of the x range
    :param xmax: maximum value of the x range
    :param xresolution: resolution of the plot

    :return: np-arrays with x-coordinates, y-coordinates of function f, y-coordinates of Taylor Polynomial
    """

    # parse symbolic representation of function
    f = coefficients[0]

    # evaluate function at x0
    fx0 = f.subs(x, x0)

    # update the x values for plotting
    xs = np.linspace(xmin, xmax, int(xresolution))
    fys = lambdify(x, f, 'numpy')(xs)
    tys = evaluate_shifted_polynomial(coefficients, x0, degree, xs)
    return xs, fys, tys, fx0


# To Do: Why does caching update_plot hang?
# @st.cache(suppress_st_warning=True)
def update_plot(x0, fx0, xs, ys, ps, visible, xmin, xmax, ymin, ymax):
    """
    Creates a Matplotlib plot if the dictionary st.session_state.handles is empty, otherwise
    updates a Matplotlib plot by modifying the plot handles stored in st.session_state.handles.
    The figure is stored in st.session_state.fig.

    :param x0: Evaluation point of the function/Taylor polynomial
    :param fx0: Function evaluated at x0
    :param xs: numpy-array of x-coordinates
    :param ys: numpy-array of f(x)-coordinates
    :param ps: numpy-array of P(x)-coordinates, where P is the Taylor polynomial
    :param visible: A flag wether the Taylor polynomial is visible or not
    :param xmin: minimum x-range value
    :param xmax: maximum x-range value
    :param ymin: minimum y-range value
    :param ymax: maximum y-range value
    :return: none.
    """

    handles = st.session_state.handles

    ax = st.session_state.mpl_fig.axes[0]

    # if the dictionary of plot handles is empty, the plot does not exist yet. We create it. Otherwise the plot exists,
    # and we can update the plot handles in fs, without having to redraw everything (better performance).
    if not handles:
        #######################
        # Initialize the plot #
        #######################

        # plot f and append the plot handle
        handles["func"] = ax.plot(xs, ys, label="function f")[0]

        # plot the Taylor polynomial
        handles["taylor"] = ax.plot(xs, ps,
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
        handles["func"].set_ydata(ys)

        # update the taylor polynomial plot
        handles["taylor"].set_xdata(xs)
        handles["taylor"].set_ydata(ps)

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
    st.session_state.mpl_fig.canvas.draw()

@st.cache(suppress_st_warning=True)
def update_coefficients(function_string, degree_max):
    """
    stores symbolic representations of the coefficients k! * df/dk(x0) of the Taylor polynomial in an array from the
    outer scope.

    This is used to cache the derivative calculation for better performance, when evaluating the Taylor polynomial
    """

    coeffs = [0 for k in range(0, degree_max + 1)]
    coeffs[0] = parse_expr(function_string)
    fac = 1
    for k in range(1, degree_max + 1):
        coeffs[k] = diff(coeffs[k - 1] * fac, x) / (fac * k)
        fac = fac * k

    return coeffs


if __name__ == '__main__':

    # maximum allowed degree of a Taylor Polynomial
    degree_max = 10

    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    # create sidebar widgets

    st.sidebar.title("Advanced settings")

    func_str = st.sidebar.text_input(label="function",
                                     value='25 + exp(x)*sin(x**2) - 10*x')

    st.sidebar.markdown("Visualization Options")

    # Good for in-classroom use
    qr = st.sidebar.checkbox(label="Display QR Code", value=False)

    visible = st.sidebar.checkbox(label='Display Taylor Polynomial', value=True)

    xcol1, xcol2 = st.sidebar.columns(2)
    with xcol1:
        xmin = st.number_input(label='xmin', value=1.)
        ymin = st.number_input(label='ymin', value=-50.)
    with xcol2:
        xmax = st.number_input(label='xmax', value=4.)
        ymax = st.number_input(label='ymax', value=50.)

    res = st.sidebar.number_input(label='resolution', value=200)

    backend = st.sidebar.selectbox(label="Backend", options=('Matplotlib', 'Altair'), index=0)

    # Create main page widgets

    tcol1, tcol2 = st.columns(2)

    with tcol1:
        st.title('Truncated Taylor Series')
    with tcol2:
        if qr:
            st.markdown('## <img src="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data='
                        'https://share.streamlit.io/joergbrech/truncated-taylor-series/main)" alt='
                        '"https://s.gwdg.de/PST5dv" width="200"/> https://s.gwdg.de/PST5dv',
                        unsafe_allow_html=True)

    # prepare matplotlib plot
    if 'Matplotlib' in backend:

        def clear_figure():
            del st.session_state['mpl_fig']
            del st.session_state['handles']
        xkcd = st.sidebar.checkbox("use xkcd-style", value=True, on_change=clear_figure)

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
            max_value=degree_max,
            value=int(0)
        )

    # update the data
    coefficients = update_coefficients(func_str, degree_max)
    xs, ys, ps, fx0 = update_data(coefficients, degree, x0, xmin, xmax, res)

    if 'Matplotlib' in backend:

        if xkcd:
            # set rc parameters to xkcd style
            plt.xkcd()
        else:
            # reset rc parameters to default
            plt.rcdefaults()

        # initialize the Matplotlib figure and initialize an empty dict of plot handles
        if 'mpl_fig' not in st.session_state:
            st.session_state.mpl_fig = plt.figure(figsize=(8, 3))
            st.session_state.mpl_fig.add_axes([0., 0., 1., 1.])

        if 'handles' not in st.session_state:
            st.session_state.handles = {}

    if 'Altair' in backend and 'chart' not in st.session_state:
        # initialize empty chart
        st.session_state.chart = st.empty()

    # update plot
    if 'Matplotlib' in backend:
        update_plot(x0, fx0, xs, ys, ps, visible, xmin, xmax, ymin, ymax)
        st.pyplot(st.session_state.mpl_fig)
    else:
        df = pd.DataFrame(data=np.array([xs, ys, ps], dtype=np.float64).transpose(),
                          columns=["x", "function", "Taylor polynomial at x0"])
        chart = alt.Chart(df) \
            .transform_fold(["function", "Taylor polynomial at x0"], as_=["legend", "y"]) \
            .mark_line(clip=True) \
            .encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=(xmin, xmax))),
                y=alt.Y('y:Q', scale=alt.Scale(domain=(ymin, ymax))),
                color=alt.Color('legend:N',
                                scale=alt.Scale(range=["green", "blue"]),
                                legend=alt.Legend(orient='bottom'))
            )\
            .interactive()
        pnt_data = pd.DataFrame({'x': [float(x0),], 'y': [float(fx0),]})
        pnt = alt.Chart(pnt_data)\
            .mark_point(clip=True, color='white')\
            .encode(
                x='x:Q',
                y='y:Q',
            )\
            .interactive()
        altair_chart = (chart + pnt).properties(width=800, height=400)
        st.session_state.chart.altair_chart(altair_chart, use_container_width=True)
