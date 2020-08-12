import os
from shutil import copyfile
import subprocess
import tempfile
import numpy as np


LIMIT = 2000


base = r'''\begin{tikzpicture}[
        %scale=.5
    ]
    \begin{axis}[
        %cycle list name=exotic,
        %width=.5\textwidth,
        %height=\axisdefaultheight,
        %every axis plot/.append style={ultra thick},
        %ymin=0,
        %ymax=100,
        %xmin=0,
        %xmax=100,
        no markers,
        legend style={at={(1.1, 1.0)},
			anchor=north west,
			font=\scriptsize},'''


addplot = r'''
        \addplot+[
            %smooth
        ] plot coordinates {};'''


final = r'''
    \end{axis}
\end{tikzpicture}
'''


texdir = 'texplots'


def plot(fname, title, xlabel, ylabel, data, xlog=False, barplot=False):
    """creates the tikzpicture in a local tex and pdf file"""
    if not os.path.isdir(texdir):
        os.makedirs(texdir)
    fname = os.path.join(texdir, fname)
    write_tikzpicture(fname, title, xlabel, ylabel, data, xlog, barplot)
    wdir = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        copyfile(os.path.join(wdir, fname + '.tex'), 'plot.tex')
        copyfile(os.path.join(os.path.dirname(__file__), 'standalone_template.tex'), 'tmp.tex')
        try:
            FNULL = open(os.devnull, 'w')
            subprocess.run(['pdflatex', 'tmp.tex'], stdout=FNULL, stderr=subprocess.STDOUT)
            copyfile('tmp.pdf', os.path.join(wdir, os.path.basename(fname + '.pdf')))
        except FileNotFoundError:
            print('Generating the plots failed, make sure that pdflatex and pgfplots are correctly installed!')
        os.chdir(wdir)


def write_tikzpicture(fname, title, xlabel, ylabel, data, xlog, barplot):
    """stores the data in a local tikzpicture tex file"""
    xlabel = '\t\txlabel={' + xlabel + '},'
    ylabel = '\t\tylabel={' + ylabel + '},'
    title = '\t\ttitle={' + title + '},'
    xmode = '\t\txmode=log,' if xlog else '\t\txmode=normal,'
    if barplot:
        init = base.replace('%ymin', 'ymin') + '\n\t\tybar stacked,\n\t\tbar width=3 pt,'
    else:
        init = base
    tex = '\n'.join([init, xlabel, ylabel, title, xmode, '\t]'])
    if len(data) == 1 and len(data[0][0]) == 2:
        values = data[0][::int(np.ceil(len(data[0]) / LIMIT))]
        coord = '{' + ' '.join('({}, {})'.format(xval, yval) for xval, yval in values) + '}'
        tex = tex + addplot.format(coord)
    else:
        for name, values in data:
            if values != []:
                values = values[::int(np.ceil(len(values) / LIMIT))]
                coord = '{' + ' '.join('({}, {})'.format(xval, yval) for xval, yval in values) + '}'
                legend = '\n\t\t' + r'\addlegendentry{' + name.replace('_', ' ') + '}'
                tex = tex + addplot.format(coord) + legend

    tex = tex + final
    with open(fname + '.tex', 'w') as texfile:
        texfile.write(tex)
