import math
import os
import re
import sys
import pickle
import shutil


start = ':: '
end = ' ::'
max_width = 80
min_width = 50


def to_line(text):
    return start + text + end


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def cleanup():
    shutil.rmtree('imgs', ignore_errors=True)
    shutil.rmtree('texplots', ignore_errors=True)
    shutil.rmtree('probgf/__pycache__', ignore_errors=True)
    shutil.rmtree('probgf/.pytest_cache', ignore_errors=True)
    shutil.rmtree('tests/__pycache__', ignore_errors=True)
    shutil.rmtree('.pytest_cache', ignore_errors=True)
    shutil.rmtree('.cache', ignore_errors=True)
    for _, dirs, files in os.walk(os.getcwd()):
        for dirname in dirs:
            if dirname.startswith('CV_'):
                shutil.rmtree(dirname, ignore_errors=True)
        for fname in files:
            if fname == '.config':
                os.remove(fname)
            if fname.endswith('.npy'):
                os.remove(fname)
            if fname.startswith('fig_'):
                os.remove(fname)
            if fname.startswith('report_'):
                os.remove(fname)
            if fname.startswith('pred_'):
                os.remove(fname)


class ConsoleOutput:

    def __init__(self, kfolds, cpus):
        self.p_fmt = '{:4.1f}%'
        self.width_prog = len(self.p_fmt.format(100))
        self.width_text = min(max(kfolds * self.width_prog + kfolds - 1, min_width), max_width - len(start) - len(end))
        self.max_per_line = (max_width - len(start) - len(end)) // (self.width_prog + 1)
        self.prog_lines = [(line * self.max_per_line, (line + 1) * self.max_per_line) for line in range(math.ceil(kfolds / self.max_per_line))]
        self.cpus = cpus


    def empty(self, printit=True):
        output = to_line(' ' * self.width_text)
        if printit:
            print(output)
        return output


    def clear_for_prog(self, printit=True):
        output = '\n'.join(self.empty(False) for _ in range(len(self.prog_lines)))
        if printit:
            print(output)
        return output


    def centered(self, text, printit=True, emptystart=False, emptyend=False):
        output = to_line(text.center(self.width_text))
        if emptystart:
            output = self.empty(False) + '\n' + output
        if emptyend:
            output = output + '\n' + self.empty(False)
        if printit:
            print(output)
        return output


    def progress(self, progress_list, active_split, printit=True):
        output = ''
        if self.cpus == 1: # only print the progress that is not 0.0
            centered = self.p_fmt.format(progress_list[active_split])
            output = output + self.centered(centered, False)
        else: # print all progress simultaneously
            for l_start, l_end in self.prog_lines:
                line_prog = progress_list[l_start:l_end]
                centered = ' '.join([self.p_fmt.format(prog).center(self.width_prog) for prog in line_prog])
                output = output + self.centered(centered, False) + '\n'
            output = output[:-1]
        if printit:
            for _ in range(len(self.prog_lines)):
                sys.stdout.write("\033[F") # cursor up one line
            print(output)    
        return output


def draw_structure(edges, positions, fname, max_y=4, labels=True):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print('Could not store figure with STRF structure, make sure that networkx and matplotlib is correctly installed!')
        return
    plt.clf()
    G = nx.Graph()
    edges = [edge for edge in edges if all([positions[node][1] < max_y for node in edge])]
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    if labels:
        nx.draw_networkx(G, pos=positions, font_size=8)
    else:
        nx.draw_networkx_nodes(G, pos=positions, node_size=50)
        nx.draw_networkx_edges(G, pos=positions)
    plt.savefig(fname, bbox_inches='tight', pad_inches=-0.2)


def project_value(value, src_min, src_max, dst_min, dst_max):
    """project value from source interval to destination interval"""
    scaled = float(value - src_min) / float(src_max - src_min)
    return dst_min + (scaled * (dst_max - dst_min))


def find_method(method_name, methods, error_msg):
    for name, method in methods:
        if name == method_name:
            chosen_method = method
            break
    else:
        raise RuntimeError('{} method "{}" not found. Implemented methods:\n'.format(error_msg, method_name) + \
            '   ' + '\n   '.join(['{}: {}'.format(name, meth.__doc__) for name, meth in methods]))
    return chosen_method
