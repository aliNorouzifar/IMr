import networkx as nx
import copy
from pm4py import util as pmutil
from pm4py.algo.discovery.dfg.variants import native, performance, freq_triples, case_attributes
from pm4py.objects.conversion.log import converter as log_conversion
from pm4py.util import xes_constants as xes_util
from pm4py.util import exec_utils
from enum import Enum
import pkgutil
from pm4py.util import constants
from enum import Enum
from typing import Optional, Dict, Any, Union, Tuple
from pm4py.objects.log.obj import EventLog, EventStream
import pandas as pd
from collections import Counter
from pm4py.objects.log import obj as log_instance


def n_edges(net, S, T):
    net_c = copy.deepcopy(net)
    edges_reweight = list(nx.edge_boundary(net_c, S, T, data='weight', default=1))
    return sum(weight for u, v, weight in edges_reweight if (u in S and v in T))


def max_flow_graph(net):
    flow_graph = {}
    for x in net.nodes:
        for y in net.nodes:
            if (x != y):
                flow_graph[(x, y)] = nx.algorithms.flow.maximum_flow(net, x, y, capacity='weight')[0]
    return flow_graph

def add_SE(net, s):
    if s & set(net.successors('start')):
        s.add('start')
    if s & set(net.predecessors('end')):
        s.add('end')
    return s

def export(log):
    new_log = log_instance.EventLog()
    for t in log:
        new_t = log_instance.Trace()
        for e in t:
            new_t.append(e)
        new_log.append(new_t)
    return new_log

