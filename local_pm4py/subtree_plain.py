import collections
from copy import copy
import time
from pm4py.algo.discovery.dfg.utils.dfg_utils import infer_start_activities, infer_end_activities

from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py import util as pmutil
import local_pm4py.split_functions.split as split
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.statistics.end_activities.log import get as end_activities_get
from pm4py.statistics.start_activities.log import get as start_activities_get
from pm4py.util import exec_utils
from pm4py.util import constants
from enum import Enum
from pm4py.objects.log import obj as log_instance
from pm4py.util import xes_constants
from local_pm4py.dfg import algorithm as dfg_discovery
import networkx as nx
import pm4py
from pm4py.algo.discovery.dfg.utils.dfg_utils import get_activities_from_dfg
import copy
from collections import Counter
from local_pm4py.candidate_search.search import find_possible_partitions
from local_pm4py.base_case.check import check_base_case
from local_pm4py.functions.functions import max_flow_graph
from local_pm4py.cut_quality.cost_functions import cost_functions
from local_pm4py.functions.functions import n_edges

def artificial_start_end(log):
    st = 'start'
    en = 'end'
    activity_key = xes_constants.DEFAULT_NAME_KEY
    start_event = log_instance.Event()
    start_event[activity_key] = st

    end_event = log_instance.Event()
    end_event[activity_key] = en

    for trace in log:
        trace.insert(0, start_event)
        trace.append(end_event)
    return log

def generate_nx_graph_from_dfg(dfg):
    dfg_acts = set()
    for x in dfg:
        dfg_acts.add(x[0])
        dfg_acts.add(x[1])
    G = nx.DiGraph()
    for act in dfg_acts:
        G.add_node(act)
    for edge in dfg:
        G.add_edge(edge[0], edge[1], weight=dfg[edge])
    return G

def filter_start(net, sa, thr):
    sa_new = set()
    for a in sa:
        act_ratio = n_edges(net, {'start'}, {a})/ net.out_degree(a, weight='weight')
        start_ratio = n_edges(net, {'start'}, {a})/ net.out_degree('start', weight='weight')
        if act_ratio>= thr or start_ratio >= thr:
            sa_new.add(a)
    return sa_new

def filter_end(net, sa, thr):
    en_new = set()
    for a in sa:
        act_ratio = n_edges(net, {a}, {'end'})/ net.out_degree(a, weight='weight')
        end_ratio = n_edges(net, {a}, {'end'})/ net.in_degree('end', weight='weight')
        if act_ratio>= thr or end_ratio >= thr:
            en_new.add(a)
    return en_new


class SubtreePlain(object):
    def __init__(self, logp,logm, dfg, master_dfg, initial_dfg, activities, counts, rec_depth, noise_threshold=0,
                 start_activities=None, end_activities=None, initial_start_activities=None,
                 initial_end_activities=None, parameters=None, real_init=True, sup= None, ratio = None, size_par = None, rules = None):

        if real_init:
            self.master_dfg = copy.copy(master_dfg)
            self.initial_dfg = copy.copy(initial_dfg)
            self.counts = counts
            self.rec_depth = rec_depth
            self.noise_threshold = noise_threshold
            self.start_activities = pm4py.get_start_activities(logp)
            self.start_activitiesM = pm4py.get_start_activities(logm)
            self.end_activities = pm4py.get_end_activities(logp)
            self.end_activitiesM = pm4py.get_end_activities(logm)
            self.initial_start_activities = initial_start_activities
            if self.initial_start_activities is None:
                self.initial_start_activities = infer_start_activities(master_dfg)
            self.initial_end_activities = initial_end_activities
            if self.initial_end_activities is None:
                self.initial_end_activities = infer_end_activities(master_dfg)

            self.log = logp
            self.log_art = artificial_start_end(copy.deepcopy(logp))
            self.logM = logm
            self.logM_art = artificial_start_end(copy.deepcopy(logm))
            self.inverted_dfg = None
            self.original_log = logp
            self.activities = None

            self.initialize_tree(dfg, logp,logm, initial_dfg, activities, parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, rules = rules)


    def initialize_tree(self, dfg, logp,logm, initial_dfg, activities, second_iteration=False, end_call=True,
                        parameters=None, sup= None, ratio = None, size_par = None, rules = None):


        if activities is None:
            self.activities = get_activities_from_dfg(dfg)
        else:
            self.activities = copy.copy(activities)
        self.detected_cut = None
        self.children = []
        self.log = logp
        self.log_art = artificial_start_end(logp.__deepcopy__())
        self.logM = logm
        self.logM_art = artificial_start_end(logm.__deepcopy__())
        self.original_log = logp
        self.parameters = parameters

        self.detect_cut(second_iteration=False, parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, rules = rules)


    def detect_cut(self, second_iteration=False, parameters=None, sup= None, ratio = None, size_par = None, rules = None):
        ratio = ratio
        sup_thr = sup

        logP_var = Counter([tuple([x['concept:name'] for x in t]) for t in self.log])
        logM_var = Counter([tuple([x['concept:name'] for x in t]) for t in self.logM])


        if parameters is None:
            parameters = {}
        activity_key = exec_utils.get_param_value(constants.PARAMETER_CONSTANT_ACTIVITY_KEY, parameters,
                                                  pmutil.xes_constants.DEFAULT_NAME_KEY)

        # check base cases:
        isbase, cut = check_base_case(self, logP_var,logM_var, sup, ratio, size_par)

        if isbase==False:
            dfg2 = dfg_discovery.apply(self.log_art, variant=dfg_discovery.Variants.FREQUENCY)
            netP = generate_nx_graph_from_dfg(dfg2)
            del dfg2[('start', 'end')]

            dfg2M = dfg_discovery.apply(self.logM_art, variant=dfg_discovery.Variants.FREQUENCY)
            netM = generate_nx_graph_from_dfg(dfg2M)
            del dfg2M[('start', 'end')]

            if parameters == {}:
                feat_scores_togg = collections.defaultdict(lambda: 1, {})
                feat_scores = collections.defaultdict(lambda: 1, {})
                for x in dfg2.keys():
                    feat_scores[x] = 1
                    feat_scores_togg[x] = 1
                for y in dfg2M.keys():
                    feat_scores[y] = 1
                    feat_scores_togg[y] = 1


            possible_partitions = find_possible_partitions(netP,rules,set(self.start_activities),set(self.end_activities))
            # print('#possible_cuts:')
            # print(sum([len(tt[2]) for tt in possible_partitions]))
            if len(possible_partitions)==0 or (len(possible_partitions)==1 and possible_partitions[0][2]=={'loop_tau'}):
                print('no cut available')
                rules2 = {}
                for r in rules.keys():
                        rules2[r] = []
                possible_partitions = find_possible_partitions(netP, rules2, set(self.start_activities),set(self.end_activities))
                # print(sum([len(tt[2]) for tt in possible_partitions]))

            cut = []

            dfgP = dfg_discovery.apply(self.log_art, variant=dfg_discovery.Variants.FREQUENCY)
            dfgM = dfg_discovery.apply(self.logM_art, variant=dfg_discovery.Variants.FREQUENCY)
            activitiesM = set(a for x in logM_var.keys() for a in x)


            #########################
            fP = max_flow_graph(netP)
            fM = max_flow_graph(netM)


            start_acts_P = set([x[1] for x in dfgP if (x[0] == 'start')])-{'end'}
            end_acts_P = set([x[0] for x in dfgP if (x[1] == 'end')])-{'start'}


            ratio_backup = ratio

            for pp in possible_partitions:
                A = pp[0] - {'start', 'end'}
                B = pp[1] - {'start', 'end'}

                start_A_P = set([x[1] for x in dfgP if ((x[0] == 'start') and (x[1] in A))])
                end_A_P = set([x[0] for x in dfgP if (x[0] in A and (x[1] == 'end'))])
                input_B_P = set([x[1] for x in dfgP if ((x[0] not in B) and (x[1] in B))])
                output_B_P = set([x[0] for x in dfgP if ((x[0] in B) and (x[1] not in B))])

                start_A_M = set([x[1] for x in dfgM if ((x[0] == 'start') and (x[1] in A))])
                end_A_M = set([x[0] for x in dfgM if (x[0] in A and (x[1] == 'end'))])
                input_B_M = set([x[1] for x in dfgM if ((x[0] not in B) and (x[1] in B))])
                output_B_M = set([x[0] for x in dfgM if ((x[0] in B) and (x[1] not in B))])

                type = pp[2]
                if len(set(activitiesM).intersection(A))==0 or len(set(activitiesM).intersection(B))==0:
                    ratio = 0
                else:
                    ratio = ratio_backup

                #####################################################################
                # seq check
                if "seq" in type:
                    cost_seq_P = cost_functions.cost_seq(netP, A, B, sup, fP)
                    cost_seq_M = cost_functions.cost_seq(netM, A.intersection(activitiesM), B.intersection(activitiesM), sup, fM)
                    cut.append(((A, B), 'seq', cost_seq_P, cost_seq_M, cost_seq_P - ratio* size_par * cost_seq_M))
                #####################################################################

                #####################################################################
                # xor check
                if "exc" in type:
                    cost_exc_P = cost_functions.cost_exc(netP, A, B)
                    cost_exc_M = cost_functions.cost_exc(netM, A.intersection(activitiesM), B.intersection(activitiesM))
                    cut.append(((A, B), 'exc', cost_exc_P, cost_exc_M, cost_exc_P - ratio* size_par * cost_exc_M))
                #####################################################################

                #####################################################################
                # xor-tau check
                if n_edges(netP,{'start'},{'end'})>0:
                    missing_exc_tau_P = 0
                    missing_exc_tau_P += max(0, sup_thr * len(self.log) - cost_functions.n_edges(netP,{'start'},{'end'}))

                    missing_exc_tau_M = 0
                    missing_exc_tau_M += max(0, sup_thr * len(self.logM) - cost_functions.n_edges(netM, {'start'}, {'end'}))

                    cost_exc_tau_P = missing_exc_tau_P
                    cost_exc_tau_M = missing_exc_tau_M
                    cut.append(((A.union(B), set()), 'exc2', cost_exc_tau_P, cost_exc_tau_M,cost_exc_tau_P - ratio * size_par * cost_exc_tau_M,1))
                #####################################################################

                #####################################################################
                # parallel check
                if "par" in type:
                    cost_par_P = cost_functions.cost_par(netP, A, B,sup)
                    cost_par_M = cost_functions.cost_par(netM, A.intersection(activitiesM), B.intersection(activitiesM), sup)
                    cut.append(((A, B), 'par', cost_par_P, cost_par_M, cost_par_P - ratio * size_par * cost_par_M))
                #####################################################################

                #####################################################################
                # loop check
                if "loop" in type:
                    cost_loop_P = cost_functions.cost_loop(netP, A, B, sup, start_A_P, end_A_P, input_B_P, output_B_P)
                    cost_loop_M = cost_functions.cost_loop(netM, A, B, sup, start_A_M, end_A_M, input_B_M, output_B_M)

                    if cost_loop_P is not False:
                        cut.append(((A, B), 'loop', cost_loop_P, cost_loop_M, cost_loop_P - ratio * size_par * cost_loop_M))

                if "loop_tau" in type:
                    missing_loopP = 0
                    missing_loopM = 0

                    MMP = max(sum(self.start_activities.values()),sum(self.end_activities.values()))
                    for x in start_acts_P:
                        for y in end_acts_P:
                            L1P = max(0, MMP * sup_thr * (
                                    self.start_activities[x] / (sum(self.start_activities.values()))) * (
                                              self.end_activities[y] / (sum(self.end_activities.values()))) - dfgP[
                                          (y, x)])
                            missing_loopP += L1P
                    MMM = max(sum(self.start_activitiesM.values()), sum(self.end_activitiesM.values()))
                    for x in start_acts_P.intersection(self.start_activitiesM.keys()):
                        for y in end_acts_P.intersection(self.end_activitiesM.keys()):
                            L1M = max(0, MMM * sup_thr * (
                                    self.start_activitiesM[x] / (sum(self.start_activitiesM.values()))) * (
                                              self.end_activitiesM[y] / (sum(self.end_activitiesM.values()))) -
                                      dfgM[(y, x)])
                            missing_loopM += L1M

                    cost_loop_P = missing_loopP
                    cost_loop_M = missing_loopM

                    cut.append(((start_acts_P, end_acts_P), 'loop_tau', cost_loop_P, cost_loop_M,
                                    cost_loop_P - ratio * size_par * cost_loop_M, 1))



            sorted_cuts = sorted(cut, key=lambda x: (x[4], x[2],['exc','exc2','seq','par','loop','loop_tau'].index(x[1]), -(len(x[0][0]) * len(x[0][1]) / (len(x[0][0]) + len(x[0][1])))))
            if len(sorted_cuts) != 0:
                cut = sorted_cuts[0]
            else:
                cut = ('none', 'none', 'none','none','none', 'none')

        # print(cut)

        map_cut_op = {'par': 'parallel', 'seq': 'sequential', 'exc': 'concurrent', 'exc2': 'concurrent',
                      'loop': 'loopCut', 'loop1': 'loopCut', 'loop_tau': 'loopCut'}
        map_cut_op2 = {'par': 'par', 'seq': 'seq', 'exc': 'exc', 'exc2': 'exc',
                      'loop': 'loop', 'loop1': 'loop1', 'loop_tau': 'loop_tau'}
        if cut[1] in map_cut_op.keys():
            self.detected_cut = map_cut_op[cut[1]]
            LAP, LBP = split.split(map_cut_op2[cut[1]], [cut[0][0], cut[0][1]], self.log, activity_key)
            LAM, LBM = split.split(map_cut_op2[cut[1]], [cut[0][0], cut[0][1]], self.logM, activity_key)
            new_logs = [[LAP, LAM], [LBP, LBM]]
            for l in new_logs:
                new_dfg = [(k, v) for k, v in dfg_inst.apply(l[0], parameters=parameters).items() if v > 0]
                activities = attributes_get.get_attribute_values(l[0], activity_key)
                start_activities = list(start_activities_get.get_start_activities(l[0], parameters=parameters).keys())
                end_activities = list(end_activities_get.get_end_activities(l[0], parameters=parameters).keys())
                self.children.append(
                    SubtreePlain(l[0], l[1], new_dfg, self.master_dfg, self.initial_dfg, activities, self.counts,
                                 self.rec_depth + 1,
                                 noise_threshold=self.noise_threshold, start_activities=start_activities,
                                 end_activities=end_activities,
                                 initial_start_activities=self.initial_start_activities,
                                 initial_end_activities=self.initial_end_activities,
                                 parameters=parameters, sup=sup, ratio=ratio, size_par=size_par, rules=rules))
        elif cut[1] == 'none':
            self.detected_cut = 'flower'


def make_tree(logp, logm, dfg, master_dfg, initial_dfg, activities, c, recursion_depth, noise_threshold, start_activities,
              end_activities, initial_start_activities, initial_end_activities, parameters=None, sup= None, ratio = None, size_par = None, rules = None):

    tree = SubtreePlain(logp,logm, dfg, master_dfg, initial_dfg, activities, c, recursion_depth, noise_threshold,
                        start_activities,
                        end_activities, initial_start_activities, initial_end_activities, parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, rules= rules)

    return tree
