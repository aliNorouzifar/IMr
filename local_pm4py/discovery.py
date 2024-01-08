from typing import Optional, Dict, Any, Union, Tuple
from pm4py.objects.log.obj import EventLog, EventStream
from pm4py.objects.petri_net.obj import PetriNet, Marking
import pandas as pd
from pm4py import util as pmutil
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py.objects.conversion.process_tree import converter as tree_to_petri
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.util import filtering_utils
from pm4py.objects.process_tree.utils import generic
from pm4py.objects.process_tree.utils.generic import tree_sort
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.statistics.end_activities.log import get as end_activities_get
from pm4py.statistics.start_activities.log import get as start_activities_get
from pm4py.util import exec_utils
from pm4py.util import variants_util
from pm4py.util import xes_constants
from pm4py.util import constants
from enum import Enum
import deprecation
# from local_pm4py.algo.discovery.inductive import algorithm as inductive_miner
from local_pm4py.subtree_plain import SubtreePlain
from pm4py.algo.discovery.dfg.utils.dfg_utils import get_activities_self_loop
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.objects.process_tree.obj import Operator
from pm4py.util import exec_utils, xes_constants
from pm4py.util import constants
from enum import Enum


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY



def apply_bi(logp, logm, parameters: Optional[Dict[Any, Any]] = None, sup= None, ratio = None, size_par = None, rules =None) -> Tuple[PetriNet, Marking, Marking]:

    process_tree = apply_tree(logp, logm, parameters, sup=sup, ratio=ratio, size_par=size_par, rules=rules)
    net, initial_marking, final_marking = tree_to_petri.apply(process_tree)

    return net, initial_marking, final_marking



def apply_tree(logp,logm, parameters=None, sup= None, ratio = None, size_par = None, rules= None):
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters,
                                              pmutil.xes_constants.DEFAULT_NAME_KEY)


    dfgp = [(k, v) for k, v in dfg_inst.apply(logp, parameters=parameters).items() if v > 0]
    dfgm = [(k, v) for k, v in dfg_inst.apply(logm, parameters=parameters).items() if v > 0]

    c = Counts()
    activitiesp = attributes_get.get_attribute_values(logp, activity_key)
    start_activitiesp = list(start_activities_get.get_start_activities(logp, parameters=parameters).keys())
    end_activitiesp = list(end_activities_get.get_end_activities(logp, parameters=parameters).keys())
    contains_empty_traces = False
    traces_length = [len(trace) for trace in logp]
    if traces_length:
        contains_empty_traces = min([len(trace) for trace in logp]) == 0

    recursion_depth = 0
    sub = make_tree(logp,logm, dfgp, dfgp, dfgp, activitiesp, c, recursion_depth, 0.0, start_activitiesp,
                            end_activitiesp,
                            start_activitiesp, end_activitiesp, parameters, sup= sup, ratio = ratio, size_par = size_par, rules= rules)

    process_tree = get_repr(sub, 0, contains_empty_traces=contains_empty_traces)
    # Ensures consistency to the parent pointers in the process tree
    fix_parent_pointers(process_tree)
    # Fixes a 1 child XOR that is added when single-activities flowers are found
    fix_one_child_xor_flower(process_tree)
    # folds the process tree (to simplify it in case fallthroughs/filtering is applied)
    process_tree = generic.fold(process_tree)
    # sorts the process tree to ensure consistency in different executions of the algorithm
    tree_sort(process_tree)

    return process_tree


def make_tree(logp, logm, dfg, master_dfg, initial_dfg, activities, c, recursion_depth, noise_threshold, start_activities,
              end_activities, initial_start_activities, initial_end_activities, parameters=None, sup= None, ratio = None, size_par = None, rules = None):

    tree = SubtreePlain(logp,logm, dfg, master_dfg, initial_dfg, activities, c, recursion_depth, noise_threshold,
                        start_activities,
                        end_activities, initial_start_activities, initial_end_activities, parameters=parameters, sup= sup, ratio = ratio, size_par = size_par, rules= rules)
    return tree


def get_repr(spec_tree_struct, rec_depth, contains_empty_traces=False):

    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, spec_tree_struct.parameters,
                                              xes_constants.DEFAULT_NAME_KEY)

    base_cases = ('empty_log', 'single_activity')
    cut = ('concurrent', 'sequential', 'parallel', 'loopCut')
    # note that the activity_once_per_trace is not included here, as it is can be dealt with as a parallel cut
    fall_throughs = ('empty_trace', 'strict_tau_loop', 'tau_loop', 'flower')

    # if a cut was detected in the current subtree:
    if spec_tree_struct.detected_cut in cut:
        if spec_tree_struct.detected_cut == "sequential":
            final_tree_repr = ProcessTree(operator=Operator.SEQUENCE)
        elif spec_tree_struct.detected_cut == "loopCut":
            final_tree_repr = ProcessTree(operator=Operator.LOOP)
        elif spec_tree_struct.detected_cut == "concurrent":
            final_tree_repr = ProcessTree(operator=Operator.XOR)
        elif spec_tree_struct.detected_cut == "parallel":
            final_tree_repr = ProcessTree(operator=Operator.PARALLEL)

        if not (spec_tree_struct.detected_cut == "loopCut" and len(spec_tree_struct.children) >= 3):
            for ch in spec_tree_struct.children:
                # get the representation of the current child (from children in the subtree-structure):
                child = get_repr(ch, rec_depth + 1)
                # add connection from child_tree to child_final and the other way around:
                final_tree_repr.children.append(child)
                child.parent = final_tree_repr

        else:
            child = get_repr(spec_tree_struct.children[0], rec_depth + 1)
            final_tree_repr.children.append(child)
            child.parent = final_tree_repr

            redo_child = ProcessTree(operator=Operator.XOR)
            for ch in spec_tree_struct.children[1:]:
                child = get_repr(ch, rec_depth + 1)
                redo_child.children.append(child)
                child.parent = redo_child

            final_tree_repr.children.append(redo_child)
            redo_child.parent = final_tree_repr

        if spec_tree_struct.detected_cut == "loopCut" and len(spec_tree_struct.children) < 3:
            while len(spec_tree_struct.children) < 2:
                child = ProcessTree()
                final_tree_repr.children.append(child)
                child.parent = final_tree_repr
                spec_tree_struct.children.append(None)

    if spec_tree_struct.detected_cut in base_cases:
        # in the base case of an empty log, we only return a silent transition
        if spec_tree_struct.detected_cut == "empty_log":
            return ProcessTree(operator=None, label=None)
        # in the base case of a single activity, we return a tree consisting of the single activity
        elif spec_tree_struct.detected_cut == "single_activity":
            if len(spec_tree_struct.log[0]) != 0:
                act_a = spec_tree_struct.log[0][0][activity_key]
            else:
                l = spec_tree_struct.log
                l = sorted(l, key=lambda x: len(x))
                act_a = l[-1][0][activity_key]
            return ProcessTree(operator=None, label=act_a)

    if spec_tree_struct.detected_cut in fall_throughs:
        if spec_tree_struct.detected_cut == "empty_trace":
            # should return XOR(tau, IM(L') )
            final_tree_repr = ProcessTree(operator=Operator.XOR)
            final_tree_repr.children.append(ProcessTree(operator=None, label=None))
            # iterate through all children of the current node
            for ch in spec_tree_struct.children:
                child = get_repr(ch, rec_depth + 1)
                final_tree_repr.children.append(child)
                child.parent = final_tree_repr

        elif spec_tree_struct.detected_cut == "strict_tau_loop" or spec_tree_struct.detected_cut == "tau_loop":
            # should return LOOP( IM(L'), tau)
            final_tree_repr = ProcessTree(operator=Operator.LOOP)
            # iterate through all children of the current node
            if spec_tree_struct.children:
                for ch in spec_tree_struct.children:
                    child = get_repr(ch, rec_depth + 1)
                    final_tree_repr.children.append(child)
                    child.parent = final_tree_repr
            else:
                for ch in spec_tree_struct.activities:
                    child = get_transition(ch)
                    final_tree_repr.append(child)
                    child.parent = final_tree_repr

            # add a silent tau transition as last child of the current node
            final_tree_repr.children.append(ProcessTree(operator=None, label=None))

        elif spec_tree_struct.detected_cut == "flower":
            # should return something like LOOP(XOR(a,b,c,d,...), tau)
            final_tree_repr = ProcessTree(operator=Operator.LOOP)
            xor_child = ProcessTree(operator=Operator.XOR, parent=final_tree_repr)
            # append all the activities in the current subtree to the XOR part to allow for any behaviour
            for ch in spec_tree_struct.activities:
                child = get_transition(ch)
                xor_child.children.append(child)
                child.parent = xor_child
            final_tree_repr.children.append(xor_child)
            # now add the tau to the children to get the wanted output
            final_tree_repr.children.append(ProcessTree(operator=None, label=None))

    return final_tree_repr


def get_transition(label):
    """
    Create a node (transition) with the specified label in the process tree
    """
    return ProcessTree(operator=None, label=label)

class Counts(object):
    """
    Shared variables among executions
    """

    def __init__(self):
        """
        Constructor
        """
        self.num_places = 0
        self.num_hidden = 0
        self.num_visible_trans = 0
        self.dict_skips = {}
        self.dict_loops = {}

    def inc_places(self):
        """
        Increase the number of places
        """
        self.num_places = self.num_places + 1

    def inc_no_hidden(self):
        """
        Increase the number of hidden transitions
        """
        self.num_hidden = self.num_hidden + 1

    def inc_no_visible(self):
        """
        Increase the number of visible transitions
        """
        self.num_visible_trans = self.num_visible_trans + 1


def fix_parent_pointers(pt):
    """
    Ensures consistency to the parent pointers in the process tree

    Parameters
    --------------
    pt
        Process tree
    """
    for child in pt.children:
        child.parent = pt
        if child.children:
            fix_parent_pointers(child)


def fix_one_child_xor_flower(tree):
    """
    Fixes a 1 child XOR that is added when single-activities flowers are found

    Parameters
    --------------
    tree
        Process tree
    """
    if tree.parent is not None and tree.operator is Operator.XOR and len(tree.children) == 1:
        for child in tree.children:
            child.parent = tree.parent
            tree.parent.children.append(child)
            del tree.parent.children[tree.parent.children.index(tree)]
    else:
        for child in tree.children:
            fix_one_child_xor_flower(child)