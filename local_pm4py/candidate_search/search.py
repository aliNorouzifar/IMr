from local_pm4py.functions.functions import n_edges
from local_pm4py.functions.functions import add_SE
from local_pm4py.candidate_search.is_allowed import is_allowed
import networkx as nx


def adj(node_set, net):
    adj_set = set()
    for node in node_set:
        adj_set = adj_set.union(set(net.neighbors(node)))
    return adj_set

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



def find_possible_partitions(net,rules,sa,ea):
    thr = 0.001
    sa = filter_start(net, sa, thr)
    ea = filter_end(net, ea, thr)
    activity_list = set(net.nodes)-{'start','end'}
    queue = [(set(), {'start'})]
    visited = []
    valid = []
    st_net = {x:net.succ['start'][x]['weight'] for x in net.succ['start'] if x!='end'}
    en_net = {x: net.pred['end'][x]['weight'] for x in net.pred['end'] if x!='end'}
    while len(queue) != 0:
        current = queue.pop()
        for x in current[1]:
            new_state = current[0].union({x})
            if x in sa:
                new_state.add('start')
            if x in ea:
                new_state.add('end')

            if new_state not in visited:
                new_adj = current[1].union(adj({x},net)) - new_state
                visited.append(new_state)


                B = activity_list - new_state
                if (len(B) == 0) or (len(B) == len(activity_list)):
                    queue.append((new_state, new_adj))
                    if (len(B) == 0) and (n_edges(net, set(en_net.keys()), set(st_net.keys()))>0):
                        na, block = is_allowed(new_state, B, rules, st_net, en_net)
                        possible_cuts = {'loop_tau'} - na
                        if (len(possible_cuts) > 0)  and (sa!=ea):
                            valid.append((new_state, B, possible_cuts))
                    continue
                B = add_SE(net, B)
                netA = net.subgraph(new_state)
                netB = net.subgraph(B)

                # 'start' ~> netB
                if 'start' in netB:
                    startB = set(netB.nodes) - set(nx.descendants(netB, 'start')) - {'start','end'}
                else:
                    startB = set(netB.nodes) - {'start','end'}
                # 'end' ~> netA
                if 'end' in netA:
                    endA = set(netA.nodes) - set(nx.ancestors(netA, 'end')) - {'start','end'}
                else:
                    endA = set(netA.nodes) - {'start','end'}
                # 'end' ~> netB
                if 'end' in netB:
                    endB = set(netB.nodes) - set(nx.ancestors(netB, 'end')) - {'start', 'end'}
                else:
                    endB = set(netB.nodes) - {'start','end'}

                na, block = is_allowed(new_state, B, rules,st_net,en_net)
                possible_cuts = set()
                if len(endB)==0:
                    possible_cuts.add("seq")
                    if len(endA)==0:
                        possible_cuts.add("loop")
                        if len(startB)==0 and (B not in visited):
                            possible_cuts.add("exc")
                            possible_cuts.add("par")
                elif len(endA)==0:
                    possible_cuts.add("loop")
                # r_p = possible_cuts
                possible_cuts = possible_cuts - na
                if len(possible_cuts)>0:
                    valid.append((new_state, B, possible_cuts))

                if block==False:
                    queue.append((new_state, new_adj))

    return valid