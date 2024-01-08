import networkx as nx
from collections import Counter
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import time

EXISTENCE = "existence"
EXACTLY_ONE = "exactly_one"
INIT = "init"
RESPONDED_EXISTENCE = "responded_existence"
RESPONSE = "response"
PRECEDENCE = "precedence"
SUCCESSION = "succession"
ALTRESPONSE = "altresponse"
ALTPRECEDENCE = "altprecedence"
ALTSUCCESSION = "altsuccession"
CHAINRESPONSE = "chainresponse"
CHAINPRECEDENCE = "chainprecedence"
CHAINSUCCESSION = "chainsuccession"
ABSENCE = "absence"
COEXISTENCE = "coexistence"
NONCOEXISTENCE = "noncoexistence"
NONSUCCESSION = "nonsuccession"
NONCHAINSUCCESSION = "nonchainsuccession"

def n_edges(net, S, T, scaling = None):
    net_c = copy.deepcopy(net)
    if scaling == None:
        edges_reweight = list(nx.edge_boundary(net_c, S, T, data='weight', default=1))
    else:
        edges = list(nx.edge_boundary(net_c, S, T, data='weight', default=1))
        edges_reweight = []
        for ed in edges:
            edges_reweight.append((ed[0],ed[1], ed[2]*scaling[(ed[0], ed[1])]))
            # net_c[ed[0]][ed[1]]['weight'] = net_c[sc[0]][sc[1]]['weight'] * scaling[sc]
        # edges = edges_reweight
    return sum(weight for u, v, weight in edges_reweight if (u in S and v in T))

def drop_SE(s):
    return s-{'start','end'}

def add_SE(net, s):
    if s & set(net.successors('start')):
        s.add('start')
    if s & set(net.predecessors('end')):
        s.add('end')
    return s

def add_SS(s):
    s.add('start')
    return s

def add_EE(s):
    s.add('end')
    return s

def r_to_s(net):
    return (set(nx.descendants(net, 'start')) == (set(net.nodes) - {'start'}))


def r_from_e(net):
    return (set(nx.ancestors(net, 'end')) == (set(net.nodes) - {'end'}))

def dfg_extract(log):
    dfgs = map((lambda t: [(t[i - 1], t[i]) for i in range(1, len(t))]), log)
    return Counter([dfg for lista in dfgs for dfg in lista])

def lal(net,a):
    return net.out_degree(weight='weight')[a]

def lAl(net,A):
    return sum([net.out_degree(weight='weight')[a] for a in A])

def toggle(dic):
    dic_new = defaultdict(lambda: 1, {})
    for x in dic:
        # dic_new[x] = (1-dic[x])+1
        dic_new[x] = 1/dic[x]
    return dic_new


def cost_seq(net, A, B, in_set, out_set, sup, flow, scores):
    scores_toggle = toggle(scores)
    c1 = n_edges(net, B, A, scaling=scores_toggle)

    c2 = 0
    for x in A:
        for y in B:
            c2 += max(0, scores[(x, y)] * net.out_degree(x, weight='weight') * sup * (net.out_degree(y, weight='weight') / (
                        sum([net.out_degree(p, weight='weight') for p in B]) + sum([net.out_degree(p, weight='weight') for p in A]))) - flow[(x, y)])

    c3 = 0
    # for x in out_set:
    #     for y in in_set:
    #         c3 += max(0, scores[(x, y)] * n_edges(net, {x}, B.union({'end'}), scaling=scores) * sup * (n_edges(net, A.union({'start'}), {y}, scaling=scores) /
    #                                                                                                    (n_edges(net, A.union({'start'}), B.union({'end'}), scaling=scores))) - n_edges(net, {x}, {y}, scaling=scores))

    # c3 = 0
    # for x in out_set:
    #     for y in in_set:
    #         # c3 += max(0, scores[(x, y)] * net.out_degree(x, weight='weight') * sup * (net.out_degree(y, weight='weight') / (sum([net.out_degree(p, weight='weight') for p in B]) + sum([net.out_degree(p, weight='weight') for p in A]))) - n_edges(net, {x}, {y}, scaling=scores))
    #         if n_edges(net,out_set, in_set)==0:
    #             print('hi')
    #         c3 += max(0, scores[(x, y)] * n_edges(net,{x}, in_set) * sup * (n_edges(net,out_set, {y}) / n_edges(net,out_set, in_set)) - n_edges(net, {x}, {y}, scaling=scores))

    return c1 + c2 + c3

def fit_seq(log_var,A,B):
    count = 0
    for tr in log_var:
        for i in range(0,len(tr)-1):
            if (tr[i] in B) and (tr[i+1] in A):
                count += log_var[tr]
                break
    fit = 1-(count/sum(log_var.values()))
    return fit

def fit_exc(log_var,A,B):
    count = 0
    for tr in log_var:
        if set(tr).issubset(A) or set(tr).issubset(B):
            count += log_var[tr]
    fit = (count/sum(log_var.values()))
    return fit

def fit_loop(log_var,A,B,A_end,A_start):
    count = 0
    for tr in log_var:
        if len(tr)==0:
            continue
        if (tr[0] in B) or (tr[-1] in B):
            count += log_var[tr]
            continue
        for i in range(0,len(tr)-1):
            if (tr[i+1] in B) and (tr[i] in A):
                if (tr[i] not in A_end):
                    count += log_var[tr]
                break
            if (tr[i] in B) and (tr[i+1] in A):
                if (tr[i+1] not in A_start):
                    count += log_var[tr]
                break
    fit = 1 - (count / sum(log_var.values()))
    return fit


def cost_exc(net, A, B, scores):
    scores_toggle = toggle(scores)
    c1 = n_edges(net, A, B, scaling =scores_toggle)
    c1 += n_edges(net,B ,A, scaling =scores_toggle)
    return c1


def cost_par(net, A, B, sup, scores):
    c1 = 0
    c2 = 0
    for a in A:
        for b in B:
            c1 += max(0, scores[(a, b)] * (net.out_degree(a, weight='weight') * sup * net.out_degree(b, weight='weight')) / (
                        (sum([net.out_degree(p, weight='weight') for p in B])) + (sum([net.out_degree(p, weight='weight') for p in A]))) - n_edges(net, {a}, {b}, scaling=scores))
            c2 += max(0, scores[(b, a)] * (net.out_degree(b, weight='weight') * sup * net.out_degree(a, weight='weight')) / (
                        (sum([net.out_degree(p, weight='weight') for p in B])) + (sum([net.out_degree(p, weight='weight') for p in A]))) - n_edges(net, {b}, {a}, scaling=scores))

    return c1+c2


def cost_loop(net, A, B, sup, start_A, end_A, input_B, output_B, scores):
    scores_toggle = toggle(scores)

    flag_loop_valid = False

    if n_edges(net, B, start_A) != 0:
        if n_edges(net, end_A, B) != 0:
            flag_loop_valid = True
        else:
            return False
    else:
        return False

    BotoAs_P = n_edges(net, output_B, start_A)
    AetoBi_P = n_edges(net, end_A, input_B)
    M_P = max(BotoAs_P, AetoBi_P)



    c1 = n_edges(net, {'start'}, B, scaling=scores_toggle)
    c1 += n_edges(net, B, {'end'}, scaling=scores_toggle)

    c2 = n_edges(net, A - end_A, B, scaling= scores_toggle)

    c3 = n_edges(net, B, A - start_A, scaling=scores_toggle)

    c4 = 0
    if len(output_B) != 0:
        for a in start_A:
            for b in output_B:
                c4 += max(0, scores[(b, a)] * M_P * sup * (n_edges(net,{'start'},{a})/n_edges(net, {'start'}, start_A)) * (n_edges(net, {b}, start_A)/ n_edges(net, output_B, start_A))- n_edges(net, {b}, {a}, scaling=scores))

    c5 = 0
    if len(input_B) != 0:
        for a in end_A:
            for b in input_B:
               c5 +=  max(0, scores[(a,b)] * M_P * sup * (n_edges(net,{a}, {'end'})/n_edges(net, end_A, {'end'})) * (n_edges(net, end_A, {b})/ n_edges(net, end_A, input_B))- n_edges(net, {a}, {b}, scaling=scores))


    # if sup*M_P==0:
    #     return False
    # if (c4+c5)/(2*sup*M_P)>0.3:
    #     return False

    return c1 + c2 + c3 + c4 + c5

def visualisecpcm(cuts, ratio, size_par):
    cp = [x[2] for x in cuts]
    cm = [x[3] for x in cuts]
    tt = [str(x[1])+", "+str(x[0]) for x in cuts]
    diff = [x[2] - ratio * size_par * x[3] for x in cuts]
    color_fit = [x[5] for x in cuts]
    min_value = min(diff)
    min_index = diff.index(min_value)
    edge = [20]*len(diff)
    edge[min_index]= 100

    fig, ax = plt.subplots()
    s = ax.scatter(cp, cm, c=color_fit, cmap='inferno',s=edge)
    ax.set_xlabel(r'cp', fontsize=15)
    ax.set_ylabel(r'cm', fontsize=15)
    fig.colorbar(s, ax=ax)



    from matplotlib.widgets import Cursor
    # Defining the cursor
    cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                    color='r', linewidth=1)

    # cursor grid lines
    lnx = plt.plot([60, 60], [0, 1.5], color='black', linewidth=0.3)
    lny = plt.plot([0, 100], [1.5, 1.5], color='black', linewidth=0.3)
    lnx[0].set_linestyle('--')
    lny[0].set_linestyle('None')
    # annotation
    annot = ax.annotate("", xy=(0, 0), xytext=(5, 5), textcoords="offset points")
    annot.set_visible(False)
    # xy limits
    plt.xlim(min(cp) * 0.95, max(cp) * 1.05)
    plt.ylim(min(cm) * 0.95, max(cm) * 1.05)

    def hover(event):
        # check if event was in the axis
        if event.inaxes == ax:
            cont, ind = s.contains(event)
            if cont:
                # change annotation position
                annot.xy = (event.xdata, event.ydata)
                print((event.xdata, event.ydata))
                print("{}".format(', '.join([tt[n] for n in ind["ind"]])))
                # write the name of every point contained in the event
                annot.set_text("{}".format('\n '.join([tt[n] for n in ind["ind"]])))
                annot.set_visible(True)
                fig.canvas.draw()
            else:
                annot.set_visible(False)
        # else:
        #     lnx[0].set_visible(False)
        #     lny[0].set_visible(False)

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


def check_base_case(self, logP, logM, sup_thr, ratio, size_par):
    activitiesP = set(a for x in logP.keys() for a in x)

    if len(activitiesP) <= 1:
        base_check = True
        counter = logP[()]
        counterM = logM[()]
        len_logP = sum(logP.values())
        acc_contP = sum([len(x) * logP[x] for x in logP])
        len_logM = sum(logM.values())
        acc_contM = sum([len(x) * logM[x] for x in logM])

        # empty check
        if (counter == len_logP) or (len_logP == 0):
            self.detected_cut = 'empty_log'
            cut = ('none', 'empty_log', 'none', 'none')
        else:
            # xor check
            cost_single_exc = max(0, sup_thr * len_logP - counter) - ratio * size_par * max(0,sup_thr * len_logM - counterM)
            if (counter > (sup_thr / 2) * len_logP) and (cost_single_exc <= 0):
            # if (cost_single_exc <= 0):
                cut = (({activitiesP.pop()}, set()), 'exc', 'none', 'none')
            else:
                # loop check
                del logP[()]
                if acc_contP > 0:
                    p_prime_Lp = (len_logP - counter) / ((len_logP - counter) + acc_contP)
                else:
                    p_prime_Lp = 'nd'

                if acc_contM > 0:
                    p_prime_Lm = (len_logM - counterM) / ((len_logM - counterM) + acc_contM)
                else:
                    p_prime_Lm = 'nd'

                if p_prime_Lm != 'nd':
                    cost_single_loop = max(0, sup_thr/2 - abs(p_prime_Lp - 0.5)) - ratio * size_par * max(0,sup_thr/2 - abs(p_prime_Lm - 0.5))
                else:
                    cost_single_loop = max(0, sup_thr/2 - ratio * size_par * abs(p_prime_Lp - 0.5))

                if (abs(p_prime_Lp - 0.5) > sup_thr / 2) and (cost_single_loop <= 0):
                # if (cost_single_loop <= 0):
                    cut = (({activitiesP.pop()}, set()), 'loop1', 'none', 'none')
                else:
                    # single activity
                    self.detected_cut = 'single_activity'
                    cut = ('none', 'single_activity', 'none', 'none')
    else:
        base_check = False
        cut = "not_base"

    return base_check, cut




def get_all_connected_subgraphs(G):
    """Get all connected subgraphs by a recursive procedure"""
    # G = N.to_undirected(reciprocal=False, as_view=False)
    con_comp = [c for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)]

    def recursive_local_expand(node_set, possible, excluded, results, max_size):
        """
        Recursive function to add an extra node to the subgraph being formed
        """
        results.append(node_set)
        if len(node_set) == max_size:
            return
        for j in possible - excluded:
            new_node_set = node_set | {j}
            excluded = excluded | {j}
            new_possible = (possible | set(G.neighbors(j))) - excluded
            recursive_local_expand(new_node_set, new_possible, excluded, results, max_size)

    results = []
    for cc in con_comp:
        max_size = len(cc)

        excluded = set()
        for i in G:
            excluded.add(i)
            recursive_local_expand({i}, set(G.neighbors(i)) - excluded, excluded, results, max_size)

    results.sort(key=len)

    return results



def find_possible_partitions3(net,rules,sa,ea):

    def adj(node_set, net):
        adj_set = set()
        for node in node_set:
            adj_set = adj_set.union(set(net.neighbors(node)))
        return adj_set

    activity_list = set(net.nodes)-{'start','end'}
    queue = [(set(), {'start'})]
    # queue.append()
    visited = []
    valid = []
    # reserve_list =[]
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

            # new_state = add_SE(net, new_state)
            if new_state not in visited:
                new_adj = current[1].union(adj({x},net)) - new_state
                visited.append(new_state)


                B = activity_list - new_state
                if (len(B) == 0) or (len(B) == len(activity_list)):
                    queue.append((new_state, new_adj))
                    if (len(B) == 0) and (n_edges(net, set(en_net.keys()), set(st_net.keys()), scaling=None)>0):
                        na, block, n_conflicts = is_allowed(new_state, B, rules, st_net, en_net)
                        possible_cuts = {'loop_tau'} - na
                        if len(possible_cuts) > 0:
                            valid.append((new_state, B, possible_cuts))


                    continue
                B = add_SE(net, B)
                netA = net.subgraph(new_state)
                netB = net.subgraph(B)
                # BB = net.subgraph(B)
                # na = {}
                block = False

                # 'start' ~> netB
                if 'start' in netB:
                    startB = set(netB.nodes) - set(nx.descendants(netB, 'start')) - {'start','end'}
                else:
                    startB = set(netB.nodes)
                # 'end' ~> netA
                if 'end' in netA:
                    endA = set(netA.nodes) - set(nx.ancestors(netA, 'end')) - {'start','end'}
                else:
                    endA = set(netA.nodes)
                # 'end' ~> netB
                if 'end' in netB:
                    endB = set(netB.nodes) - set(nx.ancestors(netB, 'end')) - {'start', 'end'}
                else:
                    endB = set(netB.nodes)

                na, block, n_conflicts = is_allowed(new_state, B, rules,st_net,en_net)
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
                # else:
                #     reserve_list.append((n_conflicts,(new_state, B, r_p)))
                if block==False:
                    queue.append((new_state, new_adj))
    # if len(valid)==0:
    #     m_conf = min([n[0] for n in reserve_list])
    #     more = [c[1] for c in reserve_list if c[0]==m_conf]
    #     valid = valid + more
    #     print(f'conf:{m_conf}')
    return valid




def is_allowed(S1,S2,rules,st_net,en_net):
    exclude = []
    n_conf = 0
    block = False

    # if (S2-{'start', 'end'}) == {'A_Incomplete'}:
    #     print('hh')

    # o_S1 = sorted([rules['order'].index(x) for x in S1-{'start', 'end'} if x in rules['order']])
    # o_S2 = sorted([rules['order'].index(x) for x in S2-{'start', 'end'} if x in rules['order']])
    # if len(o_S1)>0 and len(o_S2)>0 and o_S1[-1]>o_S2[0]:
    #     exclude.append('seq')
    #     block=True



    s_count = 0
    e_count = 0
    for n in S2-{'start','end'}:
        if n in st_net:
            s_count += st_net[n]
        if n in en_net:
            e_count += en_net[n]
    if s_count>= 0.05 * sum(st_net.values()) or e_count>= 0.05 * sum(en_net.values()):
        n_conf+=1
        exclude.append('loop')

    EXISTENCE_acts = set([r for r in rules[EXISTENCE]])
    EXACTLY_ONE_acts = set([r for r in rules[EXACTLY_ONE]])

    if S1.intersection(EXACTLY_ONE_acts) == S1:
        n_conf += 1
        exclude.append('loop')

    if S2.intersection(EXISTENCE_acts) == S2:
        n_conf += 1
        exclude.append('loop')


    for r in rules[NONSUCCESSION]:
        if r[0] in S1 and r[1] in S2:
            n_conf += 1
            exclude.append('seq')
            # exclude.append('loop')
            exclude.append('par')
            # block = True
        elif r[0] in S2 and r[1] in S1:
            n_conf += 1
            exclude.append('par')
            exclude.append('loop')
            # block = True
        elif (r[0] in S1 and r[1] in S1) or (r[0] in S2 and r[1] in S2):
            exclude.append('loop')
            # exclude.append('loop_tau')


    for r in rules[RESPONDED_EXISTENCE]:
        if r[0] in S1 and r[1] in S2:
            n_conf += 1
            exclude.append('exc')
            # exclude.append('loop')
        elif r[0] in S2 and r[1] in S1:
            n_conf += 1
            exclude.append('exc')


    for r in rules[NONCOEXISTENCE]:
        if r[0] in S1 and r[1] in S2:
            exclude.append('par')
            exclude.append('seq')
            exclude.append('loop')
        elif r[0] in S2 and r[1] in S1:
            exclude.append('par')
            exclude.append('seq')
            exclude.append('loop')
        elif (r[0] in S1 and r[1] in S1) or (r[0] in S2 and r[1] in S2):
            # exclude.append('loop_tau')
            print("ss")

        # if (r[0] in S1 and r[1] in S1):
        #     block = True

    for r in rules[RESPONSE]:
        if (r[1], r[0]) not in rules[PRECEDENCE]:
            if r[0] in S1 and r[1] in S2:
                n_conf += 1
                exclude.append('exc')
                exclude.append('loop')
                exclude.append('par')
            if r[0] in S2 and r[1] in S1:
                n_conf += 1
                exclude.append('exc')
                exclude.append('seq')
                exclude.append('par')
                block = True
        else:
            if r[0] in S1 and r[1] in S2:
                n_conf += 1
                exclude.append('exc')
                exclude.append('loop')
                exclude.append('par')

            if r[0] in S2 and r[1] in S1:
                n_conf += 1
                exclude.append('exc')
                exclude.append('loop')
                exclude.append('par')



    for r in rules[PRECEDENCE]:
        if (r[1], r[0]) not in rules[RESPONSE]:
            if r[0] in S1 and r[1] in S2:
                n_conf += 1
                exclude.append('par')
                exclude.append('exc')
            if r[0] in S2 and r[1] in S1:
                n_conf += 1
                exclude.append('exc')
                exclude.append('seq')
                exclude.append('par')
                exclude.append('loop')
                block = True
        else:
            if r[0] in S1 and r[1] in S2:
                n_conf += 1
                exclude.append('par')
                exclude.append('exc')
                exclude.append('seq')
                exclude.append('loop')
            if r[0] in S2 and r[1] in S1:
                n_conf += 1
                exclude.append('exc')
                exclude.append('par')
                exclude.append('loop')



    return set(exclude), block, n_conf


def max_flow_graph(net):
    flow_graph = {}
    for x in net.nodes:
        for y in net.nodes:
            if (x != y):
                flow_graph[(x, y)] = nx.algorithms.flow.maximum_flow(net, x, y, capacity='weight')[0]
    return flow_graph



def noise_filtering(dfg0, nt):
    dfg = copy.deepcopy(dfg0)
    log_size = sum([dfg[x] for x in dfg if x[0] == 'start'])
    noisy_edges = sorted([(x,dfg[x]) for x in dfg if (dfg[x]/log_size) < nt], key=lambda z:z[1])
    net = generate_nx_graph_from_dfg(dfg0)
    for ne in noisy_edges:
        net_copy = copy.deepcopy(net)
        nodes_set = set(net_copy.nodes)
        net_copy.remove_edge(ne[0][0],ne[0][1])
        if (set(nx.ancestors(net_copy, 'end')) == nodes_set-{'end'}):
            if(set(nx.descendants(net_copy, 'start')) == nodes_set-{'start'}):
                del dfg[ne[0]]
                net = net_copy
    return dfg

def generate_nx_graph_from_dfg(dfg):
    dfg_acts = set()
    for x in dfg:
        dfg_acts.add(x[0])
        dfg_acts.add(x[1])
    G = nx.DiGraph()
    for act in dfg_acts:
        G.add_node(act)
    for edge in dfg:
        G.add_edge(edge[0], edge[1])
    return G