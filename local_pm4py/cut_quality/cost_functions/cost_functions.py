from local_pm4py.functions.functions import n_edges


def cost_seq(net, A, B, sup, flow):
    c1 = n_edges(net, B, A)

    c2 = 0
    for x in A:
        for y in B:
            c2 += max(0, net.out_degree(x, weight='weight') * sup * (net.out_degree(y, weight='weight') / (
                        sum([net.out_degree(p, weight='weight') for p in B]) + sum([net.out_degree(p, weight='weight') for p in A]))) - flow[(x, y)])
    c3 = 0

    return c1 + c2 + c3


def cost_exc(net, A, B):
    c1 = n_edges(net, A, B)
    c1 += n_edges(net,B ,A)
    return c1


def cost_par(net, A, B, sup):
    c1 = 0
    c2 = 0
    for a in A:
        for b in B:
            c1 += max(0, (net.out_degree(a, weight='weight') * sup * net.out_degree(b, weight='weight')) / (
                        (sum([net.out_degree(p, weight='weight') for p in B])) + (sum([net.out_degree(p, weight='weight') for p in A]))) - n_edges(net, {a}, {b}))
            c2 += max(0, (net.out_degree(b, weight='weight') * sup * net.out_degree(a, weight='weight')) / (
                        (sum([net.out_degree(p, weight='weight') for p in B])) + (sum([net.out_degree(p, weight='weight') for p in A]))) - n_edges(net, {b}, {a}))

    return c1+c2


def cost_loop(net, A, B, sup, start_A, end_A, input_B, output_B):

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



    c1 = n_edges(net, {'start'}, B)
    c1 += n_edges(net, B, {'end'})

    c2 = n_edges(net, A - end_A, B)

    c3 = n_edges(net, B, A - start_A)

    c4 = 0
    if len(output_B) != 0:
        for a in start_A:
            for b in output_B:
                c4 += max(0, M_P * sup * (n_edges(net,{'start'},{a})/n_edges(net, {'start'}, start_A)) * (n_edges(net, {b}, start_A)/ n_edges(net, output_B, start_A))- n_edges(net, {b}, {a}))

    c5 = 0
    if len(input_B) != 0:
        for a in end_A:
            for b in input_B:
               c5 +=  max(0, M_P * sup * (n_edges(net,{a}, {'end'})/n_edges(net, end_A, {'end'})) * (n_edges(net, end_A, {b})/ n_edges(net, end_A, input_B))- n_edges(net, {a}, {b}))


    # if sup*M_P==0:
    #     return False
    # if (c4+c5)/(2*sup*M_P)>0.3:
    #     return False

    return c1 + c2 + c3 + c4 + c5