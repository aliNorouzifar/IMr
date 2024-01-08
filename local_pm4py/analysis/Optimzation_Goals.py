from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

def apply(LPlus, LMinus, dfg_net, sa, ea):
    measures = {}
    print('hi')
    alp = dfg_alignment.apply(LPlus, dfg_net, sa, ea)
    fp_inf = replay_fitness.evaluate(alp, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    fp = fp_inf['averageFitness']
    roc_data = [('p',x['fitness']) for x in alp]


    # alp = alignments.apply_log(LPlus, net, i_m, i_f)
    # fp_inf = replay_fitness.evaluate(alp, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    # fp = fp_inf['averageFitness']


    ################################################################################
    # prec_Plus = precision_evaluator.apply(LPlus, net, i_m, i_f,
    #                                       variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    ################################################################################

    alm = dfg_alignment.apply(LMinus, dfg_net, sa, ea)
    fm_inf = replay_fitness.evaluate(alm, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    fm = fm_inf['averageFitness']
    roc_data += [('n',x['fitness']) for x in alm]

    # alm = alignments.apply_log(LMinus, net, i_m, i_f)
    # fm_inf = replay_fitness.evaluate(alm, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    # fm = fm_inf['averageFitness']


    measures['acc'] = round(fp - fm,2)
    measures['F1'] = round(2 * ((fp*(1-fm))/(fp+(1-fm))),2)
    # measures['precision'] = prec_Plus
    measures['fitP'] = round(fp,2)
    measures['fitM'] = round(fm,2)
    measures['roc_data'] = roc_data


    return measures


def apply_petri(LPlus, LMinus, net, i_m, i_f):
    measures = {}


    alp = alignments.apply_log(LPlus, net, i_m, i_f)
    fp_inf = replay_fitness.evaluate(alp, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    fp = fp_inf['averageFitness']
    fp_pef = fp_inf['percentage_of_fitting_traces']/100
    # roc_data = [('p', x['fitness']) for x in alp]


    ################################################################################
    prec_Plus = precision_evaluator.apply(LPlus, net, i_m, i_f,
                                          variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    ################################################################################


    alm = alignments.apply_log(LMinus, net, i_m, i_f)
    fm_inf = replay_fitness.evaluate(alm, variant=replay_fitness.Variants.ALIGNMENT_BASED)
    fm = fm_inf['averageFitness']
    fm_pef = fm_inf['percentage_of_fitting_traces']/100
    # roc_data += [('n', x['fitness']) for x in alm]


    measures['acc'] = round(fp - fm,2)
    measures['F1'] = round(2 * ((fp*(1-fm))/(fp+(1-fm))),2)
    measures['precision'] = round(prec_Plus,2)
    measures['fitP'] = round(fp,2)
    measures['fitM'] = round(fm,2)
    measures['acc_perf'] = round(fp_pef - fm_pef,2)
    measures['F1_perf'] = round(2 * ((fp_pef*(1-fm_pef))/(fp_pef+(1-fm_pef))),2)

    TP = fp_pef*len(LPlus)
    FP = fm_pef*len(LMinus)
    FN = (1-fp_pef)*len(LPlus)
    TN = (1-fm_pef)*len(LMinus)
    measures['acc_ML'] = (TP+TN)/(TP+TN+FP+FN)
    if (TP + FP) != 0:
        measures['prc_ML'] = TP / (TP + FP)
    else:
        measures['prc_ML'] = 'ignore'

    if (TP + FN) != 0:
        measures['rec_ML'] = TP / (TP + FN)
    else:
        measures['rec_ML'] = 'ignore'
    # measures['roc_data'] = roc_data


    return measures



