



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
                    # cost_single_loop = max(0, sup_thr/2 - ratio * size_par * abs(p_prime_Lp - 0.5))
                    cost_single_loop = max(0, sup_thr / 2 - abs(p_prime_Lp - 0.5))

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