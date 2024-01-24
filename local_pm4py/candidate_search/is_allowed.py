

EXISTENCE = "existence"
EXACTLY_ONE = "exactly_one"
INIT = "init"
END = "end"
RESPONDED_EXISTENCE = "responded_existence"
RESPONSE = "response"
PRECEDENCE = "precedence"
COEXISTENCE = "coexistence"
NONCOEXISTENCE = "noncoexistence"
NONSUCCESSION = "nonsuccession"
ATMOST_ONE = "atmost1"

def is_allowed(S1,S2,rules,st_net,en_net):
    exclude = []
    block = False

    for r in rules[ATMOST_ONE]:
        if r in S1:
            exclude.append('loop')
            exclude.append('loop_tau')
        elif r in S2:
            exclude.append('loop')
            exclude.append('loop_tau')

    for r in rules[EXISTENCE]:
        if r in S1:
            exclude.append('exc')
        elif r in S2:
            exclude.append('exc')
            exclude.append('loop')



    for r in rules[NONSUCCESSION]:
        if r[0] in S1 and r[1] in S2:
            exclude.append('seq')
            exclude.append('loop')
            exclude.append('loop_tau')
            exclude.append('par')
            # block = True
        elif r[0] in S2 and r[1] in S1:
            exclude.append('par')
            exclude.append('loop')
            exclude.append('loop_tau')
        elif r[0] in S1 and r[1] in S1:
            exclude.append('loop')
            exclude.append('loop_tau')
        elif r[0] in S2 and r[1] in S2:
            exclude.append('loop')
            exclude.append('loop_tau')



    for r in rules[RESPONDED_EXISTENCE]:
        if r[0] in S1 and r[1] in S2:
            exclude.append('exc')
            exclude.append('loop')
        elif r[0] in S2 and r[1] in S1:
            exclude.append('exc')


    for r in rules[NONCOEXISTENCE]:
        if r[0] in S1 and r[1] in S2:
            exclude.append('par')
            exclude.append('seq')
            exclude.append('loop')
            exclude.append('loop_tau')
        elif r[0] in S2 and r[1] in S1:
            exclude.append('par')
            exclude.append('seq')
            exclude.append('loop')
            exclude.append('loop_tau')
        elif r[0] in S1 and r[1] in S1:
            exclude.append('loop')
            exclude.append('loop_tau')
        elif r[0] in S2 and r[1] in S2:
            exclude.append('loop')
            exclude.append('loop_tau')


    for r in rules[COEXISTENCE]:
        if r[0] in S1 and r[1] in S2:
            exclude.append('exc')
            exclude.append('loop')
        elif r[0] in S2 and r[1] in S1:
            exclude.append('exc')
            exclude.append('loop')


    for r in rules[RESPONSE]:
        if (r[1], r[0]) not in rules[PRECEDENCE]:
            if r[0] in S1 and r[1] in S2:
                exclude.append('exc')
                exclude.append('loop')
                exclude.append('par')
            if r[0] in S2 and r[1] in S1:
                exclude.append('exc')
                exclude.append('seq')
                exclude.append('par')
                block = True

        else:
            if r[0] in S1 and r[1] in S2:
                exclude.append('exc')
                exclude.append('loop')
                exclude.append('par')
            if r[0] in S2 and r[1] in S1:
                exclude.append('exc')
                exclude.append('seq')
                exclude.append('par')

    for r in rules[PRECEDENCE]:
        if (r[1], r[0]) not in rules[RESPONSE]:
            if r[0] in S1 and r[1] in S2:
                exclude.append('par')
                exclude.append('exc')

            if r[0] in S2 and r[1] in S1:
                exclude.append('exc')
                exclude.append('loop')
                exclude.append('seq')
                exclude.append('par')

                block = True
        else:
            if r[0] in S1 and r[1] in S2:
                exclude.append('par')
                exclude.append('exc')

            if r[0] in S2 and r[1] in S1:
                exclude.append('exc')
                exclude.append('loop')
                exclude.append('seq')
                exclude.append('par')

    return set(exclude), block

