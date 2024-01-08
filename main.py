
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from local_pm4py.analysis import gui
import time
import pm4py
from pm4py.objects.log import obj as log_instance
from local_pm4py import discovery


support, ratio, LPlus_LogFile, LMinus_LogFile= gui.input()
logP = xes_importer.apply(LPlus_LogFile)

if LMinus_LogFile != '':
    logM = xes_importer.apply(LMinus_LogFile)
else:
    logM = log_instance.EventLog()
    logM.append(log_instance.Trace())


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

allowed_templates = {RESPONDED_EXISTENCE, RESPONSE,COEXISTENCE, PRECEDENCE, NONCOEXISTENCE,EXISTENCE, EXACTLY_ONE,INIT,END,NONSUCCESSION}

conf = 1
print(f'conf: {conf}')
rules = pm4py.discover_declare(logP, min_support_ratio=1 - conf, min_confidence_ratio=conf,allowed_templates=allowed_templates)
for r in allowed_templates:
    if r not in rules.keys():
        rules[r] = []


ratio = 0.0
start = time.time()
net, initial_marking, final_marking = discovery.apply_bi(logP,logM, sup=support, ratio=ratio, size_par=len(logP)/max(1,len(logM)),rules = rules)
end = time.time()

print("run time:")
print(end-start)

parameters = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT:"pdf"}
gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters)
pn_visualizer.view(gviz)

# file_name = "petri_r"+str(ratio)+"_s"+str(support)
# pm4py.write_pnml(net, initial_marking, final_marking, file_name)


