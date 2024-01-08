import PySimpleGUI as sg

def input():
    sg.theme("DarkGrey7")
    layout = [  [sg.Text('Settings', font='Any 20')],
                    [sg.Text('support ', justification='center', font='Any 14'), sg.Slider(range=(0,1), resolution=0.1, orientation='h', border_width =2, s=(100,20), key='-sup-')],
                    [sg.Text('ratio      ', font='Any 14') , sg.Slider(range=(0,1), resolution=0.1, orientation='h', border_width =2,s=(100,20), key='-r-')],
                    [sg.Text('Desirable Log(.xes)    ', font='Any 14'), sg.FileBrowse(key="-Desirable Log-", font='Any 14')],
                    [sg.Text('Undesirable Log(.xes)', font='Any 14'), sg.FileBrowse(key="-Undesirable Log-", font='Any 14')],
                    [sg.Button('Run IMbi', font='Any 14')]]

    window = sg.Window('Inputs', layout, size=(600, 300))

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Run IMbi":
            break

    window.close()
    return float(values["-sup-"]), float(values["-r-"]), values["-Desirable Log-"], values["-Undesirable Log-"]


def output(acc,F1,acc_s,F1_s,fitp,prc,time):
    layout = [[sg.Text("Report", font='Any 20')],
              [sg.Text("alignment accuracy:     " + acc + "%", font=("Consolas", 14))],
              [sg.Text("alignment F1-score:     " + F1 + "%", font=("Consolas", 14))],
              [sg.Text("trace accuracy:         " + acc_s + "%", font=("Consolas", 14))],
              [sg.Text("trace F1-score:         " + F1_s + "%", font=("Consolas", 14))],
              [sg.Text("alignment fitness (L+): " + fitp + "%", font=("Consolas", 14))],
              [sg.Text("precision:              " + prc + "%", font=("Consolas", 14))],
              [sg.Text("execution time:         " + time + "sec.", font=("Consolas", 14))],
              [sg.Button("Exit")]]

    window = sg.Window("Outputs", layout, size=(400, 300))
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
    window.close()

