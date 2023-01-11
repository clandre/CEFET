import wfdb as wf
from matplotlib import pyplot as plt
import numpy as np
import os
from biosppy.signals import ecg
from scipy import signal

# 1 ) Achar nome dos dados
root = "mit-bih-arrhythmia-database-1.0.0/"
files = os.listdir(root)
files = list(filter(lambda x: x.endswith(".dat"), files))


for filename in files:

# 2) Importar dados e anotações para cada arquivo de dados
    datfile = f'{root}/{filename.split(".")[0]}'
    record = wf.rdsamp(datfile)
    annotation = wf.rdann(datfile, 'atr')

    print('Arquivo aberto:', filename.split(".")[0])
    print('Frequência utilizada:', record[1].get('fs'))
    print('Dimensões dos dados:', record[0].shape)
    print('Quantidade de Anotações:', len(annotation.num))

# 3) Pegar os valores do ECG e as anotações

    data = record[0].transpose()
    cat = np.array(annotation.symbol)
    rate = np.zeros_like(cat, dtype='float')
    for catid, catval in enumerate(cat):
        if catval in ('N','L','R'):
            rate[catid] = 0.0 
        elif catval in ('A','a','J','S','j'):
            rate[catid] = 1.0 
        elif catval in ('V','E'):
            rate[catid] = 2.0 
        elif catval == 'F':
            rate[catid] = 3.0 
        elif catval in ('Q', 'P','/','f','u'):
            rate[catid] = 4.0 
        else:
            rate[catid] == 100.0
    rates = np.zeros_like(data[0], dtype='float')
    rates[annotation.sample] = rate    

# 4) Processando os canais

    for channelid, channel in enumerate(data):
        chname = record[1].get('sig_name')[channelid]
        print('ECG channel type:', chname)

# 5) Gerando os R-PICOS com as anotações
        out = ecg.ecg(signal=channel, sampling_rate=record[1].get('fs'), show=False)
        rpeaks = np.zeros_like(channel, dtype='float')
        rpeaks[out['rpeaks']] = 1.0  

# 6) Retirando os batimentos cardiacos
        beatstoremove = np.array([0])
        beats = np.split(channel, out['rpeaks'])

# 7) Dividir em batimentos individuais
        for idx, idxval in enumerate(out['rpeaks']):
            firstround = idx == 0
            lastround = idx == len(beats) - 1        

            if (firstround or lastround):
                continue

            fromidx = 0 if idxval < 10 else idxval - 10
            toidx = idxval + 10
            catval = rates[fromidx:toidx].max()

# 8) Remover os picos sem classificação
            if catval == 100.0:
                beatstoremove = np.append(beatstoremove, idx)
                continue

# 9) Adicionar dados extras além do pico e normalização dos dados

            beats[idx] = np.append(beats[idx], beats[idx+1][:40])
            beats[idx] = (beats[idx] - beats[idx].min()) / beats[idx].ptp()

# 10) mudar a frequência de 360 para 125.
            newsize = int((beats[idx].size * 125 / 360) + 0.5)
            beats[idx] = signal.resample(beats[idx], newsize)

# 11) Pular os que são muito longos
            if (beats[idx].size > 187):
                beatstoremove = np.append(beatstoremove, idx)
                continue

# 12) Fazer o padding com zeros
            zerocount = 187 - beats[idx].size
            beats[idx] = np.pad(beats[idx], (0, zerocount), 'constant', constant_values=(0.0, 0.0))

# 13) Adicionar a classificação
            beats[idx] = np.append(beats[idx], catval)


# 14) Remover beats sem classifcação
        beatstoremove = np.append(beatstoremove, len(beats)-1)
        beats = np.delete(beats, beatstoremove)

        # Save to CSV file.
        savedata = np.array(list(beats[:]), dtype=np.float)
        outfn = 'data_ecg/'+filename.split(".")[0]+'_'+chname+'.csv'
        print('    Generating ', outfn)
        with open(outfn, "wb") as fin:
            np.savetxt(fin, savedata, delimiter=",", fmt='%f')



