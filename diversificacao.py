"""
A briga dos dois maiores gestores da historia: diversificação.

Buffett é claro, “Diversification is a protection against ignorance,[It] makes very little sense for those who know what they’re doing.”

Em contrapartida "Renaissance has gathered, the firm only profits on barely more than 50 percent of its trades, a sign of how challenging it is to try to beat the market—and how foolish it is for most investors to try."

São duas formas de investir, se dedicar muita a analisar poucos investimenos onde você tenha alta acertividade. Ou procurar investimentos marginalmente melhnores que jogar moeda e diversificar.

Qual a melhor? Depende, vamos supor que o Buffet faça 5 investimentos por ano. Supondo que o Simons acerte 51% do tempo, quantos trades ele tem que fazer para barrar o Buffet?

"""
import numpy as np
import pandas as pd
ret_pos = 0.1

def sharpe(n,p):
    return np.sqrt(n)*(2*p-1)/(2*np.sqrt(p*(1-p)))

def n_for_sharpe(p,n2,p2):
    sharpe_target = sharpe(n2,p2)
    fac_aux = (2*p-1)/(2*np.sqrt(p*(1-p)))
    return (sharpe_target/fac_aux)**2

p_jim  = 0.51
n_buffet = [3,5,10,20]

n_plot = 50
n_min = .55
n_max = .80
idx_aux = [n_min + (n_max-n_min)*i/n_plot for i in range(n_plot+1)]

trades_info = pd.DataFrame(data=0,columns=n_buffet,index = idx_aux)
sharpe_info = pd.DataFrame(data=0,columns=n_buffet,index = idx_aux)

for col_i in range(len(n_buffet)):
    trades_info.iloc[:,col_i]=[n_for_sharpe(p_jim,n_buffet[col_i],p2) for p2 in trades_info.index]
    sharpe_info.iloc[:,col_i]=[sharpe(n_buffet[col_i],p2) for p2 in sharpe_info.index]

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib

# Plot
xLim = [trades_info.index[0],trades_info.index[-1]]
# color_list = ["#1f2a65", "#c05131"]
matplotlib.rcParams['animation.embed_limit'] = 100
plt.style.use('seaborn-pastel')
fig,ax = plt.subplots(figsize=(10,5))

row_max = len(trades_info)
row_aux = int(row_max*1.2)
row_max -=1

def plot_frame(t_i):
    col_i, row_i = divmod(t_i,row_aux)
    row_i = np.min((row_i,row_max))+1
    y_max = trades_info.iloc[-1,col_i]
    ax.clear()
    ax.set_xlim(xLim[0], xLim[1])
    ax.set_ylim(0, y_max)
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.ylabel('Trades necessários para o Jim Simons')
    plt.xlabel('Assertividade do Buffet')
    for col_j in range(col_i+1):
        if col_j == col_i:
            ax.plot(trades_info.iloc[:row_i, col_j], label=f"{n_buffet[col_j]} Trades")
        else:
            ax.plot(trades_info.iloc[:, col_j], label=f"{n_buffet[col_j]} Trades")
        plt.legend(loc='upper left',title="Número de trades do Buffet:")


plt.rcParams['animation.ffmpeg_path'] = "K:\\Users\\SUFIX\\Leo\\Git\\ExplicaPython\\ffmpeg\\bin\\ffmpeg"

animator  = FuncAnimation(fig,plot_frame,frames=range(0,row_aux*trades_info.shape[1]),interval = 200)
# Writer = animation.writers['ffmpeg']
writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
animator.save('jim_buffet.mp4',writer=writer)














