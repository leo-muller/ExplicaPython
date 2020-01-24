import scipy.stats as si

import numpy as np
import pandas as pd

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib

def euro_vanilla_call(S, K, T, rd, rf, sigma):
    """
    Modelo de Garman Kohlhagen
    :param S:
    :param K:
    :param T:
    :param rd: Taxa de juros
    :param rf: Taxa de adm
    :param sigma:
    :return:
    """


    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (rd - rf - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    call = (np.exp(-rf * T)*S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-rd * T) * si.norm.cdf(d2, 0.0, 1.0))

    return call

r_cdi = np.log(1+5.0/100)
r_adm = np.log(1+2.0/100)
n_sim = 10000
sharpe = [0,0.7,1.5]
p_fee = 0.2

cost_list = pd.DataFrame(index = [float(i/200) for i in range(1,40)], data = 0,columns = sharpe)
sharpe_cliente = pd.DataFrame(index = [float(i/200) for i in range(1,40)], data = 0,columns = sharpe)
cdi_6m = np.exp(r_cdi / 2)
cdi_1y = np.exp(r_cdi)

for vol_i in cost_list.index:
    for sharpe_i in cost_list.columns:
        S1 = np.exp(np.random.normal((r_cdi - r_adm + sharpe_i * vol_i) / 2, vol_i / np.sqrt(2), n_sim))
        pfee_1 = [np.max((0, S_i - cdi_6m)) * p_fee for S_i in S1]
        S1 = [S_i - pf_i for S_i, pf_i in zip(S1,pfee_1)]

        S2 = S1 * np.exp(np.random.normal((r_cdi - r_adm + sharpe_i * vol_i) / 2, vol_i / np.sqrt(2), n_sim))
        S2_marca_agua = [np.max((cdi_6m, S_i)) * cdi_6m for S_i in S1]
        pfee_2 = [np.max((0, S_i - S_adj)) * p_fee for S_i, S_adj in zip(S1, S2_marca_agua)]
        S2 = [S_i - pf_i for S_i, pf_i in zip(S2, pfee_2)]
        alpha = [S_i - cdi_1y for S_i in S2]

        cost_list.loc[vol_i, sharpe_i]=np.exp((r_adm + np.average(pfee_1)+np.average(pfee_2)))-1
        sharpe_cliente.loc[vol_i, sharpe_i]=np.mean(alpha)/np.std(alpha)

# Plot
yLim = [float(cost_list.min().min()),float(cost_list.max().max())]
yLim2 = [np.max(( -3,    float(sharpe_cliente.min().min()))),float(sharpe_cliente.max().max())]
xLim = [0,float(cost_list.index[-1])]
color_list = ["#1f2a65", "#c05131","#a92a31"]
matplotlib.rcParams['animation.embed_limit'] = 40
plt.style.use('seaborn-pastel')
fig,ax = plt.subplots(figsize=(10,5))

plot_n = len(cost_list)
t_i_max = plot_n + 10

def plot_frame(t_i):
    if t_i < t_i_max:
        plot_cost(t_i)
    else:
        plot_sharpe(t_i-t_i_max)


def plot_cost(t_i):
    ax.clear()
    ax.set_xlim(xLim[0], xLim[1])
    ax.set_ylim(yLim[0], yLim[1])
    plt.xlabel('Volatilidade')
    plt.ylabel('Custo Total')
    ax.set_title('Custo do Fundo',fontsize=16)
    for idx_i in range(3):
        ax.plot(cost_list.iloc[:t_i, idx_i], label=f"Sharpe {cost_list.columns[idx_i]:.1f}", color=color_list[idx_i])
    plt.legend(loc='upper left')

def plot_sharpe(t_i):
    ax.clear()
    ax.set_xlim(xLim[0], xLim[1])
    ax.set_ylim(yLim2[0], yLim2[1])
    plt.xlabel('Volatilidade')
    plt.ylabel('Sharpe Liquido Cliente')
    ax.set_title('Risco x Retorno do Fundo',fontsize=16)
    for idx_i in range(3):
        ax.plot(sharpe_cliente.iloc[:t_i, idx_i], label=f"Sharpe {sharpe_cliente.columns[idx_i]:.1f}", color=color_list[idx_i])
    plt.legend(loc='lower right')

# plt.rcParams['animation.ffmpeg_path'] = "K:\\Users\\SUFIX\\Leo\\Git\\ExplicaPython\\ffmpeg\\bin\\ffmpeg"
animator  = FuncAnimation(fig,plot_frame,frames=range(0,t_i_max+plot_n),interval = 25)
# Writer = animation.writers['ffmpeg']
writer = animation.FFMpegWriter(fps=12, metadata=dict(artist='Me'), bitrate=1800)
animator.save('fund_cost.mp4',writer=writer)






