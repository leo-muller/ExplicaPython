"""
Escolhendo fundos usando Sharpe ou retorno
"""

import numpy as np
import pandas as pd

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(52)

n_carteira = 5
fund_list = {
    'Bom':{'Sharpe':1,'Quant':10,'Vol':6},
    'Louco':{'Sharpe':0,'Quant':20,'Vol':15},
    'Ruido':{'Sharpe':0,'Quant':20,'Vol':6}
}

ret_fact = 52*100
sd_fact = np.sqrt(52)*100
pos0 = 0

for idx in fund_list.keys():
    fund_list[idx]['Ret'] = fund_list[idx]['Vol']*fund_list[idx]['Sharpe']/ret_fact
    fund_list[idx]['Vol']=fund_list[idx]['Vol']/sd_fact
    posN = pos0 + fund_list[idx]['Quant']
    fund_list[idx]['Range']=list(range(pos0,posN))
    pos0 = posN

n_weeks = 52 * 30
ret_win = 52

funds_ts = pd.DataFrame({f"{fund_label}_{pos}":
                             np.random.normal(fund_info['Ret'], fund_info['Vol'], n_weeks)
                         for fund_label, fund_info in fund_list.items()
                         for pos in range(1, fund_info['Quant'] + 1)
                         })

retorno_ts = funds_ts.rolling(ret_win).mean()[ret_win:]
sharpe_ts = retorno_ts/(funds_ts.rolling(ret_win).std()[ret_win:])
funds_ts = funds_ts[ret_win:]

retorno_ts = retorno_ts.rank(axis=1, ascending = False)
sharpe_ts = sharpe_ts.rank(axis=1, ascending = False)

carteira_ts_sharpe = pd.DataFrame(0,columns=['retorno']+[nome for nome in fund_list.keys()],index = range(len(sharpe_ts)))
carteira_ts_ret = pd.DataFrame(0,columns=['retorno']+[nome for nome in fund_list.keys()],index = range(len(sharpe_ts)))

for i in range(len(sharpe_ts)-1):
    valid_sharpe = (sharpe_ts.iloc[i]<(n_carteira+0.1)).values
    carteira_ts_sharpe.iloc[i+1,0]=funds_ts.iloc[i+1,valid_sharpe].mean()

    valid_return = (retorno_ts.iloc[i]<(n_carteira+0.1)).values
    carteira_ts_ret.iloc[i + 1, 0] = funds_ts.iloc[i + 1, valid_return].mean()

    for pos_i, key_i in enumerate(fund_list.keys(), 1):
        carteira_ts_sharpe.iloc[i + 1, pos_i] = np.sum(valid_sharpe[fund_list[key_i]['Range']])
        carteira_ts_ret.iloc[i + 1, pos_i] = np.sum(valid_return[fund_list[key_i]['Range']])

print(carteira_ts_sharpe.mean())
print(carteira_ts_ret.mean())

# Rolling
win_ma = 26
carteira_ts_sharpe['Ret_MV'] =52*carteira_ts_sharpe.loc[:,'retorno'].rolling(len(carteira_ts_sharpe),min_periods=win_ma).mean()
carteira_ts_sharpe['Sharpe_MV'] =carteira_ts_sharpe['Ret_MV'] /(np.sqrt(52)*carteira_ts_sharpe.loc[:,'retorno'].rolling(len(carteira_ts_sharpe),min_periods=win_ma).std())
carteira_ts_sharpe['Sharpe_MV'] = carteira_ts_sharpe['Sharpe_MV'].bfill()
carteira_ts_sharpe['Fund_Prop'] = carteira_ts_sharpe.iloc[:,1].rolling(len(carteira_ts_sharpe),min_periods=win_ma).mean()/n_carteira
carteira_ts_sharpe.retorno = carteira_ts_sharpe.retorno.cumsum()

carteira_ts_ret['Ret_MV'] =52*carteira_ts_ret.loc[:,'retorno'].rolling(len(carteira_ts_sharpe),min_periods=win_ma).mean()
carteira_ts_ret['Sharpe_MV'] =carteira_ts_ret['Ret_MV'] /(np.sqrt(52)*carteira_ts_sharpe.loc[:,'retorno'].rolling(len(carteira_ts_sharpe),min_periods=win_ma).std())
carteira_ts_ret['Sharpe_MV'] = carteira_ts_ret['Sharpe_MV'].bfill()
carteira_ts_ret['Fund_Prop'] = carteira_ts_ret.iloc[:,1].rolling(len(carteira_ts_sharpe),min_periods=win_ma).mean()/n_carteira
carteira_ts_ret.retorno = carteira_ts_ret.retorno.cumsum()

# Plot
yLim = [np.min((carteira_ts_sharpe.retorno.min(),carteira_ts_ret.retorno.min())), 1.07*np.max((carteira_ts_sharpe.retorno.max(),carteira_ts_ret.retorno.max()))]
xLim = [0,len(carteira_ts_sharpe)]
color_list = ["#1f2a65", "#c05131"]
matplotlib.rcParams['animation.embed_limit'] = 40
plt.style.use('seaborn-pastel')
fig,ax = plt.subplots(figsize=(10,5))
# ax.ylabel('Rentabilidade acumulada')
# ax.xlabel('Tempo')

dY = yLim[1]-yLim[0]
dX = xLim[1]-xLim[0]

pos1 = [0.01*dX,0.85*dY]
pos2 = [0.01*dX,0.93*dY]
fund_name = list(fund_list.keys())[0]

addX = [0.3*dX,0.54*dX]

def plot_frame(t_i):
    ax.clear()
    ax.set_xlim(xLim[0], xLim[1])
    ax.set_ylim(yLim[0], yLim[1])
    ax.plot(carteira_ts_ret.iloc[:t_i, 0], label="Retorno", color=color_list[0])
    ax.plot(carteira_ts_sharpe.iloc[:t_i, 0], label="Sharpe", color=color_list[1])
    plt.text(pos1[0], pos1[1], "RESULTADO", color=color_list[0], fontweight='bold', size=18)
    plt.text(pos2[0], pos2[1], "CONSISTENCIA", color=color_list[1], fontweight='bold', size=18)
    if t_i > win_ma:
        plt.text(pos1[0]+addX[0],pos1[1],f"Sharpe: {carteira_ts_ret.Sharpe_MV.iloc[t_i]:4.02f}",color=color_list[0],fontweight='bold',size=16)
        plt.text(pos1[0]+addX[1],pos1[1],f"{fund_name}: {100*carteira_ts_ret.Fund_Prop.iloc[t_i]:.0f}%",color=color_list[0],fontweight='bold',size=16)
        plt.text(pos2[0]+addX[0],pos2[1],f"Sharpe: {carteira_ts_sharpe.Sharpe_MV.iloc[t_i]:4.02f}",color=color_list[1],fontweight='bold',size=16)
        plt.text(pos2[0]+addX[1],pos2[1],f"{fund_name}: {100*carteira_ts_sharpe.Fund_Prop.iloc[t_i]:.0f}%",color=color_list[1],fontweight='bold',size=16)


# plt.rcParams['animation.ffmpeg_path'] = "K:\\Users\\SUFIX\\Leo\\Git\\ExplicaPython\\ffmpeg\\bin\\ffmpeg"
animator  = FuncAnimation(fig,plot_frame,frames=range(0,xLim[1],5),interval = 25)
# Writer = animation.writers['ffmpeg']
writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
animator.save('sharpe_vs_ret.mp4',writer=writer)















