import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib

class risk_parity:
    def __init__(self, covariance, portfolio=None):
        self.covariance = covariance
        if portfolio is not None:
            self.portfolio = portfolio

    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self,covariance):
        self._n_assets = covariance.shape[0]
        self._covariance = covariance
        self._port_std = None

    @property
    def portfolio(self):
        return self._portfolio

    @portfolio.setter
    def portfolio(self,portfolio):
        self._portfolio = portfolio
        self.basic_calculation()

    @property
    def risk_contribution(self):
        return self._risk_contribution

    @property
    def risk_total(self):
        return self._calc_sd_all

    def basic_calculation(self):
        self._calc_sigma_p = self._covariance @ self._portfolio
        self._calc_sd_all = float(np.sqrt(self._portfolio.T @ self._calc_sigma_p))
        self._risk_contribution = self._calc_sigma_p*(self._portfolio/self._calc_sd_all)

    def calc_hessian_sd(self):
        calc_sigma_p_adj = self._calc_sigma_p/(self._calc_sd_all**3.0)
        hess_sd = self._covariance/self._calc_sd_all- np.array([
            [self._calc_sigma_p[i,0]*calc_sigma_p_adj[j,0] for i in range(self._n_assets)]
            for j in range(self._n_assets)
        ])
        return hess_sd

    def calc_hessian_rb(self):
        hess_sd = self.calc_hessian_sd()
        d_sigma_dx = self._calc_sigma_p / self._calc_sd_all

        hess_rb = [
            [-self._portfolio[i,0]*hess_sd[i,j]-(d_sigma_dx[i,0] if i==j else 0) for i in range(self._n_assets)]
            for j in range(self._n_assets)
        ]
        return hess_rb

    def compute_rb(self,rb_target=None):

        if rb_target is None:
            rb_target = np.ones((self._n_assets,1))

        # Chute inicial
        self.portfolio = rb_target / np.reshape(np.sqrt(np.diagonal(self._covariance)),(self._n_assets,1))

        ERRO_MAX = 1e-8
        erro = rb_target- self.risk_contribution
        erro_sum = np.sum(erro**2)
        n_count = 0

        while erro_sum > ERRO_MAX:
            hess = self.calc_hessian_rb()
            portfolio_delta = np.linalg.solve(hess,-erro)
            self.portfolio = self.portfolio + portfolio_delta
            erro_old = erro_sum
            erro = rb_target - self.risk_contribution
            erro_sum = np.sum(erro ** 2)

            if erro_sum> erro_old:
                print(f"Loop: {n_count:2} Erro: {erro_sum:7.3f} (Aumentando Erro)")
                self.portfolio = self.portfolio - 0.9* portfolio_delta
                erro = rb_target - self.risk_contribution
            else:
                print(f"Loop: {n_count:2} Erro: {erro_sum:7.3f}")
            if n_count > 10:
                break
            else:
                n_count+=1


if __name__ == '__main__':

    # Index
    index_ts = pd.read_csv('K:\\Users\SUFIX\Leo\Git\QuantBackEnd\Excel\Index_TS.csv').set_index('Unnamed: 0')
    col_raw = ['IMA-B','IRF-M','Ibovespa','Real','CDI']
    index_ts.columns = col_raw
    cdi_ts = index_ts.iloc[:,-1]
    index_ts = index_ts.iloc[:,:-1]

    n_assets = 4
    peso  = np.reshape(np.array([0.3,0.3,0.3,0.1]),(n_assets,1))
    peso = peso/np.sum(peso[:3,0])
    win = 52
    ret_ts = np.log(index_ts).diff()

    teste = None

    PnL = pd.DataFrame(index=ret_ts.index, data={'PnL': [1],'PnL Fixed':[1]})
    for pos_i in range(win + 1, ret_ts.shape[0]):
        pos_0 = pos_i - win
        cov_i = (ret_ts.iloc[pos_0:pos_i,:]).cov()
        if teste is None:
            teste = risk_parity(cov_i.values, peso)
        teste.covariance = cov_i.values
        teste.compute_rb(peso)
        erro = np.sum(np.abs(peso - teste.risk_contribution))
        if erro < 0.05:
            port = teste.portfolio/np.sum(teste.portfolio[:3,0])
        ret_asset = index_ts.iloc[pos_i]/index_ts.iloc[pos_i-1]-1
        ret_i=np.sum(port[:,0]*ret_asset)
        PnL.iloc[pos_i,0]=PnL.iloc[pos_i-1,0]*(1+ret_i)
        ret_i = np.sum(peso[:, 0] * ret_asset)
        PnL.iloc[pos_i,1]=PnL.iloc[pos_i-1,1]*(1+ret_i)


    index_ts = index_ts.iloc[win:]
    PnL = PnL.iloc[win:]
    cdi_ts = cdi_ts.iloc[win:]

    cdi_ts = cdi_ts / cdi_ts.iloc[0]
    index_ts = index_ts/index_ts.iloc[0]
    index_ts.iloc[:,3]=index_ts.iloc[:,3]*cdi_ts.values

    ret_cdi = cdi_ts.values[1:]/cdi_ts.values[:-1]
    ret_assets = np.array([index_ts.iloc[1:,k].values/index_ts.values[:-1,k]-ret_cdi for k in range(4)])
    ret_opt = [PnL.iloc[1:,i].values/PnL.iloc[:-1,i].values-ret_cdi for i in range(2)]

    name_aux = []
    sharpe = []

    def compute_sharpe(x):
        mean_i = np.average(x)
        std_i = np.std(x)
        sharpe_i = mean_i * np.sqrt(52) / std_i
        return sharpe_i

    new_index = []
    for i in range(4):
        sharpe_i =compute_sharpe(ret_assets[i])
        new_index.append(f"{index_ts.columns[i]} ({sharpe_i:.2f})")
        sharpe.append(sharpe_i)

    index_ts.columns = new_index
    index_ts.iloc[:,np.argsort(sharpe)]

    sharpe_i = compute_sharpe(ret_opt[0])
    index_ts[f"Carteira Risco ({sharpe_i:.2f})"]=PnL.values[:,0]
    sharpe_i = compute_sharpe(ret_opt[1])
    index_ts[f"Carteira Tamanho ({sharpe_i:.2f})"]=PnL.values[:,1]
    index_ts["CDI"]=cdi_ts.values
    index_ts = index_ts.iloc[:,[6, 0, 1, 2, 3, 5,4]]
    index_ts.index = pd.to_datetime(index_ts.index,format="%d/%m/%Y")

    # Print risk ves tamanho
    risk_contribution = peso / np.sum(peso)
    teste.portfolio = peso
    tam_contribution = teste.risk_contribution/np.sum(teste.risk_contribution)
    resumo = pd.DataFrame(index = col_raw[:4],data={
        'port risk':port[:,0],
        'port size':peso[:,0],
        'cont risk':risk_contribution[:,0],
        'cont size':tam_contribution[:,0],
    })
    print(resumo)
    print(np.log(PnL).diff().std()*np.sqrt(252))

    # Plot
    matplotlib.rcParams['animation.embed_limit'] = 40
    plt.style.use('seaborn-pastel')
    fig,ax = plt.subplots(figsize=(10,5))
    y_lim = (np.min(index_ts.min()),np.max(index_ts.max()))
    color_list = ['#1b365d', '#71bfde', '#dc6b2f', '#f5d52f', '#a92b0c', '#8b8b8b', '#f5cfb3']

    def plot_frame(t_i):
        col_i,row_i = divmod(t_i,index_ts.shape[0])
        ax.clear()
        ax.set_xlim(index_ts.index[0], index_ts.index[-1])
        ax.set_ylim(y_lim[0],y_lim[1])
        print(t_i)
        plt.ylabel('Rentabilidade')
        for i in range(col_i):
            ax.plot(index_ts.index,index_ts.iloc[:, i].values,label=index_ts.columns[i],color=color_list[i])
        ax.plot(index_ts.index[:(row_i+1)],index_ts.iloc[:(row_i+1), col_i].values,label=index_ts.columns[col_i],color=color_list[col_i])
        plt.legend(loc='upper left', title="Ativo:")

    plt.rcParams['animation.ffmpeg_path'] = "K:\\Users\\SUFIX\\Leo\\Git\\ExplicaPython\\ffmpeg\\bin\\ffmpeg"
    writer = animation.FFMpegWriter(fps=40, metadata=dict(artist='Me'), bitrate=1800)
    # Writer = animation.writers['ffmpeg']

    animator = FuncAnimation(fig, plot_frame, frames=range(0, np.prod(index_ts.shape), 3), interval=25)
    animator.save('carteira.mp4',writer=writer)



