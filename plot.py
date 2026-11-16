"""Plot the notebook's aggregate model and decile-portfolio results."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                    'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#d8d9d4',
                    'axes.labelcolor':'#626a6e','text.color':'#202c32','xtick.color':'#626a6e',
                    'ytick.color':'#202c32','figure.facecolor':'#faf9f6','axes.facecolor':'#faf9f6',
                    'svg.fonttype':'none'})
COLORS={'SAE':'#2a5663','Ridge':'#ad7651','MLP':'#87958a','PCA+Ridge':'#9a9a93'}

def main(output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(HERE/'results/model-comparison.csv')
    portfolio=pd.read_csv(HERE/'results/decile-portfolios.csv')
    fig,ax=plt.subplots(figsize=(8,4.5))
    order=metrics.sort_values('rank_ic_mean')
    ax.barh(order.model,order.rank_ic_mean,color=[COLORS[x] for x in order.model],height=.58)
    for y,v in enumerate(order.rank_ic_mean):ax.text(v+.0015,y,f'{v:.3f}',va='center')
    ax.set(xlim=(0,.095),xlabel='Mean monthly Spearman correlation',title='Ranking future stock returns')
    ax.tick_params(axis='y',length=0)
    fig.tight_layout(pad=2)
    fig.savefig(output_dir/'neural-network-ranking.svg');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4.7))
    ordered=portfolio.set_index('model').loc[['SAE','Ridge','MLP','PCA+Ridge']].iloc[::-1]
    for ax,col,mult,label in [(axes[0],'avg_monthly_long_short',100,'Mean monthly return (%)'),(axes[1],'sharpe_monthly',1,'Monthly Sharpe ratio')]:
        vals=ordered[col]*mult
        ax.barh(ordered.index,vals,color=[COLORS[x] for x in ordered.index],height=.58)
        for y,v in enumerate(vals):ax.text(v+vals.max()*.025,y,f'{v:.3f}',va='center',fontsize=10)
        ax.set(xlim=(0,vals.max()*1.30),xlabel=label)
        ax.tick_params(axis='y',length=0)
    fig.suptitle('Long–short decile portfolios · 2011–2021',fontsize=15,y=.98)
    fig.tight_layout(pad=2)
    fig.savefig(output_dir/'neural-network-deciles.svg');plt.close(fig)

    loads=pd.read_csv(HERE/'results/factor-loadings.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,5.5))
    for k,ax in enumerate(axes):
        rows=loads[loads.factor==k].sort_values('weight')
        names=rows.characteristic.str.replace('characteristic_','',regex=False)
        ax.barh(names,rows.weight,color=['#2a5663' if v>=0 else '#ad7651' for v in rows.weight],height=.65)
        ax.axvline(0,color='#9a9a93',lw=.7)
        ax.set(title=f'Latent factor {k}',xlabel='Approximate encoder weight')
        ax.tick_params(axis='y',length=0)
    fig.suptitle('Characteristics associated with two latent factors',fontsize=15)
    fig.tight_layout(pad=2)
    fig.savefig(output_dir/'neural-network-loadings.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path('figures'))
    main(p.parse_args().output_dir)
