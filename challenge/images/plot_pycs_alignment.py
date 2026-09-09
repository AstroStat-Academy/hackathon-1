"""Generate pycs_alignment.png beside this script using NumPy and Matplotlib."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(18)
def signal(t):
    return 20 - .5*np.exp(-.5*((t-38)/8)**2) - .3*np.exp(-.5*((t-83)/11)**2) + .065*np.sin(t/8)
ta = np.sort(rng.uniform(5, 110, 65))
tb = np.sort(rng.uniform(22, 130, 65))
err = .027
a = signal(ta) + rng.normal(0, err, len(ta))
b = signal(tb-20) + .45 + rng.normal(0, err, len(tb))
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
color_a, color_b = 'firebrick', 'steelblue'
for ax in axes:
    ax.grid(alpha=.15)
    ax.set_xlim(0, 135)
    ax.set_ylim(20.68, 19.22)
    ax.set_ylabel('Magnitude (brighter ↑)')
axes[0].errorbar(ta,a,yerr=err,fmt='o',ms=3.5,lw=.6,color=color_a,label='Image A')
axes[0].errorbar(tb,b,yerr=err,fmt='o',ms=3.5,lw=.6,color=color_b,label='Image B')
axes[0].set_title('1  One quasar, two delayed observations',loc='left',fontsize=12,pad=15,weight='bold')
axes[0].set_xlabel('Observation time [days]')
axes[0].annotate('', xy=(38,19.34),xytext=(58,19.34),arrowprops={'arrowstyle':'<->','color':'#333333'})
axes[0].text(48,19.29,'20 days',ha='center',fontsize=10)
axes[0].vlines([38,58],19.37,[19.53,19.98],linestyle=':',color='#999999',lw=1)
axes[0].legend(loc='upper right',frameon=False,fontsize=10)
axes[1].errorbar(ta,a,yerr=err,fmt='o',ms=3.5,lw=.6,color=color_a,label='A: unchanged',zorder=3)
axes[1].errorbar(tb-20,b-.45,yerr=err,fmt='o',ms=3.5,lw=.6,color=color_b,label='B: −20 days, −0.45 mag',zorder=3)
t = np.linspace(0,115,500)
axes[1].plot(t,signal(t),color='#262626',lw=1.6,label='Common signal (simulation)')
axes[1].set_title('2  Shift onto the same underlying history',loc='left',fontsize=12,pad=15,weight='bold')
axes[1].set_xlabel('Aligned time [days]')
axes[1].legend(loc='lower left',frameon=False,fontsize=10)
fig.tight_layout(pad=0.8, w_pad=2)
fig.savefig(Path(__file__).with_name('pycs_alignment.png'),dpi=180,facecolor='white',bbox_inches='tight',pad_inches=0.08)
plt.close(fig)
