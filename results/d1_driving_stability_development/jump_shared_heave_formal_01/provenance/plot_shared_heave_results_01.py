"""Plot saved training/evaluation evidence only; never construct a simulator."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('formal',type=Path)
    parser.add_argument('evaluation',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir()
    with gzip.open(args.formal/'episodes.jsonl.gz','rt') as f: episodes=[json.loads(x) for x in f]
    jumps=[e for e in episodes if e['spec']['request_tick'] is not None]
    x=np.array([e['end_total_transitions'] for e in jumps])
    goal=np.array([e['spec']['net_clearance_m'] for e in jumps])*1000
    p=np.array([e['progress']['progress'] for e in jumps])
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    axes[0].plot(x,goal,label='Requested net gap',color='#344054',lw=1.5)
    axes[0].scatter(x,p*goal,label='Credited gap (capped at goal)',s=9,color='#1570ef')
    axes[0].set_ylabel('Training task gap (mm)');axes[0].legend(frameon=False)
    axes[0].set_title('Shared-heave PPO — training signals, not independent qualification')
    axes[1].plot(x,[e['return'] for e in jumps],color='#1570ef',lw=1)
    axes[1].set_ylabel('Episode return');axes[1].set_xlabel('Actual training transitions')
    for ax in axes:
        for boundary in [32768,65536]:ax.axvline(boundary,color='#98a2b3',lw=.8,ls='--')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    fig.savefig(args.output/'training.png',dpi=180);plt.close(fig)
    summary=json.loads((args.evaluation/'summary.json').read_text())
    names=['no_jump_hold','jump_late_a','jump_late_b','jump_lower_friction','jump_higher_friction']
    fig,axes=plt.subplots(5,2,figsize=(12,12),sharex=True,layout='constrained')
    for row,name in enumerate(names):
        for cond,color,label in [('zero','#667085','Zero residual'),('final_policy_131072','#1570ef','Final PPO')]:
            records=json.loads((args.evaluation/(name+'__'+cond)/'records.json').read_text())
            native=records['intervals'];ends=np.array([r['end_time_s'] for r in native])
            gap=np.array([max(0.,r['endpoint_min_gap_m']-r['contact_margin_m']) for r in native])*1000
            axes[row,0].plot(ends,gap,color=color,lw=1,label=label)
            states=records['endpoints'];t=np.array([s['time_s'] for s in states])
            axes[row,1].plot(t,[s['body_vx_mps'] for s in states],color=color,lw=1,label=label)
        if row:
            request = 1.75 if name in ('jump_late_a','jump_lower_friction') else 2.25
            axes[row,0].hlines(20,request,request+1.2,color='#b42318',ls='--',lw=.8)
        for bound in [-.03,.03]:axes[row,1].hlines(bound,4.,6.,color='#b42318',ls='--',lw=.8)
        score=summary['scores'][name+'__final_policy_131072']
        axes[row,0].set_title(name+' / PPO '+('PASS' if score['passed'] else 'FAIL'),loc='left',fontsize=10)
        axes[row,0].set_ylabel('Native net gap (mm)');axes[row,1].set_ylabel('Body forward speed (m/s)')
        for ax in axes[row]:
            ax.axvspan(4.,6.,color='#12b76a',alpha=.06);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(frameon=False);axes[-1,0].set_xlabel('Episode time (s)');axes[-1,1].set_xlabel('Episode time (s)')
    fig.suptitle('Frozen evaluation: red thresholds apply in their shown windows; green = late settling\nGap curves alone do not certify unloaded flight',fontsize=12)
    fig.savefig(args.output/'evaluation.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
