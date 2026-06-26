import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# per-step telemetry (from phase2_4b_math_results.txt)
g256_kl  = [0.000,0.004,0.007,0.013,0.014,0.016,0.025,0.032,0.031,0.029,0.039,0.037,0.052,0.043,0.052]
g2048_kl = [0.000,0.005,0.008,0.041,0.019,0.026,0.030,0.038,0.067,0.087,0.062,0.054,0.078,0.065,0.069,0.065]
lora_kl  = [0.000,0.001,0.001,0.001,0.001,0.002,0.002,0.003,0.004,0.006,0.008,0.006,0.011,0.013,0.013]
g256_r   = [0.08,0.04,0.04,0.17,0.08,0.08,0.04,0.08,0.12,0.29,0.04,0.04,0.00,0.04,0.08]
g2048_r  = [0.12,0.04,0.12,0.21,0.08,0.04,0.04,0.21,0.17,0.46,0.21,0.17,0.08,0.08,0.17,0.12]
lora_r   = [0.00,0.08,0.00,0.17,0.04,0.00,0.00,0.04,0.17,0.21,0.08,0.00,0.00,0.04,0.04]
C = {"g2048":"#d1495b","g256":"#edae49","lora":"#30638e"}

def ema(x,a=0.5):
    o=[x[0]]
    for v in x[1:]: o.append(a*v+(1-a)*o[-1])
    return o

fig,ax=plt.subplots(1,2,figsize=(11,4.2))
# (1) KL — the load-bearing figure
ax[0].plot(g2048_kl,color=C["g2048"],lw=2.2,marker="o",ms=3,label="Map-G2048 (2048 params)")
ax[0].plot(g256_kl, color=C["g256"], lw=2.0,marker="s",ms=3,label="Map-G256 (256 params)")
ax[0].plot(lora_kl, color=C["lora"], lw=2.0,marker="^",ms=3,label="LoRA-r8 (16.5M params)")
ax[0].axhline(0.05,ls="--",c="gray",lw=1,alpha=0.6); ax[0].text(0.3,0.053,"leverage threshold ~0.05",fontsize=8,color="gray")
ax[0].set_title("Policy movement: KL(π‖base) per RL step",fontsize=11,weight="bold")
ax[0].set_xlabel("GRPO step"); ax[0].set_ylabel("mean KL"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
# (2) reward (EMA-smoothed)
ax[1].plot(ema(g2048_r),color=C["g2048"],lw=2.2,label="Map-G2048")
ax[1].plot(ema(g256_r), color=C["g256"], lw=2.0,label="Map-G256")
ax[1].plot(ema(lora_r), color=C["lora"], lw=2.0,label="LoRA-r8")
ax[1].set_title("Reward (correct-rate) per step, EMA-smoothed",fontsize=11,weight="bold")
ax[1].set_xlabel("GRPO step"); ax[1].set_ylabel("mean reward"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
plt.tight_layout(); plt.savefig("fig_training_curves.png",dpi=140)

# (3) accuracy bar with Wilson CI
fig2,ax2=plt.subplots(figsize=(6.5,4))
labels=["baseline\n(0)","Map-G256\n(256)","LoRA-r8\n(16.5M)","Map-G2048\n(2048)"]
acc=[0.295,0.380,0.390,0.485]; lo=[0.236,0.316,0.325,0.417]; hi=[0.362,0.449,0.459,0.554]
cols=["#999999",C["g256"],C["lora"],C["g2048"]]
err=[[a-l for a,l in zip(acc,lo)],[h-a for a,h in zip(acc,hi)]]
ax2.bar(labels,acc,color=cols,yerr=err,capsize=5,width=0.6)
ax2.axhline(0.295,ls="--",c="#999999",lw=1,alpha=0.7)
ax2.axhspan(0.236,0.362,color="#999999",alpha=0.12)
for i,a in enumerate(acc): ax2.text(i,hi[i]+0.012,f"{a:.1%}",ha="center",fontsize=9,weight="bold")
ax2.set_title("MATH-500 accuracy (n=200, Wilson 95% CI)",fontsize=11,weight="bold")
ax2.set_ylabel("accuracy"); ax2.set_ylim(0,0.62); ax2.grid(axis="y",alpha=0.25)
plt.tight_layout(); plt.savefig("fig_accuracy.png",dpi=140)
print("saved fig_training_curves.png + fig_accuracy.png")
