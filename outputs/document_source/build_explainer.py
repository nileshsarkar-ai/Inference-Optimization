#!/usr/bin/env python3
"""Build the five-page research explainer from preserved local measurements.

Run: python3 outputs/document_source/build_explainer.py
Requires: reportlab, matplotlib, numpy. No GPU or network is used.
"""
import csv
import json
import os
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
WORK = DATA.parent / 'work' / 'pdf_explainer'
WORK.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(WORK / 'mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
OUT = DATA / 'Inference_Optimization_Explained.pdf'
REPO = 'https://github.com/nileshsarkar-ai/Inference-Optimization'
TREE = REPO + '/tree/master/outputs/'
WAND = 'https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility'
URLS = {
    1: 'https://www.usenix.org/system/files/osdi24-agrawal.pdf',
    2: 'https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html',
    3: 'https://docs.vllm.ai/en/stable/design/cuda_graphs/',
    4: 'https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf',
    5: TREE + 'scheduling_results/',
    6: TREE + 'current_model_summary/',
}

def load(path):
    return json.loads((DATA / path).read_text())

policies = ['default', 'fixed32', 'context_budget']
policy_names = ['vLLM default', 'Fixed limit 32', 'Context budget']
palette = ['#243B53', '#8999A9', '#007F7A']
summaries = {p: [load(f'scheduling_results/repeat-{r}/{p}/summary.json') for r in range(3)] for p in policies}
phase = load('scheduling_results/analysis/phase_diagnostics.json')['observations']
with (DATA / 'current_model_summary/summary.csv').open() as f:
    capture_summary = list(csv.DictReader(f))
env = load('scheduling_results/repeat-0/environment.json')
assert env['vllm_version'] == '0.28.0'
assert all(s['requests'] == 240 for ss in summaries.values() for s in ss)
assert len(phase) == 9
assert all(r['phases'][0]['slo_met'] == 96 and r['phases'][2]['slo_met'] == 0 for r in phase)
means = {p: {k: mean(s[k] for s in ss) for k in ['slo_goodput_rps', 'output_tokens_s', 'p99_ttft_s', 'p99_stream_gap_s', 'mean_gpu_busy_percent']} for p, ss in summaries.items()}
delta = [100 * (summaries['context_budget'][r]['slo_goodput_rps'] / summaries['default'][r]['slo_goodput_rps'] - 1) for r in range(3)]
capture_folders = ['experiment_data_gemma12b', 'experiment_data_qwen35', 'experiment_data_gemma_e4b']
model_names = ['Gemma 4 12B', 'Qwen3.5-9B', 'Gemma 4 E4B']
capture_records = []
for folder in capture_folders:
    capture_records.append({p: [json.loads(x) for x in (DATA / folder / p / 'measurements.jsonl').read_text().splitlines()] for p in ['default', 'matched', 'coarse']})

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#596778', 'xtick.color': '#243B53', 'ytick.color': '#243B53', 'axes.labelcolor': '#243B53', 'text.color': '#243B53', 'axes.axisbelow': True, 'pdf.fonttype': 42})

def graphs():
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.35), sharex=True)
    for ax, recs, name in zip(axes, capture_records, model_names):
        for p, label, color, ls, marker in zip(['default', 'matched', 'coarse'], ['Default (35 sizes)', 'Matched (17)', 'Coarse (9)'], [palette[0], palette[2], palette[1]], ['-', '--', ':'], ['o','s','^']):
            rr = recs[p]
            assert len(rr) == 24
            # The source fields hold complete-generation throughput, not kernel-only speed.
            batches = sorted(set(r['batch'] for r in rr))
            ys = [mean(r['output_tokens_s'] for r in rr if r['batch'] == b) for b in batches]
            ax.plot(batches, ys, label=label, color=color, lw=1.55, linestyle=ls, zorder=3)
            ax.scatter([r['batch'] for r in rr], [r['output_tokens_s'] for r in rr], color=color, s=12, marker=marker, alpha=.6, zorder=4)
        ax.set_title(name, loc='left', pad=3)
        ax.set_ylabel('Output tokens/s')
        ax.grid(axis='y', color='#E0E6EC', linewidth=.6)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=8)
    axes[-1].set_xticks([24,31,48,63,80,95,112,127])
    axes[-1].set_xlabel('Requests submitted together in a fixed batch')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.53, 1.01), fontsize=8.5)
    fig.subplots_adjust(top=.91,bottom=.10,left=.13,right=.99,hspace=.53)
    fig.savefig(FIG / 'capture_measurements.png', dpi=220)
    fig.savefig(FIG / 'capture_measurements.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.10))
    metrics = [('slo_goodput_rps','SLO goodput (requests/s) - higher is better'), ('output_tokens_s','Output tokens/s - higher is better'), ('p99_ttft_s','p99 first-token wait (s) - lower is better'), ('mean_gpu_busy_percent','GPU kernel-busy time (%) - diagnostic')]
    for ax,(metric,title) in zip(axes.flat,metrics):
        for i,(p,color) in enumerate(zip(policies,palette)):
            vals = [r[metric] for r in summaries[p]]
            ax.bar(i, mean(vals), color=color, alpha=.75, width=.65, zorder=2)
            ax.scatter(np.array([-.10,0,.10])+i, vals, color=color, edgecolor='white', linewidth=.5, s=24, zorder=3)
        ax.set_xticks(range(3), ['vLLM\ndefault','Fixed\nlimit 32','Context\nbudget'])
        ax.set_title(title, loc='left', fontsize=8.5, pad=6)
        ax.grid(axis='y', color='#E0E6EC',linewidth=.6)
        ax.set_ylim(bottom=0)
        if metric == 'mean_gpu_busy_percent': ax.set_ylim(0,105)
    fig.subplots_adjust(left=.09,right=.99,top=.94,bottom=.10,wspace=.30,hspace=.54)
    fig.savefig(FIG / 'scheduling_measurements.png',dpi=220)
    fig.savefig(FIG / 'scheduling_measurements.pdf')
    plt.close(fig)

graphs()

# Embedded fonts preserve readable typography when the PDF leaves this Mac.
fontdir = Path('/System/Library/Fonts/Supplemental')
if (fontdir / 'Arial.ttf').exists():
    for name, file in [('Body','Arial.ttf'),('Bold','Arial Bold.ttf'),('Italic','Arial Italic.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fontdir / file)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Italic', boldItalic='Bold')
else:
    # ReportLab's built-in fonts permit reproduction on Linux.
    for name, target in [('Body','Helvetica'),('Bold','Helvetica-Bold'),('Italic','Helvetica-Oblique')]:
        pdfmetrics.registerFont(pdfmetrics.Font(name,target,'WinAnsiEncoding'))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Italic',boldItalic='Bold')

NAVY = colors.HexColor('#183047')
TEAL = colors.HexColor('#007F7A')
INK = colors.HexColor('#273B4E')
MUTED = colors.HexColor('#556879')
PALE = colors.HexColor('#EDF6F5')
LINE = colors.HexColor('#D7E1E8')
PAGE_W, PAGE_H = A4
LEFT, RIGHT = 47, 47
WIDTH = PAGE_W - LEFT - RIGHT
styles = {
    'body': ParagraphStyle('body',fontName='Body',fontSize=10.2,leading=14.4,textColor=INK,spaceAfter=0),
    'small': ParagraphStyle('small',fontName='Body',fontSize=8.6,leading=11.6,textColor=MUTED),
    'caption': ParagraphStyle('caption',fontName='Body',fontSize=8.1,leading=10.5,textColor=MUTED),
    'h2': ParagraphStyle('h2',fontName='Bold',fontSize=12.5,leading=16,textColor=NAVY),
    'callout': ParagraphStyle('callout',fontName='Bold',fontSize=14,leading=19,textColor=NAVY),
    'cell': ParagraphStyle('cell',fontName='Body',fontSize=9.2,leading=12.2,textColor=INK),
    'tablehead': ParagraphStyle('tablehead',fontName='Bold',fontSize=8.7,leading=11.2,textColor=colors.white),
}
c = canvas.Canvas(str(OUT),pagesize=A4)
c.setTitle('Inference Optimization: the problem, experiments and evidence')
c.setAuthor('Inference Optimization research project')
c.setSubject('A five-page explanation of useful GPU utilization and measured inference scheduling results')
y = 0
page_bottoms=[]

def link(url,text):
    return f'<link href="{url}" color="#007F7A"><u>{text}</u></link>'

def ref(n):
    return link(URLS[n],f'[{n}]')

def start(number,eyebrow,title,sub):
    global y
    c.setFillColor(TEAL); c.rect(LEFT,PAGE_H-42,32,3,fill=1,stroke=0)
    c.setFont('Bold',8.2);c.setFillColor(MUTED);c.drawString(LEFT+42,PAGE_H-42,'INFERENCE OPTIMIZATION')
    c.setFont('Body',8);c.drawRightString(PAGE_W-RIGHT,PAGE_H-42,f'RESEARCH EXPLAINER  /  {number:02d}')
    c.setFillColor(TEAL);c.setFont('Bold',9.3);c.drawString(LEFT,PAGE_H-74,eyebrow.upper())
    c.setFillColor(NAVY);c.setFont('Bold',25);c.drawString(LEFT,PAGE_H-106,title)
    y=PAGE_H-122
    para(sub,'small',gap=19)

def para(text,style='body',gap=9,width=WIDTH,x=LEFT):
    global y
    p=Paragraph(text,styles[style]); _,h=p.wrap(width,1000);p.drawOn(c,x,y-h);y-=h+gap
    if y < 55: raise ValueError(f'Page content exceeds footer: {c.getPageNumber()}, y={y}')

def heading(text): para(text,'h2',gap=6)

def callout(text):
    global y
    p=Paragraph(text,styles['callout']);_,h=p.wrap(WIDTH-28,1000)
    c.setFillColor(PALE);c.roundRect(LEFT,y-h-24,WIDTH,h+24,7,fill=1,stroke=0)
    c.setFillColor(TEAL);c.rect(LEFT,y-h-24,3,h+24,fill=1,stroke=0)
    p.drawOn(c,LEFT+14,y-h-12);y-=h+38

def table(rows,widths,header=True,font='cell'):
    global y
    cells=[[Paragraph(str(v),styles['tablehead' if header and i==0 else font]) for v in row] for i,row in enumerate(rows)]
    t=Table(cells,colWidths=widths,hAlign='LEFT')
    ts=[('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-1),.4,LINE)]
    if header: ts += [('BACKGROUND',(0,0),(-1,0),NAVY)]
    t.setStyle(TableStyle(ts));_,h=t.wrap(WIDTH,1000);t.drawOn(c,LEFT,y-h);y-=h+12
    if y<55:raise ValueError(f'Table overflow, page{c.getPageNumber()} y{y}')

def figure(filename,height):
    global y
    c.drawImage(str(FIG/filename),LEFT,y-height,width=WIDTH,height=height,mask='auto');y-=height+5

def end(number,shortsource):
    page_bottoms.append({'page':number,'content_bottom_pt':round(y,2)})
    c.setStrokeColor(LINE);c.line(LEFT,42,PAGE_W-RIGHT,42)
    c.setFont('Body',7.3);c.setFillColor(MUTED);c.drawString(LEFT,29,shortsource)
    c.drawRightString(PAGE_W-RIGHT,29,f'{number} / 5')
    c.showPage()

start(1,'01 / The question','What are we trying to improve?','A plain-language explanation of the project, using our completed A100 experiments. Evidence recorded 8 September 2026.')
para('<b>Inference</b> is using an already trained model to produce an answer. A user sends a prompt, the model processes it, and the answer appears a small piece at a time. These pieces are called <b>tokens</b>; a token can be a word, part of a word or punctuation. The GPU performs the model’s numerical calculations.')
heading('Why one request can affect another')
para('A text-generating model first processes the input (<b>prefill</b>), then produces output tokens step by step (<b>decode</b>). A long document and a short question can put different demands on these stages. A serving engine groups requests into batches and shares GPU time between them. Decisions that improve total throughput can also delay individual responses. '+ref(1))
para('Imagine 96 short questions arriving, then 48 long documents, followed by 96 more short questions. Our experiment uses that length pattern. During the long-prompt phase, each request brings <b>16 times as many input tokens</b>, although this does not imply 16 times the processing time. If work arrives faster than it finishes, a backlog builds. Switching back to short prompts does not instantly clear that backlog: a new short request may spend its entire latency allowance waiting for earlier work.')
callout('Can inference scheduling complete more requests within latency targets on the same GPU when a burst of long prompts delays subsequent short requests?')
heading('What “more useful GPU work” means here')
para('The goal is to get more timely answers from the same hardware. <b>Admission</b> decides when a waiting request enters the engine. <b>Scheduling</b> decides how admitted work shares execution. Our first candidate changes admission outside an existing engine; it does not replace all of the engine’s scheduling logic.')
para('<b>Throughput</b> counts work completed per second, such as output tokens/s. <b>Latency</b> is the time a user waits. A <b>service-level objective (SLO)</b> is a declared latency target; a request must meet our targets to count toward timely completed work.')
para('A GPU reporting 99% utilization may simply have a kernel (a GPU program) running almost all the time. Two schedules can keep it active for the same second yet finish different amounts of work, or make different users wait. Busy time therefore cannot tell us which schedule produces more timely answers or uses the processing units more efficiently. We measure completed work and latency alongside it. '+ref(2))
heading('The hypothesis remains a question')
para('Perhaps estimates of processing cost and time remaining before a deadline can guide better admission decisions. Equally, an extra gate may hold back useful work and make performance worse. Our measurements must distinguish those possibilities; an improvement is not assumed.')
end(1,'Concept sources: Sarathi-Serve [1]; NVIDIA NVML [2]. All numbered sources are clickable.')

start(2,'02 / System and rationale','What did we run, and why?','Two separate studies test different decisions in the same inference stack. Neither changes the trained model weights.')
table([
    ['Component','Recorded system'],
    ['GPU','One NVIDIA A100 PCIe 40 GB (40,960 MiB); 250 W power limit; driver 595.58.03'],
    ['Software','Python 3.12.13; vLLM 0.28.0; PyTorch 2.13.0; Transformers 5.16.1'],
    ['Model execution','BF16 precision; tensor parallelism 1 (the model runs on one GPU); text-only use'],
    ['Input material','Real WikiText-2 test text, sliced into exact token lengths; controlled request arrivals'],
    ['Scheduling settings','Gemma 4 12B; context limit 4,096; at most 128 sequences and 4,096 batched tokens; chunked prefill on; prefix caching off; requested GPU-memory fraction 0.85'],
], [113,WIDTH-113])
heading('Study A: execution configuration')
para('The CPU normally launches GPU operations; issuing those launches has overhead. A <b>CUDA graph</b> records a reusable execution plan so operations can be replayed with less launch overhead. We asked whether the sizes of these prepared plans matter for throughput. A plan for 32 requests may require padding when the relevant batch has 31; a matched size might avoid that extra slot. However, storing more plans consumes GPU memory. This padding-versus-memory tradeoff motivates the test; our records do not directly trace which graph is dispatched. '+ref(3))
para('We used three current models: <b>Gemma 4 12B, Qwen3.5-9B and Gemma 4 E4B</b>. Earlier Qwen2.5-1.5B and Qwen2.5-7B models served as implementation controls. This checks whether one configuration observation repeats across models; it does not test scheduling generalization.')
heading('Study B: admission under a workload change')
para('We then asked whether a simple limit based on request size helps when prompt lengths change abruptly. The baseline is <b>vLLM</b>, the software that runs and batches requests on the GPU. We compare its normal submission path with two external admission gates: a fixed request limit and our context-budget candidate.')
para('The idea behind a size-aware gate is to admit fewer large requests before they overwhelm resources. The danger is holding back requests that could have been processed together. A controlled burst lets us observe that tradeoff and the queue’s recovery. Unlike Study A’s fixed batches, requests arrive over time. Study A does not prove the cause of Study B’s delays.')
heading('How this relates to existing research')
para('Sarathi-Serve studies how to combine prefill and decode while balancing throughput and latency. SOLA uses request and system state to improve service-target attainment. These establish that scheduling already matters; they do not show that our simple rule beats a current engine. Their hardware, workloads and baselines differ from ours. '+ref(1)+' '+ref(4))
para('Exact model/data revisions, settings, traces, code, logs and raw records are retained in the '+link(TREE+'Experiment_Inventory.md','experiment inventory')+' and '+link(WAND,'W&B project')+'.','small',gap=0)
end(2,'System details: saved GPU/environment records. The A100 experiments are completed.')

start(3,'03 / Study A: completed','Did capture-size tuning help?','An offline, fixed-batch comparison. This tests whole-generation throughput, including prefill and host work.')
para('<b>Protocol.</b> Each model used batches of 24, 31, 48, 63, 80, 95, 112 and 127 requests. Every request had 128 input and 128 output tokens. We compared default (35 capture sizes), matched (17) and coarse (9) sets, each with a maximum capture size of 256. Each configuration had three warmed timing repetitions.',gap=7)
para('<b>Count.</b> 8 batches x 3 settings x 3 repetitions = <b>72 timed calls per model</b>; 216 across the three models below, and 360 including the two earlier controls. Initialization and ten separate profiler passes are excluded from the timing results.','small',gap=10)
figure('capture_measurements.png',310)
para('<b>Figure 1. Our actual measurements.</b> Dots show all three timing repetitions; lines are their means. Close points overlap. Read the horizontal axis as submitted batch size and the vertical axis as completed output tokens per second; higher is better. Compare configurations within a panel, not model answer quality across panels. '+ref(6),'caption',gap=10)
heading('What the graph shows')
para('Matched settings differed from default by about <b>-0.6% to +0.3%</b> across the current models; coarse settings lost up to <b>3.8%</b>. These small matched differences do not establish an improvement. Each configuration’s repetitions share a process, and configuration order was fixed.',gap=7)
para('Why might matching fail to help? The default already prepares 35 sizes, and graph-size overhead may be a small fraction of full generation time. Those are plausible explanations, not measured causes. The result tells us manual matching did not produce a useful gain here; it does not tell us exactly why.','small',gap=8)
para('For Gemma 4 12B, default throughput falls from about <b>2,130 to 1,914 output tokens/s</b> between batches 95 and 127: <b>10.1% lower</b>. Logs also reach 99.7% KV-cache occupancy, including warmup. The cache stores intermediate model state. This coincidence does not prove that cache pressure caused the drop, and cache occupancy is not GPU compute utilization.','body',gap=0)
end(3,'Figure source: preserved measurement JSONL files; numerical summary [6]. No simulated data.')

start(4,'04 / Study B: completed','Did our admission rule help?','Gemma 4 12B on the same A100. All 2,160 requests completed; each generated exactly 128 output tokens.')
para('<b>Protocol.</b> Each saved trace had 96 short (128-token), 48 long (2,048-token), then 96 short prompts. Exponential arrival gaps averaged 8 requests/s. Three independent engine runs rotated policy order; each policy replayed the same trace within a run. Warmup was excluded and requests drained between trials. '+ref(5),gap=7)
para('<b>Policies.</b> Default submits directly to vLLM. Fixed-32 admits at most 32 unfinished requests in arrival order. Context budget reserves input length + 128 tokens per admitted request under a 20,000-token cap, choosing the oldest fitting request with two-second age protection. This is an estimate, not measured KV memory.','small',gap=8)
para('<b>Targets:</b> time to first token (TTFT) at most 2 s, then mean time per output token at most 100 ms. Waiting outside the engine counts. <b>SLO goodput</b> = requests meeting both targets / full replay-and-drain seconds. The frozen pass rule required at least 5% more goodput than both controls in every paired run, without worse p99 TTFT or p99 stream gap (time between streamed updates).','small',gap=6)
figure('scheduling_measurements.png',277)
para('<b>Figure 2. Our actual measurements.</b> Dots are three independent engine runs; bars are means. Higher top panels are better; lower first-token wait is better. “p99” is the 99th percentile, describing the slow end of the distribution. GPU busy time is a diagnostic, not our success criterion. '+ref(5),'caption',gap=10)
rows=[['Mean across 3 runs','Goodput\n(req/s)','Output\n(tokens/s)','p99 TTFT\n(s)','GPU busy\n(%)']]
for p,n in zip(policies,policy_names):
    m=means[p]
    rows.append([n,f"{m['slo_goodput_rps']:.3f}",f"{m['output_tokens_s']:.1f}",f"{m['p99_ttft_s']:.2f}",f"{m['mean_gpu_busy_percent']:.2f}"])
table(rows,[144,85,91,85,WIDTH-405])
para('<b>The candidate failed all six paired comparisons.</b> Goodput was 20.0-24.0% lower than default vLLM, even though mean GPU busy time was slightly higher. The p99 columns average per-run p99 values, not pooled percentiles. No statistical-significance or answer-quality equivalence claim is made.','body',gap=0)
end(4,'Figure source: nine policy/run summaries [5]. All axes start at zero; raw outputs are preserved.')

start(5,'05 / Interpretation and next step','What can we honestly conclude?','The experiments reveal a reproducible problem in this workload and reject the tested rule. They do not deliver an improved scheduler.')
heading('The most relevant finding is poor burst recovery')
para('A later analysis of the saved records found <b>96/96 initial short requests met the targets, but 0/96 later short requests did</b>, in every policy and repetition. Only 4-9 of the 48 long requests passed. The context-budget candidate needed about 20-22 seconds after the final arrival to finish, versus 10-12 seconds for default vLLM. '+link(TREE+'scheduling_results/analysis/phase_diagnostics.json','Phase records')+'.',gap=8)
para('The gate did not rescue the later short requests. Similar numbers passed across policies, but the candidate took longer to finish all work, lowering goodput through its time denominator. This finite burst at one arrival rate does not measure sustainable capacity or show that scheduling can remove the misses at this load.',gap=9)
heading('Is this a sensible way to investigate?')
para('The comparison is useful, but the gate may be too restrictive. Each long request reserves 2,048 + 128 = <b>2,176 estimated tokens</b>, so a 20,000-token budget admits at most <b>nine all-long requests</b>, versus 32 in the fixed control. This can prevent useful batching while requests wait outside the engine. The GPU can remain busy with the admitted work even as total completion slows. These are admitted requests, not a count of simultaneous GPU executions.',gap=8)
para('That mechanism is a hypothesis, not an isolated causal result: the candidate also changes request ordering and age protection. Its two-second age rule activates when the first-token allowance is already used up, and its reservation is neither measured processing cost nor actual live KV memory. The next comparison must separate these decisions.','small',gap=8)
para('The hardware was busy almost continuously, but achieved processing-unit efficiency was not measured. Scheduling was tested on only one model, and output hashes differed across policies. We cannot claim GPU compute-efficiency gains, generalization to other models or equal answer quality.',gap=9)
heading('One next experiment: test adaptation fairly')
para('On separate calibration traces, measure waiting and processing cost and tune a simple fixed limit. Freeze that baseline and a candidate using cost and time remaining before deadlines. Compare both with default vLLM on unseen burst traces at several predeclared arrival rates, using balanced independent engine runs. Keep fairness rules matched and include a version with adaptation removed to isolate its effect.',gap=8)
para('Evaluate goodput, throughput, recovery after the burst and long-request tail latency; account for every request and any drops. Profile separately if claiming hardware-efficiency changes. The result may support adaptation, reject it, or show the load exceeds what the hardware can serve within those targets.',gap=11)
heading('Sources and reproducible records')
refs=[
    (1,'Agrawal et al. Sarathi-Serve. OSDI 2024. Prefill/decode scheduling and latency-throughput tradeoffs.'),
    (2,'NVIDIA. NVML utilization definition. Kernel-busy time is not achieved compute efficiency.'),
    (3,'vLLM. CUDA Graphs design documentation. Capture/replay and dispatch behavior.'),
    (4,'Hong et al. SOLA. MLSys 2025. Request- and system-state-aware SLO scheduling.'),
    (5,'Our scheduling study. Protocol, traces, request records, telemetry and per-run summaries.'),
    (6,'Our capture study. Per-model measurements, means and raw-data provenance.'),
]
for n,title in refs: para(ref(n)+' '+title,'caption',gap=3)
para(link(REPO,'Repository')+'  |  '+link(TREE+'References.md','Full project bibliography')+'  |  '+link(TREE+'Experiment_Inventory.md','Complete inventory')+'  |  '+link(WAND,'W&B records'),'caption',gap=0)
end(5,'Read this as an evidence-backed research question, not a claim that optimization has been solved.')
c.save()
(HERE/'build_summary.json').write_text(json.dumps({'pdf':str(OUT.name),'pages':5,'page_bottoms':page_bottoms,'capture_observations_shown':216,'scheduling_requests':2160,'scheduling_metric_observations_shown':36,'policy_means':means,'candidate_goodput_percent_vs_default':delta,'sources':URLS},indent=2)+'\n')
print(json.dumps({'output':str(OUT),'pages':5,'page_bottoms':page_bottoms},indent=2))
