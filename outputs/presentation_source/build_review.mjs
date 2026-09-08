import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {FileBlob,Presentation,PresentationFile} from '@oai/artifact-tool';
import {finalizePresentation,applyPresentationChartFont} from '/Users/nileshsarkar/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/artifact_tool_utils.mjs';

const ROOT=process.env.INFERENCE_WORKSPACE??process.env.SATURATE_WORKSPACE??'/Users/nileshsarkar/Documents/Codex/2026-09-08/so-x20';
const SKILL='/Users/nileshsarkar/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const PY='/Users/nileshsarkar/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const ASSET_DIR=process.env.TEMPLATE_ASSET_DIR??`${ROOT}/outputs/presentation_source`;
const TEMPLATE=`${ASSET_DIR}/template_reference.pptx`;
const FONT='Helvetica Neue', BLUE='#3D8DFF';
const studies=JSON.parse(await fs.readFile(`${ROOT}/outputs/current_model_summary/suite.json`,'utf8'));
const scheduling=JSON.parse(await fs.readFile(`${ROOT}/outputs/scheduling_results/analysis/validation.json`,'utf8'));
const suffix=process.argv[2]??'final';
const source={
 sola:'https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf',
 duet:'https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf',
 prism:'https://www.usenix.org/system/files/nsdi25-yang.pdf',
 prism26:'https://www.usenix.org/conference/osdi26/technical-sessions',
 dynamo:'https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing',
 aegaeon:'https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf',
 nvidia:'https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html',
 vllm:'https://docs.vllm.ai/en/stable/design/cuda_graphs/',
 nvml:'https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html',
 wandb:'https://wandb.ai/nileshsarkar-ai/saturatellm-feasibility'
};
const original=await PresentationFile.importPptx(await FileBlob.load(`${ASSET_DIR}/literature_base.pptx`));
const proto=original.toProto();
const blank=structuredClone(proto.slides[1]);
blank.elements=blank.elements.filter(e=>['533','532','3'].includes(e.id));
const cover=structuredClone(proto.slides[0]);
proto.slides=[cover,...Array.from({length:14},()=>structuredClone(blank))];
proto.slides.forEach((s,i)=>{s.id=`review-${i+1}`;s.index=i;delete s.notesSlide;});
proto.charts=[];
const p=Presentation.load(proto);
const sl=n=>p.slides.items[n-1];
const shape=(n,id)=>sl(n).shapes.items.find(s=>s.id===String(id));
function textStyle(s,size=24,bold=false,color='#000000'){
 s.text.style={typeface:FONT,fontSize:size,bold,color,verticalAlignment:'top',alignment:'left',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};
}
function add(n,text,x,y,w,h,size=24,bold=false,color='#000000'){
 const s=sl(n).shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none'}});
 s.text=text;textStyle(s,size,bold,color);return s;
}
function set(n,id,text,frame,size=24,bold=false){const s=shape(n,id);s.text=text;textStyle(s,size,bold);if(frame)s.position={left:frame[0],top:frame[1],width:frame[2],height:frame[3]};return s;}
function title(n,text){set(n,533,text,[41.3,36,1197,94],38);}
function foot(n,items){
 const s=shape(n,3);s.position={left:41.3,top:663,width:1130,height:29};textStyle(s,15);
 s.text=[{bulletCharacter:'',marginLeft:0,indent:0,runs:items.flatMap(([label,url],i)=>[
  ...(i?[{run:'  ·  ',textStyle:{typeface:FONT,fontSize:'15px'}}]:[]),
  {run:label,textStyle:{typeface:FONT,fontSize:'15px',color:'#235E9C'},...(url?{link:{uri:url,isExternal:true}}:{})}
 ])}];
}
function note(n,t){sl(n).speakerNotes.textFrame.setText(t);}
function sections(n,parts,x=41.3,y=190,w=574,h=445,size=25){
 const s=add(n,'',x,y,w,h,size);s.text=parts.flatMap(([a,b])=>[
  {bulletCharacter:'',marginLeft:0,indent:0,spaceBefore:0,spaceAfter:650,runs:[{run:a,textStyle:{bold:true,typeface:FONT,fontSize:`${size+4}px`}}]},
  {bulletCharacter:'',marginLeft:0,indent:0,spaceBefore:0,spaceAfter:2000,runs:[{run:b,textStyle:{typeface:FONT,fontSize:`${size}px`}}]}
 ]);return s;
}
async function img(n,name,x,y,w,h,alt){sl(n).images.add({blob:new Uint8Array(await fs.readFile(`${ROOT}/outputs/research_sources/${name}`)),contentType:'image/png',fit:'contain',position:{left:x,top:y,width:w,height:h},alt});}
function table(n,values,frame,widths,size=22){
 const [x,y,w,h]=frame,t=sl(n).tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
 t.styleOptions={headerRow:false,bandedRows:false};
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:'#FFFFFF',textStyle:{typeface:FONT,fontSize:size,color:'#000000'},margins:{left:10,right:10,top:8,bottom:6},anchor:'top'});
 t.borders.assign({fill:'#C6C6C6',width:.8});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:'#EAF6FD',textStyle:{typeface:FONT,fontSize:size,bold:true,color:'#000000'}});
 return t;
}
const short=m=>m.replace('google/','').replace('Qwen/','').replace('gemma-4-','Gemma 4 ');
const confs=['default','matched','coarse'];
const colors={default:'#222222',matched:BLUE,coarse:'#909AA8'};
const labels={default:'Current default',matched:'Manual matched set',coarse:'Coarse grid'};
for(let n=2;n<=15;n++)set(n,532,String(n),null,15);

const REPO='https://github.com/nileshsarkar-ai/Inference-Optimization';
const DATA=`${REPO}/tree/master/outputs`;
source.sarathi='https://www.usenix.org/system/files/osdi24-agrawal.pdf';
const shortPlain=m=>short(m).replace('Qwen2.5','Qwen2.5');
function link(n,label,url,x,y,w,h,size=20){const s=add(n,'',x,y,w,h,size);s.text=[{runs:[{run:label,textStyle:{fontSize:`${size}px`,typeface:FONT,color:'#235E9C'},link:{uri:url,isExternal:true}}]}];return s;}
for(const s of sl(1).shapes.items){const t=s.text.toString();if(t==='SaturateLLM'){s.text='Inference\nOptimization';textStyle(s,82);s.position={left:41,top:174,width:1180,height:190};}if(t.includes('8 September 2026')){s.text='8 September 2026     Research question and measured A100 results';textStyle(s,18);}if(t.startsWith('SOTA evidence')){s.text='Large language models (LLMs) provide the first test bed';textStyle(s,24);}}
set(1,6,'RESEARCH PROJECT',[41,41,1100,45],18);
set(1,5,'More useful GPU work\nthrough inference scheduling',[41,380,1150,130],40);
note(1,'Inference Optimization investigates how scheduling, admission, batching and request/stage routing can increase useful inference work per GPU. LLMs are the first experimental domain. The measured configuration and scheduling controls do not establish an improved policy.');

title(2,'Inference, GPU utilization and the metrics we use');
sections(2,[['Inference','A model processes a request. An LLM first processes the prompt (prefill), then generates tokens (decode).'],['GPU busy time','NVIDIA telemetry (NVML) measures time with a GPU kernel running. It does not measure achieved compute efficiency.']],41,163,567,390,24);
sections(2,[['Service-level objective (SLO)','A latency target. TTFT is time to the first token. TPOT is average time per later token. p99 is the 99th percentile.'],['Useful work per GPU','Throughput counts output tokens/s. SLO goodput counts requests meeting both latency targets per second.']],657,163,581,390,24);
add(2,'Our target: more completed inference work within the latency targets.',41,585,1197,48,28,true);
foot(2,[['NVIDIA metric definition',source.nvml],['Our protocol and measurements',source.wandb]]);
note(2,'A token is a unit of text represented by a model tokenizer. Scheduling SLO: TTFT ≤2 seconds and mean TPOT ≤100 milliseconds, with all queueing measured from scheduled arrival. Goodput denominator is the full replay-and-drain duration. Different published papers use different latency targets. NVML busy time is not achieved SM utilization. '+source.nvml);

title(3,'Research question and hypothesis');
sections(3,[['Research question','Can scheduling decisions increase requests meeting latency targets on the same GPU as prompt lengths and arrival rates change?'],['First investigation','We vary GPU replay batch sizes, then compare ways of deciding when waiting requests enter the inference engine.']],41,170,574,420,25);
sections(3,[['Hypothesis','Accounting for each request’s processing demand may improve useful work compared with a fixed request limit.'],['What could go wrong','An extra queue can leave capacity unused or delay requests. The engine’s existing batching may already work better.']],657,170,581,420,25);
foot(3,[['Scheduling background: SOLA',source.sola],['Routing background: Dynamo',source.dynamo]]);
note(3,'This is an open question, not an established optimization. The initial scheduling candidate estimates demand using prompt length plus the known 128-token output budget. It changes admission to one vLLM engine. It does not test multi-worker routing or generalization to non-LLM workloads. Future decisions can include batching and request/stage routing, but one bounded next experiment appears on slide14.');

title(4,'State of the art in inference resource use');
add(4,'SOTA means current leading techniques. Performance depends on the workload.\nvLLM is the open-source inference engine we use as the baseline.',41,140,1197,62,25);
table(4,[['Decision','What existing systems optimize','Examples'],['Routing and pooling','Choose workers using cached context and active load','Dynamo docs (2026)\nAegaeon (SOSP 2025)'],['Scheduling and batching','Mix prompt processing and token generation','SOLA (MLSys 2025)\nSarathi-Serve (OSDI 2024)'],['GPU execution','Overlap work and choose GPU replay batch sizes','DuetServe (ICML 2026)\nvLLM and NVIDIA docs (2026)'],['Resource allocation','Separate CPU/GPU stages and share GPU memory','Prism recommendations (NSDI 2025)\nPrism LLMs (OSDI 2026)']],[41,231,1197,372],[233,535,429],21);
foot(4,[['SOLA',source.sola],['DuetServe',source.duet],['Dynamo',source.dynamo],['Full references: slide 15',null]]);
note(4,'vLLM is the open-source LLM inference engine used for our controls. This is a map of systems, not a cross-paper performance leaderboard. Cached context is prior-token state reusable by a worker. Prism recommendation-model serving and Prism multi-LLM memory ballooning are distinct systems. Full sources: '+Object.values(source).join('\n'));

title(5,'SOLA: more requests meet the latency targets');
await img(5,'sola-figure1.png',41,222,588,320,'Original SOLA Figure 1 comparing vLLM and SOLA request latency');
sections(5,[['Published result','SLO attainment rises from 65% to 98% compared with the paper’s vLLM baseline.'],['Measured setting','Llama3-70B on four A100s. ShareGPT at 4.6 requests/s. Targets: TTFT 500 ms and TPOT 200 ms.']],675,183,563,390,25);
add(5,'Original Figure 1. Each point is a request.',41,580,610,38,21);
foot(5,[['Hong et al., SOLA, MLSys 2025. Figure 1',source.sola]]);
note(5,'SOLA uses request and system state to guide scheduling. The original figure shows per-request TTFT/TPOT. The authors’ 65% and 98% SLO-attainment values are not GPU busy percentages and are not our results. The paper’s vLLM version differs from our vLLM0.28.0. '+source.sola);

title(6,'DuetServe: overlap and dispatch improve throughput');
await img(6,'duet-figure6-legend.png',230,153,220,143,'Original DuetServe Figure6 legend');
await img(6,'duet-figure6-azure-throughput.png',41,313,588,294,'Original DuetServe Figure6 AzureCode throughput panel');
sections(6,[['Published result','SGLang-Default: 12.43 requests/s. DuetServe: 13.57 requests/s. This is a calculated 9.2% increase.'],['Measured setting','Qwen3-8B on one H100 80 GB. The incoming rate is 16 requests/s. DuetServe shares prefill/decode resources and looks ahead at execution.']],675,183,563,430,24);
foot(6,[['Gao et al., DuetServe, ICML 2026. Figure 6 and §5.2',source.duet]]);
note(6,'QPS means offered queries/requests per second. Original Figure6 AzureCode panel and legend. The 12.43 and13.57 exact values come from §5.2. Calculated gain=(13.57/12.43-1)*100=9.17%. SGLang-Default is the stated numerical baseline. Paper vLLM0.10.1/SGLang0.5 baseline versions differ from our environment. '+source.duet);

title(7,'Prism: the problem also appears in recommendations');
await img(7,'prism-figure17.png',41,262,620,250,'Original Prism recommendation-model Figure17 goodput comparison');
sections(7,[['Published result','Prism reports 5–9× higher goodput per GPU node for the shown recommendation models.'],['Why resources matter','CPU and memory limits can leave GPUs waiting. Prism separates CPU-heavy and GPU-heavy work and adds CPU nodes.']],705,180,533,395,25);
add(7,'Eight A100 80 GB GPUs per node.\nThe 9× case also uses GPU partitioning (MIG).',41,556,620,70,21);
foot(7,[['Yang et al., GPU-Disaggregated DLRM Serving, NSDI 2025. Figure 17',source.prism]]);
note(7,'Prism recommendation-model serving is distinct from the2026 LLM system. Figure17 measures total goodput on an eight-GPU node with additional CPU nodes. Baseline(n) identifies monolithic DLRM instances on the GPU node. Prism(CN,HN) identifies counts of CPU-side and GPU-side inference instances. The9× Model-XS configuration uses MIG, which partitions a GPU. This is evidence that inference utilization concerns extend beyond LLMs. It does not validate our candidate beyond LLMs. '+source.prism);

title(8,'Experiment 1: changing CUDA graph batch sizes');
add(8,'CUDA graphs replay recorded GPU operations. Captured batch sizes affect padding and memory use.',41,139,1197,66,25);
table(8,[['Configuration','Captured batch sizes','Question'],['Current default','35 sizes selected by vLLM','How well does the current engine work?'],['Coarse grid','9 powers-of-two sizes','Does a sparse set lose throughput?'],['Matched set','Coarse grid plus 8 tested sizes (17 total)','Does exact coverage help?']],[41,234,1197,251],[257,461,479],22);
add(8,'Same A100 40 GB, vLLM 0.28.0, BF16 precision and capture ceiling 256.',41,515,1197,39,23);
add(8,'128 input + 128 output tokens. 8 batches × 3 configurations × 3 repeats = 72 calls/model.',41,565,1197,61,23);
foot(8,[['vLLM CUDA graph design',source.vllm],['Measured data and W&B',source.wandb]]);
note(8,'All five models use the same fixed batches[24,31,48,63,80,95,112,127]. Default35sizes=[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]. Coarse=[1,2,4,8,16,32,64,128,256]. Matched=[1,2,4,8,16,24,31,32,48,63,64,80,95,112,127,128,256]. Means describe three warmed repetitions within one process per configuration. Fresh process per configuration, fixed configuration order default/matched/coarse, randomized case order after warmup. Active engine batches can differ from submitted fixed sizes. Measurements include prompt processing and host work, but exclude initialization/profiler passes. BF16 is16-bit floating-point precision. All model/dataset revisions and exact settings are saved. '+source.vllm);

const round=v=>Number(v.toFixed(6));
function chartCapture(n,study,x,y,w,h){
 const series=[];
 for(const k of confs){
  const stats=study.configs[k].stats;
  series.push({name:`${labels[k]} mean`,xValues:stats.map(r=>r.batch),values:stats.map(r=>round(r.mean)),line:{fill:colors[k],width:2},fill:colors[k],marker:{symbol:'none',size:2},valuesFormatCode:'0.000000'});
  series.push({name:`${labels[k]} all measured repetitions`,xValues:stats.flatMap(r=>r.values.map(()=>r.batch)),values:stats.flatMap(r=>r.values.map(round)),line:{fill:colors[k],width:0},fill:colors[k],marker:{symbol:k==='default'?'circle':k==='matched'?'diamond':'triangle',size:4},valuesFormatCode:'0.000000'});
 }
 const max=Math.max(...confs.flatMap(k=>study.configs[k].stats.flatMap(r=>r.values))),unit=max>15000?5000:max>6000?2000:1000;
 const c=sl(n).charts.add('scatter',{position:{left:x,top:y,width:w,height:h},series,hasLegend:false,scatterOptions:{style:'lineWithMarkers'},yAxis:{min:0,max:Math.ceil(max/unit)*unit,majorUnit:unit,numberFormatCode:'#,##0',textStyle:{typeface:FONT,fontSize:18},majorGridlines:{fill:'#DDDDDD',width:.7}},xAxis:{min:20,max:130,majorUnit:20,numberFormatCode:'0',textStyle:{typeface:FONT,fontSize:18}}});applyPresentationChartFont(c,{fontFamily:FONT});
}
function captureLegend(n,y){for(const [i,k] of confs.entries())add(n,`${k==='default'?'●':k==='matched'?'◆':'▲'} ${labels[k]}`,105+i*388,y,385,34,22,false,colors[k]);}
title(9,'Current models: all 216 timing measurements');
add(9,'Output tokens/s (higher is better). Points show every repeat. Lines show means.',41,137,1197,55,23);
for(const [i,s] of studies.entries()){const x=41+i*411;add(9,short(s.model),x,212,379,35,23,true);chartCapture(9,s,x,259,379,299);add(9,'Requests in fixed batch',x+60,568,300,31,20);}
captureLegend(9,614);foot(9,[['Source: raw timings and W&B runs',source.wandb]]);
note(9,'All216raw timing observations are native chart points, plus72computed means. Points at the same batch size can overlap because repeated measurements and configurations are close. No horizontal/vertical jitter. Numeric x-axis preserves uneven batch spacing. Different panel y-axis ranges allow within-model comparison. Source files: '+studies.flatMap(s=>confs.map(k=>s.configs[k].measurement_source)).join('\n')+'\nCapture repetitions are within-process, not independent confidence intervals. '+studies.flatMap(s=>confs.map(k=>s.configs[k].wandb_url)).join('\n'));

const earlier=[];
for(const [model,folder] of [['Qwen/Qwen2.5-1.5B','experiment_data'],['Qwen/Qwen2.5-7B','experiment_data_qwen7b_complete']]){
 const configs={};for(const k of confs){const rows=(await fs.readFile(`${ROOT}/outputs/${folder}/${k}/measurements.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);const batches=[...new Set(rows.map(r=>r.batch))].sort((a,b)=>a-b);configs[k]={stats:batches.map(batch=>{const values=rows.filter(r=>r.batch===batch).map(r=>r.output_tokens_s);return{batch,values,mean:values.reduce((a,b)=>a+b,0)/values.length};})};}earlier.push({model,folder,configs});}
title(10,'Earlier Qwen controls: all 144 timing measurements');
add(10,'Output tokens/s (higher is better). Two older controls, with every repeat and configuration mean.',41,137,1197,63,23);
for(const [i,s] of earlier.entries()){const x=41+i*617;add(10,short(s.model),x,216,577,35,25,true);chartCapture(10,s,x,258,577,294);add(10,'Requests in fixed batch',x+180,565,355,35,21);}
captureLegend(10,614);foot(10,[['Source: earlier Qwen raw timings and W&B',source.wandb]]);
note(10,'All144measurements appear as native chart points, plus48computed means. These are supplementary older-model controls, not claims about current model releases. The fixed-batch protocol is on slide8. Values round to6decimal places in editable workbooks. Sources: '+earlier.flatMap(s=>confs.map(k=>`outputs/${s.folder}/${k}/measurements.jsonl`)).join('\n'));

const deltas=studies.flatMap(s=>s.matched_delta_percent),maxLoss=Math.max(...studies.flatMap(s=>s.configs.coarse.stats.map((c,i)=>100*(1-c.mean/s.configs.default.stats[i].mean))));
title(11,'What the capture controls tell us');
sections(11,[['The current default is strong',`Across current models, matched-set differences range from ${Math.min(...deltas).toFixed(1)}% to +${Math.max(...deltas).toFixed(1)}%. Coarse grids lose up to ${maxLoss.toFixed(1)}%.`],['Memory pressure remains relevant','Gemma 4 12B throughput falls 10.1% from batch 95 to 127. Sampled logs also reach 99.7% KV-cache occupancy.']],41,178,574,405,25);
sections(11,[['What this establishes','The settings affect measured throughput. Exact capture matching alone has not shown a useful improvement.'],['Why investigate admission next?','A fixed batch omits changing arrivals and waiting. We next test how a queue policy handles a burst of long prompts.']],657,178,581,405,25);
add(11,'KV cache holds prior-token attention state. Its occupancy is a memory metric.',41,601,1197,41,22);
foot(11,[['Source: recorded timings and sampled engine logs',source.wandb]]);
note(11,'Gemma throughput decline and high KV occupancy are observations from the same study, not a causal demonstration that occupancy caused the decline. Occupancy includes sampled engine logs and warmup. No achieved SM counters were collected. Fixed configuration order and within-process repeats limit interpretation of sub-percent changes. Older Qwen controls likewise do not establish a matched-set gain.');

title(12,'Experiment 2: when should requests enter the engine?');
add(12,'Same Gemma 4 12B and A100. Replay the same request sequence for each policy.',41,137,1197,53,24);
table(12,[['Policy','Admission rule before the vLLM engine'],['vLLM default','Send every request on arrival. vLLM manages continuous batching.'],['Fixed limit 32','Admit at most 32 unfinished requests, in arrival order.'],['Context budget','Reserve input + 128 output tokens per request, up to 20,000 total.']],[41,208,1197,231],[277,920],22);
add(12,'Trace: 96 short, 48 long, 96 short prompts (128 / 2,048 tokens), at 8 arrivals/s.',41,461,1197,36,23);
add(12,'240 requests × 3 policies × 3 independent engine runs = 2,160 requests.',41,502,1197,35,23);
add(12,'Targets: TTFT ≤ 2 s and mean TPOT ≤ 100 ms. Admission waiting counts.',41,543,1197,35,23);
add(12,'Pass rule: ≥5% higher goodput than both controls in all 3 runs, with no worse p99 first-token delay or stream gap.',41,589,1197,61,23,true);
foot(12,[['Exact protocol, traces and W&B',source.wandb]]);
note(12,'vLLM is the engine used by all policies. A trace is an exact sequence of request prompts and scheduled arrival times. Poisson arrivals at8requests/s, seeds20260908,20260909,20260910. Three fresh engine processes each run all policies, with balanced rotated order. Each request generates128tokens. Context-budget policy chooses oldest fitting request and protects requests waiting≥2seconds by reserving room before bypassing them. Reservation is an estimate based on known output budget, not actual KV bytes. All queueing counts from scheduled arrival. The ≥5% threshold is a predeclared practical criterion, not a significance test. A stream gap is the interval between generated output chunks. Saved requests here have one token per chunk. Source: outputs/scheduling_validation/PROTOCOL.md and outputs/scheduling_results. '+scheduling.wandb_runs.join('\n'));

title(13,'Scheduling results: all four metrics and repetitions');
add(13,'Gray dots show 3 independent runs per policy. Blue markers show means. Points may overlap.',41,135,1197,45,22);
const policies=['default','fixed32','context_budget'],policyLabels=['vLLM default','Fixed 32','Context budget'];
const metrics=[['slo_goodput_rps','SLO goodput (requests/s), higher is better',4,1],['output_tokens_s','Output tokens/s, higher is better',1000,250],['p99_ttft_s','p99 first-token delay (s), lower is better',30,10],['mean_gpu_busy_percent','NVML GPU busy time (%)',105,25]];
for(const [idx,[metric,label,max,unit]] of metrics.entries()){
 const x=41+(idx%2)*617,y=196+Math.floor(idx/2)*223;
 add(13,label,x,y,577,35,22,true);
 const ser=[];
 for(let rep=0;rep<3;rep++)ser.push({name:`Independent run ${rep+1}`,values:policies.map(k=>round(scheduling.observations.find(r=>r.policy===k&&r.repeat===rep)[metric])),line:{fill:'#555555',width:0},fill:'#555555',marker:{symbol:'circle',size:6},valuesFormatCode:'0.000000'});
 ser.push({name:'Mean',values:policies.map(k=>round(scheduling.observations.filter(r=>r.policy===k).reduce((a,r)=>a+r[metric],0)/3)),line:{fill:BLUE,width:0},fill:BLUE,marker:{symbol:'diamond',size:8},valuesFormatCode:'0.000000'});
 const c=sl(13).charts.add('line',{position:{left:x,top:y+40,width:577,height:173},categories:policyLabels,series:ser,hasLegend:false,yAxis:{min:0,max,majorUnit:unit,numberFormatCode:metric==='slo_goodput_rps'?'0.0':'#,##0',textStyle:{typeface:FONT,fontSize:16},majorGridlines:{fill:'#DDDDDD',width:.7}},xAxis:{textStyle:{typeface:FONT,fontSize:17}},lineOptions:{smooth:false}});applyPresentationChartFont(c,{fontFamily:FONT});
}
foot(13,[['Source: request and telemetry records, three independent engine runs',source.wandb]]);
note(13,'Four native charts contain36per-run observations plus12computed means. All values come from outputs/scheduling_results/analysis/validation.json. Means and points are different series; some nearly identical points overlap. Blue markers show the mean, and gray points show all three runs. Policy categories are discrete, so markers have no connecting line. GPU busy percentages span99.1–100.0% but useful work differs. NVML busy is not achieved SM utilization. TTFT includes external waiting. Per-run p99means are not pooled-request p99. Goodput uses the full replay/drain time. '+scheduling.wandb_runs.join('\n'));

title(14,'What we learned and the next research question');
sections(14,[['Measured finding','The context-budget rule loses 20–24% SLO goodput versus vLLM default across paired runs. Its p99 first-token delay increases.'],['Problem exposed by the trace','All 96 short requests after the long burst miss our latency targets under every policy and repetition. Recovery is poor in this workload.']],41,174,574,425,24);
sections(14,[['Question to investigate next','Can admission based on service cost and latency targets preserve useful concurrency and improve recovery after a long-prompt burst?'],['One next experiment','Calibrate on separate data. Compare one adaptive rule with vLLM and a tuned fixed limit on unseen burst traces at several arrival rates.']],657,174,581,425,24);
foot(14,[['Measured comparison',source.wandb],['Request-level phase analysis',`${REPO}/blob/master/outputs/scheduling_results/analysis/phase_diagnostics.json`]]);
note(14,'Candidate context_budget relative to current default across paired repeats: '+JSON.stringify(scheduling.comparisons.filter(x=>x.baseline==='default'))+'. High GPU busy time alone does not establish more useful GPU work. All outputs meet the recorded length requirement, but output hashes differ across policies and this study does not establish answer-quality equivalence. The proposed cost-aware policy is an untested research direction. Post-hoc phase analysis finds 96/96 initial short requests pass and 0/96 recovery short requests pass in every trial. This diagnoses deadline misses in this finite burst, not sustainable capacity. The 20,000-token gate fits at most nine all-long requests versus 32 admitted requests for fixed32; admitted requests may queue inside vLLM and are not simultaneous GPU work. The test changes concurrency, ordering and age protection together, so it cannot isolate their causal effects. Next: use separate calibration to select baseline settings, diagnose queueing and service times, and evaluate one frozen adaptive rule on held-out traces at predeclared loads. Compare default vLLM and the strongest calibrated fixed rule; include a version without the adaptive decision to isolate its effect. Record phase-specific SLO attainment, long-request latency, all completed outputs and any drops, plus goodput and tail latency. Keep timing runs separate from detailed GPU profiling. No claim of achieved SM or memory-bandwidth efficiency follows from NVML busy time. Other inference model families and multi-worker routing remain future scope, not validated transfer.');

title(15,'References and experiment records');
const refs=[
 ['[1] Hong et al. SOLA. MLSys 2025. Figure 1',source.sola],
 ['[2] Gao et al. DuetServe. ICML 2026. Figure 6',source.duet],
 ['[3] Yang et al. GPU-disaggregated DLRM serving. NSDI 2025',source.prism],
 ['[4] Agrawal et al. Sarathi-Serve. OSDI 2024',source.sarathi],
 ['[5] Aegaeon: GPU pooling for concurrent LLM serving. SOSP 2025',source.aegaeon],
 ['[6] Prism: GPU memory ballooning for LLMs. OSDI 2026',source.prism26],
 ['[7] NVIDIA Dynamo: KV-aware routing documentation',source.dynamo],
 ['[8] NVIDIA: tuning CUDA graph batch sizes. August 2026',source.nvidia],
 ['[9] vLLM: CUDA graph design documentation',source.vllm],
 ['[10] NVIDIA: NVML utilization metric definition',source.nvml],
 ['[11] Model and WikiText revisions: saved manifests',`${REPO}/blob/master/README.md`],
 ['[12] Our measurements: raw records and W&B runs',source.wandb]
];
for(const [i,[lab,url]] of refs.entries())link(15,lab,url,41,142+i*38,1197,33,21);
add(15,'Paper figures retain their original data. Our plots use saved A100 measurements.',41,615,1197,34,21);
foot(15,[['Full bibliography and reproducibility files',`${REPO}/blob/master/README.md`],['W&B experiment project',source.wandb]]);
note(15,'Complete bibliography and model/data links:\n'+(await fs.readFile(`${ROOT}/outputs/References.md`,'utf8')).replaceAll('SaturateLLM','Inference Optimization').replace(/This engineering report limits what we can claim as novel\./g,'This report discusses capture-size selection.'));

const candidate=`${ROOT}/work/slides/finalizer/Inference-Optimization-${suffix}-candidate.pptx`;
const output=suffix==='final'?`${ROOT}/outputs/Inference_Optimization_Final.pptx`:`${ROOT}/work/review_exports/Inference-Optimization-${suffix}.pptx`;
await fs.mkdir(path.dirname(output),{recursive:true});
await(await PresentationFile.exportPptx(p)).save(candidate);
const checked=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:output,pythonExecutable:PY,
 integrityValidatorPath:`${SKILL}/container_tools/inspect_presentation_package_integrity.py`,layoutValidatorPath:`${SKILL}/container_tools/inspect_presentation_layout_geometry.py`,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[4,8,12].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:15,requiredNativeTableOwnerSlides:[4,8,12],requiredNativeChartOwnerSlides:[9,10,13],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[FONT],referencePath:TEMPLATE,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(TEMPLATE)).digest('hex')},verifyArtifactToolImport:true,receiptPath:`${ROOT}/work/slides/finalizer/Inference-Optimization-${suffix}.json`});
console.log(JSON.stringify({output,package:checked.packageIntegrity?.status,layout:checked.presentationLayout?.findingCount,warnings:checked.presentationLayout?.warnings}));
const final=await PresentationFile.importPptx(await FileBlob.load(output));
const render=`${ROOT}/work/slides/render-inference-${suffix}`;await fs.mkdir(render,{recursive:true});
for(const [i,s] of final.slides.items.entries()){const b=await final.export({slide:s,format:'png',scale:1.5});await fs.writeFile(`${render}/slide-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log(JSON.stringify({render}));
