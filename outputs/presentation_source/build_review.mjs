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
 tensormux:'https://www.tensormux.com/',
 muxbench:'https://www.tensormux.com/blogs/sla-benchmark',
 gateway:'https://github.com/KrxGu/Tensormux',
 tensorpath:'https://github.com/tensormux/Tensorpath',
 nsys:'https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html',
 ncu:'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html',
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
const order=[1,2,'literature',3,4,5,6,'tensormux','tensorpath','experiments',7,8,11,'results',9,10,12,13,14,'extension',15,'industry_refs'];
proto.slides=[cover,...Array.from({length:order.length-1},()=>structuredClone(blank))];
proto.slides.forEach((s,i)=>{s.id=`review-${i+1}`;s.index=i;delete s.notesSlide;});
proto.charts=[];
const p=Presentation.load(proto);
const sl=n=>p.slides.items[order.indexOf(n)];
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
const labels={default:'vLLM default captures',matched:'Exact batch matches',coarse:'Coarse capture grid'};
for(const n of order.slice(1))set(n,532,String(order.indexOf(n)+1),null,15);

const REPO='https://github.com/nileshsarkar-ai/Inference-Optimization';
const DATA=`${REPO}/tree/master/outputs`;
source.sarathi='https://www.usenix.org/system/files/osdi24-agrawal.pdf';
const shortPlain=m=>short(m).replace('Qwen2.5','Qwen2.5');
function link(n,label,url,x,y,w,h,size=20){const s=add(n,'',x,y,w,h,size);s.text=[{runs:[{run:label,textStyle:{fontSize:`${size}px`,typeface:FONT,color:'#235E9C'},link:{uri:url,isExternal:true}}]}];return s;}
for(const s of sl(1).shapes.items){const t=s.text.toString();if(t==='SaturateLLM'){s.text='Inference\nOptimization';textStyle(s,82);s.position={left:41,top:231,width:1180,height:190};}if(t.includes('8 September 2026')||t.startsWith('SOTA evidence'))s.text='';}
set(1,6,'',[41,41,1100,45],18);
set(1,5,'Improving GPU Utilization for LLM Inference',[41,449,1150,52],32);
note(1,'Inference Optimization investigates how scheduling, admission, batching and request/stage routing can increase useful inference work per GPU. LLMs are the first experimental domain. The measured configuration and scheduling controls do not establish an improved policy.');


function block(n,heading,body,x,y,w,h=105,size=23){add(n,heading,x,y,w,36,size+4,true);add(n,body,x,y+43,w,h,size);}
function divider(n,heading,sub){set(n,533,heading,[65,231,1150,112],66);set(n,3,'',null,15);add(n,sub,69,369,1090,94,29);}
divider('literature','Previous Literature Review','');
divider('experiments','Our Experiments','');
divider('results','Our Results','');

title(2,'Research question');
add(2,'Which limits to GPU utilization can we reduce to improve LLM throughput within latency targets?',41,144,1197,104,33,true);
block(2,'Initial investigation','We tested CUDA graph settings and admission during a burst of long prompts on one A100 40 GB.',41,284,565,115,25);
block(2,'Extension','Profile where GPU capacity remains unused and test whether a targeted change improves complete inference.',657,284,581,115,25);
add(2,'Hypothesis: reducing a measured scheduling or execution bottleneck can increase throughput while preserving response-time targets.',41,488,1197,94,27,true);
add(2,'SLO goodput counts requests meeting latency targets per second. The recoverable capacity remains to be measured.',41,605,1197,53,23);
foot(2,[['Research question and evidence',`${REPO}/blob/master/outputs/Research_Question.md`]]);

title(3,'What existing inference systems optimize');
add(3,'Selected systems address request scheduling, GPU execution overlap and resource allocation.',41,143,1197,70,26);
table(3,[['System','What it tries to improve','How the selected experiment tests it'],['SOLA\nMLSys 2025','Request order and work per GPU iteration, using request and system state','Compare first-token and per-token latency against the paper’s vLLM baseline'],['DuetServe\nICML 2026','Sharing GPU execution resources between prefill and decode','Increase offered request rate and compare completed requests/s'],['Prism\nNSDI 2025','CPU/GPU resource use in recommendation inference','Vary CPU-side and GPU-side instances and compare goodput per GPU node']],[41,239,1197,323],[225,501,471],23);
add(3,'Their gains use different hardware, workloads and latency targets. They do not form a single performance ranking.',41,598,1197,58,23);
foot(3,[['SOLA',source.sola],['DuetServe',source.duet],['Prism recommendation serving',source.prism]]);

title(4,'SOLA: scheduling to meet latency targets');
await img(4,'sola-figure1.png',41,229,588,309,'Original SOLA Figure 1, showing per-request first-token and output-token latency');
add(4,'Read the graph: each point is a request.\nx = first-token delay (TTFT), y = time per output token (TPOT). The shaded corner meets both targets.',41,556,588,92,22);
block(4,'Goal and method','SOLA changes request order and work per iteration using current request and engine state.',683,153,555,92,24);
block(4,'Experiment shown','Llama3-70B on 4 A100s. ShareGPT at 4.6 requests/s. Targets: TTFT 0.5 s and TPOT 0.2 s.',683,308,555,104,24);
block(4,'Result and relevance','Target attainment rises from 65% to 98%. Scheduling can change useful completions. This is the paper’s result, under its own setup.',683,478,555,120,24);
foot(4,[['Hong et al., SOLA, MLSys 2025. Original Figure 1',source.sola]]);

title(5,'DuetServe: sharing GPU resources across inference stages');
await img(5,'duet-figure6-legend.png',41,169,216,140,'Original DuetServe Figure 6 legend');
add(5,'QPS is incoming requests/s.\nThe vertical axis counts completed requests/s. Higher lines mean greater throughput.',288,170,340,122,23);
await img(5,'duet-figure6-azure-throughput.png',41,326,588,294,'Original DuetServe Figure 6 AzureCode throughput panel');
block(5,'Goal and method','DuetServe overlaps prompt processing with token generation using adaptive GPU resource sharing.',683,153,555,95,24);
block(5,'Experiment shown','Qwen3-8B on 1 H100 80 GB. The AzureCode workload increases incoming rate from 10 to 16 requests/s.',683,314,555,100,24);
block(5,'Result and relevance','At 16 incoming requests/s: 12.43 for SGLang-Default, 13.57 for DuetServe (+9.2%, calculated). Execution scheduling affects completed work.',683,476,555,139,24);
foot(5,[['Gao et al., DuetServe, ICML 2026. Figure 6 and §5.2',source.duet]]);

title(6,'Prism: recommendation inference across CPU and GPU nodes');
add(6,'Original Figure 17. Taller bars mean higher goodput under a 25 ms service-latency target.',41,173,623,70,24);
await img(6,'prism-figure17.png',41,280,623,253,'Original Prism Figure 17, GPU-node goodput for two recommendation models');
add(6,'x = two recommendation models.\nPrism (CN, HN) labels CPU-side and GPU-side instance counts. OOM means out of memory.',41,557,623,91,22);
block(6,'Goal and method','Separate CPU-heavy and GPU-heavy model stages so CPU or memory limits do not strand GPU capacity.',711,153,527,113,24);
block(6,'Experiment shown','One node with 8 A100 80 GB GPUs, plus separate CPU nodes. Vary the number of inference instances.',711,325,527,107,24);
block(6,'Result and relevance','Reported goodput rises 5–9×. The 9× case uses GPU partitioning (MIG). Added CPU resources matter, so this is not a same-total-hardware gain.',711,489,527,135,23);
foot(6,[['Yang et al., GPU-Disaggregated DLRM Serving, NSDI 2025. Figure 17',source.prism]]);

title('tensormux','Tensormux: routing across inference replicas');
add('tensormux','Company benchmark, July 2026. Llama 3.1 8B, BF16, vLLM 0.23.0.\n4 H100 80 GB GPUs, one replica/GPU. Fixed capacity, 1,024 input / 256 output tokens, concurrency 128.',41,142,1197,90,24);
table('tensormux',[['Routing strategy','p95 TTFT (ms)'],['Least Request','394'],['Multi-strategy','399'],['Least Latency','442'],['Throughput','450'],['Random','461']],[41,261,589,314],[311,278],23);
add('tensormux','TTFT measures first-token delay.\nAll strategies meet the 1,000 ms target.',41,598,589,56,22);
block('tensormux','What it does','Routes requests across existing engines. The platform also advertises autoscaling and cache reuse.',682,257,556,86,24);
block('tensormux','Reported performance','About 2,200 output tokens/s per GPU and 80% GPU utilization. Throughput is similar across strategies.',682,414,556,100,24);
add('tensormux','This uniform workload establishes SLA compliance. It does not quantify recovered GPU capacity.',682,574,556,79,23,true);
foot('tensormux',[['Tensormux benchmark: Table 1 and methodology',source.muxbench],['Platform',source.tensormux],['Gateway scope',source.gateway]]);

title('tensorpath','TensorPath: optimizing GPU kernels');
add('tensorpath','Tensormux’s separate Forge project generates Triton kernels and checks them against a PyTorch reference.',41,144,1197,69,26);
add('tensorpath','3.66×',41,266,570,111,78,true);
add('tensorpath','Reported RMSNorm speedup\nagainst PyTorch eager',41,390,570,79,30,true);
add('tensorpath','RTX 4070, FP16\nBatch 16, hidden size 4,096\nRMSNorm normalizes model activations.',41,508,570,119,25);
block('tensorpath','What the result covers','One operation at one shape. The repository lists integration into the serving runtime as future work.',682,259,556,107,24);
block('tensorpath','Relation to our execution study','We changed CUDA graph capture sizes in vLLM and timed full generation. We found no established gain from exact matching.',682,426,556,132,24);
add('tensorpath','Our next test would integrate the change and measure complete serving performance.',682,590,556,66,23,true);
foot('tensorpath',[['TensorPath README: Forge, reported RMSNorm results and integration scope',source.tensorpath]]);

title(7,'Our experimental system and performance measures');
table(7,[['Component','Recorded setup'],['GPU','NVIDIA A100 PCIe, 40 GB\nJarvisLabs, 250 W limit'],['Precision','BF16\nOne GPU per model'],['Serving software','vLLM 0.28.0\nPyTorch 2.13.0'],['Runtime','Python 3.12.13\nCUDA package 13.0.96'],['Inputs','WikiText-2 test text\n128 generated tokens/request']],[41,155,589,389],[211,378],22);
block(7,'Throughput','Generated output tokens divided by elapsed seconds.',682,163,556,79,24);
block(7,'SLO goodput','Requests meeting both latency targets divided by the full replay-and-drain duration.',682,310,556,86,24);
block(7,'Latency and GPU busy time','TTFT measures first-token delay. TPOT measures average later-token delay. NVML busy time records kernel activity, not compute efficiency.',682,456,556,146,24);
add(7,'Experiment 1 uses five models and fixed batches. Experiment 2 uses Gemma 4 12B and timed arrivals.',41,598,589,59,22);
foot(7,[['Recorded system and model revisions',`${REPO}/blob/master/outputs/Experiment_Inventory.md`],['NVIDIA metric definition',source.nvml]]);

title(8,'Our experiment 1: CUDA graph configuration');
add(8,'Question: does matching recorded GPU execution sizes to our batches improve throughput?\nCUDA graphs replay recorded GPU operations. Their sizes affect padding and memory use.',41,138,1197,84,25);
table(8,[['Configuration changed','Captured sizes','Role in the comparison'],['vLLM default captures','35 sizes from the engine','Current engine baseline'],['Coarse capture grid','9 powers-of-two sizes','Test the cost of a sparse set'],['Exact batch matches','17 sizes; covers all 8 tested batches','Test whether exact coverage helps']],[41,249,1197,235],[338,439,420],22);
add(8,'Held fixed: A100 40 GB, BF16, 128 input + 128 output tokens/request, and graph ceiling 256.',41,510,1197,58,23);
add(8,'8 batches × 3 configurations × 3 warmed repeats × 5 models = 360 timing calls.\nWe time full generation calls. Initialization and separate profiler runs are excluded.',41,582,1197,68,23);
foot(8,[['Our executed capture protocol',`${REPO}/blob/master/outputs/model_suite/README.md`],['NVIDIA capture-size tradeoff',source.nvidia]]);
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
function captureLegend(n,y){for(const [i,k] of confs.entries())add(n,labels[k],58+i*402,y,394,31,21,false,colors[k]);}

title(9,'Our experiment 1 results: three current models');
add(9,'A100 40 GB, BF16, 128 input + 128 output tokens/request. Output tokens/s: higher is better.\nPoints show all 216 timings. Lines show configuration means from three warmed repeats.',41,134,1197,69,23);
for(const [i,s] of studies.entries()){
 const x=41+i*411;add(9,short(s.model),x,218,379,35,23,true);chartCapture(9,s,x,258,379,270);
 add(9,'Requests in fixed batch',x+52,537,324,28,21);
 const peak=s.configs.default.stats.reduce((a,b)=>a.mean>b.mean?a:b);
 add(9,`Default peak: ${Math.round(peak.mean).toLocaleString('en-US')} tokens/s, batch ${peak.batch}`,x,569,394,29,20);
}
captureLegend(9,601);
const deltas=studies.flatMap(s=>s.matched_delta_percent),maxLoss=Math.max(...studies.flatMap(s=>s.configs.coarse.stats.map((c,i)=>100*(1-c.mean/s.configs.default.stats[i].mean))));
add(9,`Matched changes: ${Math.min(...deltas).toFixed(1)}% to +${Math.max(...deltas).toFixed(1)}%. No established gain. Coarse grids lose up to ${maxLoss.toFixed(1)}%.`,41,635,1197,26,22,true);
foot(9,[['Source: our raw timing records and W&B',source.wandb]]);
const earlier=[];
for(const [model,folder] of [['Qwen/Qwen2.5-1.5B','experiment_data'],['Qwen/Qwen2.5-7B','experiment_data_qwen7b_complete']]){
 const configs={};for(const k of confs){const rows=(await fs.readFile(`${ROOT}/outputs/${folder}/${k}/measurements.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);const batches=[...new Set(rows.map(r=>r.batch))].sort((a,b)=>a-b);configs[k]={stats:batches.map(batch=>{const values=rows.filter(r=>r.batch===batch).map(r=>r.output_tokens_s);return{batch,values,mean:values.reduce((a,b)=>a+b,0)/values.length};})};}earlier.push({model,folder,configs});}

title(10,'Our experiment 1 results: two earlier model controls');
add(10,'A100 40 GB, BF16, 128 input + 128 output tokens/request. Output tokens/s: higher is better.\nPoints show all 144 timings. Lines show configuration means from three warmed repeats.',41,134,1197,69,23);
for(const [i,s] of earlier.entries()){
 const x=41+i*617;add(10,short(s.model),x,218,577,35,25,true);chartCapture(10,s,x,258,577,270);
 add(10,'Requests in fixed batch',x+175,537,390,29,22);
 const peak=s.configs.default.stats.reduce((a,b)=>a.mean>b.mean?a:b);
 add(10,`Default peak: ${Math.round(peak.mean).toLocaleString('en-US')} tokens/s at batch ${peak.batch}`,x+58,570,510,29,22);
}
captureLegend(10,601);
add(10,'These older-model controls test the same configuration change. They do not validate an improved scheduler.',41,635,1197,28,22,true);
foot(10,[['Source: our earlier Qwen timing records and W&B',source.wandb]]);

title(11,'Our experiment 2: admission during a long-prompt burst');
add(11,'Gemma 4 12B on the same A100. We replay saved arrivals through each admission rule before vLLM.',41,140,1197,66,25);
table(11,[['Policy changed','How requests enter the model'],['Direct vLLM submission','Submit on arrival. The engine manages continuous batching.'],['Fixed limit 32','Admit at most 32 unfinished requests, in arrival order.'],['Context-budget rule','Reserve input + 128 output tokens/request, up to 20,000 total.']],[41,224,1197,224],[350,847],22);
add(11,'Each trace: 96 short, 48 long, 96 short prompts (128 / 2,048 input tokens).\nArrivals average 8 requests/s. All requests generate 128 output tokens.',41,467,1197,68,23);
add(11,'Targets: first token ≤2 s and mean later-token time ≤100 ms, including admission waiting.\n240 requests × 3 policies × 3 engine repetitions = 2,160 completed requests.',41,548,1197,68,23);
add(11,'Pass rule: ≥5% higher goodput than both controls in every repetition, with no worse tail latency.',41,625,1197,35,22,true);
foot(11,[['Our frozen protocol and exact settings',`${REPO}/blob/master/outputs/scheduling_validation/PROTOCOL.md`],['W&B runs',source.wandb]]);

title(12,'Our experiment 2 results: throughput, latency and GPU activity');
add(12,'Gemma 4 12B, A100 40 GB, timed burst workload. Points: 3 runs/policy. Diamonds: means.',41,135,1197,59,23);
const policies=['default','fixed32','context_budget'],policyLabels=['Direct vLLM','Fixed 32','Context budget'];
const metrics=[['slo_goodput_rps','SLO goodput (requests/s), higher is better',4,1],['output_tokens_s','Output tokens/s, higher is better',1000,250],['p99_ttft_s','p99 first-token delay (s), lower is better',30,10],['mean_gpu_busy_percent','NVML GPU busy time (%)',105,25]];
for(const [idx,[metric,label,max,unit]] of metrics.entries()){
 const x=41+(idx%2)*617,y=201+Math.floor(idx/2)*220;
 add(12,label,x,y,577,35,22,true);
 const ser=[];
 for(let rep=0;rep<3;rep++)ser.push({name:`Independent run ${rep+1}`,values:policies.map(k=>round(scheduling.observations.find(r=>r.policy===k&&r.repeat===rep)[metric])),line:{fill:'#555555',width:0},fill:'#555555',marker:{symbol:'circle',size:6},valuesFormatCode:'0.000000'});
 ser.push({name:'Mean',values:policies.map(k=>round(scheduling.observations.filter(r=>r.policy===k).reduce((a,r)=>a+r[metric],0)/3)),line:{fill:BLUE,width:0},fill:BLUE,marker:{symbol:'diamond',size:8},valuesFormatCode:'0.000000'});
 const c=sl(12).charts.add('line',{position:{left:x,top:y+40,width:577,height:173},categories:policyLabels,series:ser,hasLegend:false,yAxis:{min:0,max,majorUnit:unit,numberFormatCode:metric==='slo_goodput_rps'?'0.0':'#,##0',textStyle:{typeface:FONT,fontSize:16},majorGridlines:{fill:'#DDDDDD',width:.7}},xAxis:{textStyle:{typeface:FONT,fontSize:17}},lineOptions:{smooth:false}});applyPresentationChartFont(c,{fontFamily:FONT});
}
foot(12,[['Source: our saved requests, timing summaries and telemetry',`${REPO}/blob/master/outputs/scheduling_results/analysis/validation.json`]]);

title(13,'Our results: what the scheduling comparison means');
add(13,'Means across three engine runs on Gemma 4 12B and the A100 40 GB.',41,144,1197,44,25);
const mean=(policy,metric)=>scheduling.observations.filter(r=>r.policy===policy).reduce((a,r)=>a+r[metric],0)/3;
table(13,[['Policy','SLO goodput\nrequests/s','Output\ntokens/s','p99 first-token\ndelay (s)','GPU busy\n(%)'],...policies.map((k,i)=>[policyLabels[i],mean(k,'slo_goodput_rps').toFixed(3),mean(k,'output_tokens_s').toFixed(1),mean(k,'p99_ttft_s').toFixed(2),mean(k,'mean_gpu_busy_percent').toFixed(2)])],[41,215,1197,218],[311,220,217,236,213],23);
add(13,'The context-budget rule reduces SLO goodput by 20–24% versus default vLLM.',41,463,1197,42,26,true);
add(13,'Near-100% GPU busy time coexists with missed latency targets and lower SLO goodput.',41,513,1197,42,24);
add(13,'Burst finding: in every trial, 96/96 initial short requests meet our targets.\nAfter the long prompts, 0/96 later short requests meet them.',41,557,1197,66,25);
add(13,'The p99 column averages per-run p99s. This finite workload does not measure sustainable serving capacity.',41,633,1197,27,20);
foot(13,[['Our measured comparison',`${REPO}/blob/master/outputs/scheduling_results/analysis/validation.json`],['Post-hoc phase analysis',`${REPO}/blob/master/outputs/scheduling_results/analysis/phase_diagnostics.json`]]);

title(14,'How our experiments extend this work');
add(14,'The open question is how much unused GPU capacity we can recover while meeting latency targets.',41,143,1197,102,32,true);
table(14,[['Our completed study','Measured finding','Implication for the extension'],['CUDA graph configuration\nFive models, 360 calls','Exact matching gives no established gain in the three current models.','Like TensorPath, investigate execution cost. Use the engine’s actual implementation as the baseline.'],['Admission during bursts\nGemma 4 12B, 2,160 requests','The context-budget gate lowers SLO goodput by 20–24%, despite ~99–100% GPU busy time.','Like Tensormux, study where work waits. Our initial scope is one engine on one GPU.']],[41,276,1197,262],[307,402,488],23);
add(14,'These results motivate profiling. They do not identify the causal bottleneck or demonstrate a better scheduler.',41,567,1197,54,24);
add(14,'The 47% / 53% split is illustrative. GPU busy time does not measure achieved compute efficiency.',41,630,1197,29,21);
foot(14,[['Our measured results',`${REPO}/blob/master/outputs/Results.md`],['Tensormux',source.muxbench],['TensorPath',source.tensorpath],['NVIDIA metric definition',source.nvml]]);

title('extension','Next experiment: recoverable GPU underutilization');
add('extension','One controlled study on the same model and GPU, extending our saved burst workload.',41,146,1197,72,28,true);
block('extension','1. Locate the bottleneck','Sweep arrival rate around serving capacity. Separate low demand from idle gaps with pending work. Trace CPU scheduling, GPU kernels and memory activity.',41,253,565,141,25);
block('extension','2. Test one targeted change','Choose the mechanism from the profile: admission or batching if work waits, or execution changes if a kernel dominates. Freeze the change before evaluation.',657,253,581,141,25);
add('extension','Compare current vLLM, the candidate and a control with that change disabled. Use unseen traces, balanced independent repetitions and all-request accounting.',41,473,1197,88,25);
add('extension','Success: higher throughput and SLO goodput within fixed latency targets, supported by a reduction in the identified bottleneck.',41,582,1197,65,26,true);
foot('extension',[['Nsight Systems: execution timeline',source.nsys],['Nsight Compute: kernel resource use',source.ncu],['Proposed protocol',`${REPO}/blob/master/outputs/Research_Question.md`]]);

title(15,'References and reproducibility records');
const refs=[
 ['[1] Hong et al. SOLA: state-aware scheduling. MLSys 2025. Figure 1',source.sola],
 ['[2] Gao et al. DuetServe: adaptive GPU multiplexing. ICML 2026. Figure 6',source.duet],
 ['[3] Yang et al. GPU-disaggregated DLRM serving. NSDI 2025. Figure 17',source.prism],
 ['[4] NVIDIA. Tuning CUDA graph batch sizes. Engineering report, August 2026',source.nvidia],
 ['[5] vLLM. CUDA graph design and serving implementation documentation',source.vllm],
 ['[6] NVIDIA. NVML GPU-utilization metric definition',source.nvml],
 ['[7] Official model and WikiText revisions: saved experiment manifests',`${REPO}/blob/master/outputs/Experiment_Inventory.md`],
 ['[8] Our experiment code, raw results, graph data and full bibliography',REPO],
 ['[9] Our W&B project and exact run records',source.wandb]
];
for(const [i,[lab,url]] of refs.entries())link(15,lab,url,41,158+i*48,1197,39,23);
add(15,'Paper figures retain their original measurements. Our figures use saved A100 experiment records.',41,616,1197,41,22);
foot(15,[['Full references and additional background',`${REPO}/blob/master/outputs/References.md`]]);
title('industry_refs','Industry sources and profiling references');
const industryRefs=[
 ['[10] Tensormux. Inference control plane and stated product scope',source.tensormux],
 ['[11] Tensormux. H100 serving benchmark, 3 July 2026. Table 1',source.muxbench],
 ['[12] Tensormux Gateway. Routing implementation and scope',source.gateway],
 ['[13] TensorPath. Forge results, hardware conditions and integration limits',source.tensorpath],
 ['[14] NVIDIA Nsight Systems. Post-Collection Analysis Guide',source.nsys],
 ['[15] NVIDIA Nsight Compute. Profiling Guide',source.ncu]
];
for(const [i,[lab,url]] of industryRefs.entries())link('industry_refs',lab,url,41,168+i*60,1197,45,24);
add('industry_refs','Tensormux and TensorPath results are self-reported industry evidence.\nTheir setups differ from our A100 experiments. No direct performance ranking follows.',41,568,1197,84,24);
foot('industry_refs',[['Full bibliography and evidence boundaries, checked 9 September 2026',`${REPO}/blob/master/outputs/References.md`]]);

const presenterNotes=JSON.parse(await fs.readFile(`${ROOT}/outputs/presentation_source/presenter_notes.json`,'utf8'));
for(const n of order){const entry=presenterNotes[String(n)];if(!entry)throw Error(`Missing presenter notes: ${n}`);note(n,entry);}
const candidate=`${ROOT}/work/slides/finalizer/Inference-Optimization-${suffix}-candidate.pptx`;
const output=suffix==='final'?`${ROOT}/outputs/Inference_Optimization_Final.pptx`:`${ROOT}/work/review_exports/Inference-Optimization-${suffix}.pptx`;
await fs.mkdir(path.dirname(output),{recursive:true});
await(await PresentationFile.exportPptx(p)).save(candidate);
const checked=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:output,pythonExecutable:PY,
 integrityValidatorPath:`${SKILL}/container_tools/inspect_presentation_package_integrity.py`,layoutValidatorPath:`${SKILL}/container_tools/inspect_presentation_layout_geometry.py`,
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[4,8,11,12,13,18,19].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:22,requiredNativeTableOwnerSlides:[4,8,11,12,13,18,19],requiredNativeChartOwnerSlides:[15,16,17],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[FONT],referencePath:TEMPLATE,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(TEMPLATE)).digest('hex')},verifyArtifactToolImport:true,receiptPath:`${ROOT}/work/slides/finalizer/Inference-Optimization-${suffix}.json`});
console.log(JSON.stringify({output,package:checked.packageIntegrity?.status,layout:checked.presentationLayout?.findingCount,warnings:checked.presentationLayout?.warnings}));
const final=await PresentationFile.importPptx(await FileBlob.load(output));
const render=`${ROOT}/work/slides/render-inference-${suffix}`;await fs.mkdir(render,{recursive:true});
for(const [i,s] of final.slides.items.entries()){const b=await final.export({slide:s,format:'png',scale:1.5});await fs.writeFile(`${render}/slide-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log(JSON.stringify({render}));
