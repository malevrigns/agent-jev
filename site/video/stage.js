"use strict";
/* The demo video as a pure function of time: render(t) sets every pixel's state for second t.
   Every number comes from stage_data.js (recorded runs); nothing here is typed in by hand. */
const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x));
const seg=(t,a,b)=>clamp((t-a)/(b-a));
const eo=t=>1-Math.pow(1-t,3), eio=t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct=p=>p<0.001?"<0.1%":p>0.999?">99.9%":(p*100).toFixed(1)+"%";
const fmt=(n,d=0)=>Number(n).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});
const argmax=a=>a.indexOf(Math.max(...a));
const MARK='<svg viewBox="0 0 32 32"><rect x="1" y="1" width="30" height="30" rx="8" fill="#141416" stroke="rgba(236,224,200,.25)"/><rect x="8" y="17" width="4" height="8" rx="1.5" fill="#a7813f"/><rect x="14" y="8" width="4" height="17" rx="1.5" fill="#f4dba3"/><rect x="20" y="13" width="4" height="12" rx="1.5" fill="#d9b46c"/></svg>';

document.head.insertAdjacentHTML("beforeend",`<style>
.verdict{display:inline-block;padding:8px 14px;border-radius:9px;font:600 19px/1 var(--sans)}
.verdict.bad{color:var(--bad);background:rgba(240,127,113,.1);border:1px solid rgba(240,127,113,.4)}
.verdict.good{color:var(--ok);background:rgba(134,212,164,.08);border:1px solid rgba(134,212,164,.35)}
.k{font-size:19px;color:var(--muted)}
.pnum{margin-top:6px;font:500 76px/1 var(--num)}.pnum.bad{color:var(--bad)}.pnum.good{color:var(--ok)}
.req{margin:18px 0 0;font:18px/1.62 var(--mono);color:#e1dacd;white-space:pre}
.req .s{color:#a9d7b5}.req .k2{color:#e7c47f}
.usage{position:absolute;left:30px;right:30px;bottom:26px;display:flex;flex-wrap:wrap;gap:10px}
.uc{padding:8px 13px;border-radius:9px;border:1px solid var(--line-2);font:17px/1 var(--mono);color:var(--text-2);background:rgba(0,0,0,.25)}
.uc b{color:var(--gold-2);font-weight:600}
.small .br{font-size:18px;margin-top:9px;grid-template-columns:minmax(0,1fr) 1fr 110px}
.small .track{height:9px}
.cmdbox{display:inline-block;margin-top:40px;padding:20px 30px;border:1px solid var(--line-2);border-radius:16px;background:#0b0b0d;font:26px/1 var(--mono);color:#e8e2d6}
.cmdbox span{color:var(--gold)}
.links{display:flex;justify-content:center;gap:40px;margin-top:34px;font:500 26px/1 var(--mono);color:var(--gold-2)}
.legend{position:absolute;left:30px;bottom:24px;display:flex;gap:26px;font-size:18px;color:var(--muted)}
.legend i{display:inline-block;width:18px;height:11px;border-radius:3px;margin-right:9px;vertical-align:1px}
.pass-note{font:17px/1.4 var(--mono);color:var(--dim)}
#sc3 .req{font-size:16.5px;line-height:1.66}
#sc4 .br{margin-top:7px;font-size:18px}
#sc4 .qa{margin-top:14px}
#sc4 .qt{font-size:21px}
#sc4 .kind{font-size:14px;padding:5px 8px}
#sc4 .layers{height:270px}
#sc5 .frow{padding:9px 30px}
#sc5 .frow .cmd{font-size:21px}
#sc5 .why{font-size:18px;margin-top:-9px}
</style>`);

/* ---------- builders ---------- */
function rowLabel(q,i){
  const k=q.keys[i],c=q.candidates[i];
  if(q.type==="boolean")return k+" · "+c;
  if(q.type==="score")return "L"+k+" · "+c.replace(/^\d+\s*-\s*/,"");
  return k;
}
/* Distribution rows whose bars and percentages run from time a (staggered by step). */
function rowsHTML(labels,probs,a,{step=0.06,cls=""}={}){
  const w=argmax(probs);
  return labels.map((lab,i)=>`<div class="br${i===w?" win":""}"><span class="lab">${esc(lab)}</span><span class="track ${cls}"><i data-w="${a+i*step},0.9,${probs[i]}"></i></span><span class="pc" data-num="${a+i*step},0.9,0,${probs[i]}" data-mode="pct">0.0%</span></div>`).join("");
}
function qaHTML(q,a){
  return `<div class="qa" data-in="${a-0.25}" data-d="0.4"><div><span class="kind">${q.type.toUpperCase()}</span><span class="qt">${esc(q.text)}</span></div>${rowsHTML(q.keys.map((_,i)=>rowLabel(q,i)),q.probs,a)}</div>`;
}
const head=(title,sub)=>`<div class="head"><div class="h1" data-in="0.05">${title}</div><div class="sub" data-in="0.3">${sub}</div></div>`;

/* ---------- scenes ---------- */
const S=[];let T0=0;
function scene(dur,chap,zh,en,html,draw){S.push({start:T0,dur,chap,zh,en,html,draw});T0+=dur}

// 0 · title
scene(5,"开场","AgentJev：给 AI Agent 用的 0.6B 决策模型","A 0.6B decision model for AI agents.",`
<div class="title-wrap">
  <div class="big-mark" data-in="0.15">${MARK}</div>
  <div class="wordmark" data-in="0.4">AGENT<b>JEV</b></div>
  <div class="display" data-in="0.75" data-d="0.9" data-y="34">Agent 的<span class="grad">反射弧</span></div>
  <div class="tagline" data-in="1.45">State in. A distribution out. Nothing decoded.</div>
  <div class="chips" data-in="2.05"><span class="chip"><b>0.6B</b> 决策模型</span><span class="chip">一次前向</span><span class="chip">生成 <b>0</b> token</span><span class="chip">Apache-2.0</span></div>
</div>`);

// 1 · why
const gateQ=V.gate.gates.questions[0];
scene(9,"为什么","每一步只需要一个判断，常见做法却是让大模型写一段话再去解析","Most agent steps are decisions, not writing.",
head(`大部分步骤，<span class="g">不是写作题</span>`,"测试过了没有？下一步调哪个工具？这条命令能不能执行？")+`
<div class="card" style="left:120px;top:330px;width:820px;height:500px;padding:30px" data-in="0.6">
  <div style="display:flex;justify-content:space-between;align-items:center"><span class="lbl">拿写作者当开关 · 示意</span><span class="cnt" id="w-cnt">0 token</span></div>
  <div class="qt" style="margin-top:22px">“测试都通过了吗？请用 JSON 回答。”</div>
  <div class="tokens">${V.why_tokens.map((t,i)=>`<span class="tk" data-show="${(1.7+i*0.2).toFixed(2)}">${esc(t)}</span>`).join("")}</div>
  <div style="position:absolute;left:30px;right:30px;bottom:26px;font-size:20px;color:var(--muted)">逐个 token 解码，再从文本里解析出字符串。没有概率，格式也可能坏掉。</div>
</div>
<div class="card goldb" style="left:980px;top:330px;width:820px;height:500px;padding:30px" data-in="0.8">
  <div style="display:flex;justify-content:space-between;align-items:center"><span class="lbl gold">AgentJev · 实测</span><span class="cnt gold">0 token</span></div>
  <div class="qt" style="margin-top:22px">同一个问题，问一段真实的 pytest 失败输出</div>
  <div class="pulse"><i id="w-pulse"></i></div>
  <div class="qa" style="margin-top:0"><div><span class="kind">BOOLEAN</span><span class="qt">${esc(gateQ.text)}</span></div>${rowsHTML(gateQ.keys.map((_,i)=>rowLabel(gateQ,i)),gateQ.probs,2.0)}
  <div class="meta" data-in="2.6">value <b>false</b> · P(true) <b>${pct(gateQ.probs[0])}</b> · 服务端 <b>${V.gate.gates.wall_ms_median.toFixed(1)} ms</b></div></div>
  <div style="position:absolute;left:30px;right:30px;bottom:26px;font-size:20px;color:var(--muted)">一次前向，在你给的选项上直接给出分布。拿它和你自己的门槛比。</div>
</div>`,(lt,el)=>{
  const shown=[...el.querySelectorAll(".tk")].filter(t=>lt>=+t.dataset.show);
  el.querySelectorAll(".tk").forEach(t=>t.classList.toggle("cur",t===shown[shown.length-1]&&shown.length<V.why_tokens.length));
  el.querySelector("#w-cnt").textContent=shown.length+" token";
  const p=seg(lt,1.25,2.05), pulse=el.querySelector("#w-pulse");
  pulse.style.left=(-30+130*eo(p))+"%";pulse.style.opacity=p>0&&p<1?1:0;
});

// 2 · quickstart terminal
const serve1="python -m jev_service.server --checkpoint models/agentjev-coding-gates/model.safetensors \\";
const serve2="    --model-path models/agentjev-coding-gates --temperatures models/agentjev-coding-gates/temperatures.json --port 8149";
const okLine=l=>esc(l).replace(/ok$/,'<span class="ok">ok</span>');
const termLines=[
  [0.5,`<span class="pr">$ </span><span data-type="0.5,1.8" data-full="python fetch_v1_weights.py --variant coding-gates"></span>`],
  [2.0,`<span class="dm"># 约 1.2 GB，这里略去下载的过程</span>`],
  ...V.fetch.map((l,i)=>[2.35+i*0.3,okLine(l)]),
  [4.75,`<span class="hl">${esc(V.fetch_summary)}</span>`],
  [5.6,`<span class="pr">$ </span><span data-type="5.6,6.8" data-full="${esc(serve1)}"></span>`],
  [6.85,`<span data-type="6.85,8.0" data-full="${esc(serve2)}"></span>`],
  [8.3,`<span class="dm"># 加载权重并预热：本机实测 ${V.load_seconds.toFixed(1)} s</span>`],
  [9.5,`<span class="ok">${esc(V.ready)}</span>`],
];
scene(13,"两条命令跑起来","拉取固定版本的权重并逐个校验，再启动本地服务","Fetch pinned, hash-checked weights. Start the local server.",
head(`两条命令，<span class="g">跑起来</span>`,"权重固定在 Hugging Face 的一个修订上，逐个核对大小和 SHA-256；服务只绑定 127.0.0.1。")+`
<div class="card term" style="left:120px;top:300px;width:1680px;height:575px" data-in="0.3">
  <div class="bar"><span class="dots"><i></i><i></i><i></i></span><span class="lbl" style="letter-spacing:.08em;text-transform:none">agent-jev · 本机实录 · ${esc(V.env.gpu.replace("NVIDIA GeForce ",""))}</span></div>
  <div style="position:absolute;left:0;right:0;top:61px;bottom:0;overflow:hidden"><div class="body" id="t-body">${termLines.map(([a,h])=>`<div data-show="${a}">${h}</div>`).join("")}</div></div>
</div>`,(lt,el)=>{
  const body=el.querySelector("#t-body"), box=body.parentElement;
  body.style.transform=`translateY(${-Math.max(0,body.scrollHeight-box.clientHeight)}px)`;
});

// 3 · the README example, both models
const R=V.http, rq=R.request, [qd,qn]=rq.questions;
const sq=s=>`<span class="s">${esc(JSON.stringify(s))}</span>`, kq=s=>`<span class="k2">${esc(JSON.stringify(s))}</span>`;
const reqHTML=`{
  ${kq("state")}: ${sq(rq.state)},
  ${kq("questions")}: [
    {${kq("id")}: ${sq(qd.id)}, ${kq("type")}: ${sq(qd.type)},
     ${kq("question")}: ${sq(qd.question)},
     ${kq("criteria")}: {${kq("true")}: ${sq(qd.criteria.true)},
                  ${kq("false")}: ${sq(qd.criteria.false)}}},
    {${kq("id")}: ${sq(qn.id)}, ${kq("type")}: ${sq(qn.type)},
     ${kq("question")}: ${sq(qn.question)},
     ${kq("options")}: {${kq("debug_failure")}: ${sq(qn.options.debug_failure)},
                 ${kq("submit_patch")}: ${sq(qn.options.submit_patch)}}}
  ]
}`;
function modelCard(top,who,m,a,good){
  const keys=Object.keys(qn.options), probs=keys.map(k=>m.next[k]);
  return `<div class="card${good?" goldb":""}" style="left:880px;top:${top}px;width:920px;height:270px;padding:26px 30px" data-in="${a-0.7}">
    <div style="display:flex;align-items:center;gap:18px"><span class="lbl${good?" gold":""}">${who}</span><span class="verdict ${good?"good":"bad"}" data-in="${a+1.5}">${good?"判对了：有 1 个失败":"判错了：明明有 1 个失败"}</span></div>
    <div style="display:grid;grid-template-columns:290px 1fr;gap:40px;margin-top:20px">
      <div><div class="k">P(全部通过)</div><div class="pnum ${good?"good":"bad"}" data-num="${a},1.2,0,${m.done},2">0.00</div><div class="track ${good?"good":"bad"}" style="margin-top:16px"><i data-w="${a},1.2,${m.done}"></i></div></div>
      <div class="small"><div class="k">${esc(qn.question)}</div>${rowsHTML(keys,probs,a+0.4)}</div>
    </div></div>`;
}
const U=R.usage;
scene(12,"README 示例","README 的例子“23 过 1 败”：v1 认为全部通过的概率是 0.84，新权重给出 0.24","v1 calls “23 passed, 1 failed” green at 0.84. coding-gates says 0.24.",
head(`README 自己的例子`,"“23 tests passed, 1 failed.” 测试都通过了吗？同一个请求，分别发给 v1 和新权重。")+`
<div class="card" style="left:120px;top:300px;width:720px;height:575px;padding:28px 30px" data-in="0.3">
  <div class="lbl">POST /api/evaluate · 请求体</div>
  <div class="req" data-in="0.6">${reqHTML}</div>
  <div class="usage" data-in="7.4"><span class="uc">候选路径 <b>${U.candidate_paths}</b></span><span class="uc">骨干 <b>${U.input_path_tokens} → ${U.backbone_input_tokens}</b> token</span><span class="uc">生成 <b>0</b> token</span><span class="uc">服务端中位 <b>${R.median_ms.toFixed(1)} ms</b></span></div>
</div>
${modelCard(300,"v1",R.v1,2.3,false)}
${modelCard(605,"coding-gates · 新权重",R.gates,4.6,true)}`);

// 4 · a real pytest failure through the console
const GS=V.gate.gates, gstate=V.gate.gates.state;
scene(10,"真实 pytest 输出","真实的 pytest 失败输出：三道题一次前向，判定没通过、下一步修断言、失败 1 个","Real pytest output. Three questions, one forward pass.",
head(`真实的 pytest <span class="g">失败输出</span>`,"同一状态的三道题放进一个请求：状态只编码一次，一次前向全部答完。")+`
<div class="card" style="left:120px;top:280px;width:1680px;height:610px;overflow:hidden" data-in="0.25">
  <div class="cbody" style="height:calc(100% - 58px)">
    <div class="cstate"><div class="lbl">State · pytest -q --tb=short</div><pre data-type="0.6,2.4" data-full="${esc(gstate)}"></pre></div>
    <div class="cpass"><div class="lbl">Qwen3</div><div class="layers">${"<i></i>".repeat(28)}</div><div class="passl">一次前向<b data-num="2.5,0.8,0,${GS.usage.backbone_input_tokens},0">0</b>token</div><div class="passl">生成</div><div class="zero">0</div></div>
    <div class="cdist" style="padding-top:22px"><div class="lbl">Distribution · ${GS.questions.length} 题 · ${GS.usage.candidate_paths} 条候选</div>${GS.questions.map((q,i)=>qaHTML(q,3.6+i*0.45)).join("")}</div>
  </div>
  <div class="foot" data-in="5.6" data-y="0"><span>候选路径 <b>${GS.usage.candidate_paths}</b></span><span>骨干 token <b>${fmt(GS.usage.input_path_tokens)} → ${fmt(GS.usage.backbone_input_tokens)}</b></span><span>生成 <b class="g">0</b> token</span><span>服务端 <b>${GS.wall_ms_median.toFixed(1)} ms</b> · 同一请求 15 次的中位数</span></div>
</div>`,(lt,el)=>{
  const n=Math.round(28*seg(lt,2.5,3.3)), off=lt>5.4;
  el.querySelectorAll(".layers i").forEach((l,j)=>l.classList.toggle("on",j<n&&!off));
});

// 5 · Claude Code hook
const hookRows=V.hook.map((h,i)=>{
  const a=0.9+i*0.72, d=h.decision||"none";
  const why=h.reason?`<div class="why ${d}" data-in="${a+0.35}">${esc(h.reason)}</div>`:"";
  const chg=h.v1.decision!==h.decision?`<div class="why" data-in="${a+0.7}">v1 对同一条命令：${h.v1.decision||"不表态"} · L${h.v1.level} · P(安全) ${h.v1.p_safe.toFixed(2)}<span class="chg">← 新权重改判</span></div>`:"";
  return `<div class="frow" data-in="${a}" data-y="0"><span class="cmd">${esc(h.command)}</span><span class="ps">P(安全) <span data-num="${a},0.8,0,${h.p_safe},2">0.00</span><span class="track" style="display:block"><i data-w="${a},0.8,${h.p_safe}"></i></span></span><span class="lv${h.level===3?" l3":""}">L${h.level}</span><span class="badge ${d}">${d==="none"?"不表态":d}</span>${why}${chg}</div>`;
}).join("");
scene(11,"Claude Code 钩子","接进 Claude Code：高危的拦下，拿不准的弹窗问你，其余不表态","A PreToolUse hook: deny the dangerous, ask when unsure, stay silent otherwise.",
head(`接进 <span class="g">Claude Code</span>`,"PreToolUse 钩子：Bash、Write、Edit 执行之前先问 AgentJev。它从不输出 allow，不会绕过你的权限设置。")+`
<div class="card" style="left:120px;top:290px;width:1680px;height:590px;overflow:hidden" data-in="0.3">
  <div class="term"><div class="bar"><span class="dots"><i></i><i></i><i></i></span><span class="lbl" style="letter-spacing:.08em;text-transform:none">PreToolUse · Bash · 新权重 · agentjev_hook.py 的真实输出</span></div></div>
  ${hookRows}
</div>`);

// 6 · one prefix tree
const TR=V.trie, rowsT=[];TR.questions.forEach((q,qi)=>q.cands.forEach(c=>rowsT.push({qi,q:q.q,c})));
const depth=Math.max(...rowsT.map(r=>TR.state+r.q+r.c));
scene(8,"为什么快","整棵前缀树一次前向：每案中位时延 278 ms → 43.9 ms","One forward pass over one tree: 278 ms → 43.9 ms per case.",
head(`状态<span class="g">只读一次</span>`,`同一状态下 ${rowsT.length} 条候选路径 [状态][问题][候选]，合成一棵前缀树，一次前向编码完。`)+`
<div class="card" style="left:120px;top:300px;width:1080px;height:575px;padding:28px 30px" data-in="0.3">
  <div style="display:flex;align-items:baseline;gap:14px"><span class="pnum" style="color:var(--text);font-size:64px" data-num="2.3,1.3,${TR.path_tokens},${TR.trie_tokens},0">${fmt(TR.path_tokens)}</span><span class="k">骨干 token</span><span class="k" id="tr-l" style="margin-left:auto"></span></div>
  <div id="tr-stage" style="position:absolute;left:30px;right:30px;top:130px;height:370px">${rowsT.map(()=>`<div class="tok s"></div><div class="tok q"></div><div class="tok c"></div>`).join("")}</div>
  <div class="legend"><span><i style="background:#a7813f"></i>状态</span><span><i style="background:rgba(147,180,255,.6)"></i>问题</span><span><i style="background:#f4dba3"></i>候选</span><span>块宽与真实 token 数成正比</span></div>
</div>
<div class="card" style="left:1240px;top:300px;width:560px;height:575px" data-in="0.5">
  <div class="stat" data-in="3.7"><div class="k">每案中位时延</div><div class="from">${V.speed.p50[0].toFixed(1)} ms</div><div class="to"><span data-num="3.7,1.1,${V.speed.p50[0]},${V.speed.p50[1]},1">${V.speed.p50[0]}</span><small>ms</small></div></div>
  <div class="stat" data-in="4.1"><div class="k">吞吐 · 每请求 20 案</div><div class="from">${V.speed.tp[0].toFixed(2)} 案/秒</div><div class="to"><span data-num="4.1,1.1,${V.speed.tp[0]},${V.speed.tp[1]},1">${V.speed.tp[0]}</span><small>案/秒</small></div></div>
  <div class="stat" data-in="4.5"><div class="k">每案骨干 token</div><div class="from">${fmt(V.speed.tokens[0])}</div><div class="to"><span data-num="4.5,1.1,${V.speed.tokens[0]},${V.speed.tokens[1]},0">${fmt(V.speed.tokens[0])}</span></div></div>
</div>`,(lt,el)=>{
  const st=el.querySelector("#tr-stage"), Wd=1020, Hd=370, pitch=Hd/rowsT.length, h=Math.min(16,pitch-6), k=Wd/depth;
  const p=eio(seg(lt,2.2,3.6)), y=i=>i*pitch+(pitch-h)/2, mid=y((rowsT.length-1)/2);
  const gTop=qi=>{const ix=rowsT.map((r,i)=>r.qi===qi?i:-1).filter(i=>i>=0);return y((ix[0]+ix[ix.length-1])/2)};
  const toks=st.children, first={};
  rowsT.forEach((r,i)=>{
    const s=toks[i*3],q=toks[i*3+1],c=toks[i*3+2], qf=!(r.qi in first);first[r.qi]=1;
    const sy=y(i)+(mid-y(i))*p, qy=y(i)+(gTop(r.qi)-y(i))*p;
    Object.assign(s.style,{left:"0px",top:sy+"px",width:(TR.state*k-3)+"px",height:h+"px",opacity:i===0?1:1-p});
    Object.assign(q.style,{left:TR.state*k+"px",top:qy+"px",width:(r.q*k-3)+"px",height:h+"px",opacity:qf?1:1-p});
    Object.assign(c.style,{left:(TR.state+r.q)*k+"px",top:y(i)+"px",width:(r.c*k-3)+"px",height:h+"px"});
  });
  el.querySelector("#tr-l").textContent=p<.5?`${rowsT.length} 条独立路径 · 状态重复 ${rowsT.length} 次`:"1 棵前缀树 · 状态只编码一次";
});

// 7 · coding gates results
const RS=V.results, cells=V.grid;
scene(8.5,"编码门控的结果","测试集上，失败被判通过 123 → 0；代价是没训练过的运行器上，多拦下 8 次通过的运行","Failing runs called green: 123 → 0 of 348. The cost: 8 passing runs blocked on unseen runners.",
head(`测试没过，<span class="g">就不能说通过</span>`,"编码门控新权重：用 2,100 次真实测试运行训练，每个留出集只读一次。")+`
<div class="card" style="left:120px;top:290px;width:1000px;height:590px;padding:28px 30px" data-in="0.3">
  <div class="lbl">测试集 · ${RS.fp[2]} 个失败案例里，被判成“已通过”的</div>
  <div class="bignum" style="margin-top:20px"><s>${RS.fp[0]}</s><span class="ar">→</span><span data-num="2.4,1.4,${RS.fp[0]},${RS.fp[1]},0">${RS.fp[0]}</span></div>
  <div class="dgrid" id="dg" style="margin-top:34px">${cells.map(()=>"<i></i>").join("")}</div>
</div>
<div class="kvs" style="position:absolute;left:1160px;top:290px;width:640px">
  <div class="kv" data-in="3.4"><div class="k">没训练过的运行器 · 失败被判通过</div><div class="v"><s>${RS.ood_fp[0]}</s>${RS.ood_fp[1]}<small> / ${RS.ood_fp[2]}</small></div></div>
  <div class="kv" data-in="3.8"><div class="k">没训练过的运行器 · 实际通过却被拦下（代价）</div><div class="v warn"><s>${RS.ood_fb[0]}</s>${RS.ood_fb[1]}<small> / ${RS.ood_fb[2]}</small></div></div>
  <div class="kv" data-in="4.2"><div class="k">五种没训练过的问法 · 门控准确率</div><div class="v"><s>${(RS.para[0]*100).toFixed(1)}%</s>${(RS.para[1]*100).toFixed(1)}%</div></div>
  <div class="kv" data-in="4.6"><div class="k">Typed Decisions 测试集 top-1</div><div class="v"><s>${(RS.typed[0]*100).toFixed(2)}%</s>${(RS.typed[1]*100).toFixed(2)}%</div></div>
</div>`,(lt,el)=>{
  const cs=el.querySelector("#dg").children;
  for(let i=0;i<cells.length;i++){
    const [a,b]=cells[i], red=a>=.5, p=eo(seg(lt,2.4+(i%36)*0.02,3.2+(i%36)*0.02));
    const on=red?1-p:0;
    cs[i].style.background=on>0?`rgba(240,127,113,${0.12+0.88*on})`:"rgba(236,224,200,.08)";
    cs[i].style.boxShadow=on>.5?`0 0 12px rgba(240,127,113,${.5*on})`:"none";
    cs[i].style.opacity=eo(seg(lt,0.6+i*0.002,1.2+i*0.002));
  }
});

// 8 · outro
scene(5.5,"开源","开源，Apache-2.0：代码在 GitHub，权重在 Hugging Face","Open source under Apache-2.0. Code on GitHub, weights on Hugging Face.",`
<div class="title-wrap" style="top:180px">
  <div class="big-mark" data-in="0.1">${MARK}</div>
  <div class="wordmark" data-in="0.3">AGENT<b>JEV</b></div>
  <div class="h1" style="margin-top:46px;font-size:72px" data-in="0.55">把这一下反射，<span class="grad">放进你的 Agent 循环</span></div>
  <div data-in="1.0"><div class="cmdbox"><span>$ </span>python fetch_v1_weights.py --variant coding-gates</div></div>
  <div class="links" data-in="1.4"><span>github.com/malevrigns/agent-jev</span><span>huggingface.co/aimeigaoshou/agent-jev</span></div>
  <div class="chips" data-in="1.8"><span class="chip">一张 <b>8 GB</b> 显卡就够</span><span class="chip">输出带完整分布</span><span class="chip">Apache-2.0</span></div>
</div>`);

const DURATION=T0;
window.DURATION=DURATION;
window.CHAPTERS=S.slice(0,-1).map(s=>({t:s.start,title:s.chap}));

/* ---------- mount ---------- */
const root=document.getElementById("scenes");
root.innerHTML=S.map((s,i)=>`<div class="scene" id="sc${i}">${s.html}<div class="cap" data-in="0.35"><div class="zh">${esc(s.zh)}</div><div class="en">${esc(s.en)}</div></div></div>`).join("");
const els=S.map((_,i)=>document.getElementById("sc"+i));
// typed text keeps its full value in data-full; pre-measure nothing else
function drawAttrs(r,lt){
  r.querySelectorAll("[data-in]").forEach(el=>{
    const a=+el.dataset.in, d=+(el.dataset.d||.55), y=el.dataset.y!==undefined?+el.dataset.y:18, p=eo(seg(lt,a,a+d));
    el.style.opacity=p;el.style.transform=y?`translateY(${((1-p)*y).toFixed(2)}px)`:"";
    if(el.classList.contains("cap"))el.style.transform=`translateX(-50%) translateY(${((1-p)*y).toFixed(2)}px)`;
  });
  r.querySelectorAll("[data-type]").forEach(el=>{
    const [a,b]=el.dataset.type.split(",").map(Number), full=el.dataset.full, p=seg(lt,a,b);
    el.innerHTML=esc(full.slice(0,Math.round(full.length*p)))+(p>0&&p<1?'<span class="caret"></span>':"");
  });
  r.querySelectorAll("[data-w]").forEach(el=>{const [a,d,v]=el.dataset.w.split(",").map(Number);el.style.width=(eo(seg(lt,a,a+d))*v*100)+"%"});
  r.querySelectorAll("[data-num]").forEach(el=>{
    const [a,d,from,to,dec]=el.dataset.num.split(",").map(Number), v=from+(to-from)*eo(seg(lt,a,a+d));
    el.textContent=el.dataset.mode==="pct"?pct(v):fmt(v,dec);
  });
  r.querySelectorAll("[data-show]").forEach(el=>{el.style.display=lt>=+el.dataset.show?"":"none"});
}
const topBar=document.querySelector(".top"), chap=document.getElementById("chap"), prog=document.getElementById("prog");
const orbs=[...document.querySelectorAll(".orb")];
window.render=function(t){
  // background: slow drift
  const o=[[ -260,-200,760,"rgba(217,180,108,.26)",.07],[1340,40,640,"rgba(160,110,50,.22)",.05],[700,700,520,"rgba(217,180,108,.14)",.04]];
  orbs.forEach((el,i)=>{const [x,y,s,c,f]=o[i];Object.assign(el.style,{left:x+Math.sin(t*f*2+i)*60+"px",top:y+Math.cos(t*f*2+i*2)*40+"px",width:s+"px",height:s+"px",background:`radial-gradient(circle,${c},transparent 62%)`})});
  let cur=0;
  S.forEach((s,i)=>{
    const lt=t-s.start, on=lt>=0&&lt<s.dur;
    els[i].style.display=on?"block":"none";
    if(!on)return;cur=i;
    const fin=i===0?0.7:0.45, fout=i===S.length-1?0.9:0.4;
    els[i].style.opacity=Math.min(seg(lt,0,fin),1-seg(lt,s.dur-fout,s.dur));
    drawAttrs(els[i],lt);
    if(s.draw)s.draw(lt,els[i]);
  });
  const edge=cur===0||cur===S.length-1;
  topBar.style.opacity=edge?0:Math.min(seg(t-S[cur].start,0,.45),1-seg(t-S[cur].start,S[cur].dur-.4,S[cur].dur));
  chap.innerHTML=edge?"":`${String(cur).padStart(2,"0")} / ${String(S.length-2).padStart(2,"0")}<span>${esc(S[cur].chap)}</span>`;
  prog.style.width=(t/DURATION*100)+"%";
};
window.render(0);
