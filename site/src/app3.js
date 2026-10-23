/* ---------- tooltip ---------- */
const tip=$("#tip");
function showTip(text,x,y){
  tip.textContent=text;tip.classList.add("show");
  const r=tip.getBoundingClientRect();
  tip.style.left=Math.max(8,Math.min(innerWidth-r.width-8,x+14))+"px";
  tip.style.top=(y+r.height+24>innerHeight?y-r.height-14:y+18)+"px";
}
const hideTip=()=>tip.classList.remove("show");
addEventListener("scroll",hideTip,{passive:true});

/* ---------- coding gates: every failing test case, both models ---------- */
(function(){
  const G=D.grid, seg=$("#grid-seg"), fp=$("#fp-now"), run=runner();
  const runs=G.test.filter(c=>c.k==="r"), sums=G.test.filter(c=>c.k==="s");
  const cells=(box,list)=>{box.innerHTML=list.map((_,i)=>`<i data-i="${i}"></i>`).join("");return $$("i",box)};
  const parts=[[cells($("#grid-runs"),runs),runs,false],[cells($("#grid-sums"),sums),sums,false],[cells($("#grid-ood"),G.ood),G.ood,true]];
  let model=null, touched=false;
  function paint(m){
    if(m===model)return;model=m;pressed(seg,$(`[data-model=${m}]`,seg));
    const key=m==="v1"?"a":"b";
    parts.forEach(([els,list,ood])=>els.forEach((el,i)=>{
      const c=list[i], p=c[key], green=p>=.5;
      el.className=ood?(c.pass?(green?"okp":"blocked"):(green?"bad":"")):(green?"bad":"");
    }));
    const from=+fp.textContent, to=G.test.filter(c=>c[key]>=.5).length, alive=run.start();
    tween(900,t=>fp.textContent=Math.round(from+(to-from)*t),alive);
  }
  paint("v1");
  seg.addEventListener("click",e=>{const b=e.target.closest("button");if(b){touched=true;paint(b.dataset.model)}});
  onView($("#grid-runs"),()=>setTimeout(()=>{if(!touched)paint("gates")},RM?0:1600),.5);
  parts.forEach(([els,list,ood])=>{
    const box=els[0]&&els[0].parentElement;if(!box)return;
    const text=i=>{const c=list[i];return `${c.t}\n${ood?(c.pass?"实际：通过":"实际：失败"):"实际：失败"}\nP(全部通过)  v1 ${c.a.toFixed(3)}  →  新权重 ${c.b.toFixed(3)}`};
    box.addEventListener("pointerover",e=>{const el=e.target.closest("i[data-i]");if(el)showTip(text(+el.dataset.i),e.clientX,e.clientY)});
    box.addEventListener("pointermove",e=>{if(tip.classList.contains("show")&&e.target.closest("i[data-i]"))showTip(tip.textContent,e.clientX,e.clientY)});
    box.addEventListener("pointerleave",hideTip);
  });
})();

/* ---------- Typed Decisions board ---------- */
(function(){
  const NAME={"AgentJev-0.6B, this run":"AgentJev-0.6B · v1","Laya, published checkpoint":"Laya · 已发布检查点","AgentJev phase 4, before this run":"AgentJev · 第 4 阶段","Prior, label frequency":"先验 · 标签频率","Uniform":"均匀分布"};
  const KIND={"Specialist":"专用","Generalist, zero-shot":"通用 · 零样本","Reference":"参照"};
  const box=$("#board");
  box.innerHTML=D.board.map((b,i)=>`<div class="brow${i===0?" me":b.kind==="Reference"?" ref":""}"><span class="nm" title="${esc(b.model)}">${esc(NAME[b.model]||b.model)}<small>${esc(KIND[b.kind]||b.kind)}</small></span><span class="bt"><i data-w="${b.top1}"></i></span><span class="v">${b.top1.toFixed(2)}%</span></div>`).join("");
  onView(box,()=>$$(".bt>i",box).forEach((el,i)=>setTimeout(()=>el.style.width=el.dataset.w+"%",RM?0:i*90)),.3);
})();

/* ---------- Claude Code hook feed ---------- */
(function(){
  const rows=$("#feed-rows"), run=runner(), dec={deny:"deny",ask:"ask"};
  rows.innerHTML=D.hook.map(h=>{
    const d=h.decision||"none", v=h.v1, changed=v.decision!==h.decision;
    const why=h.reason?`<div class="why ${d}">${esc(h.reason)}</div>`:"";
    const diff=changed?`<div class="why">v1：${v.decision?dec[v.decision]:"不表态"} · L${v.level} · P(安全) ${v.p_safe.toFixed(2)}<span class="diff">← 新权重改判</span></div>`:"";
    return `<div class="frow"><span class="cmd">${esc(h.command)}</span><span class="psafe"><span>P(安全) ${h.p_safe.toFixed(2)}</span><span class="track"><i style="width:${h.p_safe*100}%"></i></span></span><span class="lvl l${h.level}">L${h.level}</span><span class="badge ${d}">${d==="none"?"不表态":d}</span>${why}${diff}</div>`;
  }).join("");
  const sm=D.hook.map(h=>h.server_ms), hm=D.hook.map(h=>h.hook_ms);
  $("#feed-foot").textContent=`每次调用：钩子进程 ${fmt(Math.min(...hm),0)}–${fmt(Math.max(...hm),0)} ms（含 Python 启动），其中服务端 ${fmt(Math.min(...sm),1)}–${fmt(Math.max(...sm),1)} ms。`;
  async function play(){
    const alive=run.start(), els=$$(".frow",rows);
    els.forEach(e=>e.classList.remove("show"));
    for(const e of els){await sleep(420);if(!alive())return;e.classList.add("show")}
  }
  onView(rows,play,.3);
  $("#feed-replay").addEventListener("click",play);
})();

/* ---------- quickstart: code blocks and a recorded terminal ---------- */
(function(){
  const Q=D.quick, T=D.terminal, seg=$("#var-seg"), term=$("#term"), run=runner();
  let variant="v1";
  const tabs={install:"bash",ask:"curl"};
  function fill(){
    $("#code-install").textContent=Q.install[tabs.install];
    $("#code-fetch").textContent=Q.fetch[variant];
    $("#code-serve").textContent=Q.serve[variant];
    $("#code-ask").textContent=Q.ask[tabs.ask];
    $("#var-note").innerHTML=Q.note[variant];
  }
  $$(".tabs[data-tabs]").forEach(g=>g.addEventListener("click",e=>{const b=e.target.closest("[role=tab]");if(!b)return;tabs[g.dataset.tabs]=b.dataset.k;selected(g,b);fill()}));
  seg.addEventListener("click",e=>{const b=e.target.closest("button");if(!b||b.dataset.v===variant)return;variant=b.dataset.v;pressed(seg,b);fill();play()});
  fill();
  async function play(){
    const alive=run.start(), rec=T[variant];
    $("#term-title").textContent=`agent-jev · 本机实录 · ${variant==="v1"?"v1":"coding-gates"}`;
    term.innerHTML="";
    const line=(cls,text)=>{const s=document.createElement("span");if(cls)s.className=cls;s.textContent=text;term.appendChild(s);term.appendChild(document.createTextNode("\n"));term.scrollTop=term.scrollHeight;return s};
    for(const step of rec){
      if(!alive())return;
      if(step.cmd!==undefined){
        const p=line("pr","$ "), c=document.createElement("span");c.className="cm";p.after(c);
        await tween(Math.min(1100,step.cmd.length*22),t=>{c.textContent=step.cmd.slice(0,Math.round(step.cmd.length*t))},alive,linear);
        await sleep(step.pause||350);
      }else{
        for(const o of step.out){if(!alive())return;line(step.cls||"",o);await sleep(step.gap===undefined?60:step.gap)}
        await sleep(step.pause||250);
      }
    }
  }
  onView(term,play,.3);
  $("#term-replay").addEventListener("click",play);
})();
