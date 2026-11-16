/* ---------- demo video ---------- */
(function(){
  const v=$("#video"), cover=$("#v-cover"), box=$("#chapters"), chs=D.video.chapters;
  const clock=t=>Math.floor(t/60)+":"+String(Math.floor(t%60)).padStart(2,"0");
  box.innerHTML=chs.map((c,i)=>`<button type="button" class="chapter" data-i="${i}"><span>${clock(c.t)}</span><b>${esc(c.title)}</b></button>`).join("");
  const start=()=>{cover.classList.add("hide");v.controls=true;const p=v.play();if(p&&p.catch)p.catch(()=>{})};
  cover.addEventListener("click",start);
  box.addEventListener("click",e=>{const b=e.target.closest(".chapter");if(!b)return;v.currentTime=chs[+b.dataset.i].t+.05;start()});
  v.addEventListener("play",()=>cover.classList.add("hide"));
  v.addEventListener("timeupdate",()=>{
    let cur=0;chs.forEach((c,i)=>{if(v.currentTime>=c.t)cur=i});
    $$(".chapter",box).forEach((b,i)=>b.classList.toggle("on",i===cur&&!v.paused));
  });
})();

/* ---------- why: a writer decoding vs one forward pass ---------- */
(function(){
  const toks=D.why_tokens, box=$("#w-tokens"), cnt=$("#w-cnt"), run=runner();
  const gate=D.hero[0].gates.questions[0], bars=$("#jev-bars");
  bars.innerHTML=`<div class="qa"><div class="qa-h"><span class="kind">BOOLEAN</span><span class="qa-t">${esc(gate.text)}</span></div>${rowsHTML(gate)}<div class="qa-m"></div></div>`;
  async function play(){
    const alive=run.start();
    box.innerHTML="";cnt.textContent="0 token";
    $$(".br",bars).forEach(r=>{r.dataset.p=0;$(".track>i",r).style.width="0";$(".pc",r).textContent="0.0%"});
    $(".qa-m",bars).innerHTML="";
    const pulse=$(".jev-pulse");pulse.classList.remove("go");void pulse.offsetWidth;pulse.classList.add("go");
    await sleep(380);if(!alive())return;
    setRows(bars,gate.probs,alive,700);$(".qa-m",bars).innerHTML=metaHTML(gate)+` · 服务端 <b>${fmt(D.hero[0].gates.wall_ms_median,1)} ms</b>`;
    for(let i=0;i<toks.length;i++){
      if(!alive())return;
      $$(".tk.cur",box).forEach(t=>t.classList.remove("cur"));
      box.insertAdjacentHTML("beforeend",`<span class="tk cur">${esc(toks[i])}</span>`);
      cnt.textContent=(i+1)+" token";
      await sleep(115);
    }
    $$(".tk.cur",box).forEach(t=>t.classList.remove("cur"));
  }
  onView($("#lane-writer"),play,.45);
  $(".duel").addEventListener("click",()=>play());
})();

/* ---------- primitives: one trace, three answers ---------- */
(function(){
  const t=D.examples.trace, byId=Object.fromEntries(t.questions.map(q=>[q.id,q]));
  $("#trace-line").textContent=D.trace_state_compact;
  const b=byId.needs_review, c=byId.action, s=byId.risk;
  $("#pb-q").textContent=b.text;
  $("#pb-split").innerHTML=`<div class="t" style="flex-grow:1">true <span>${pct(b.probs[0])}</span></div><div class="f" style="flex-grow:1">false <span>${pct(b.probs[1])}</span></div>`;
  $("#pc-q").textContent=c.text;
  $("#pc-bars").innerHTML=rowsHTML(c);
  $("#ps-q").textContent=s.text;
  const n=s.keys.length, pos=i=>4+92*i/(n-1);
  $("#ps-scale").innerHTML=`<div class="rail"></div>`+s.keys.map((k,i)=>`<div class="tick" style="left:${pos(i)}%"><b>L${k}</b></div><div class="mass${i===s.level?" win":""}" style="left:${pos(i)}%" title="${esc(s.candidates[i])}"></div>`).join("")+`<div class="needle" data-v="score ${s.score.toFixed(2)}"></div>`;
  onView($("#primitives .bento"),()=>{
    const [t1,f1]=$$("#pb-split>div");t1.style.flexGrow=Math.max(b.probs[0],.12);f1.style.flexGrow=Math.max(b.probs[1],.12);
    setRows($("#pc-bars"),c.probs);$("#pc-m").innerHTML=`value <b>${esc(c.value)}</b> · top <b>${pct(c.top_probability)}</b> · margin <b>${c.margin.toFixed(2)}</b>`;
    $$("#ps-scale .mass").forEach((m,i)=>m.style.height=(4+s.probs[i]*46)+"px");
    $("#ps-scale .needle").style.left=pos(s.score)+"%";
    $("#ps-m").innerHTML=`level <b>L${s.level}</b> · score = Σ i·Pᵢ = <b>${s.score.toFixed(2)}</b>`;
  },.3);
})();

/* ---------- how: eighteen paths, one prefix tree ---------- */
(function(){
  const T=D.trie, stage=$("#tree-stage"), seg=$("#tree-seg"), n=$("#tree-n"), l=$("#tree-l");
  const rows=[];T.questions.forEach((q,qi)=>q.cands.forEach((c,ci)=>rows.push({qi,ci,q:q.q,c})));
  const depth=Math.max(...rows.map(r=>T.state+r.q+r.c));
  stage.innerHTML=rows.map(()=>`<div class="tok s"></div><div class="tok q"></div><div class="tok c"></div>`).join("");
  const els=$$(".tok",stage);
  let view="paths", counter=runner();
  const groupTop=qi=>{const ix=rows.map((r,i)=>r.qi===qi?i:-1).filter(i=>i>=0);return(ix[0]+ix[ix.length-1])/2};
  function layout(){
    const W=stage.clientWidth, H=stage.clientHeight, pitch=H/rows.length, k=W/depth, h=Math.min(14,pitch-6);
    const y=i=>i*pitch+(pitch-h)/2;
    const firstOf={};
    rows.forEach((r,i)=>{
      const [s,q,c]=els.slice(i*3,i*3+3), tree=view==="tree";
      const sy=tree?y((rows.length-1)/2):y(i), qy=tree?y(groupTop(r.qi)):y(i);
      const qFirst=!(r.qi in firstOf);firstOf[r.qi]=true;
      Object.assign(s.style,{left:"0px",top:sy+"px",width:Math.max(2,T.state*k-2)+"px",height:h+"px"});
      Object.assign(q.style,{left:T.state*k+"px",top:qy+"px",width:Math.max(2,r.q*k-2)+"px",height:h+"px"});
      Object.assign(c.style,{left:(T.state+r.q)*k+"px",top:y(i)+"px",width:Math.max(2,r.c*k-2)+"px",height:h+"px"});
      s.classList.toggle("ghost",tree&&i>0);q.classList.toggle("ghost",tree&&!qFirst);
    });
  }
  function show(v){
    if(v===view)return;view=v;pressed(seg,$(`[data-view=${v}]`,seg));layout();
    const alive=counter.start(), from=+n.textContent.replace(/,/g,""), to=v==="tree"?T.trie_tokens:T.path_tokens;
    tween(900,t=>n.textContent=fmt(Math.round(from+(to-from)*t),0),alive);
    l.textContent=v==="tree"?"1 棵前缀树 · 状态只编码一次":`${rows.length} 条独立路径 · 状态重复 ${rows.length} 次`;
  }
  layout();l.textContent=`${rows.length} 条独立路径 · 状态重复 ${rows.length} 次`;
  addEventListener("resize",layout);
  seg.addEventListener("click",e=>{const b=e.target.closest("button");if(b)show(b.dataset.view)});
  onView(stage,()=>setTimeout(()=>show("tree"),RM?0:1100),.5);
})();

/* ---------- count-up on stat tiles ---------- */
$$(".stat .to").forEach(el=>onView(el,()=>{
  const node=el.firstChild;if(!node||node.nodeType!==3)return;
  const raw=node.textContent.trim(), m=raw.match(/^[\d,.]+/);if(!m)return;
  const to=parseFloat(m[0].replace(/,/g,"")), dec=(m[0].split(".")[1]||"").length, grp=m[0].includes(",");
  tween(1100,t=>node.textContent=(grp?fmt(to*t,dec):(to*t).toFixed(dec))+raw.slice(m[0].length));
},.6));
