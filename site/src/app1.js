"use strict";
/* ---------- helpers ---------- */
const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt=(n,d=1)=>Number(n).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=p=>p<0.001?"<0.1%":p>0.999?">99.9%":(p*100).toFixed(1)+"%";
const RM=matchMedia("(prefers-reduced-motion: reduce)").matches;
const wait=ms=>new Promise(r=>setTimeout(r,ms));
const sleep=ms=>wait(RM?0:ms);
const easeOut=t=>1-Math.pow(1-t,3), linear=t=>t;
/* Run fn(eased progress) every frame for ms; resolves early when alive() turns false. */
function tween(ms,fn,alive=()=>true,ease=easeOut){
  return new Promise(res=>{
    if(RM||ms<=0){fn(1);res();return}
    const t0=performance.now();
    (function frame(now){
      if(!alive()){res();return}
      const t=Math.min(1,(now-t0)/ms);fn(ease(t));
      t<1?requestAnimationFrame(frame):res();
    })(t0);
  });
}
/* Call fn once, the first time el is on screen. */
function onView(el,fn,threshold=.3){
  if(!el)return;
  if(!("IntersectionObserver" in window)){fn();return}
  const io=new IntersectionObserver(es=>{if(es.some(e=>e.isIntersecting)){io.disconnect();fn()}},{threshold});
  io.observe(el);
}
/* Track whether el is on screen; returns a getter. */
function visibility(el){
  let v=false;
  if(!("IntersectionObserver" in window))return()=>true;
  new IntersectionObserver(es=>{v=es[es.length-1].isIntersecting},{threshold:.15}).observe(el);
  return()=>v;
}
/* A restartable animation: start() cancels the previous run and returns its alive() check. */
function runner(){let id=0;return{start(){const mine=++id;return()=>mine===id},stop(){id++}}}
function pressed(group,btn){$$("button",group).forEach(b=>b.setAttribute("aria-pressed",String(b===btn)))}
function selected(group,btn){$$("[role=tab]",group).forEach(b=>b.setAttribute("aria-selected",String(b===btn)))}
const argmax=a=>a.indexOf(Math.max(...a));

/* ---------- copy buttons ---------- */
function copyText(btn,text){
  const label=btn.dataset.label||(btn.dataset.label=btn.textContent);
  const done=msg=>{btn.textContent=msg;btn.classList.add("done");setTimeout(()=>{btn.textContent=label;btn.classList.remove("done")},1400)};
  const fallback=()=>{
    const ta=document.createElement("textarea");ta.value=text;ta.style.cssText="position:fixed;opacity:0;top:0";document.body.appendChild(ta);ta.select();
    let ok=false;try{ok=document.execCommand("copy")}catch(e){}
    ta.remove();done(ok?"已复制":"请手动复制");
  };
  if(navigator.clipboard&&window.isSecureContext)navigator.clipboard.writeText(text).then(()=>done("已复制"),fallback);else fallback();
}
document.addEventListener("click",e=>{
  const b=e.target.closest("[data-copy],[data-copy-from]");if(!b)return;
  copyText(b,b.dataset.copy!==undefined?b.dataset.copy:$("#"+b.dataset.copyFrom).textContent);
});

/* ---------- nav, progress, active section ---------- */
const nav=$("#nav"), prog=$("#progress"), navLinks=$$("#nav-links a");
const navTargets=navLinks.map(a=>document.getElementById(a.getAttribute("href").slice(1)));
let scrollQueued=false;
function onScroll(){
  scrollQueued=false;
  nav.classList.toggle("scrolled",scrollY>12);
  const max=document.documentElement.scrollHeight-innerHeight;
  prog.style.width=(max>0?Math.min(1,scrollY/max)*100:0)+"%";
  let cur=-1;navTargets.forEach((t,i)=>{if(t&&t.getBoundingClientRect().top<innerHeight*.4)cur=i});
  navLinks.forEach((a,i)=>a.classList.toggle("active",i===cur));
}
addEventListener("scroll",()=>{if(!scrollQueued){scrollQueued=true;requestAnimationFrame(onScroll)}},{passive:true});
onScroll();
const menuBtn=$("#menu-btn");
menuBtn.addEventListener("click",()=>{const open=!nav.classList.contains("open");nav.classList.toggle("open",open);menuBtn.setAttribute("aria-expanded",String(open));menuBtn.setAttribute("aria-label",open?"关闭菜单":"打开菜单")});
navLinks.forEach(a=>a.addEventListener("click",()=>{nav.classList.remove("open");menuBtn.setAttribute("aria-expanded","false")}));

/* ---------- reveal on scroll, card spotlight ---------- */
if("IntersectionObserver" in window&&!RM){
  const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add("in");io.unobserve(e.target)}}),{threshold:.12,rootMargin:"0px 0px -6% 0px"});
  $$(".reveal").forEach(el=>io.observe(el));
}else $$(".reveal").forEach(el=>el.classList.add("in"));
document.addEventListener("pointermove",e=>{
  const c=e.target.closest&&e.target.closest(".card");if(!c)return;
  const r=c.getBoundingClientRect();c.style.setProperty("--mx",(e.clientX-r.left)+"px");c.style.setProperty("--my",(e.clientY-r.top)+"px");
},{passive:true});
requestAnimationFrame(()=>requestAnimationFrame(()=>document.body.classList.add("loaded")));

/* ---------- distribution rows (console, primitives, why) ---------- */
function rowLabel(q,i){
  const k=q.keys[i],c=q.candidates[i];
  if(q.type==="boolean")return k+" · "+c;
  if(q.type==="score")return "L"+k+" · "+c.replace(/^\d+\s*-\s*/,"");
  return k;
}
function metaHTML(q){
  if(q.type==="boolean")return `value <b>${q.value}</b> · P(true) <b>${pct(q.probs[0])}</b>`;
  if(q.type==="choice")return `value <b>${esc(q.value)}</b> · margin <b>${q.margin.toFixed(2)}</b>`;
  return `level <b>L${q.level}</b> · score <b>${q.score.toFixed(2)}</b> / ${q.keys.length-1}`;
}
function rowsHTML(q){
  const w=argmax(q.probs);
  return q.keys.map((k,i)=>`<div class="br${i===w?" win":""}" title="${esc(q.candidates[i])}"><span class="lab">${esc(rowLabel(q,i))}</span><span class="track"><i></i></span><span class="pc">0.0%</span></div>`).join("");
}
/* Move every bar in box to probs (CSS animates the width); count the percentages. */
function setRows(box,probs,alive,ms=900){
  const rows=$$(".br",box), w=argmax(probs);
  const from=rows.map(r=>+(r.dataset.p||0));
  rows.forEach((r,i)=>{r.classList.toggle("win",i===w);r.dataset.p=probs[i];$(".track>i",r).style.width=(probs[i]*100)+"%"});
  return tween(ms,t=>rows.forEach((r,i)=>{$(".pc",r).textContent=pct(from[i]+(probs[i]-from[i])*t)}),alive);
}

/* ---------- hero console: recorded requests, replayed ---------- */
(function(){
  const H=D.hero, pre=$("#c-pre"), qs=$("#c-qs"), foot=$("#c-foot"), verdict=$("#c-verdict"), tok=$("#c-tok");
  const tabs=$("#c-tabs"), modelSeg=$("#c-model"), run=runner(), seen=visibility($("#console"));
  let idx=0, model="gates", shown=false;
  tabs.innerHTML=H.map((h,i)=>`<button type="button" data-i="${i}" aria-pressed="${i===0}">${esc(h.label)}</button>`).join("");
  $("#c-layers").innerHTML="<i></i>".repeat(28);
  const layers=$$("#c-layers i");
  const verdictText={deny:"deny",ask:"ask",none:"不表态"};
  function finish(h,alive){
    const s=h[model], u=s.usage, ms=s.wall_ms_median||u.wall_ms;
    $$(".qa",qs).forEach((box,qi)=>{setRows(box,s.questions[qi].probs,alive);$(".qa-m",box).innerHTML=metaHTML(s.questions[qi])});
    foot.innerHTML=`<span>候选路径 <b>${u.candidate_paths}</b></span><span>骨干 token <b>${fmt(u.input_path_tokens,0)} → ${fmt(u.backbone_input_tokens,0)}</b></span><span>生成 <b class="g">0</b> token</span><span>服务端 <b>${fmt(ms,1)} ms</b>${s.wall_ms_median?" · 同一请求 15 次的中位数":""}</span>`;
    if(h.key==="hook"){
      const d=s.decision||"none";
      verdict.className="verdict "+d;
      verdict.innerHTML=`<b>${verdictText[d]}</b><span>${esc(s.reason||"不输出任何内容，按你自己的权限设置走")}</span>`;
    }
  }
  async function play(i,auto){
    const alive=run.start(); idx=i; shown=false;
    pressed(tabs,tabs.children[i]);
    const h=H[i], s=h[model];
    $("#c-src").textContent=h.source;
    $("#c-qn").textContent=`${s.questions.length} 题 · ${s.usage.candidate_paths} 条候选`;
    qs.innerHTML="";foot.classList.remove("show");foot.innerHTML="";verdict.className="verdict";verdict.innerHTML="";
    layers.forEach(l=>l.classList.remove("on"));tok.textContent="—";
    const text=h.display;
    await tween(Math.min(1600,450+text.length*1.6),t=>{pre.innerHTML=esc(text.slice(0,Math.round(text.length*t)))+(t<1?'<span class="caret"></span>':"")},alive,linear);
    if(!alive())return;
    await tween(700,t=>{const n=Math.round(t*28);layers.forEach((l,j)=>l.classList.toggle("on",j<n));tok.textContent=fmt(Math.round(t*s.usage.backbone_input_tokens),0)},alive,linear);
    if(!alive())return;
    qs.innerHTML=s.questions.map(q=>`<div class="qa"><div class="qa-h"><span class="kind">${q.type.toUpperCase()}</span><span class="qa-t">${esc(q.text)}</span></div>${rowsHTML(q)}<div class="qa-m"></div></div>`).join("");
    await wait(30);
    shown=true;finish(h,alive);
    await sleep(500);if(!alive())return;
    layers.forEach(l=>l.classList.remove("on"));
    foot.classList.add("show");if(h.key==="hook")verdict.classList.add("show");
    if(RM)return;
    await wait(6500);
    while(alive()&&!seen())await wait(500);
    if(alive())play((idx+1)%H.length,true);
  }
  tabs.addEventListener("click",e=>{const b=e.target.closest("button");if(b)play(+b.dataset.i)});
  modelSeg.addEventListener("click",e=>{
    const b=e.target.closest("button");if(!b||b.dataset.model===model)return;
    model=b.dataset.model;pressed(modelSeg,b);
    if(shown){finish(H[idx],()=>true);foot.classList.add("show");if(H[idx].key==="hook")verdict.classList.add("show")}
  });
  play(0);
})();
