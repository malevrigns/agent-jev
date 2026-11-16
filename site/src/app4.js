/* ---------- guide tabs ---------- */
(function(){
  const g=$("#gtabs"), hooks={};
  window.onGuideTab=(k,fn)=>{hooks[k]=fn};
  function open(b,focus){
    selected(g,b);$$(".gpanel").forEach(p=>p.classList.toggle("on",p.id==="g-"+b.dataset.g));
    if(focus)b.focus();
    if(hooks[b.dataset.g])hooks[b.dataset.g]();
  }
  g.addEventListener("click",e=>{const b=e.target.closest("[role=tab]");if(b)open(b)});
  g.addEventListener("keydown",e=>{
    if(e.key!=="ArrowRight"&&e.key!=="ArrowLeft")return;
    const bs=$$("[role=tab]",g), i=bs.indexOf(document.activeElement);if(i<0)return;
    open(bs[(i+(e.key==="ArrowRight"?1:bs.length-1))%bs.length],true);e.preventDefault();
  });
})();

/* ---------- request builder ---------- */
(function(){
  const SAMPLE=D.builder;
  let B, outMode="json";
  const sampleState=()=>typeof SAMPLE.state==="string"?SAMPLE.state:JSON.stringify(SAMPLE.state,null,2);
  function toModel(q){
    const m={id:q.id,type:q.type,question:q.question};
    if(q.type==="boolean")m.criteria={true:(q.criteria&&q.criteria.true)||"",false:(q.criteria&&q.criteria.false)||""};
    if(q.type==="choice")m.options=Object.entries(q.options||{}).map(([key,desc])=>({key,desc}));
    if(q.type==="score")m.levels=[...(q.levels||[])];
    return m;
  }
  function reset(){B={state:sampleState(),questions:SAMPLE.questions.map(toModel)};renderForm();update()}
  function blank(type,n){
    const id=(type==="boolean"?"check_":type==="choice"?"pick_":"level_")+n;
    if(type==="boolean")return{id,type,question:"",criteria:{true:"",false:""}};
    if(type==="choice")return{id,type,question:"",options:[{key:"a",desc:""},{key:"b",desc:""}]};
    return{id,type,question:"",levels:["",""]};
  }
  const del=(attr,label)=>`<button type="button" class="mini-btn" ${attr} aria-label="${label}">✕</button>`;
  function renderForm(){
    $("#b-state").value=B.state;
    $("#b-questions").innerHTML=B.questions.map((q,qi)=>{
      let body="";
      if(q.type==="boolean")body=["true","false"].map(k=>`<div class="opt cr"><span class="ix">${k}</span><input type="text" data-q="${qi}" data-f="c${k}" value="${esc(q.criteria[k])}" placeholder="可选：什么情况算 ${k}" aria-label="${k} 准则"></div>`).join("");
      if(q.type==="choice")body=q.options.map((o,oi)=>`<div class="opt"><input type="text" data-q="${qi}" data-o="${oi}" data-f="key" value="${esc(o.key)}" aria-label="选项 id"><input type="text" data-q="${qi}" data-o="${oi}" data-f="desc" value="${esc(o.desc)}" placeholder="把意思写在这里" aria-label="选项描述">${del(`data-del="${qi}:${oi}"`,"删除选项")}</div>`).join("")+`<button type="button" class="mini-btn" data-addopt="${qi}" style="align-self:flex-start">+ 选项</button>`;
      if(q.type==="score")body=q.levels.map((l,li)=>`<div class="opt lv"><span class="ix">L${li}</span><input type="text" data-q="${qi}" data-o="${li}" data-f="level" value="${esc(l)}" placeholder="${li===0?"最低一档":"更高一档"}" aria-label="等级 ${li}">${del(`data-del="${qi}:${li}"`,"删除等级")}</div>`).join("")+`<button type="button" class="mini-btn" data-addopt="${qi}" style="align-self:flex-start">+ 等级（往高处加）</button>`;
      return `<div class="qcard"><div class="row"><span class="kind">${q.type.toUpperCase()}</span><input type="text" class="qid" data-q="${qi}" data-f="id" value="${esc(q.id)}" aria-label="题目 id"><button type="button" class="mini-btn" data-delq="${qi}">删除</button></div><input type="text" data-q="${qi}" data-f="question" value="${esc(q.question)}" placeholder="题目文字" aria-label="题目文字">${body}</div>`;
    }).join("");
  }
  $("#b-form").addEventListener("input",e=>{
    const t=e.target;
    if(t.id==="b-state"){B.state=t.value;update();return}
    if(t.dataset.q===undefined)return;
    const q=B.questions[+t.dataset.q], f=t.dataset.f, oi=+t.dataset.o;
    if(f==="id")q.id=t.value;else if(f==="question")q.question=t.value;
    else if(f==="ctrue")q.criteria.true=t.value;else if(f==="cfalse")q.criteria.false=t.value;
    else if(f==="key")q.options[oi].key=t.value;else if(f==="desc")q.options[oi].desc=t.value;
    else if(f==="level")q.levels[oi]=t.value;
    update();
  });
  $("#b-form").addEventListener("click",e=>{
    const t=e.target.closest("button");if(!t)return;
    if(t.dataset.add)B.questions.push(blank(t.dataset.add,B.questions.length+1));
    else if(t.dataset.delq!==undefined)B.questions.splice(+t.dataset.delq,1);
    else if(t.dataset.addopt!==undefined){const q=B.questions[+t.dataset.addopt];if(q.type==="choice")q.options.push({key:String.fromCharCode(97+q.options.length%26)+(q.options.length>=26?q.options.length:""),desc:""});else q.levels.push("")}
    else if(t.dataset.del){const[qi,oi]=t.dataset.del.split(":").map(Number);const q=B.questions[qi];(q.type==="choice"?q.options:q.levels).splice(oi,1)}
    else if(t.id==="b-reset"){reset();return}
    else return;
    renderForm();update();
  });
  /* The server's own serialization: sorted keys, compact separators (jev_service/contract.py). */
  function canonical(v){
    if(Array.isArray(v))return"["+v.map(canonical).join(",")+"]";
    if(v&&typeof v==="object")return"{"+Object.keys(v).sort().map(k=>JSON.stringify(k)+":"+canonical(v[k])).join(",")+"}";
    return JSON.stringify(v);
  }
  function parseState(text){
    const s=text.trim();
    if(!s)return{error:"state 不能为空"};
    if(s.startsWith("{")||s.startsWith("[")){try{const v=JSON.parse(s);if(v&&typeof v==="object")return{value:v,text:canonical(v),json:true}}catch(e){return{error:"state 看起来像 JSON，但解析失败："+e.message}}}
    return{value:text,text,json:false};
  }
  function payloadOf(){
    const st=parseState(B.state), issues=[];if(st.error)issues.push(st.error);
    const ids=new Set(), qs=[];let paths=0;
    B.questions.forEach((q,i)=>{
      const tag=`第 ${i+1} 题`;
      if(!q.id.trim())issues.push(`${tag}：id 不能为空`);else if(ids.has(q.id))issues.push(`${tag}：id “${q.id}” 重复`);ids.add(q.id);
      if(!q.question.trim())issues.push(`${tag}：题目文字不能为空`);
      const out={id:q.id,type:q.type,question:q.question};let cands=[];
      if(q.type==="boolean"){
        const c={};if(q.criteria.true.trim())c.true=q.criteria.true;if(q.criteria.false.trim())c.false=q.criteria.false;
        if(Object.keys(c).length)out.criteria=c;cands=[c.true||"TRUE",c.false||"FALSE"];
      }
      if(q.type==="choice"){
        const o={}, keys=new Set();
        q.options.forEach((x,j)=>{if(!x.key.trim())issues.push(`${tag}：第 ${j+1} 个选项缺少 id`);else if(keys.has(x.key))issues.push(`${tag}：选项 id “${x.key}” 重复`);keys.add(x.key);if(!x.desc.trim())issues.push(`${tag}：选项 “${x.key||j+1}” 没有描述`);o[x.key]=x.desc});
        if(q.options.length<2||q.options.length>255)issues.push(`${tag}：Choice 需要 2–255 个选项`);
        out.options=o;cands=q.options.map(x=>x.desc);
      }
      if(q.type==="score"){
        q.levels.forEach((l,j)=>{if(!l.trim())issues.push(`${tag}：等级 L${j} 没有描述`)});
        if(q.levels.length<2||q.levels.length>10)issues.push(`${tag}：Score 需要 2–10 个等级`);
        out.levels=[...q.levels];cands=[...q.levels];
      }
      if(new Set(cands).size!==cands.length)issues.push(`${tag}：同一道题的候选描述不能重复`);
      paths+=cands.length;qs.push({out,cands});
    });
    if(!B.questions.length)issues.push("至少需要一道题");
    if(B.questions.length>128)issues.push("一次最多 128 道题");
    if(paths>1024)issues.push("一次最多 1,024 条候选路径");
    return{st,issues,paths,payload:{state:st.json?st.value:B.state,questions:qs.map(x=>x.out)},qs};
  }
  function py(v,ind=0){
    const pad=" ".repeat(ind), pad2=" ".repeat(ind+4);
    if(v===true)return"True";if(v===false)return"False";if(v===null)return"None";
    if(typeof v==="string"||typeof v==="number")return JSON.stringify(v);
    if(Array.isArray(v))return v.length?"[\n"+v.map(x=>pad2+py(x,ind+4)).join(",\n")+",\n"+pad+"]":"[]";
    const ks=Object.keys(v);return ks.length?"{\n"+ks.map(k=>pad2+JSON.stringify(k)+": "+py(v[k],ind+4)).join(",\n")+",\n"+pad+"}":"{}";
  }
  function render(p){
    const body=JSON.stringify(p.payload,null,2);
    if(outMode==="json")return body;
    if(outMode==="curl")return"curl -s http://127.0.0.1:8149/api/evaluate \\\n  -H 'Content-Type: application/json' \\\n  -d '"+JSON.stringify(p.payload).replace(/'/g,"'\\''")+"'";
    if(outMode==="ps")return"$body = @'\n"+body+"\n'@\nInvoke-RestMethod -Uri http://127.0.0.1:8149/api/evaluate -Method Post `\n  -ContentType 'application/json; charset=utf-8' `\n  -Body ([Text.Encoding]::UTF8.GetBytes($body)) | ConvertTo-Json -Depth 8";
    const st=p.st.json?py(p.st.value):JSON.stringify(B.state), qs=p.payload.questions;
    let call;
    if(qs.length===1&&qs[0].type==="boolean")call=`gate = jev.decide_boolean(\n    state,\n    ${JSON.stringify(qs[0].question)},\n    criteria=${py(qs[0].criteria||{true:"The statement is true.",false:"The statement is false."},4)},\n)\nprint(gate["decision"], gate["prob_true"])`;
    else if(qs.length===1&&qs[0].type==="choice")call=`route = jev.decide_choice(\n    state,\n    ${JSON.stringify(qs[0].question)},\n    options=${py(qs[0].options,4)},\n)\nprint(route["best_action"], route["probability"], route["margin"])`;
    else if(qs.length===1&&qs[0].type==="score")call=`risk = jev.score(\n    state,\n    ${JSON.stringify(qs[0].question)},\n    levels=${py(qs[0].levels,4)},\n)\nprint(risk["level"], risk["expected_score"])`;
    else call=`# 同一状态的多道题放进一次调用，状态只编码一次\nresult = jev.evaluate(state, ${py(qs)})\nfor answer in result["answers"]:\n    print(answer["id"], answer["distribution"])`;
    return"from agentjev_client import AgentJev\n\njev = AgentJev(\"http://127.0.0.1:8149\")\n"+(p.st.json?"# dict / list 原样发送，服务端按键排序、紧凑序列化\n":"")+"state = "+st+"\n\n"+call;
  }
  function update(){
    const p=payloadOf();
    $("#b-issues").innerHTML=p.issues.length?p.issues.slice(0,6).map(s=>`<div class="b-issue">${esc(s)}</div>`).join("")+(p.issues.length>6?`<div class="b-issue">还有 ${p.issues.length-6} 处</div>`:""):`<div class="b-ok">符合服务端的规则，可以直接发送</div>`;
    $("#b-meter").innerHTML=`<span>题目 <b>${B.questions.length}</b> / 128</span><span>候选路径 <b>${p.paths}</b> / 1024</span><span>state：<b>${p.st.error?"—":p.st.json?"JSON（服务端排序键、紧凑序列化）":"文字"}</b></span>`;
    $("#b-out").textContent=render(p);
    const first=p.qs[0];
    if(first&&!p.st.error){
      const s=p.st.text.startsWith("[STATE]")?p.st.text.slice(7):" "+p.st.text;
      $("#b-peek").innerHTML=`<span class="m">[STATE]</span>${esc(s)}\n<span class="m">[QUESTION]</span> ${esc(first.out.question)}\n<span class="m">[CANDIDATE]</span> ${esc(first.cands[0]||"")}`;
    }else $("#b-peek").textContent="—";
  }
  $("#b-tabs").addEventListener("click",e=>{const t=e.target.closest("[data-out]");if(!t)return;outMode=t.dataset.out==="client"?"client":t.dataset.out;selected($("#b-tabs"),t);update()});
  $("#b-copy").addEventListener("click",e=>copyText(e.currentTarget,$("#b-out").textContent));
  reset();
})();

/* ---------- reading the distribution: a threshold over real probes ---------- */
(function(){
  const P=D.probes, range=$("#th"), out=$("#th-out"), seg=$("#th-seg"), box=$("#th-rows"), sum=$("#th-sum");
  let model="gates";
  function paint(){
    const th=+range.value, list=P[model];
    out.textContent=th.toFixed(2);range.style.setProperty("--p",((th-.05)/.9*100)+"%");box.style.setProperty("--p",(th*100)+"%");
    let pass=0,wrong=0;
    box.innerHTML=list.map(r=>{
      const ok=r.p_true>=th, right=ok===r.gold_true;pass+=ok;wrong+=!right;
      const badge=ok?(right?"none":"deny"):(right?"none":"ask");
      return `<div class="trow"><span class="s">${esc(r.state)}</span><span class="gold-t"><span class="track"><i style="width:${r.p_true*100}%"></i></span><span class="mark"></span></span><span class="pv">${r.p_true.toFixed(3)}</span><span class="out badge ${badge}">${ok?"放行":"拦下"}${right?"":" ✗"}</span></div>`;
    }).join("");
    sum.textContent=`放行 ${pass} / ${list.length} · 判错 ${wrong}（实际：${list.filter(r=>r.gold_true).length} 段全部通过）`;
  }
  range.addEventListener("input",paint);
  seg.addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;model=b.dataset.model;pressed(seg,b);paint()});
  paint();
})();

/* ---------- batching: twenty test cases, replayed at recorded speed ---------- */
(function(){
  const R=D.race, clock=$("#race-clock"), btn=$("#race-play"), run=runner();
  const cum=a=>a.reduce((acc,x)=>(acc.push((acc.length?acc[acc.length-1]:0)+x),acc),[]);
  const lanes=[["before",cum(R.before)],["after",cum(R.after)]];
  lanes.forEach(([k])=>$("#race-"+k).innerHTML="<i></i>".repeat(20));
  const total=Math.max(...lanes.map(([,c])=>c[c.length-1]));
  function frame(ms){
    clock.textContent=fmt(ms,0)+" ms";
    lanes.forEach(([k,c])=>{
      $$("#race-"+k+" i").forEach((el,i)=>el.classList.toggle("on",ms>=c[i]));
      const done=c.filter(x=>x<=ms).length;
      $("#race-"+k+"-v").textContent=done===20?fmt(c[19],0)+" ms":done+" / 20";
    });
  }
  function play(){
    const alive=run.start();btn.textContent="重放";
    if(RM){frame(total);return}
    const t0=performance.now();
    (function tick(now){if(!alive())return;const ms=Math.min(total,now-t0);frame(ms);if(ms<total)requestAnimationFrame(tick)})(t0);
  }
  frame(0);
  btn.addEventListener("click",play);
  let ran=false;onGuideTab("batch",()=>{if(!ran){ran=true;setTimeout(play,300)}});
})();
