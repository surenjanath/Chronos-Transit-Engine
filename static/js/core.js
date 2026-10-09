// core.js — the shared shell: theme, live status, row links, the dock's log console.
(function(){
  const $=s=>document.querySelector(s);
  // theme
  const root=document.documentElement;
  $('#navTheme')?.addEventListener('click',e=>{
    e.preventDefault();
    const next=root.getAttribute('data-theme')==='dark'?'light':'dark';
    root.setAttribute('data-theme',next);
    try{ localStorage.setItem('cte-theme',next); }catch(_){}
  });
  // whole-row links in ledgers
  document.addEventListener('click',e=>{
    const row=e.target.closest('tr[data-href]');
    if(row && !e.target.closest('a,button,form,input,select')) location.href=row.dataset.href;
  });
  // confirm destructive or long actions
  document.addEventListener('submit',e=>{
    const msg=e.target.dataset.confirm;
    if(msg && !window.confirm(msg)) e.preventDefault();
  });
  // segmented source switch on forms (works without a round trip)
  document.querySelectorAll('[data-switch]').forEach(group=>{
    group.addEventListener('click',e=>{
      const btn=e.target.closest('[data-value]'); if(!btn) return;
      e.preventDefault();
      group.querySelectorAll('[data-value]').forEach(b=>b.classList.toggle('on',b===btn));
      document.querySelector(group.dataset.switch).value=btn.dataset.value;
      document.querySelectorAll('[data-when]').forEach(f=>{ f.hidden=f.dataset.when!==btn.dataset.value; });
    });
  });
  // dock console
  const live=$('#live'), con=$('#console');
  async function tail(){
    if(!live || live.hidden) return;
    try{
      const d=await (await fetch(live.dataset.url)).json();
      con.innerHTML=d.lines.map(l=>`<span class="${l.level==='INFO'?'':'bad'}">${l.time}  ${l.level.padEnd(7)} ${l.stage.padEnd(20)} ${l.message.replace(/</g,'&lt;')}</span>`).join('\n');
      con.scrollTop=con.scrollHeight;
    }catch(_){}
  }
  $('#logToggle')?.addEventListener('click',()=>{ live.hidden=!live.hidden; tail(); });
  $('#liveClose')?.addEventListener('click',()=>{ live.hidden=true; });
  // live status: the header reads the pipeline; pages marked data-autoreload refresh when its state changes
  const status=$('#liveStatus'), body=document.body;
  let signature=null;
  async function poll(){
    if(document.hidden) return;
    try{
      const d=await (await fetch(body.dataset.live)).json();
      const n=d.active.length;
      if(status){
        const first=d.active.find(a=>a.stage!=='QUEUED')||d.active[0];
        status.textContent=n?`${n} running · ${first.stage.replace(/_/g,' ').toLowerCase()} ${Math.round(first.progress*100)}%`
          :d.training.status==='running'?`Training · ${Math.round(d.training.progress*100)}%`
          :d.autopilot!=null?'Autopilot · '+(d.autopilot||'starting')
          :d.benchmark.status==='running'?`Benchmark · ${d.benchmark.done}/${d.benchmark.total}`
          :d.validation==='running'?'Verifying':'Idle';
        status.classList.toggle('busy',n>0||d.training.status==='running'||d.validation==='running'||d.benchmark.status==='running'||d.autopilot!=null);
      }
      const tl=$('#liveTitle'); if(tl) tl.textContent=status.textContent;
      const mark=document.querySelector('[data-mark="live"]'); if(mark) mark.classList.toggle('on',n>0);
      const sig=[n,d.latest_finished,d.runs,d.pending,d.validation,d.training.status,d.benchmark.status,d.autopilot!=null].join('|');
      if(signature!==null && sig!==signature && body.hasAttribute('data-autoreload') && !document.querySelector('input:focus,textarea:focus,select:focus')) location.reload();
      signature=sig;
      document.dispatchEvent(new CustomEvent('cte:live',{detail:d}));
      tail();
    }catch(_){}
  }
  if(body.dataset.live){ poll(); setInterval(poll,2500); }
})();
