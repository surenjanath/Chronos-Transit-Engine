// calculator.js — sliders mirror the number boxes; results and the model transit refresh as you move them.
(function(){
  const form=document.getElementById('calcForm');
  const first=jsonData('calcCurve');
  const system=CTESystem(document.getElementById('calcSystem'),jsonData('calcGeometry'),{hours:first.hours,flux:first.flux});
  const draw=(c,g)=>system.update(g,c);
  const ranges=[...form.querySelectorAll('input[type=range]')];
  const toPos=(r,v)=>{ const lo=+r.dataset.min, hi=+r.dataset.max; v=Math.min(Math.max(v,lo),hi);
    return 1000*('log' in r.dataset?Math.log(v/lo)/Math.log(hi/lo):(v-lo)/(hi-lo)); };
  const toVal=r=>{ const lo=+r.dataset.min, hi=+r.dataset.max, p=r.value/1000;
    return Number(('log' in r.dataset?lo*Math.pow(hi/lo,p):lo+p*(hi-lo)).toPrecision(5)); };
  const sync=()=>ranges.forEach(r=>{ r.value=toPos(r,+form[r.dataset.for].value); });
  sync();
  let timer=null, seq=0;
  const fmt=(v,d)=>v==null?'—':Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
  async function compute(){
    const mine=++seq, qs=new URLSearchParams(new FormData(form)); qs.delete('record');
    const res=await fetch(form.dataset.url+'?'+qs);
    if(mine!==seq) return;
    const box=document.getElementById('calcWarnings');
    if(!res.ok){ box.innerHTML='<p class="flagline high">One of the inputs is out of range.</p>'; return; }
    const d=await res.json();
    document.querySelectorAll('[data-out]').forEach(el=>{ el.textContent=fmt(d.parameters[el.dataset.out],+el.dataset.d); });
    box.innerHTML=''; d.warnings.forEach(w=>{ const p=document.createElement('p'); p.className='flagline high'; p.textContent=w; box.append(p); });
    draw(d.curve,d.geometry);
    history.replaceState(null,'','?'+qs);
  }
  const queue=()=>{ clearTimeout(timer); timer=setTimeout(()=>compute().catch(()=>{}),70); };
  ranges.forEach(r=>r.addEventListener('input',()=>{ form[r.dataset.for].value=toVal(r); queue(); }));
  form.querySelectorAll('input[type=number]').forEach(n=>n.addEventListener('input',()=>{ sync(); queue(); }));
  form.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',()=>{
    const p=JSON.parse(b.dataset.preset); for(const k in p) form[k].value=p[k]; sync(); queue(); }));
})();
