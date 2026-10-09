// overview.js — the radial classification dial and the 14-day throughput plot.
(function(){
  const d=jsonData('overviewData');
  // radial: one ring, a segment per class, ticks around it; identity is carried by the labels beside each arc
  const svg=document.getElementById('radial'), NS='http://www.w3.org/2000/svg';
  const total=d.classes.reduce((a,c)=>a+c.count,0), cx=140, cy=140, R=92;
  const add=(tag,attrs,text)=>{ const e=document.createElementNS(NS,tag); for(const k in attrs) e.setAttribute(k,attrs[k]); if(text!=null) e.textContent=text; svg.append(e); return e; };
  for(let i=0;i<72;i++){ const a=i/72*2*Math.PI, r1=118, r2=i%6?123:128;
    add('line',{x1:cx+r1*Math.sin(a),y1:cy-r1*Math.cos(a),x2:cx+r2*Math.sin(a),y2:cy-r2*Math.cos(a),stroke:'var(--faint)','stroke-width':1}); }
  add('circle',{cx,cy,r:R,fill:'none',stroke:'var(--rule)','stroke-width':16});
  const tones=['var(--ink)','var(--mute)','var(--faint)'];
  let a0=0;
  d.classes.forEach((c,i)=>{
    if(!total||!c.count) return;
    const frac=c.count/total, gap=d.classes.filter(x=>x.count).length>1?0.012:0, a1=a0+frac;
    const s=(a0+gap)*2*Math.PI, e=(a1-gap)*2*Math.PI;
    if(frac>=0.999) add('circle',{cx,cy,r:R,fill:'none',stroke:tones[i],'stroke-width':16});
    else add('path',{d:`M${cx+R*Math.sin(s)} ${cy-R*Math.cos(s)} A${R} ${R} 0 ${e-s>Math.PI?1:0} 1 ${cx+R*Math.sin(e)} ${cy-R*Math.cos(e)}`,fill:'none',stroke:tones[i],'stroke-width':16})
      .append(Object.assign(document.createElementNS(NS,'title'),{textContent:`${c.label}: ${c.count} (${Math.round(frac*100)}%)`}));
    a0=a1;
  });
  const lead=d.classes[0];
  add('text',{x:cx,y:cy+4,'text-anchor':'middle','font-size':44,'font-weight':500,fill:'var(--ink)','letter-spacing':'-0.03em'},total?Math.round(lead.count/total*100)+'%':'—');
  add('text',{x:cx,y:cy+26,'text-anchor':'middle','font-size':12,fill:'var(--mute)'},total?'classified planet':'no runs yet');

  const idx=d.daily.labels.map((_,i)=>i);
  CTEPlot(document.getElementById('dailyPlot'),{height:190,
    series:[{name:'Completed runs',x:idx,y:d.daily.runs,mode:'line',tone:'ink',width:1.8},
            {name:'Planet candidates',x:idx,y:d.daily.candidates,mode:'line',tone:'mute',width:1.8,dash:true}],
    xFmt:i=>Number.isInteger(i)&&d.daily.labels[i]?d.daily.labels[i]:'', yFmt:y=>Number.isInteger(y)?String(y):''});
})();
