// neural.js — the classifier drawn as it is built: two inputs, two convolutional branches, a merged head, three answers.
// Signals travel left to right in waves. With real data (a finished run) the inputs are the run's own folded curves and
// the answers its own probabilities; without (a run in progress) the network idles until its stage arrives.
// CTENeural(el, {global:[], local:[], probs:{planet,binary,noise}, final:{...}|null, state:'idle'|'thinking'|'done'}) -> {set(state)}
(function(){
  const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const TAU=6.283185307;
  const GLOBAL=[['conv 16',5],['conv 32',7],['conv 64',9]], LOCAL=[['conv 24',6],['conv 48',8]], HEAD=[['dense 256',10],['dense 64',7]];
  const OUT=[['Planet','planet'],['Eclipsing binary','binary'],['Noise','noise']];
  const rnd=i=>{ const x=Math.sin(i*127.1+311.7)*43758.5453; return x-Math.floor(x); };
  window.CTENeural=function(el,opt){
    el.classList.add('neural'); el.innerHTML=''; el.style.height=(el.dataset.height||320)+'px'; el.style.position='relative'; el.style.overflow='hidden';
    const cv=document.createElement('canvas'); el.append(cv);
    let state=opt.state||'idle', W=0,H=0, t0=performance.now(), raf=0;
    const reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    function layout(){
      W=el.clientWidth; H=el.clientHeight;
      const inW=Math.min(170,W*0.2), outX=W-Math.min(190,W*0.24), x0=inW+34, x1=outX-30, top=26, bot=H-18, mid=(top+bot)/2;
      const colX=k=>x0+(x1-x0)*k/5;                              // six columns between the inputs and the answers
      const col=(k,n,y0,y1)=>Array.from({length:n},(_,i)=>({x:colX(k),y:y0+(y1-y0)*(n===1?0.5:i/(n-1)),k}));
      const g=GLOBAL.map((l,i)=>col(i,l[1],top+6,mid-14)), lo=LOCAL.map((l,i)=>col(i+0.5,l[1],mid+14,bot-6));
      const head=HEAD.map((l,i)=>col(3.6+i*0.9,l[1],top+34,bot-34));
      const outs=OUT.map((o,i)=>({x:outX,y:mid+(i-1)*Math.min(58,(bot-top)/3.4)}));
      const edges=[]; let id=0;
      const link=(A,B,stage)=>A.forEach((a,i)=>{ for(let j=0;j<B.length;j++){ if(rnd(id++)<Math.min(1,3.2/B.length)) edges.push({a,b:B[j],stage,seed:rnd(id+9)}); } });
      link(g[0],g[1],1); link(g[1],g[2],2); link(lo[0],lo[1],1.5); link(g[2],head[0],3); link(lo[1],head[0],3); link(head[0],head[1],4);
      head[1].forEach((a,i)=>outs.forEach((b,j)=>edges.push({a,b,stage:5,seed:rnd(i*7+j),out:j})));
      const gIn={x:8,y:top,w:inW,h:mid-14-top}, lIn={x:8,y:mid+14,w:inW,h:bot-(mid+14)};
      g[0].forEach((b,i)=>edges.push({a:{x:gIn.x+gIn.w,y:gIn.y+gIn.h*(0.2+0.6*i/(g[0].length-1))},b,stage:0,seed:rnd(i+40)}));
      lo[0].forEach((b,i)=>edges.push({a:{x:lIn.x+lIn.w,y:lIn.y+lIn.h*(0.2+0.6*i/(lo[0].length-1))},b,stage:0,seed:rnd(i+80)}));
      return {g,lo,head,outs,edges,gIn,lIn,outX,top,bot,mid,colX};
    }
    let L=null;
    function spark(g,box,data,C,label){
      g.strokeStyle=C.rule; g.strokeRect(box.x+.5,box.y+.5,box.w-1,box.h-1);
      g.fillStyle=C.mute; g.textAlign='left'; g.textBaseline='bottom'; g.fillText(label,box.x,box.y-3);
      const rows=Array.isArray(data&&data[0])?data:[data||[]];
      rows.forEach((row,r)=>{ if(!row||!row.length) return; const lo=Math.min(...row), hi=Math.max(...row), span=(hi-lo)||1, h=box.h/rows.length;
        g.strokeStyle=r===0?C.ink:C.mute; g.lineWidth=r===0?1.5:1; g.beginPath();
        row.forEach((v,i)=>{ const x=box.x+4+(box.w-8)*i/(row.length-1), y=box.y+r*h+4+(h-8)*(1-(v-lo)/span); i?g.lineTo(x,y):g.moveTo(x,y); }); g.stroke(); });
      g.lineWidth=1;
    }
    function draw(now){
      raf=0;
      const dpr=window.devicePixelRatio||1; if(el.clientWidth!==W||el.clientHeight!==H||!L){ L=layout(); cv.width=W*dpr; cv.height=H*dpr; }
      const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
      const C={ink:cssv('--ink'),mute:cssv('--mute'),faint:cssv('--faint'),rule:cssv('--rule'),canvas:cssv('--canvas'),gap:cssv('--gap'),ok:cssv('--ok')};
      g.font='11px '+getComputedStyle(document.body).fontFamily;
      const period=state==='thinking'?2600:4200, wave=state==='idle'?-1:(((now-t0)%period)/period)*7-0.5;   // position of the wave, in stages
      const lit=s=>state==='idle'?0:Math.max(0,1-Math.abs(wave-s)/0.9);
      // inputs: synthetic placeholders are never drawn; an input box stays empty until there is real data
      spark(g,L.gIn,opt.global,C,'Whole orbit · 201 bins'); spark(g,L.lIn,opt.local,C,opt.local?'The transit · 61 bins (first of 4 views)':'The transit · 4 views × 61 bins');
      if(!opt.global){ g.fillStyle=C.mute; g.textAlign='center'; g.textBaseline='middle'; g.fillText('waiting for the folded curve',L.gIn.x+L.gIn.w/2,L.gIn.y+L.gIn.h/2); g.fillText('waiting',L.lIn.x+L.lIn.w/2,L.lIn.y+L.lIn.h/2); }
      // connections
      for(const e of L.edges){ const a=lit(e.stage+0.5); g.globalAlpha=0.10+0.5*a*(0.4+0.6*e.seed); g.strokeStyle=C.ink; g.beginPath(); g.moveTo(e.a.x,e.a.y); g.lineTo(e.b.x,e.b.y); g.stroke(); }
      g.globalAlpha=1;
      // travelling pulses
      if(state!=='idle' && !reduce) for(const e of L.edges){ const u=wave-e.stage+ (e.seed-0.5)*0.5; if(u<0||u>1) continue;
        g.fillStyle=C.gap; g.globalAlpha=0.85*(1-Math.abs(u-0.5)*1.2); g.beginPath(); g.arc(e.a.x+(e.b.x-e.a.x)*u,e.a.y+(e.b.y-e.a.y)*u,1.8,0,TAU); g.fill(); }
      g.globalAlpha=1;
      // units
      const cols=[...L.g.map((c,i)=>[c,i+1,GLOBAL[i][0],-1]),...L.lo.map((c,i)=>[c,i+1.5,LOCAL[i][0],1]),...L.head.map((c,i)=>[c,i+4,HEAD[i][0],-1])];
      for(const [c,s,name,side] of cols){
        c.forEach((n,i)=>{ const a=lit(s)*(0.35+0.65*rnd(i*13+s*29)); g.fillStyle=C.canvas; g.beginPath(); g.arc(n.x,n.y,4.2,0,TAU); g.fill();
          g.strokeStyle=C.ink; g.globalAlpha=0.55; g.stroke(); g.globalAlpha=a; g.fillStyle=C.ink; g.beginPath(); g.arc(n.x,n.y,4.2,0,TAU); g.fill(); g.globalAlpha=1; });
        g.fillStyle=C.mute; g.textAlign='center'; g.textBaseline=side<0?'bottom':'top'; g.fillText(name,c[0].x,side<0?c[0].y-9:c[c.length-1].y+9);
      }
      // answers
      const p=opt.probs, best=p?OUT.reduce((b,o,i)=>p[o[1]]>p[OUT[b][1]]?i:b,0):-1, shown=state==='done'||(p&&wave>5.4)||(p&&reduce);
      L.outs.forEach((o,i)=>{ const v=p&&shown?p[OUT[i][1]]:0, a=p?(shown?0.25+0.75*v:lit(6)*0.4):lit(6)*0.5;
        g.fillStyle=C.canvas; g.beginPath(); g.arc(o.x,o.y,7,0,TAU); g.fill(); g.strokeStyle=C.ink; g.stroke();
        g.globalAlpha=a; g.fillStyle=i===best&&shown?C.gap:C.ink; g.beginPath(); g.arc(o.x,o.y,7,0,TAU); g.fill(); g.globalAlpha=1;
        g.textAlign='left'; g.textBaseline='middle'; g.fillStyle=C.ink; g.fillText(OUT[i][0],o.x+14,o.y-7);
        const bw=Math.max(40,W-o.x-22); g.fillStyle=C.rule; g.fillRect(o.x+14,o.y+3,bw,4);
        if(p&&shown){ g.fillStyle=i===best?C.gap:C.ink; g.fillRect(o.x+14,o.y+3,bw*v,4); g.fillStyle=C.mute; g.textAlign='right'; g.fillText((v*100).toFixed(1)+'%',o.x+14+bw,o.y-7); } });
      g.fillStyle=C.mute; g.textAlign='left'; g.textBaseline='top';
      g.textAlign='right'; g.fillText(state==='idle'?'Has not seen this star yet':state==='thinking'?'Thinking…':'Five networks, averaged',W-4,2);
      if(!reduce && (state!=='idle')) raf=requestAnimationFrame(draw);
    }
    const kick=()=>{ if(!raf) raf=requestAnimationFrame(draw); };
    new ResizeObserver(()=>{ L=null; kick(); }).observe(el);
    new MutationObserver(kick).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
    kick();
    return {set(s){ if(s!==state){ state=s; t0=performance.now(); kick(); } }, update(o){ Object.assign(opt,o); kick(); }};
  };
})();
