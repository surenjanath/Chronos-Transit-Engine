// plot.js — canvas XY plot in the station's ink-on-canvas vocabulary: hairline grid, crosshair readout, drag to zoom.
// CTEPlot(el, {series:[{name,x,y,mode:'points'|'line',tone:'ink'|'mute'|'gap',size,width,alpha,dash,hover}], xLabel, yLabel,
//              markers:[{x,label}], zoom, xDomain, yDomain, xFmt, yFmt, height})
(function(){
  const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const TONE={ink:'--ink',mute:'--mute',gap:'--gap',faint:'--faint'};
  function ticks(lo,hi,count){
    const span=hi-lo; if(!(span>0)) return [lo];
    const s0=Math.pow(10,Math.floor(Math.log10(span/count))), err=span/count/s0;
    const step=s0*(err>=7.5?10:err>=3.5?5:err>=1.5?2:1), out=[];
    for(let v=Math.ceil(lo/step)*step; v<=hi+step*1e-9; v+=step) out.push(Math.abs(v)<step*1e-9?0:v);
    return out;
  }
  function autoFmt(lo,hi){
    const span=Math.abs(hi-lo), d=span===0?2:Math.max(0,Math.min(6,Math.ceil(-Math.log10(span/5))+1));
    return v=>v.toFixed(d);
  }
  window.CTEPlot=function(el,opt){
    const height=opt.height||240;
    el.classList.add('plot'); el.style.height=height+'px'; el.innerHTML='';
    const named=opt.series.filter(s=>s.name && s.legend!==false);
    if(named.length>1 || opt.zoom){
      const lg=document.createElement('div'); lg.className='plot-legend';
      if(named.length>1) named.forEach(s=>{
        const sp=document.createElement('span'), i=document.createElement('i');
        if(s.mode==='points') i.className='pt'; 
        i.style[s.mode==='points'?'background':'borderTopColor']=`var(${TONE[s.tone||'ink']})`;
        if(s.dash) i.style.borderTopStyle='dashed';
        sp.append(i,document.createTextNode(s.name)); lg.append(sp);
      });
      if(opt.zoom){ const h=document.createElement('span'); h.style.marginLeft='auto'; h.className='zoomhint'; h.textContent='Drag to zoom'; lg.append(h); }
      el.before(lg); el._legend=lg;
    }
    const cv=document.createElement('canvas'), tip=document.createElement('div');
    tip.className='plot-tip'; tip.hidden=true; el.append(cv,tip);
    let width=el.clientWidth||600, view=null, hover=null, drag=null, geom={};
    const full=(()=>{ if(opt.xDomain) return opt.xDomain; let lo=Infinity,hi=-Infinity;
      for(const s of opt.series) for(const v of s.x){ if(v<lo) lo=v; if(v>hi) hi=v; }
      return isFinite(lo)?[lo,hi===lo?lo+1:hi]:[0,1]; })();

    function draw(){
      const dpr=window.devicePixelRatio||1; cv.width=width*dpr; cv.height=height*dpr;
      const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,width,height);
      const [x0,x1]=view||full; let y0=Infinity,y1=-Infinity;
      if(opt.yDomain) [y0,y1]=opt.yDomain;
      else{
        for(const s of opt.series) for(let i=0;i<s.x.length;i++){ const v=s.y[i]; if(v==null||s.x[i]<x0||s.x[i]>x1) continue; if(v<y0) y0=v; if(v>y1) y1=v; }
        if(!isFinite(y0)){ y0=0; y1=1; }
        const pad=(y1-y0||Math.abs(y1)||1)*0.08; y0-=pad; y1+=pad;
      }
      const fy=opt.yFmt||autoFmt(y0,y1), fx=opt.xFmt||autoFmt(x0,x1), yt=ticks(y0,y1,Math.max(3,Math.floor(height/48)));
      g.font='11px '+getComputedStyle(document.body).fontFamily;
      const l=Math.max(...yt.map(v=>g.measureText(fy(v)).width))+(opt.yLabel?30:12), r=10, t=8, b=opt.xLabel?40:24;
      geom={x0,x1,y0,y1,l,r,t,b};
      const sx=v=>l+(v-x0)/(x1-x0)*(width-l-r), sy=v=>t+(1-(v-y0)/(y1-y0))*(height-t-b);
      const mute=cssv('--mute'), rule=cssv('--rule');
      g.strokeStyle=rule; g.lineWidth=1; g.fillStyle=mute; g.textBaseline='middle'; g.textAlign='right';
      for(const v of yt){ const y=Math.round(sy(v))+.5; g.beginPath(); g.moveTo(l,y); g.lineTo(width-r,y); g.stroke(); g.fillText(fy(v),l-7,y); }
      g.textAlign='center'; g.textBaseline='top';
      for(const v of ticks(x0,x1,Math.max(3,Math.floor(width/95)))) g.fillText(fx(v),sx(v),height-b+7);
      if(opt.xLabel) g.fillText(opt.xLabel,l+(width-l-r)/2,height-15);
      if(opt.yLabel){ g.save(); g.translate(10,t+(height-t-b)/2); g.rotate(-Math.PI/2); g.textBaseline='middle'; g.fillText(opt.yLabel,0,0); g.restore(); }
      g.save(); g.beginPath(); g.rect(l,t,width-l-r,height-t-b); g.clip();
      for(const m of opt.markers||[]){
        if(m.x<x0||m.x>x1) continue;
        g.strokeStyle=cssv('--gap'); g.setLineDash([3,4]); g.lineWidth=1; g.beginPath(); g.moveTo(sx(m.x),t); g.lineTo(sx(m.x),height-b); g.stroke(); g.setLineDash([]);
        if(m.label){ g.fillStyle=cssv('--ink'); g.textAlign='left'; g.textBaseline='top'; g.fillText(m.label,sx(m.x)+5,t+2); }
      }
      for(const s of opt.series){
        const color=cssv(TONE[s.tone||'ink']); g.globalAlpha=s.alpha==null?1:s.alpha;
        if(s.mode==='points'){
          g.fillStyle=color; const size=s.size||2;
          for(let i=0;i<s.x.length;i++){ const v=s.y[i]; if(v==null||s.x[i]<x0||s.x[i]>x1) continue;
            if(size<=2.5) g.fillRect(sx(s.x[i])-size/2,sy(v)-size/2,size,size); else{ g.beginPath(); g.arc(sx(s.x[i]),sy(v),size/2,0,6.2832); g.fill(); } }
        }else{
          g.strokeStyle=color; g.lineWidth=s.width||1.6; g.lineJoin='round'; g.setLineDash(s.dash?[5,4]:[]); g.beginPath(); let pen=false;
          for(let i=0;i<s.x.length;i++){ const v=s.y[i]; if(v==null){ pen=false; continue; } const px=sx(s.x[i]),py=sy(v); pen?g.lineTo(px,py):g.moveTo(px,py); pen=true; }
          g.stroke(); g.setLineDash([]);
        }
      }
      g.globalAlpha=1;
      if(hover){ g.strokeStyle=mute; g.lineWidth=1; g.beginPath(); g.moveTo(sx(hover.x),t); g.lineTo(sx(hover.x),height-b); g.stroke();
        for(const row of hover.rows){ g.fillStyle=cssv('--canvas'); g.beginPath(); g.arc(sx(hover.x),sy(row.y),5,0,6.2832); g.fill();
          g.fillStyle=cssv(TONE[row.tone||'ink']); g.beginPath(); g.arc(sx(hover.x),sy(row.y),3.2,0,6.2832); g.fill(); } }
      if(drag){ g.fillStyle=cssv('--ink'); g.globalAlpha=.1; g.fillRect(Math.min(drag[0],drag[1]),t,Math.abs(drag[1]-drag[0]),height-t-b); }
      g.restore();
    }
    const toX=px=>geom.x0+(px-geom.l)/(width-geom.l-geom.r)*(geom.x1-geom.x0);
    function nearest(s,target){ let best=-1,bd=Infinity; for(let i=0;i<s.x.length;i++){ if(s.y[i]==null) continue; const d=Math.abs(s.x[i]-target); if(d<bd){ bd=d; best=i; } } return best; }
    el.addEventListener('mousemove',e=>{
      const rect=el.getBoundingClientRect(), px=e.clientX-rect.left, py=e.clientY-rect.top;
      if(drag){ drag[1]=px; draw(); return; }
      if(px<geom.l||px>width-geom.r){ hover=null; tip.hidden=true; draw(); return; }
      const hs=opt.series.filter(s=>s.hover!==false&&s.x.length); if(!hs.length) return;
      const i0=nearest(hs[0],toX(px)); if(i0<0) return;
      const snap=hs[0].x[i0], tol=(geom.x1-geom.x0)*0.02;
      const rows=hs.flatMap(s=>{ const i=s===hs[0]?i0:nearest(s,snap); return i>=0&&Math.abs(s.x[i]-snap)<=tol?[{name:s.name,tone:s.tone,y:s.y[i]}]:[]; });
      hover={x:snap,rows};
      const fx=opt.xFmt||autoFmt(geom.x0,geom.x0+(geom.x1-geom.x0)*0.1), fy=opt.yFmt||(v=>Math.abs(v)>=1000?v.toFixed(0):Number(v.toPrecision(6)).toString());
      tip.innerHTML=`<div style="opacity:.7">${opt.xLabel?opt.xLabel.split(' (')[0]+': ':''}${fx(snap)}</div>`+rows.map(r=>`<div>${r.name}: <b>${fy(r.y)}</b></div>`).join('');
      tip.hidden=false; tip.style.left=Math.min(px+14,width-180)+'px'; tip.style.top=Math.max(py-10,0)+'px'; draw();
    });
    el.addEventListener('mouseleave',()=>{ hover=null; drag=null; tip.hidden=true; draw(); });
    if(opt.zoom){
      el.addEventListener('mousedown',e=>{ const px=e.clientX-el.getBoundingClientRect().left; drag=[px,px]; hover=null; tip.hidden=true; });
      el.addEventListener('mouseup',()=>{
        if(drag&&Math.abs(drag[1]-drag[0])>8){ view=[Math.max(toX(Math.min(drag[0],drag[1])),full[0]),Math.min(toX(Math.max(drag[0],drag[1])),full[1])];
          const h=el._legend&&el._legend.querySelector('.zoomhint'); if(h) h.textContent='Double-click to reset zoom'; }
        drag=null; draw();
      });
      el.addEventListener('dblclick',()=>{ view=null; const h=el._legend&&el._legend.querySelector('.zoomhint'); if(h) h.textContent='Drag to zoom'; draw(); });
    }
    new ResizeObserver(e=>{ width=Math.max(200,e[0].contentRect.width); draw(); }).observe(el);
    new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
    draw();
    return {redraw:draw};
  };
  window.jsonData=id=>{ const el=document.getElementById(id); return el?JSON.parse(el.textContent):null; };
})();
