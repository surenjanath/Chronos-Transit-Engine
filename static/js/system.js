// system.js — animated visual model of a transit: the companion crossing the stellar disc (to scale),
// a top-down view of the orbit (to scale), and the light curve those geometries are expected to produce.
// CTESystem(el, {k, a_rs, b, period_days}, {hours:[], flux:[]}, observed?{hours:[], flux:[]})
(function(){
  const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  window.CTESystem=function(el,geom,curve,observed){
    const H=opt_h(el); el.classList.add('plot'); el.style.height=H+'px'; el.innerHTML='';
    const cv=document.createElement('canvas'); el.append(cv);
    let W=el.clientWidth||600, playing=true, phase=0, last=null, raf=null;
    let fmin, fmax, t0, t1;
    const rescale=()=>{ const obs=observed?observed.flux.filter(v=>v!=null):[]; fmin=Math.min(...curve.flux,...obs); fmax=Math.max(1,...obs); t0=curve.hours[0]; t1=curve.hours[curve.hours.length-1]; };
    rescale();
    const interp=t=>{ const n=curve.hours.length, u=(t-t0)/(t1-t0)*(n-1), i=Math.max(0,Math.min(n-2,Math.floor(u))); return curve.flux[i]+(curve.flux[i+1]-curve.flux[i])*(u-i); };
    function opt_h(e){ return +e.dataset.height||320; }
    function draw(){
      const dpr=window.devicePixelRatio||1; cv.width=W*dpr; cv.height=H*dpr;
      const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
      const dark=document.documentElement.getAttribute('data-theme')==='dark';
      const ink=cssv('--ink'), mute=cssv('--mute'), rule=cssv('--rule'), canvas=cssv('--canvas'), gap=cssv('--gap');
      const t=t0+phase*(t1-t0), ang=2*Math.PI*(t/24)/geom.period_days;
      const stripH=96, skyH=H-stripH-8, skyW=W*0.62;
      g.font='11px '+getComputedStyle(document.body).fontFamily; g.textBaseline='top';

      // --- sky view: what a telescope would resolve if it could ---
      const R=Math.min(skyH*0.4, skyW*0.2), cx=skyW/2, cy=skyH/2+6;
      const grad=g.createRadialGradient(cx,cy,0,cx,cy,R);
      grad.addColorStop(0,dark?'#f1f1ec':'#fbfaf5'); grad.addColorStop(0.7,dark?'#d6d6d0':'#eeede6'); grad.addColorStop(1,dark?'#8c8c86':'#c9c8c0');
      g.fillStyle=grad; g.beginPath(); g.arc(cx,cy,R,0,6.2832); g.fill();
      g.strokeStyle=ink; g.lineWidth=1; g.beginPath(); g.arc(cx,cy,R,0,6.2832); g.stroke();
      g.strokeStyle=mute; g.setLineDash([2,4]); g.beginPath(); g.moveTo(0,cy-geom.b*R); g.lineTo(skyW,cy-geom.b*R); g.stroke(); g.setLineDash([]);
      const px=cx+geom.a_rs*Math.sin(ang)*R, py=cy-geom.b*Math.cos(ang)*R, pr=Math.max(geom.k*R,1.6);
      g.fillStyle=dark?'#161615':ink; g.beginPath(); g.arc(px,py,pr,0,6.2832); g.fill();
      if(dark){ g.strokeStyle=ink; g.lineWidth=.8; g.stroke(); }
      g.fillStyle=mute; g.textAlign='left';
      g.fillText(`Sky view, to scale · Rp/R★ = ${geom.k.toFixed(4)} · b = ${(+geom.b).toFixed(2)}`,0,0);

      // --- orbit seen from above ---
      const ox=skyW+(W-skyW)/2, oy=skyH/2+6, Ro=Math.min((W-skyW)*0.38, skyH*0.4), rs=Math.max(Ro/geom.a_rs,1.5);
      g.strokeStyle=rule; g.lineWidth=1; g.beginPath(); g.arc(ox,oy,Ro,0,6.2832); g.stroke();
      g.fillStyle=dark?'#d6d6d0':'#eeede6'; g.beginPath(); g.arc(ox,oy,rs,0,6.2832); g.fill(); g.strokeStyle=ink; g.stroke();
      g.fillStyle=ink; g.beginPath(); g.arc(ox+Ro*Math.sin(ang),oy+Ro*Math.cos(ang),Math.max(geom.k*rs,2.2),0,6.2832); g.fill();
      g.strokeStyle=mute; g.setLineDash([2,4]); g.beginPath(); g.moveTo(ox,oy+rs+2); g.lineTo(ox,oy+Ro+14); g.stroke(); g.setLineDash([]);
      g.fillStyle=mute; g.textAlign='center'; g.fillText('to observer',ox,oy+Ro+16);
      g.textAlign='left'; g.fillText(`Orbit from above · a/R★ = ${geom.a_rs.toFixed(2)}`,skyW+4,0);

      // --- the light curve this geometry predicts ---
      const y0=H-stripH, l=44, r=8, sx=v=>l+(v-t0)/(t1-t0)*(W-l-r), pad=(fmax-fmin)*0.15||1e-4;
      const sy=v=>y0+8+(1-(v-(fmin-pad))/((fmax+pad)-(fmin-pad)))*(stripH-30);
      g.strokeStyle=rule; g.beginPath(); g.moveTo(l,sy(1)); g.lineTo(W-r,sy(1)); g.stroke();
      g.fillStyle=mute; g.textAlign='right'; g.textBaseline='middle'; g.fillText('1.0000',l-6,sy(1)); g.fillText(fmin.toFixed(4),l-6,sy(fmin));
      if(observed){ g.fillStyle=mute; for(let i=0;i<observed.hours.length;i++){ const v=observed.flux[i]; if(v==null) continue; g.beginPath(); g.arc(sx(observed.hours[i]),sy(v),1.8,0,6.2832); g.fill(); } }
      g.strokeStyle=ink; g.lineWidth=1.8; g.beginPath(); curve.hours.forEach((h,i)=>{ i?g.lineTo(sx(h),sy(curve.flux[i])):g.moveTo(sx(h),sy(curve.flux[i])); }); g.stroke();
      g.strokeStyle=gap; g.lineWidth=1; g.beginPath(); g.moveTo(sx(t),y0+4); g.lineTo(sx(t),H-18); g.stroke();
      g.fillStyle=gap; g.beginPath(); g.arc(sx(t),sy(interp(t)),4,0,6.2832); g.fill();
      g.fillStyle=mute; g.textAlign='center'; g.textBaseline='top';
      [t0,t0/2,0,t1/2,t1].forEach(h=>g.fillText(h.toFixed(1),sx(h),H-14));
      g.textAlign='left'; g.fillStyle=ink;
      g.fillText(`${t>=0?'+':''}${t.toFixed(2)} h · flux ${interp(t).toFixed(5)}${playing?'':' · paused'}`,l,y0-8);
    }
    function frame(ts){
      if(playing){ if(last!=null) phase=(phase+(ts-last)/7000)%1; last=ts; draw(); raf=requestAnimationFrame(frame); }
    }
    const still=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if(still){ playing=false; phase=0.5; }
    el.title='Click to pause or play; move the pointer to scrub while paused';
    el.addEventListener('click',()=>{ playing=!playing; last=null; if(playing) raf=requestAnimationFrame(frame); else draw(); });
    el.addEventListener('mousemove',e=>{ if(playing) return; const x=e.clientX-el.getBoundingClientRect().left; phase=Math.max(0,Math.min(1,(x-44)/(W-52))); draw(); });
    new ResizeObserver(e=>{ W=Math.max(260,e[0].contentRect.width); draw(); }).observe(el);
    new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
    if(playing) raf=requestAnimationFrame(frame); else draw();
    return {update(g2,c2){ Object.assign(geom,g2); curve.hours=c2.hours; curve.flux=c2.flux; rescale(); draw(); }};
  };

  // Heat strip: one row per observed transit, columns across the transit window, ink density = fractional depth.
  window.CTEStack=function(el,rows,hours){
    el.innerHTML=''; el.classList.add('plot');
    const rowH=Math.max(5,Math.min(16,Math.floor(220/Math.max(rows.length,1)))), H=rows.length*rowH+30;
    el.style.height=H+'px';
    const cv=document.createElement('canvas'), tip=document.createElement('div'); tip.className='plot-tip'; tip.hidden=true; el.append(cv,tip);
    let W=el.clientWidth||500; const l=58, r=8, n=rows[0].values.length;
    function draw(){
      const dpr=window.devicePixelRatio||1; cv.width=W*dpr; cv.height=H*dpr;
      const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
      const ink=cssv('--ink'), mute=cssv('--mute'), cw=(W-l-r)/n;
      g.font='11px '+getComputedStyle(document.body).fontFamily;
      rows.forEach((row,i)=>{
        row.values.forEach((v,j)=>{ if(v==null) return; g.globalAlpha=Math.max(0,Math.min(1,-v)); g.fillStyle=ink; g.fillRect(l+j*cw,i*rowH,Math.ceil(cw),rowH-1); });
        g.globalAlpha=1;
        if(rowH>=10||i%Math.ceil(10/rowH)===0){ g.fillStyle=mute; g.textAlign='right'; g.textBaseline='middle'; g.fillText('#'+row.epoch,l-8,i*rowH+rowH/2); }
      });
      g.fillStyle=mute; g.textAlign='center'; g.textBaseline='top';
      const h0=hours[0], h1=hours[hours.length-1];
      [h0,h0/2,0,h1/2,h1].forEach(h=>g.fillText(h.toFixed(1),l+(h-h0)/(h1-h0)*(W-l-r),rows.length*rowH+4));
      g.fillText('Time from mid-transit (hours)',l+(W-l-r)/2,rows.length*rowH+17);
    }
    el.addEventListener('mousemove',e=>{ const b=el.getBoundingClientRect(), i=Math.floor((e.clientY-b.top)/rowH), j=Math.floor((e.clientX-b.left-l)/((W-l-r)/n));
      if(i<0||i>=rows.length||j<0||j>=n||rows[i].values[j]==null){ tip.hidden=true; return; }
      tip.hidden=false; tip.textContent=`Transit #${rows[i].epoch}: ${(-rows[i].values[j]*100).toFixed(0)}% of fitted depth`; tip.style.left=Math.min(e.clientX-b.left+12,W-190)+'px'; tip.style.top=(e.clientY-b.top+10)+'px'; });
    el.addEventListener('mouseleave',()=>{ tip.hidden=true; });
    new ResizeObserver(e=>{ W=Math.max(260,e[0].contentRect.width); draw(); }).observe(el);
    new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
    draw();
  };
})();
