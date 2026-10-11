// map3d.js — the planetary systems as a field of stars you can fly through. Every star is at its measured place:
// X towards the centre of the Galaxy, Y the way the Galaxy turns, Z up out of its plane, the Sun at the origin.
// CTEMap3D(el, api) -> {show(on), refresh(), focus(host), state()}
(function(){
  const TAU=Math.PI*2, group=v=>Math.round(v).toLocaleString('en-US');
  const dist=ly=>ly<0.02?group(ly*63241)+' AU':ly<10?ly.toFixed(1)+' light-years':group(ly)+' light-years';
  function sprite(rgb){   // a star: a soft glow with the four spikes a telescope gives it
    const c=document.createElement('canvas'), n=64; c.width=c.height=n; const g=c.getContext('2d');
    const gr=g.createRadialGradient(n/2,n/2,0,n/2,n/2,n/2); gr.addColorStop(0,'rgba(255,255,255,1)'); gr.addColorStop(0.1,`rgba(${rgb},.95)`); gr.addColorStop(0.32,`rgba(${rgb},.28)`); gr.addColorStop(1,`rgba(${rgb},0)`);
    g.fillStyle=gr; g.fillRect(0,0,n,n); g.strokeStyle=`rgba(${rgb},.75)`; g.lineWidth=1.4;
    g.beginPath(); g.moveTo(n/2,4); g.lineTo(n/2,n-4); g.moveTo(4,n/2); g.lineTo(n-4,n/2); g.stroke(); return c;
  }
  const TINT=t=>t==null?0:t<3900?1:t<5200?2:t<6300?0:3;   // yellow for Sun-like and unknown, orange-red for cool dwarfs, blue-white for hot stars
  window.CTEMap3D=function(el,api){
    const box=document.createElement('div'); box.className='map3d'; box.hidden=true; box.tabIndex=0;
    box.innerHTML=`<canvas></canvas>
      <div class="m3-you">You are<b></b>from Earth</div>
      <div class="m3-title"><h3>Milky Way Galaxy</h3><p></p></div>
      <div class="m3-tip" hidden></div>
      <div class="m3-filter"><div class="m3-h"><span>Filter</span></div>
        <div class="m3-tabs"><button type="button" data-f="kind">Planet type</button><button type="button" data-f="facility">Found by</button><button type="button" data-f="method">Method</button></div>
        <div class="m3-opts" hidden></div><p class="m3-cur"></p></div>
      <div class="m3-gizmo"><canvas width="88" height="88"></canvas></div>
      <div class="m3-zoom"><button type="button" data-z="in" aria-label="Move closer">+</button><button type="button" data-z="out" aria-label="Move away">−</button></div>
      <div class="m3-bar"><b>View from:</b><button type="button" data-v="space" class="on">Space</button><button type="button" data-v="above">Above</button><button type="button" data-v="edge">Edge-on</button><button type="button" data-v="sun">The Sun</button>
        <span class="m3-keys">Drag · Shift-drag · Scroll · W A S D Q E</span></div>`;
    el.appendChild(box);
    const cv=box.querySelector('canvas'), g=cv.getContext('2d'), tip=box.querySelector('.m3-tip'), gz=box.querySelector('.m3-gizmo canvas').getContext('2d');
    const sprites=[sprite('255,226,90'),sprite('255,150,80'),sprite('255,196,110'),sprite('190,210,255')];
    const cam={tx:0,ty:0,tz:0,yaw:-2.2,pitch:0.42,dist:9000};
    let W=0,H=0,on=false,raf=0,anim=0,hover=null,shown=[],F=1;
    const sys=()=>api.systems();
    function basis(){ const cp=Math.cos(cam.pitch), o=[cp*Math.cos(cam.yaw),cp*Math.sin(cam.yaw),Math.sin(cam.pitch)];
      const C=[cam.tx+o[0]*cam.dist,cam.ty+o[1]*cam.dist,cam.tz+o[2]*cam.dist], f=[-o[0],-o[1],-o[2]];
      let r=[f[1],-f[0],0]; const rn=Math.hypot(r[0],r[1])||1; r=[r[0]/rn,r[1]/rn,0];            // right = forward × up(z)
      const u=[r[1]*f[2]-r[2]*f[1], r[2]*f[0]-r[0]*f[2], r[0]*f[1]-r[1]*f[0]];
      return {C,f,r,u}; }
    let B=basis();
    function proj(x,y,z){ const dx=x-B.C[0], dy=y-B.C[1], dz=z-B.C[2], zc=dx*B.f[0]+dy*B.f[1]+dz*B.f[2]; if(zc<=cam.dist*1e-4) return null;
      return [W/2+(dx*B.r[0]+dy*B.r[1]+dz*B.r[2])/zc*F, H/2-(dx*B.u[0]+dy*B.u[1]+dz*B.u[2])/zc*F, zc]; }
    function line(a,b,style,width,dash){ const p=proj(...a), q=proj(...b); if(!p||!q) return; g.strokeStyle=style; g.lineWidth=width||1; g.setLineDash(dash||[]); g.beginPath(); g.moveTo(p[0],p[1]); g.lineTo(q[0],q[1]); g.stroke(); g.setLineDash([]); }
    function ring(radius,style){ g.strokeStyle=style; g.lineWidth=1; g.beginPath(); let pen=false;
      for(let i=0;i<=120;i++){ const a=i/120*TAU, p=proj(Math.cos(a)*radius,Math.sin(a)*radius,0); if(!p){ pen=false; continue; } pen?g.lineTo(p[0],p[1]):g.moveTo(p[0],p[1]); pen=true; } g.stroke(); }
    function text(t,x,y,colour,bold){ g.font=(bold?'600 ':'')+'12px '+getComputedStyle(document.body).fontFamily; g.fillStyle=colour; g.textAlign='left'; g.textBaseline='middle'; g.fillText(t,x,y); }
    function draw(){
      raf=0; if(!on) return; const dpr=window.devicePixelRatio||1; W=box.clientWidth; H=box.clientHeight;
      if(cv.width!==Math.round(W*dpr)||cv.height!==Math.round(H*dpr)){ cv.width=Math.round(W*dpr); cv.height=Math.round(H*dpr); }
      g.setTransform(dpr,0,0,dpr,0,0); g.globalCompositeOperation='source-over'; g.fillStyle='#04050a'; g.fillRect(0,0,W,H);
      F=H*0.95; B=basis();
      // distance rings in the plane of the Galaxy, centred on the Sun, and the three axes
      const top=Math.pow(10,Math.ceil(Math.log10(cam.dist*1.5)));
      for(const r of [top/100,top/10,top]){ ring(r,'rgba(120,140,220,.22)'); const p=proj(r*Math.cos(0.6),r*Math.sin(0.6),0); if(p) text(dist(r),p[0]+4,p[1],'rgba(150,165,225,.7)'); }
      const L=top*0.55;
      line([0,0,0],[L,0,0],'rgba(255,120,110,.7)',1.2); line([0,0,0],[0,L,0],'rgba(120,220,140,.7)',1.2); line([0,0,0],[0,0,L],'rgba(120,170,255,.8)',1.2);
      line([0,0,0],[-L,0,0],'rgba(255,120,110,.25)',1,[3,5]); line([0,0,0],[0,-L,0],'rgba(120,220,140,.25)',1,[3,5]); line([0,0,0],[0,0,-L],'rgba(120,170,255,.25)',1,[3,5]);
      [[L,0,0,'X · to the centre of the Galaxy','rgba(255,150,140,.9)'],[0,L,0,'Y · the way the Galaxy turns','rgba(140,230,160,.9)'],[0,0,L,'Z · up out of the plane','rgba(150,190,255,.95)']].forEach(a=>{ const p=proj(a[0],a[1],a[2]); if(p) text(a[3],p[0]+6,p[1],a[4]); });
      const gc=proj(api.gc,0,0); if(gc&&gc[0]>-50&&gc[0]<W+50){ g.strokeStyle='rgba(255,255,255,.7)'; g.beginPath(); g.moveTo(gc[0]-7,gc[1]); g.lineTo(gc[0]+7,gc[1]); g.moveTo(gc[0],gc[1]-7); g.lineTo(gc[0],gc[1]+7); g.stroke(); text('Centre of the Galaxy · '+group(api.gc)+' light-years',gc[0]+11,gc[1],'rgba(255,255,255,.8)'); }
      // the stars, added together so that crowded regions blaze
      const crowd=0.42+0.58*Math.min(1,900/Math.max(shown.length,1));   // dimmer sprites when thousands overlap, so the core keeps its shape
      g.globalCompositeOperation='lighter'; shown=[]; const sel=api.selected(), mine=api.mine;
      for(const h of sys()){ if(!h._on) continue; const p=proj(h.x,h.y,h.z); if(!p||p[0]<-20||p[0]>W+20||p[1]<-20||p[1]>H+20) continue;
        const s=Math.max(3.2,Math.min(26,5+11*cam.dist/p[2]*0.55)); h._sx=p[0]; h._sy=p[1]; h._ss=s; shown.push(h);
        g.globalAlpha=Math.max(0.3,Math.min(1,1.25*cam.dist/p[2]))*crowd; g.drawImage(sprites[TINT(h.teff)],p[0]-s,p[1]-s,2*s,2*s); }
      g.globalAlpha=1; g.globalCompositeOperation='source-over';
      for(const h of shown) if(mine.has(h.name)){ g.strokeStyle='rgba(255,140,110,.8)'; g.lineWidth=1; g.beginPath(); g.arc(h._sx,h._sy,Math.min(7,h._ss*0.3+2.5),0,TAU); g.stroke(); }
      const sun=proj(0,0,0); if(sun){ g.strokeStyle='#ff8f7a'; g.lineWidth=1.5; g.beginPath(); g.arc(sun[0],sun[1],5,0,TAU); g.stroke(); text('Sun',sun[0]+11,sun[1],'#ff9c88',true); }
      if(sel&&sel._on){ const p=proj(sel.x,sel.y,sel.z); if(p){ g.strokeStyle='#fff'; g.lineWidth=1.5; g.beginPath(); g.arc(p[0],p[1],11,0,TAU); g.stroke(); text(sel.name,p[0]+16,p[1],'#fff',true); if(sun){ g.strokeStyle='rgba(255,255,255,.35)'; g.setLineDash([4,5]); g.beginPath(); g.moveTo(sun[0],sun[1]); g.lineTo(p[0],p[1]); g.stroke(); g.setLineDash([]); } } }
      // readouts
      box.querySelector('.m3-you b').textContent=dist(Math.hypot(...B.C));
      const all=sys(), vis=all.filter(h=>h._on), planets=vis.reduce((n,h)=>n+h.planets.length,0);
      box.querySelector('.m3-title p').textContent=vis.length<all.length?`${group(vis.length)} of ${group(all.length)} solar systems have such a planet | ${group(planets)} confirmed planets in them`:`${group(vis.length)} solar systems | ${group(planets)} confirmed planets`;
      // orientation: which way the three axes point from here
      gz.clearRect(0,0,88,88); [[1,0,0,'X','#ff8f85'],[0,1,0,'Y','#8ce6a0'],[0,0,1,'Z','#96beff']].map(a=>({a,z:a[0]*B.f[0]+a[1]*B.f[1]+a[2]*B.f[2]})).sort((p,q)=>q.z-p.z).forEach(({a})=>{
        const x=44+(a[0]*B.r[0]+a[1]*B.r[1]+a[2]*B.r[2])*30, y=44-(a[0]*B.u[0]+a[1]*B.u[1]+a[2]*B.u[2])*30; gz.strokeStyle=a[4]; gz.lineWidth=2; gz.beginPath(); gz.moveTo(44,44); gz.lineTo(x,y); gz.stroke(); gz.fillStyle=a[4]; gz.font='600 11px sans-serif'; gz.textAlign='center'; gz.textBaseline='middle'; gz.fillText(a[3],44+(x-44)*1.25,44+(y-44)*1.25); });
    }
    // a tab in the background gets no animation frames: draw at once there, so readouts stay true
    const redraw=()=>{ if(!on) return; if(document.hidden) return draw(); if(!raf) raf=requestAnimationFrame(draw); };
    function fly(to,ms){ cancelAnimationFrame(anim); if(document.hidden){ Object.assign(cam,to); return redraw(); } const from=Object.assign({},cam), t0=performance.now(); let dy=(to.yaw??from.yaw)-from.yaw; dy=((dy+Math.PI)%TAU+TAU)%TAU-Math.PI;
      const step=now=>{ const u=Math.min(1,(now-t0)/(ms||1100)), e=u<0.5?2*u*u:1-Math.pow(-2*u+2,2)/2;
        for(const k of ['tx','ty','tz','pitch']) if(to[k]!=null) cam[k]=from[k]+(to[k]-from[k])*e;
        cam.yaw=from.yaw+dy*e; if(to.dist!=null) cam.dist=from.dist*Math.pow(to.dist/from.dist,e); redraw(); if(u<1) anim=requestAnimationFrame(step); };
      anim=requestAnimationFrame(step); }
    const clamp=()=>{ cam.pitch=Math.max(-1.55,Math.min(1.55,cam.pitch)); cam.dist=Math.max(0.3,Math.min(160000,cam.dist)); };
    const pick=(x,y)=>{ let best=null,bd=14*14; for(const h of shown){ const d=(h._sx-x)**2+(h._sy-y)**2; if(d<bd){ bd=d; best=h; } } return best; };
    // pointer: drag turns the view about the point looked at; with Shift (or the right button) it slides that point sideways and up
    let drag=null; const at=e=>{ const b=cv.getBoundingClientRect(); return [e.clientX-b.left,e.clientY-b.top]; };
    cv.addEventListener('pointerdown',e=>{ cv.setPointerCapture(e.pointerId); const [x,y]=at(e); drag={x,y,moved:0,pan:e.shiftKey||e.button===2||e.button===1}; cancelAnimationFrame(anim); box.focus({preventScroll:true}); });
    cv.addEventListener('pointermove',e=>{ const [x,y]=at(e);
      if(!drag){ const h=pick(x,y); if(h!==hover){ hover=h; cv.style.cursor=h?'pointer':'grab'; }
        if(h){ tip.hidden=false; tip.innerHTML=`<b>${h.name}</b>${dist(h.d)} away · ${h.planets.length} planet${h.planets.length>1?'s':''}<span>Click to zoom</span>`; tip.style.left=Math.min(x+16,W-210)+'px'; tip.style.top=Math.max(y-12,6)+'px'; } else tip.hidden=true; return; }
      const dx=x-drag.x, dy=y-drag.y; drag.moved+=Math.abs(dx)+Math.abs(dy); drag.x=x; drag.y=y; tip.hidden=true;
      if(drag.pan){ const k=cam.dist/F; cam.tx-=(dx*B.r[0]-dy*B.u[0])*k; cam.ty-=(dx*B.r[1]-dy*B.u[1])*k; cam.tz-=(dx*B.r[2]-dy*B.u[2])*k; }
      else { cam.yaw-=dx*0.006; cam.pitch+=dy*0.006; }
      clamp(); redraw(); });
    cv.addEventListener('pointerup',e=>{ const d=drag; drag=null; if(d&&d.moved<5){ const [x,y]=at(e), h=pick(x,y); if(h) api.select(h); } });
    cv.addEventListener('contextmenu',e=>e.preventDefault());
    cv.addEventListener('wheel',e=>{ e.preventDefault(); cam.dist*=Math.exp(Math.max(-60,Math.min(60,e.deltaY))*(e.ctrlKey?0.012:0.0035)); clamp(); redraw(); },{passive:false});
    box.addEventListener('keydown',e=>{ const k=e.key.toLowerCase(), s=cam.dist*0.07, move=(v,m)=>{ cam.tx+=v[0]*m; cam.ty+=v[1]*m; cam.tz+=v[2]*m; };
      if(k==='w') move(B.f,s); else if(k==='s') move(B.f,-s); else if(k==='a') move(B.r,-s); else if(k==='d') move(B.r,s);
      else if(k==='e') cam.tz+=s; else if(k==='q') cam.tz-=s; else if(k==='arrowleft') cam.yaw+=0.08; else if(k==='arrowright') cam.yaw-=0.08;
      else if(k==='arrowup') cam.pitch+=0.06; else if(k==='arrowdown') cam.pitch-=0.06; else if(k==='+'||k==='=') cam.dist*=0.82; else if(k==='-'||k==='_') cam.dist*=1.22; else return;
      e.preventDefault(); e.stopPropagation(); clamp(); redraw(); });
    box.querySelector('.m3-zoom').addEventListener('click',e=>{ const z=e.target.dataset.z; if(z) fly({dist:cam.dist*(z==='in'?0.5:2)},350); });
    const VIEWS={space:{tx:0,ty:0,tz:0,yaw:-2.2,pitch:0.42,dist:9000},above:{tx:0,ty:0,tz:0,yaw:-Math.PI/2,pitch:1.55,dist:9000},edge:{tx:0,ty:0,tz:0,yaw:-Math.PI/2,pitch:0,dist:9000},sun:{tx:0,ty:0,tz:0,dist:60}};
    box.querySelector('.m3-bar').addEventListener('click',e=>{ const v=e.target.dataset.v; if(!v) return; box.querySelectorAll('.m3-bar [data-v]').forEach(b=>b.classList.toggle('on',b===e.target)); fly(VIEWS[v],1400); });
    // filters: one list at a time, as on the reference
    const opts=box.querySelector('.m3-opts'), cur=box.querySelector('.m3-cur'), NAMES={kind:'planet types',facility:'observatories',method:'methods'};
    function label(){ const f=api.filter(); const parts=[f.kind,f.facility,f.method&&(f.method==='other'?'other methods':f.method)].filter(Boolean); cur.textContent='Current filter: '+(parts.length?parts.join(' · '):'All planet types'); }
    box.querySelector('.m3-tabs').addEventListener('click',e=>{ const k=e.target.dataset.f; if(!k) return; const open=!opts.hidden&&opts.dataset.k===k;
      box.querySelectorAll('.m3-tabs button').forEach(b=>b.classList.toggle('on',!open&&b===e.target)); if(open){ opts.hidden=true; return; }
      const f=api.filter(), list=api.options(k); opts.dataset.k=k; opts.hidden=false;
      opts.innerHTML=`<button type="button" data-val=""${!f[k]?' class="on"':''}>All ${NAMES[k]}</button>`+list.map(o=>`<button type="button" data-val="${o[0]}"${f[k]===o[0]?' class="on"':''}>${o[1]}<span>${group(o[2])}</span></button>`).join(''); });
    opts.addEventListener('click',e=>{ const b=e.target.closest('[data-val]'); if(!b) return; api.setFilter(opts.dataset.k,b.dataset.val); opts.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b)); label(); redraw(); });
    new ResizeObserver(redraw).observe(box);
    return {
      show(v){ on=v; box.hidden=!v; if(v){ label(); redraw(); } },
      refresh(){ label(); redraw(); },
      focus(h){ if(!h) return redraw(); fly({tx:h.x,ty:h.y,tz:h.z,dist:Math.max(12,Math.min(cam.dist,Math.max(h.d*0.45,40)))},1300); },
    };
  };
})();
