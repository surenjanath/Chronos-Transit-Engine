// map.js — a pannable, zoomable map of real space, from Earth's orbit out to the whole Galaxy.
// World units are light-years (Sun at the origin, galactic axes) in the two space views and degrees in the sky view.
// CTEMap(el, {data_url, voyager1, radio_ly, galactic_centre_ly, galactic_plane, mine, focus, year, new_url})
(function(){
  const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const AU=1/63241.077, TAU=6.283185307;                       // one astronomical unit in light-years
  const SOLAR=[['Mercury',0.387],['Venus',0.723],['Earth',1],['Mars',1.524],['Jupiter',5.203],['Saturn',9.537],['Uranus',19.19],['Neptune',30.07]];
  // Where each planet is today. Solar System: mean longitude at J2000 (degrees) and its daily motion.
  const SOLAR_NOW={Mercury:[252.251,4.0923344],Venus:[181.980,1.6021303],Earth:[100.464,0.9856474],Mars:[355.453,0.5240207],Jupiter:[34.404,0.0830853],Saturn:[49.944,0.0334441],Uranus:[313.232,0.0117283],Neptune:[304.880,0.0059810]};
  const JD_NOW=Date.now()/86400000+2440587.5;
  const solarAngle=name=>{ const e=SOLAR_NOW[name]; return e?-((e[0]+e[1]*(JD_NOW-2451545.0))%360)*Math.PI/180:null; };   // anticlockwise seen from the north
  // Another star's planet: orbits since its last measured transit. At transit it lies between its star and Earth, so
  // the angle is counted from the direction of the Sun on the map. Only shown while the period is known well enough that the position has drifted under 1/20 of an orbit.
  const transitAngle=p=>{ if(p.t0==null||!p.P||!p.Pe) return null; const n=(JD_NOW-p.t0)/p.P; if(Math.abs(n)*p.Pe/p.P>0.05) return null; return -(n-Math.floor(n))*2*Math.PI; };
  const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const group=v=>Math.round(v).toLocaleString('en-US');
  const sig=(v,n=3)=>{ if(!isFinite(v)||v===0) return '0'; const d=Math.max(0,n-1-Math.floor(Math.log10(Math.abs(v)))); return v>=1000?group(+v.toPrecision(n)):(+v.toFixed(d)).toString(); };
  function fmtDist(ly){
    const au=ly/AU;
    if(au<0.05) return sig(au*149.5979)+' million km';
    if(au<10000) return sig(au)+' AU';
    return sig(ly)+(ly===1?' light-year':' light-years');
  }
  function fmtShort(ly){ const au=ly/AU; return au<10000?sig(au,2)+' AU':sig(ly,2)+' ly'; }
  function fmtYears(y){
    if(y<1e4) return sig(y)+' years';
    if(y<1e6) return group(+y.toPrecision(3))+' years';
    if(y<1e9) return sig(y/1e6)+' million years';
    return sig(y/1e9)+' billion years';
  }
  function fmtLight(ly){
    const s=ly*31557600;
    if(s<120) return sig(s)+' seconds';
    if(s<7200) return sig(s/60)+' minutes';
    if(s<172800) return sig(s/3600)+' hours';
    if(s<31557600*1.5) return sig(s/86400)+' days';
    return fmtYears(ly);
  }
  const VOY=299792.458/16.95;                                   // years Voyager 1 needs per light-year

  window.CTEMap=function(el,ctx){
    const cv=el.querySelector('canvas'), tip=el.querySelector('#mapTip'), loading=el.querySelector('#mapLoading');
    const scaleEl=el.querySelector('#mapScale'), readout=document.getElementById('mapReadout'), panel=document.getElementById('mapPanel');
    const mine=new Map((ctx.mine||[]).map(m=>[m.host,m]));
    const startHash=location.hash;   // read before the first draw rewrites it
    let W=0,H=0, systems=[], byName=new Map(), view='top', sel=null, hover=null, anim=null, raf=0, visible=[];
    const cams={space:{cx:0,cy:0,s:1}, sky:{cx:-180,cy:0,s:1}};
    const filt={method:'',small:false,temperate:false,naked:false};
    const cam=()=>view==='sky'?cams.sky:cams.space;
    const P=h=>view==='top'?[h.x,h.y]:view==='side'?[h.x,h.z]:[-h.ra,h.dec];
    const X=wx=>W/2+(wx-cam().cx)/cam().s, Y=wy=>H/2-(wy-cam().cy)/cam().s;
    const limits=()=>view==='sky'?[0.4/W,400/W]:[0.012*AU/W,170000/W];
    const clampCam=()=>{ const c=cam(),[lo,hi]=limits(); c.s=Math.min(hi,Math.max(lo,c.s));
      if(view==='sky'){ c.cx=Math.min(0,Math.max(-360,c.cx)); c.cy=Math.min(90,Math.max(-90,c.cy)); } };

    // ---------- filters ----------
    const MAIN=['Transit','Radial Velocity','Microlensing','Imaging'];
    function passes(h){
      if(filt.naked && !(h.vmag!=null && h.vmag<=6)) return false;
      if(!filt.method && !filt.small && !filt.temperate) return true;
      return h.planets.some(p=>(!filt.method || (filt.method==='other'?!MAIN.includes(p.method):p.method===filt.method))
        && (!filt.small || (p.r!=null && p.r<1.6)) && (!filt.temperate || (p.S!=null && p.S>=0.25 && p.S<=1.75)));
    }

    // ---------- drawing ----------
    function draw(){
      raf=0;
      const dpr=window.devicePixelRatio||1; W=el.clientWidth; H=el.clientHeight;
      if(cv.width!==Math.round(W*dpr)||cv.height!==Math.round(H*dpr)){ cv.width=Math.round(W*dpr); cv.height=Math.round(H*dpr); }
      clampCam();
      const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
      const C={ink:cssv('--ink'),mute:cssv('--mute'),faint:cssv('--faint'),rule:cssv('--rule'),canvas:cssv('--canvas'),gap:cssv('--gap'),ok:cssv('--ok')};
      g.font='11px '+getComputedStyle(document.body).fontFamily; g.textBaseline='middle'; g.lineWidth=1;
      const boxes=[];                                            // label rectangles already placed
      const label=(text,x,y,o={})=>{
        const w=g.measureText(text).width, al=o.align||'left', x0=al==='center'?x-w/2:al==='right'?x-w:x;
        const b=[x0-3,y-8,x0+w+3,y+8];
        if(b[2]<0||b[0]>W||b[3]<0||b[1]>H) return false;
        if(!o.force && boxes.some(q=>b[0]<q[2]&&b[2]>q[0]&&b[1]<q[3]&&b[3]>q[1])) return false;
        boxes.push(b);
        g.save(); g.textAlign='left'; g.lineJoin='round'; g.lineWidth=3; g.strokeStyle=C.canvas; g.strokeText(text,x0,y);
        g.fillStyle=o.color||C.mute; g.fillText(text,x0,y); g.restore(); return true;
      };
      const ring=(wx,wy,r,o={})=>{                               // circle of world radius r; returns its radius in pixels
        const rp=r/cam().s, x=X(wx), y=Y(wy);
        if(rp<(o.min||3)||rp>4*Math.hypot(W,H)) return rp;
        if(x+rp<0||x-rp>W||y+rp<0||y-rp>H) return rp;
        g.save(); g.strokeStyle=o.color||C.rule; g.lineWidth=o.width||1; if(o.dash) g.setLineDash(o.dash);
        g.beginPath(); g.arc(x,y,rp,0,TAU); g.stroke(); g.restore();
        if(o.label && rp>(o.labelMin||26)){                      // label on the ring, at the point nearest the middle of the view
          let ang=-Math.PI/2; const dx=W/2-x, dy=H/2-y;
          if(Math.hypot(dx,dy)>rp*0.35) ang=Math.atan2(dy,dx);
          else if(o.angle!=null) ang=o.angle;
          label(o.label,x+rp*Math.cos(ang),y+rp*Math.sin(ang)-(Math.sin(ang)<-0.3?9:Math.sin(ang)>0.3?-9:0),{align:'center',color:o.labelColor});
        }
        return rp;
      };
      const band=(wx,wy,r0,r1,alpha)=>{
        const x=X(wx), y=Y(wy), a=r0/cam().s, b=r1/cam().s;
        if(b<2||a>2*Math.hypot(W,H)) return;
        g.save(); g.globalAlpha=alpha; g.fillStyle=C.mute; g.beginPath(); g.arc(x,y,b,0,TAU); g.arc(x,y,Math.max(a,0),0,TAU,true); g.fill('evenodd'); g.restore();
      };
      const orbits=(wx,wy,list,star)=>{                          // orbits, to scale in size, seen from above, with each planet where it is today when that is known
        const amax=Math.max(...list.map(p=>p[1]))*AU/cam().s; if(!(amax>5)) return;
        if(star && star.lum){ const a=Math.sqrt(star.lum/1.1)*AU, b=Math.sqrt(star.lum/0.53)*AU; band(wx,wy,a,b,0.13);
          if(b/cam().s>40) label('Liquid-water zone',X(wx),Y(wy)+(a+b)/2/cam().s,{align:'center'}); }
        list.forEach((p,i)=>ring(wx,wy,p[1]*AU,{color:p[2]||C.ink,label:p[0],labelColor:p[2]||C.ink,labelMin:9,angle:-Math.PI/2+(i%2?0.5:-0.5)+i*0.9}));
        const home=star && !star.sun?Math.atan2(Y(0)-Y(wy),X(0)-X(wx)):0;
        list.forEach(p=>{ const rp=p[1]*AU/cam().s; if(p[3]==null||rp<9||rp>4*Math.hypot(W,H)) return;                 // the planet itself, where it is today
          g.save(); g.fillStyle=p[2]||C.ink; g.strokeStyle=C.canvas; g.lineWidth=1.5; g.beginPath(); g.arc(X(wx)+rp*Math.cos(home+p[3]),Y(wy)+rp*Math.sin(home+p[3]),3.2,0,TAU); g.fill(); g.stroke(); g.restore(); });
        if(star && !star.sun && amax<400) ring(wx,wy,AU,{color:C.ok,dash:[3,4],label:'Earth’s orbit, for scale',labelColor:C.ok,labelMin:14,angle:Math.PI/2});
      };

      if(view==='sky') drawSky(g,C,label); else {
        // ---- the Galaxy ----
        const gc=ctx.galactic_centre_ly, s=cam().s;
        if(170000/s>60){
          if(view==='top'){
            ring(gc,0,50000,{dash:[2,5],label:'Edge of the Galaxy’s disc (approximate)',min:40});
            ring(gc,0,gc,{dash:[1,5],label:'The Sun’s path around the Galaxy · one lap takes about 230 million years',min:40,angle:Math.PI*0.75});
          } else {
            g.save(); g.strokeStyle=C.rule; g.beginPath(); g.moveTo(X(gc-50000),Y(0)); g.lineTo(X(gc+50000),Y(0)); g.stroke(); g.restore();
            if(50000/s>80) label('Plane of the Galaxy, edge-on',X(gc)+60,Y(0)+12);
          }
          const gx=X(gc), gy=Y(0);
          if(gx>-20&&gx<W+20){ g.save(); g.strokeStyle=C.ink; g.beginPath(); g.moveTo(gx-6,gy); g.lineTo(gx+6,gy); g.moveTo(gx,gy-6); g.lineTo(gx,gy+6); g.stroke(); g.restore();
            label('Centre of the Galaxy · '+group(gc)+' light-years',gx+10,gy-10,{color:C.ink}); }
        }
        // ---- distance rings: powers of ten in astronomical units, then in light-years ----
        for(let k=-1;k<=4;k++){ const au=Math.pow(10,k); ring(0,0,au*AU,{label:`${au>=1?group(au):au} AU · light takes ${fmtLight(au*AU)}`,min:30,labelMin:44}); }
        for(let k=0;k<=5;k++){ const ly=Math.pow(10,k); ring(0,0,ly,{label:`${group(ly)} light-year${ly>1?'s':''} · Voyager 1 would take ${fmtYears(ly*VOY)}`,min:30,labelMin:44}); }
        // ---- the Sun's surroundings ----
        band(0,0,30*AU,50*AU,0.10); if(40*AU/s>60) label('Kuiper belt',X(0),Y(0)+40*AU/s,{align:'center'});
        band(0,0,2000*AU,100000*AU,0.07); if(100000*AU/s>80 && 2000*AU/s<Math.max(W,H)) label('Oort cloud (inferred; never seen directly)',X(0),Y(0)+Math.min(50000*AU/s,H*0.4),{align:'center'});
        ring(0,0,121*AU,{dash:[4,4],label:'Heliopause · the Sun’s wind stops here',min:22,labelMin:40,angle:Math.PI*0.25});
        ring(0,0,ctx.radio_ly,{color:C.ok,dash:[1,4],label:`Radio bubble · the farthest any human broadcast has reached (${ctx.radio_ly} light-years, since 1906)`,labelColor:C.ok,min:14,labelMin:30,angle:-Math.PI*0.25});
        orbits(0,0,SOLAR.map(p=>[p[0],p[1],p[0]==='Earth'?C.ok:null,solarAngle(p[0])]),{sun:true,lum:1});
        // Voyager 1
        const v=ctx.voyager1, vp=view==='top'?[v.x,v.y]:[v.x,v.z], vx=X(vp[0]), vy=Y(vp[1]), vr=Math.hypot(vp[0],vp[1])/s;
        if(vr>18 && vr<6*Math.hypot(W,H)){
          g.save(); g.strokeStyle=C.gap; g.setLineDash([2,3]); g.beginPath(); g.moveTo(X(0),Y(0)); g.lineTo(vx,vy); g.stroke(); g.setLineDash([]);
          g.fillStyle=C.gap; g.beginPath(); g.moveTo(vx,vy-5); g.lineTo(vx+4,vy+3); g.lineTo(vx-4,vy+3); g.closePath(); g.fill(); g.restore();
          label(`Voyager 1 · about ${group(v.au)} AU after ${ctx.year-1977} years of flight`,vx+9,vy,{color:C.gap});
        }
      }

      // ---- stars with planets ----
      visible=[];
      for(const h of systems){ if(!h._on) continue; const p=P(h), x=X(p[0]), y=Y(p[1]); if(x<-8||x>W+8||y<-8||y>H+8) continue; h._x=x; h._y=y; visible.push(h); }
      const n=visible.length, dense=n>1500?0.42:n>500?0.6:n>150?0.8:1, base=n>1500?1.3:n>500?1.7:n>120?2.2:3;
      if(view!=='sky') for(const h of visible){ if(h._amax && h._amax*AU/cam().s>5) orbits(P(h)[0],P(h)[1],h.planets.filter(p=>p.a!=null).map(p=>[p.name.replace(h.name,'').trim()||p.name,p.a,null,transitAngle(p)]),{lum:h._lum}); }
      if(sel && view!=='sky'){ const p=P(sel); g.save(); g.strokeStyle=C.ink; g.setLineDash([5,4]); g.beginPath(); g.moveTo(X(0),Y(0)); g.lineTo(X(p[0]),Y(p[1])); g.stroke(); g.restore();
        const mx=(X(0)+X(p[0]))/2, my=(Y(0)+Y(p[1]))/2; if(Math.hypot(X(p[0])-X(0),Y(p[1])-Y(0))>120) label(fmtDist(sel.d)+' from the Sun',mx,my-10,{align:'center',color:C.ink,force:true}); }
      g.save(); g.globalAlpha=dense; g.fillStyle=C.ink;
      for(const h of visible){ if(mine.has(h.name)) continue; const r=base+(n>500?0:Math.min(2.5,(h.planets.length-1)*0.45)); g.beginPath(); g.arc(h._x,h._y,r,0,TAU); g.fill(); }
      g.restore();
      for(const h of visible){ if(!mine.has(h.name)) continue; g.fillStyle=C.gap; g.beginPath(); g.arc(h._x,h._y,Math.max(base+1.2,3.4),0,TAU); g.fill(); g.strokeStyle=C.canvas; g.lineWidth=1.5; g.stroke(); g.lineWidth=1; }
      for(const h of [hover,sel]){ if(!h||!h._on) continue; if(view!=='sky' && h._amax*AU/cam().s>5) continue; const p=P(h); g.strokeStyle=h===sel?C.ink:C.mute; g.lineWidth=h===sel?1.6:1; g.beginPath(); g.arc(X(p[0]),Y(p[1]),8,0,TAU); g.stroke(); g.lineWidth=1; }
      // ---- the Sun ----
      if(view!=='sky'){ const x=X(0), y=Y(0); g.fillStyle=C.ok; g.beginPath(); g.arc(x,y,4.5,0,TAU); g.fill(); g.strokeStyle=C.canvas; g.lineWidth=1.5; g.stroke(); g.lineWidth=1;
        label(AU/cam().s>14?'The Sun':'The Sun and Earth',x+9,y+1,{color:C.ok,force:true}); }
      // ---- names ----
      const want=[sel,hover,...visible.filter(h=>mine.has(h.name))].filter(Boolean);
      const rest=n<=700?visible.slice().sort((a,b)=>b.planets.length-a.planets.length||a.d-b.d):[];
      let placed=0;
      for(const h of [...new Set([...want,...rest])]){ if(!h._on) continue; const p=P(h), forced=h===sel;
        if(label(h.name,X(p[0])+9,Y(p[1])-(forced?0:1),{color:mine.has(h.name)?C.gap:(forced||h===hover)?C.ink:C.mute,force:forced})) placed++;
        if(placed>=(n<=60?60:26)) break; }
      hud();
    }

    function drawSky(g,C,label){
      g.save(); g.strokeStyle=C.rule;
      for(let ra=0;ra<=360;ra+=30){ const x=X(-ra); if(x<-1||x>W+1) continue; g.beginPath(); g.moveTo(x,Math.max(0,Y(90))); g.lineTo(x,Math.min(H,Y(-90))); g.stroke(); label((ra/15)+'h',x+4,Math.min(H-10,Y(-90)-10)); }
      for(let dec=-90;dec<=90;dec+=30){ const y=Y(dec); if(y<-1||y>H+1) continue; g.beginPath(); g.moveTo(Math.max(0,X(0)),y); g.lineTo(Math.min(W,X(-360)),y); g.stroke(); label((dec>0?'+':'')+dec+'°',Math.max(4,X(0)+4),y-8); }
      g.strokeStyle=C.mute; g.setLineDash([5,5]); g.beginPath(); let prev=null;
      for(const q of ctx.galactic_plane){ const x=X(-q[0]), y=Y(q[1]); if(prev!=null && Math.abs(q[0]-prev)<90) g.lineTo(x,y); else g.moveTo(x,y); prev=q[0]; }
      g.stroke(); g.restore();
      const mid=ctx.galactic_plane[60]; label('Plane of the Milky Way',X(-mid[0])+8,Y(mid[1])-10);
      const gx=X(-266.405), gy=Y(-28.936); g.save(); g.strokeStyle=C.ink; g.beginPath(); g.moveTo(gx-6,gy); g.lineTo(gx+6,gy); g.moveTo(gx,gy-6); g.lineTo(gx,gy+6); g.stroke(); g.restore();
      label('Centre of the Galaxy',gx+10,gy+12,{color:C.ink});
      label('Kepler’s field of view',X(-290.67)+26/Math.max(cam().s,0.02)*0.02+16,Y(44.5)-(9/cam().s)-10,{color:C.ink});
      label('North',W/2,Math.max(10,Y(90)+10),{align:'center'}); label('East ←',Math.max(30,X(0)+60),Math.max(10,Y(90)+10));
    }

    // ---------- heads-up display ----------
    function nice(v){ const p=Math.pow(10,Math.floor(Math.log10(v))), m=v/p; return (m>=5?5:m>=2?2:1)*p; }
    function hud(){
      const s=cam().s, bar=scaleEl.querySelector('i'), txt=scaleEl.querySelector('span');
      if(view==='sky'){
        const d=nice(s*120); bar.style.width=(d/s)+'px'; txt.textContent=d>=1?d+'°':(d*60).toFixed(0)+' arcminutes';
        const w=s*W; readout.textContent=`This view of the sky is ${sig(Math.min(w,360),2)}° wide. The full Moon is half a degree across, so ${group(Math.min(w,360)/0.5)} of them would fit side by side. ${visible.length?group(visible.length)+' stars with planets are in view.':''}`;
      } else {
        const inAU=s*120<6000*AU, d=inAU?nice(s*120/AU):nice(s*120), ly=inAU?d*AU:d;
        bar.style.width=(ly/s)+'px'; txt.textContent=inAU?`${d>=1?group(d):d} AU`:`${group(d)} light-year${d===1?'':'s'}`;
        const w=s*W; readout.textContent=`This view is ${fmtDist(w)} across. Light crosses it in ${fmtLight(w)}; Voyager 1 would need ${fmtYears(w*VOY)}. ${visible.length?group(visible.length)+(visible.length===1?' star with planets is':' stars with planets are')+' in view.':'No star with known planets is in view.'}`;
      }
      const cnt=document.getElementById('mapCount'); if(cnt) cnt.textContent=systems.length?`${group(systems.filter(h=>h._on).length)} of ${group(systems.length)} systems shown`:'';
      try{ const c=cam(); history.replaceState(null,'',`#${view}/${c.cx.toPrecision(9)}/${c.cy.toPrecision(9)}/${c.s.toPrecision(6)}${sel?'/'+encodeURIComponent(sel.name):''}`); }catch(_){}
    }
    const redraw=()=>{ if(!raf) raf=requestAnimationFrame(draw); };

    // ---------- the side panel ----------
    function showPanel(){
      if(!sel){
        panel.innerHTML=`<h2 class="sheet-title">You are here</h2>
          <p class="coach">The green dot is the Sun; Earth circles it at one astronomical unit (AU), 150 million km. Scroll to zoom out and watch that distance shrink to nothing.</p>
          <dl class="kv"><dt>Sunlight reaches Earth in</dt><dd>8.3 minutes</dd><dt>Farthest spacecraft, Voyager 1</dt><dd>${group(ctx.voyager1.au)} AU</dd>
          <dt>Nearest star with planets</dt><dd>${systems[0]?esc(systems[0].name)+' · '+sig(systems[0].d)+' ly':'—'}</dd>
          <dt>Farthest on this map</dt><dd>${systems.length?group(systems[systems.length-1].d)+' ly':'—'}</dd></dl>
          <p class="coach" style="margin-top:14px">Select any star to see its planets, how long its light has been travelling, and how long a journey there would take.</p>`;
        return;
      }
      const h=sel, m=mine.get(h.name), yr=ctx.year-h.d, left=yr>0?`the year ${Math.round(yr)}`:`${group(Math.round(-yr)+1)} BCE`;
      const methods=[...new Set(h.planets.map(p=>p.method).filter(Boolean))].join(', ');
      const rows=h.planets.map(p=>`<tr><td>${esc(p.name)}<span class="sub">${p.year||''}${p.method?' · '+esc(p.method):''}</span></td><td class="r num">${p.r!=null?sig(p.r)+' R⊕':'—'}</td><td class="r num">${p.P!=null?sig(p.P)+' d':'—'}</td><td class="r num">${p.a!=null?sig(p.a)+' AU':'—'}</td></tr>`).join('');
      const rmax=Math.max(11.2,...h.planets.map(p=>p.r||0)), px=r=>Math.max(1.5,r/rmax*26);
      const discs=[['Earth',1,'ok'],...h.planets.filter(p=>p.r!=null).map(p=>[p.name.replace(h.name,'').trim()||p.name,p.r,'ink']),['Jupiter',11.2,'mute']]
        .map(d=>`<span class="map-disc"><i style="width:${2*px(d[1])}px;height:${2*px(d[1])}px;background:var(--${d[2]})"></i>${esc(d[0])}</span>`).join('');
      panel.innerHTML=`<h2 class="sheet-title">${esc(h.name)} ${m?'<span class="quiet">· analysed here</span>':''}</h2>
        <div class="fig-label">Distance from Earth</div><div class="fig-value">${sig(h.d)} <span class="fig-detail">light-years</span></div>
        <p class="coach" style="margin:10px 0 14px">The light you would see from it tonight left in ${left}.</p>
        <dl class="kv"><dt>In astronomical units</dt><dd>${group(+(h.d/AU).toPrecision(3))} AU</dd>
          <dt>Voyager 1 would arrive in</dt><dd>${fmtYears(h.d*VOY)}</dd>
          <dt>Parker Solar Probe, at its fastest</dt><dd>${fmtYears(h.d*299792.458/192)}</dd>
          <dt>An airliner</dt><dd>${fmtYears(h.d*299792.458/0.25)}</dd>
          <dt>Star</dt><dd>${h.type?esc(h.type)+' · ':''}${h.teff?group(h.teff)+' K':'—'}${h.rad?' · '+sig(h.rad)+' R☉':''}</dd>
          <dt>Brightness</dt><dd>${h.vmag!=null?'V = '+h.vmag+(h.vmag<=6?' · visible to the naked eye':h.vmag<=10?' · binoculars':' · telescope'):'—'}</dd>
          <dt>Where in the sky</dt><dd>RA ${(h.ra/15).toFixed(2)}h · Dec ${h.dec>0?'+':''}${h.dec.toFixed(1)}°</dd>
          <dt>Found by</dt><dd>${esc(methods)||'—'}</dd></dl>
        <h3 style="font-size:13px;font-weight:500;margin:18px 0 6px">${h.planets.length} known planet${h.planets.length>1?'s':''}</h3>
        <table class="mini"><thead><tr><th>Planet</th><th class="r">Radius</th><th class="r">Period</th><th class="r">Orbit size</th></tr></thead><tbody>${rows}</tbody></table>
        ${h.planets.some(p=>p.r!=null)?`<h3 style="font-size:13px;font-weight:500;margin:16px 0 8px">Sizes, to scale</h3><div class="map-discs">${discs}</div>`:''}
        <div class="actions" style="flex-wrap:wrap">
          <button type="button" class="go sm" data-act="journey">Travel there from Earth</button>
          ${h._amax?'<button type="button" class="sm" data-act="system">See its orbits</button>':''}
          ${m?`<a class="sm" href="${esc(m.url)}">Open analysis</a>`:`<a class="sm" href="${esc(ctx.new_url)}?mast_target=${encodeURIComponent(h.name)}&search=1">Analyse this star</a>`}
          <button type="button" class="text" data-act="clear">Clear</button>
        </div>`;
    }
    panel.addEventListener('click',e=>{
      const act=e.target.closest('[data-act]')?.dataset.act; if(!act||!sel) return;
      if(act==='clear'){ sel=null; showPanel(); redraw(); }
      if(act==='journey') journey(sel);
      if(act==='system'){ if(view==='sky') setView('top'); const p=P(sel); fly(p[0],p[1],sel._amax*AU*3.2/Math.min(W,H),1600); }
    });

    // ---------- motion ----------
    function fly(cx,cy,s,ms,path){
      const c=cam(), from={cx:c.cx,cy:c.cy,s:c.s}, t0=performance.now(); cancelAnimationFrame(anim);
      const [lo,hi]=limits(); s=Math.min(hi,Math.max(lo,s));
      const step=now=>{
        const u=Math.min(1,(now-t0)/ms), e=u<0.5?2*u*u:1-Math.pow(-2*u+2,2)/2;
        c.s=from.s*Math.pow(s/from.s,e);
        const k=path==='out'?(c.s-from.s)/((s-from.s)||1):path==='in'?1-(c.s-s)/((from.s-s)||1):e;   // keep the start in view while the scale changes
        c.cx=from.cx+(cx-from.cx)*k; c.cy=from.cy+(cy-from.cy)*k;
        redraw(); if(u<1) anim=requestAnimationFrame(step);
      };
      anim=requestAnimationFrame(step);
    }
    function journey(h){                                           // from Earth's orbit, straight out until the star comes into view
      if(view==='sky') setView('top');
      const p=P(h), c=cam(); cancelAnimationFrame(anim);
      c.cx=0; c.cy=0; c.s=3.2*AU/Math.min(W,H); redraw();
      const span=Math.hypot(p[0],p[1])||h.d;
      setTimeout(()=>fly(p[0]/2,p[1]/2,span/(0.62*Math.min(W,H)),Math.min(14000,3500+1100*Math.log10(span/AU))),500);
    }
    function select(h,move){
      sel=h; showPanel();
      if(h && move){ const p=P(h);
        if(view==='sky') fly(p[0],p[1],Math.min(cam().s,40/W),900);
        else { const span=Math.hypot(p[0],p[1])||1; fly(p[0]/2,p[1]/2,Math.max(span/(0.62*Math.min(W,H)),limits()[0]),1300); } }
      redraw();
    }
    function setView(v){
      view=v; document.querySelectorAll('#mapViews [data-view]').forEach(b=>b.classList.toggle('on',b.dataset.view===v));
      document.getElementById('mapPresets').style.visibility=v==='sky'?'hidden':'';
      cancelAnimationFrame(anim); redraw();
    }
    const PRESET={earth:[0,0,3.2*AU],solar:[0,0,80*AU],voyager:[0,0,420*AU],nearest:[0,0,30],galaxy:[ctx.galactic_centre_ly,0,112000]};
    function preset(k){
      if(k==='kepler'){ const ks=systems.filter(h=>Math.abs(h.ra-290.7)<8&&Math.abs(h.dec-44.5)<8&&h.d<6000); if(!ks.length) return;
        const xs=ks.map(h=>P(h)[0]), ys=ks.map(h=>P(h)[1]), x0=Math.min(0,...xs), x1=Math.max(0,...xs), y0=Math.min(0,...ys), y1=Math.max(0,...ys);
        return fly((x0+x1)/2,(y0+y1)/2,Math.max((x1-x0)/W,(y1-y0)/H)*1.15,1500); }
      const p=PRESET[k]; if(p) fly(p[0],p[1],p[2]/Math.min(W,H),1500,cam().s<p[2]/Math.min(W,H)?'out':'in');
    }

    // ---------- input ----------
    const at=e=>{ const r=cv.getBoundingClientRect(); return [e.clientX-r.left,e.clientY-r.top]; };
    function pick(x,y){ let best=null,bd=14*14; for(const h of visible){ const d=(h._x-x)**2+(h._y-y)**2; if(d<bd){ bd=d; best=h; } } return best; }
    function zoomAt(x,y,f){ const c=cam(), wx=c.cx+(x-W/2)*c.s, wy=c.cy-(y-H/2)*c.s; cancelAnimationFrame(anim); c.s*=f; clampCam(); c.cx=wx-(x-W/2)*c.s; c.cy=wy+(y-H/2)*c.s; redraw(); }
    cv.addEventListener('wheel',e=>{ e.preventDefault(); const [x,y]=at(e); zoomAt(x,y,Math.exp(Math.max(-60,Math.min(60,e.deltaY))*(e.ctrlKey?0.012:0.0035))); },{passive:false});
    const ptrs=new Map(); let drag=null, pinch=0;
    cv.addEventListener('pointerdown',e=>{ cv.setPointerCapture(e.pointerId); ptrs.set(e.pointerId,at(e)); const [x,y]=at(e); drag={x,y,moved:0}; cancelAnimationFrame(anim);
      if(ptrs.size===2){ const p=[...ptrs.values()]; pinch=Math.hypot(p[0][0]-p[1][0],p[0][1]-p[1][1]); } });
    cv.addEventListener('pointermove',e=>{
      const [x,y]=at(e);
      if(ptrs.has(e.pointerId)){
        ptrs.set(e.pointerId,[x,y]);
        if(ptrs.size===2){ const p=[...ptrs.values()], d=Math.hypot(p[0][0]-p[1][0],p[0][1]-p[1][1]); if(pinch>0&&d>0) zoomAt((p[0][0]+p[1][0])/2,(p[0][1]+p[1][1])/2,pinch/d); pinch=d; return; }
        if(drag){ const c=cam(); c.cx-=(x-drag.x)*c.s; c.cy+=(y-drag.y)*c.s; drag.moved+=Math.abs(x-drag.x)+Math.abs(y-drag.y); drag.x=x; drag.y=y; el.classList.add('dragging'); redraw(); }
        return;
      }
      const h=pick(x,y);
      if(h!==hover){ hover=h; redraw(); }
      if(h){ tip.hidden=false; tip.textContent=`${h.name} · ${sig(h.d)} light-years · ${h.planets.length} planet${h.planets.length>1?'s':''}`; tip.style.left=Math.min(x+14,W-tip.offsetWidth-4)+'px'; tip.style.top=Math.max(4,y-34)+'px'; cv.style.cursor='pointer'; }
      else { tip.hidden=true; cv.style.cursor=''; }
    });
    const up=e=>{ const was=drag; ptrs.delete(e.pointerId); if(ptrs.size<2) pinch=0; if(!ptrs.size){ drag=null; el.classList.remove('dragging'); }
      if(was && was.moved<5 && e.type==='pointerup'){ const [x,y]=at(e), h=pick(x,y); if(h) select(h,false); } };
    cv.addEventListener('pointerup',up); cv.addEventListener('pointercancel',up);
    cv.addEventListener('pointerleave',()=>{ if(hover){ hover=null; redraw(); } tip.hidden=true; });
    cv.addEventListener('dblclick',e=>{ const [x,y]=at(e); zoomAt(x,y,0.4); });
    el.addEventListener('keydown',e=>{ const c=cam(), k=e.key, d=60*c.s;
      if(k==='+'||k==='=') zoomAt(W/2,H/2,0.6); else if(k==='-'||k==='_') zoomAt(W/2,H/2,1/0.6);
      else if(k==='ArrowLeft') c.cx-=d; else if(k==='ArrowRight') c.cx+=d; else if(k==='ArrowUp') c.cy+=d; else if(k==='ArrowDown') c.cy-=d;
      else if(k==='Escape'){ sel=null; showPanel(); } else return;
      e.preventDefault(); redraw(); });
    el.querySelector('.map-zoom').addEventListener('click',e=>{ const z=e.target.dataset.zoom; if(z==='in') zoomAt(W/2,H/2,0.5); if(z==='out') zoomAt(W/2,H/2,2);
      if(z==='home'){ if(view==='sky') fly(-180,0,400/W,900); else fly(0,0,Math.min(cam().s,80*AU/Math.min(W,H)),1200,'in'); } });
    document.getElementById('mapViews').addEventListener('click',e=>{ const v=e.target.dataset.view; if(v) setView(v); });
    document.getElementById('mapPresets').addEventListener('click',e=>{ const p=e.target.dataset.preset; if(p) preset(p); });
    document.addEventListener('click',e=>{ const b=e.target.closest('[data-fly]'); if(!b) return; const h=byName.get(b.dataset.fly); if(h){ el.scrollIntoView({behavior:'smooth',block:'center'}); select(h,true); } });
    const applyFilters=()=>{ for(const h of systems) h._on=passes(h); if(sel&&!sel._on){ sel=null; showPanel(); } redraw(); };
    document.getElementById('mapFilters').addEventListener('change',()=>{
      filt.method=document.querySelector('input[name=mapMethod]:checked').value; filt.small=document.getElementById('mapSmall').checked;
      filt.temperate=document.getElementById('mapTemperate').checked; filt.naked=document.getElementById('mapNaked').checked; applyFilters(); });
    // search: stars and planets by name
    const search=document.getElementById('mapSearch'), sug=document.getElementById('mapSuggest');
    let index=[];
    const suggest=()=>{ const q=search.value.trim().toLowerCase().replace(/\s+/g,''); if(q.length<2){ sug.hidden=true; return; }
      const hits=[]; for(const it of index){ const i=it[0].indexOf(q); if(i>=0){ hits.push([i,it]); if(hits.length>400) break; } }
      hits.sort((a,b)=>a[0]-b[0]||a[1][1].length-b[1][1].length);
      sug.innerHTML=hits.slice(0,8).map(x=>`<li><button type="button" data-host="${esc(x[1][2].name)}">${esc(x[1][1])}<span class="sub">${sig(x[1][2].d)} light-years</span></button></li>`).join('')||'<li><span class="quiet" style="padding:7px 14px;display:block;font-size:13px">Nothing by that name in the archive</span></li>';
      sug.hidden=false; };
    search.addEventListener('input',suggest);
    search.addEventListener('keydown',e=>{ if(e.key==='Enter'){ e.preventDefault(); sug.querySelector('button')?.click(); } if(e.key==='Escape') sug.hidden=true; });
    sug.addEventListener('click',e=>{ const b=e.target.closest('[data-host]'); if(!b) return; const h=byName.get(b.dataset.host); sug.hidden=true; search.value=b.dataset.host; if(h){ if(!h._on){ document.querySelector('input[name=mapMethod][value=""]').checked=true; ['mapSmall','mapTemperate','mapNaked'].forEach(i=>document.getElementById(i).checked=false); Object.assign(filt,{method:'',small:false,temperate:false,naked:false}); applyFilters(); } select(h,true); } });
    document.addEventListener('click',e=>{ if(!e.target.closest('.map-search')) sug.hidden=true; });
    new ResizeObserver(redraw).observe(el);
    new MutationObserver(redraw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});

    // ---------- start ----------
    W=el.clientWidth; H=el.clientHeight;
    cams.space.s=80*AU/Math.min(W,H); cams.sky.s=400/W;
    const restore=()=>{ const m=startHash.slice(1).split('/'); if(m.length>=4 && ['top','side','sky'].includes(m[0]) && m.slice(1,4).every(v=>isFinite(+v))){
      setView(m[0]); Object.assign(cam(),{cx:+m[1],cy:+m[2],s:+m[3]}); if(m[4]){ const h=byName.get(decodeURIComponent(m[4])); if(h){ sel=h; } } return true; } return false; };
    showPanel(); redraw();
    fetch(ctx.data_url).then(r=>r.json()).then(d=>{
      systems=d.systems||[];
      for(const h of systems){ byName.set(h.name,h); h._on=true; const a=h.planets.map(p=>p.a).filter(v=>v!=null); h._amax=a.length?Math.max(...a):0;
        h._lum=(h.rad&&h.teff)?h.rad*h.rad*Math.pow(h.teff/5772,4):0;
        index.push([h.name.toLowerCase().replace(/\s+/g,''),h.name,h]); for(const id of h.ids) index.push([id.toLowerCase().replace(/\s+/g,''),id+' · '+h.name,h]);
        for(const p of h.planets) index.push([p.name.toLowerCase().replace(/\s+/g,''),p.name,h]); }
      loading.hidden=true;
      const focus=ctx.focus&&byName.get(ctx.focus);
      if(focus){ showPanel(); select(focus,true); } else if(restore()){ showPanel(); redraw(); } else { showPanel(); redraw(); }
    }).catch(()=>{ loading.textContent='The catalogue could not be loaded. The Solar System is still drawn to scale.'; });
  };
})();
