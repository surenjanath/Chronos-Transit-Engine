// orrery.js — the system view: a star, its planets' orbits to scale, and where each one is. Three views
// (system, planet, star), a habitable-zone ring, the inner Solar System for comparison, and a clock you can speed up.
// CTEOrrery(el, data, {static:true}) draws one still frame with no controls (the printed report).
(function(){
  const TAU=Math.PI*2, AU_PER_RSUN=0.00465047, JD_NOW=Date.now()/86400000+2440587.5;
  const SOLAR=[['Mercury',0.387,87.97,0.383],['Venus',0.723,224.7,0.949],['Earth',1.0,365.26,1.0]];
  const C={bg:'#05060b',line:'rgba(255,255,255,.42)',faint:'rgba(255,255,255,.16)',text:'#f1f1ee',mute:'#9aa0ad',cand:'#f0a27f',hz:'rgba(92,190,120,',sol:'rgba(140,170,255,.55)',label:'rgba(18,20,30,.82)'};
  const fmt=(v,d)=>v==null?'—':Number(v).toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:d});
  // black-body tint of a star from its temperature: red dwarfs orange, Sun-like yellow-white, hot stars blue-white
  function starColour(t){ const s=[[2500,[255,150,90]],[3700,[255,190,120]],[5200,[255,222,120]],[6000,[255,244,214]],[7500,[240,244,255]],[12000,[190,210,255]]];
    for(let i=1;i<s.length;i++) if(t<=s[i][0]){ const f=Math.max(0,(t-s[i-1][0])/(s[i][0]-s[i-1][0])); return 'rgb('+s[i-1][1].map((c,k)=>Math.round(c+(s[i][1][k]-c)*f)).join(',')+')'; }
    return 'rgb(190,210,255)'; }
  const planetColour=p=>p.status.indexOf('this engine')>=0?C.cand:(p.radius_earth>=6?'#d9b38c':p.radius_earth>=2?'#8fb8e8':'#c9c2b6');
  // where on its orbit a planet is today, when its transit time and period are known well enough to say; else null
  function today(p){ if(p.t0_jd==null||!p.period_err) return null; const n=(JD_NOW-p.t0_jd)/p.period_days; return Math.abs(n)*p.period_err/p.period_days>0.05?null:n-Math.floor(n); }

  window.CTEOrrery=function(el,data,opt){
    opt=opt||{}; const still=!!opt.static, star=data.star, planets=data.planets, hz=data.habitable_zone_au;
    const focus=planets.findIndex(p=>p.focus); let sel=focus>=0?focus:0, view='system', showHz=false, showSol=false, rate=1, clock=0, last=null, zoom=null;
    planets.forEach((p,i)=>{ p._known=today(p); p._phase0=p._known!=null?p._known:(p.t0_jd!=null?(((JD_NOW-p.t0_jd)/p.period_days)%1+1)%1:(i*0.37+0.11)%1); });
    el.classList.add('orrery'); if(still) el.classList.add('still');
    el.innerHTML=`<canvas></canvas>
      <div class="or-top"><label class="or-check"><input type="checkbox" data-k="hz"><span>Habitable zone</span></label></div>
      <div class="or-you">${data.distance_ly?`You are<b>${fmt(data.distance_ly,0)} light-years</b>from Earth`:''}</div>
      <div class="or-title"><h3>${data.host}</h3><p>${data.headline}</p></div>
      <div class="or-side"><div class="or-view-h"><span>View</span></div>
        <div class="or-tabs"><button type="button" data-view="planet">Planet</button><button type="button" data-view="system" class="on">System</button><button type="button" data-view="star">Star</button></div>
        <button type="button" class="or-compare" data-k="sol">Compare to our Solar System</button></div>
      <div class="or-info"></div>
      <div class="or-bar"><span class="or-rate-l">Rate</span><input type="range" min="-2" max="1.7" step="0.01" value="0" aria-label="Days per second"><span class="or-rate-n"></span>
        <button type="button" data-k="pause">Pause</button></div>`;
    const cv=el.querySelector('canvas'), g=cv.getContext('2d'), info=el.querySelector('.or-info'), rateN=el.querySelector('.or-rate-n');
    let W=0,H=0, playing=!still;
    // a fixed starfield, the same on every draw
    let seed=20261011; const rnd=()=>(seed=(seed*1664525+1013904223)%4294967296)/4294967296;
    const stars=Array.from({length:150},()=>[rnd(),rnd(),rnd()*1.3+.3,rnd()*.7+.2,rnd()<.12?'255,214,170':rnd()<.15?'170,190,255':'255,255,255']);
    const TILT=0.46, ROT=-0.33;   // the orbits are seen from above and to one side, as from a ship approaching the system
    const extent=()=>{ let a=Math.max(...planets.map(p=>p.a_au)); if(showHz) a=Math.max(a,hz[1]); if(showSol) a=Math.max(a,1.0); return a*1.18; };
    const wide=()=>W>900&&!still;
    const centre=()=>still?[W*0.5,H*0.5]:wide()?[(W-330)*0.5+6,(H-54)*0.47]:[W*0.5,H*0.3];
    const project=(a,th)=>{ const [cx,cy]=centre(), x=Math.cos(th)*a*zoom, y=Math.sin(th)*a*zoom*TILT; return [cx+x*Math.cos(ROT)-y*Math.sin(ROT), cy+x*Math.sin(ROT)+y*Math.cos(ROT)]; };
    const angle=p=>Math.PI/2-TAU*(p._phase0+clock/p.period_days);   // phase 0 is the transit: the planet in front of the star, towards us
    function ellipse(a,style,width,dash){ const [cx,cy]=centre(); g.beginPath(); g.ellipse(cx,cy,a*zoom,a*zoom*TILT,ROT,0,TAU); g.strokeStyle=style; g.lineWidth=width; g.setLineDash(dash||[]); g.stroke(); g.setLineDash([]); }
    let placed=[];
    function label(text,x,y,strong,colour){ g.font=(strong?'600 ':'')+'13px '+getComputedStyle(document.body).fontFamily; const w=g.measureText(text).width+12;
      for(let k=0;k<6;k++){ const hit=placed.find(b=>x+9<b[0]+b[2]&&x+9+w>b[0]&&y-11<b[1]+b[3]&&y+11>b[1]); if(!hit) break; y=hit[1]-13; }   // step above a label already there
      placed.push([x+9,y-11,w,22]); g.fillStyle=C.label; g.fillRect(x+9,y-11,w,22); g.fillStyle=colour||C.text; g.textBaseline='middle'; g.textAlign='left'; g.fillText(text,x+15,y+1); return [x+9,y-11,w,22]; }
    function backdrop(){ g.fillStyle=C.bg; g.fillRect(0,0,W,H); for(const s of stars){ g.fillStyle=`rgba(${s[4]},${s[3]})`; g.beginPath(); g.arc(s[0]*W,s[1]*H,s[2],0,TAU); g.fill(); } }
    // a star: a solid disc of radius r with a soft halo around it
    function glow(x,y,r,colour){ const soft=colour.replace('rgb','rgba'), gr=g.createRadialGradient(x,y,r*0.98,x,y,r*2.2+6); gr.addColorStop(0,soft.replace(')',',.42)')); gr.addColorStop(1,soft.replace(')',',0)'));
      g.fillStyle=gr; g.beginPath(); g.arc(x,y,r*2.2+6,0,TAU); g.fill(); g.fillStyle=colour; g.beginPath(); g.arc(x,y,r,0,TAU); g.fill(); }
    let hits=[];
    function drawSystem(){
      const room=still?[W*0.96,H*0.94]:wide()?[W-420,(H-54)*0.8]:[W*0.9,H*0.42];
      const target=Math.min(room[0]/(2*Math.cos(ROT)*1.0+2*TILT*Math.abs(Math.sin(ROT))),room[1]/(2*TILT*Math.cos(ROT)+2*Math.abs(Math.sin(ROT))))/extent(); zoom=zoom==null||still?target:zoom+(target-zoom)*0.12;
      const [cx,cy]=centre(); hits=[]; placed=[];
      if(showHz){ g.save(); g.beginPath(); g.ellipse(cx,cy,hz[1]*zoom,hz[1]*zoom*TILT,ROT,0,TAU); g.ellipse(cx,cy,hz[0]*zoom,hz[0]*zoom*TILT,ROT,0,TAU,true); g.fillStyle=C.hz+'.2)'; g.fill('evenodd'); g.restore();
        ellipse(hz[0],C.hz+'.55)',1); ellipse(hz[1],C.hz+'.55)',1); const q=project(hz[1],-0.9); g.fillStyle=C.hz+'.95)'; g.font='12px '+getComputedStyle(document.body).fontFamily; g.textAlign='left'; g.fillText('Habitable zone',q[0]+8,q[1]); }
      if(showSol) for(const s of SOLAR){ ellipse(s[1],C.sol,1,[2,5]); const q=project(s[1],Math.PI/2-TAU*(((JD_NOW+clock)/s[2])%1));
        g.fillStyle=C.sol; g.beginPath(); g.arc(q[0],q[1],3,0,TAU); g.fill(); g.font='12px '+getComputedStyle(document.body).fontFamily; g.textAlign='left'; g.textBaseline='middle'; g.fillText(s[0]+"'s orbit",q[0]+8,q[1]); }
      const order=planets.map((p,i)=>i);
      for(const i of order){ const p=planets[i], mine=p.status.indexOf('this engine')>=0; ellipse(p.a_au,i===sel?(mine?C.cand:'rgba(255,255,255,.85)'):(mine?'rgba(240,162,127,.6)':C.line),i===sel?1.6:1,p.status==='confirmed'?[]:[6,5]); }
      const rs=Math.max(star.radius_solar*AU_PER_RSUN*zoom,still?5:7); glow(cx,cy,rs,starColour(star.teff));
      const sb=label(data.host,cx+rs*0.4,cy+rs+16,false); hits.push([sb,'star']);
      for(const i of order){ const p=planets[i], q=project(p.a_au,angle(p)), r=Math.max(2.6,Math.min(9,2.2+Math.log2(1+(p.radius_earth||1))*1.9));
        g.fillStyle=planetColour(p); g.beginPath(); g.arc(q[0],q[1],r,0,TAU); g.fill();
        if(i===sel){ g.strokeStyle='rgba(255,255,255,.85)'; g.lineWidth=1; g.beginPath(); g.arc(q[0],q[1],r+4,0,TAU); g.stroke(); }
        const b=label(p.name+(p.status==='confirmed'||/candidate/i.test(p.name)?'':' · candidate'),q[0]+r*0.2,q[1]-r-12,i===sel,p.status.indexOf('this engine')>=0?C.cand:null); hits.push([b,i]); hits.push([[q[0]-r-6,q[1]-r-6,2*r+12,2*r+12],i]); }
    }
    function disc(x,y,r,colour,name,sub){ g.fillStyle=colour; g.beginPath(); g.arc(x,y,Math.max(r,1.2),0,TAU); g.fill();
      g.fillStyle=C.text; g.font='600 13px '+getComputedStyle(document.body).fontFamily; g.textAlign='center'; g.textBaseline='top'; g.fillText(name,x,y+Math.max(r,1.2)+12);
      g.fillStyle=C.mute; g.font='12px '+getComputedStyle(document.body).fontFamily; g.fillText(sub,x,y+Math.max(r,1.2)+30); }
    function row(items,colourOf){ const big=Math.max(...items.map(i=>i[1])), maxR=Math.min(H*0.2,W*0.11), px=v=>v/big*maxR, gap=Math.min(46,W*0.04);
      const total=items.reduce((s,i)=>s+2*Math.max(px(i[1]),18),0)+gap*(items.length-1); let x=(wide()?W-330:W-total>0?W:W)/2-total/2; const y=wide()?(H-54)*0.42:H*0.28;
      for(const it of items){ const r=px(it[1]), half=Math.max(r,18); x+=half; disc(x,y,r,colourOf(it),it[0],it[2]); x+=half+gap; } }
    function drawPlanet(){ const p=planets[sel], r=p.radius_earth||1;
      const items=[['Earth',1,'1.00 R⊕'],[p.name,r,fmt(r,2)+' R⊕']]; if(r>3) items.push(['Jupiter',11.21,'11.2 R⊕']); else items.push(['Neptune',3.88,'3.88 R⊕']);
      items.sort((a,b)=>a[1]-b[1]); row(items,it=>it[0]===p.name?planetColour(p):it[0]==='Earth'?'#6f9fd8':it[0]==='Neptune'?'#7f9fe0':'#d9b38c'); }
    function drawStar(){ const items=[['The Sun',1,5772],[data.host,star.radius_solar,star.teff]].sort((a,b)=>a[1]-b[1]);
      const big=items[1][1], maxR=Math.min((wide()?(H-54):H*0.5)*0.26,(wide()?W-330:W)*0.15), gap=maxR*1.1, r0=items[0][1]/big*maxR, r1=maxR;
      const mid=(wide()?W-330:W)/2+(wide()?40:0), y=wide()?(H-54)*0.44:H*0.26, xs=[mid-gap/2-r0,mid+gap/2+r1];
      items.forEach((it,k)=>{ const r=k?r1:r0; glow(xs[k],y,r,starColour(it[2])); g.fillStyle=C.text; g.font='600 13px '+getComputedStyle(document.body).fontFamily; g.textAlign='center'; g.textBaseline='top'; g.fillText(it[0],xs[k],y+maxR*1.5+14);
        g.fillStyle=C.mute; g.font='12px '+getComputedStyle(document.body).fontFamily; g.fillText(fmt(it[2],0)+' K · '+fmt(it[1],2)+' R☉',xs[k],y+maxR*1.5+32); }); }
    function facts(){
      if(view==='star'){ info.innerHTML=`<h4>${data.host}</h4><dl><dt>Type</dt><dd>${star.type}-type star</dd><dt>Temperature</dt><dd>${fmt(star.teff,0)} K</dd><dt>Size</dt><dd>${fmt(star.radius_solar,2)} × the Sun</dd><dt>Mass</dt><dd>${fmt(star.mass_solar,2)} × the Sun</dd><dt>Brightness</dt><dd>${fmt(star.luminosity_solar,2)} × the Sun</dd><dt>Habitable zone</dt><dd>${fmt(hz[0],2)}–${fmt(hz[1],2)} AU</dd></dl>`; return; }
      const p=planets[sel], inHz=p.a_au>=hz[0]&&p.a_au<=hz[1], mine=p.status.indexOf('this engine')>=0;
      info.innerHTML=`<h4>${p.name}</h4><p class="or-status${mine?' cand':''}">${p.status==='confirmed'?'Confirmed planet':p.status==='NASA candidate'?'Candidate in NASA’s list':'Candidate found by this engine'+(p.p_planet!=null?' · '+fmt(p.p_planet*100,1)+'% planet':'')}</p>
        <dl><dt>Size</dt><dd>${p.radius_earth?fmt(p.radius_earth,2)+' × Earth'+(p.kind?' · '+p.kind:''):'not measured'}</dd>
        <dt>From its star</dt><dd>${fmt(p.a_au,4)} AU</dd>
        <dt>In kilometres</dt><dd>${fmt(p.a_au*149.6,1)} million</dd>
        <dt>Its year</dt><dd>${fmt(p.period_days,p.period_days<100?3:1)} days</dd>
        <dt>Temperature</dt><dd>${p.temp_k?fmt(p.temp_k,0)+' K':'—'}</dd>
        ${p.mass_earth?`<dt>Mass</dt><dd>${fmt(p.mass_earth,2)} × Earth</dd>`:''}
        <dt>Habitable zone</dt><dd>${inHz?'inside it':p.a_au<hz[0]?fmt(hz[0]/p.a_au,1)+'× too close':fmt(p.a_au/hz[1],1)+'× too far'}</dd>
        <dt>Position today</dt><dd>${p._known!=null?fmt(p._known*100,0)+'% round since a transit':'not known'}</dd></dl>
        ${planets.length>1?`<div class="or-pick">${planets.map((q,i)=>`<button type="button" data-p="${i}"${i===sel?' class="on"':''}>${q.name.replace(data.host,'').trim()||q.name}</button>`).join('')}</div>`:''}`;
    }
    function draw(){ backdrop(); if(view==='system'||still) drawSystem(); else if(view==='planet') drawPlanet(); else drawStar(); }
    function size(){ const r=el.getBoundingClientRect(), dpr=window.devicePixelRatio||1; W=Math.max(280,r.width); H=cv.clientHeight||r.height; cv.width=W*dpr; cv.height=H*dpr; g.setTransform(dpr,0,0,dpr,0,0); if(still) zoom=null; draw(); }
    function frame(ts){ if(last!=null&&playing) clock+=(ts-last)/1000*rate; last=ts; draw(); requestAnimationFrame(frame); }
    const setRate=v=>{ rate=Math.pow(10,v); rateN.innerHTML=`<b>${rate<1?fmt(rate*24,rate*24<10?1:0)+' hours':fmt(rate,rate<10?1:0)+' day'+(rate===1?'':'s')}</b> per second`; };
    el.addEventListener('click',e=>{
      const t=e.target.closest('[data-view],[data-k],[data-p]');
      if(t){ if(t.dataset.view){ view=t.dataset.view; el.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('on',b===t)); el.dataset.view=view; }
        else if(t.dataset.p!=null){ sel=+t.dataset.p; }
        else if(t.dataset.k==='sol'){ showSol=!showSol; t.classList.toggle('on',showSol); t.textContent=showSol?'Hide our Solar System':'Compare to our Solar System'; view='system'; el.dataset.view=view; el.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('on',b.dataset.view==='system')); }
        else if(t.dataset.k==='pause'){ playing=!playing; t.textContent=playing?'Pause':'Play'; }
        else if(t.dataset.k==='hz'){ showHz=t.checked; }
        facts(); draw(); return; }
      if(e.target===cv&&view==='system'){ const b=cv.getBoundingClientRect(), x=e.clientX-b.left, y=e.clientY-b.top;
        const hit=hits.find(h=>x>=h[0][0]&&x<=h[0][0]+h[0][2]&&y>=h[0][1]&&y<=h[0][1]+h[0][3]); if(hit){ if(hit[1]==='star'){ view='star'; el.dataset.view=view; el.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('on',b.dataset.view==='star')); } else sel=hit[1]; facts(); draw(); } }
    });
    el.querySelector('input[type=range]').addEventListener('input',e=>setRate(+e.target.value));
    const want=(location.hash.match(/system=([a-z,]+)/)||[])[1]||'';   // #system=planet,hz opens the view in that state
    if(/hz/.test(want)){ showHz=true; el.querySelector('[data-k=hz]').checked=true; }
    if(/sol/.test(want)){ showSol=true; const b=el.querySelector('[data-k=sol]'); b.classList.add('on'); b.textContent='Hide our Solar System'; }
    const first=(want.match(/planet|star/)||[])[0]; if(first){ view=first; el.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('on',b.dataset.view===view)); }
    el.dataset.view=view; setRate(0); facts();
    new ResizeObserver(size).observe(el); size();
    if(!still) requestAnimationFrame(frame);
  };
  document.querySelectorAll('[data-orrery]').forEach(el=>{ const d=window.jsonData&&jsonData(el.dataset.orrery); if(d) CTEOrrery(el,d,{static:el.hasAttribute('data-static')}); });
})();
