// globe.js — an artist's impression of a planet as a lit, slowly turning sphere, painted from the only things known
// about it: its size class and its temperature. CTEGlobe.look(kind, tempK) says what is drawn and why;
// CTEGlobe.make(look, seed) returns {draw(g, x, y, radius, turn)}.
(function(){
  const hex=h=>[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];
  const mix=(a,b,t)=>[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];
  function noise(seed,period){   // value noise on a lattice that wraps in x, so the map joins up round the back of the planet
    const h=(x,y)=>{ let n=(((x%period)+period)%period)*374761393+y*668265263+seed*1442695041; n=(n^(n>>>13))*1274126177; return ((n^(n>>>16))>>>0)/4294967295; };
    const sm=t=>t*t*(3-2*t);
    return (x,y)=>{ const xi=Math.floor(x), yi=Math.floor(y), fx=sm(x-xi), fy=sm(y-yi);
      return (h(xi,yi)*(1-fx)+h(xi+1,yi)*fx)*(1-fy)+(h(xi,yi+1)*(1-fx)+h(xi+1,yi+1)*fx)*fy; };
  }
  const LOOKS={
    gasCold:{style:'bands',c:['#cdb38c','#eadfc4','#a9744d','#dcc6a2','#8e6748','#f1e8d2'],text:'A giant planet composed mainly of gas',why:'cold enough for pale ammonia clouds in bands, like Jupiter'},
    gasWater:{style:'bands',c:['#e3ebf0','#b6c9d6','#f5f7f7','#9bb3c6','#cfdde6'],text:'A giant planet composed mainly of gas',why:'at this temperature its clouds would be white water vapour'},
    gasClear:{style:'bands',c:['#3f6fb0','#5d90d2','#2e548f','#7ba7dd','#4a7dc0'],text:'A giant planet composed mainly of gas',why:'too warm for clouds, so its deep atmosphere would scatter blue'},
    gasDark:{style:'bands',c:['#3d4a5c','#57667a','#2b3545','#6f7d90','#465468'],text:'A giant planet composed mainly of gas',why:'hot enough that sodium and potassium vapour would absorb most light, leaving it dark'},
    gasHot:{style:'bands',c:['#5a3a34','#8b4b36','#39221f','#b7663b','#6d3f33'],glow:'#ff7a3a',text:'A giant planet composed mainly of gas',why:'so hot it would glow, with clouds of vaporised rock'},
    nepCold:{style:'bands',soft:1,c:['#4f86c6','#70a6de','#3c6eae','#90bce7','#5b93d0'],text:'A Neptune-like planet with a deep atmosphere',why:'cold, with methane tinting its hydrogen atmosphere blue'},
    nepWarm:{style:'bands',soft:1,c:['#7fb5a6','#abd3c5','#5c998a','#d2e8df','#93c4b6'],text:'A Neptune-like planet with a deep atmosphere',why:'warm, under pale hazes'},
    nepHot:{style:'bands',soft:1,c:['#8a7fa3','#aba1c2','#6b6287','#c5bed8','#978cb0'],text:'A Neptune-like planet with a deep atmosphere',why:'hot, wrapped in high haze'},
    lava:{style:'rock',c:['#1c1614','#3a2a24','#2a1f1b','#4a342b'],crack:'#ff6a2a',glow:'#ff5a1e',text:'A rocky world',why:'hot enough for its dayside rock to melt'},
    barren:{style:'rock',c:['#8a6f55','#b0926f','#6d5642','#c7ab86'],text:'A rocky world',why:'too hot to hold water, so bare rock'},
    temperate:{style:'rock',c:['#8d8271','#a89c88','#6f6657','#bdb29d'],cloud:1,text:'A rocky world',why:'at a temperature where water could be liquid; what its surface is like is not known'},
    ice:{style:'rock',c:['#cfd9e2','#e9f0f5','#a9bccb','#f5f8fa'],text:'A rocky world',why:'cold enough to be frozen over'},
    unknown:{style:'rock',c:['#7c7c82','#94949a','#66666c','#a8a8ae'],text:'A planet of unknown size',why:'its size has not been measured'},
    earth:{style:'earth',text:'Earth'},
  };
  // which look a planet gets from its size class and temperature (kelvin); a super-Earth is drawn as rock, which is a guess
  function look(kind,temp){
    const t=temp==null?500:temp; let key;
    if(kind==='Gas giant') key=t<150?'gasCold':t<350?'gasWater':t<900?'gasClear':t<1400?'gasDark':'gasHot';
    else if(kind==='Neptune-like') key=t<400?'nepCold':t<1000?'nepWarm':'nepHot';
    else if(kind==='Terrestrial'||kind==='Super-Earth') key=t>1300?'lava':t>320?'barren':t>200?'temperate':'ice';
    else key='unknown';
    const l=Object.assign({key},LOOKS[key]);
    if(kind==='Super-Earth') l.text='A world larger than Earth and smaller than Neptune';
    if(kind==='Terrestrial') l.text='A rocky world about the size of Earth or smaller';
    return l;
  }
  const TW=512, TH=256;
  function texture(l,seed){
    const tex=new Uint8ClampedArray(TW*TH*3), n1=noise(seed,8), n2=noise(seed+7,16), n3=noise(seed+13,32), n4=noise(seed+29,64);
    const cols=(l.c||[]).map(hex);
    for(let y=0;y<TH;y++) for(let x=0;x<TW;x++){
      const u=x/TW, v=y/TH; let c;
      const f=(n1(u*8,v*4)*0.5+n2(u*16,v*8)*0.28+n3(u*32,v*16)*0.15+n4(u*64,v*32)*0.07);
      if(l.style==='bands'){
        const warp=(n2(u*16,v*8)-0.5)*(l.soft?0.018:0.05)+(n3(u*32,v*30)-0.5)*(l.soft?0.006:0.02);
        const b=(v+warp)*(l.soft?8:19)+n1(0.5,v*9)*1.6, i=Math.floor(b), t=b-i, e=t*t*(3-2*t);
        const pick=k=>cols[(((k*7+seed)%cols.length)+cols.length)%cols.length];
        c=mix(pick(i),pick(i+1),e); const s=0.84+f*0.22+(n4(u*64,v*110)-0.5)*(l.soft?0.08:0.2); c=[c[0]*s,c[1]*s,c[2]*s];
        if(!l.soft){ const sx=(u-0.62+1)%1-0.5, sy=v-0.63, d=sx*sx*6+sy*sy*40; if(d<0.02) c=mix(c,cols[2],0.75*(1-d/0.02)); }   // one great storm
      }else if(l.style==='earth'){
        const land=f+0.10*Math.cos((v-0.45)*5)-0.04, lat=Math.abs(v-0.5)*2;
        c=land>0.5?mix(hex('#3f7a45'),hex('#a08a5f'),Math.min(1,(land-0.5)*5)):mix(hex('#123c7a'),hex('#2a66b0'),Math.max(0,land*1.8));
        if(lat>0.86) c=mix(c,[240,246,250],Math.min(1,(lat-0.86)*8));
        const cl=n2(u*16+3,v*8+1)*0.42+n3(u*32,v*16)*0.33+n4(u*64,v*32)*0.25; if(cl>0.56) c=mix(c,[250,250,250],Math.min(0.9,(cl-0.56)*5));
      }else{
        const t=Math.min(1,Math.max(0,(f-0.25)*2)); c=mix(mix(cols[2],cols[0],Math.min(1,t*2)),mix(cols[1],cols[3],Math.max(0,t*2-1)),t);
        const crater=n4(u*64,v*32); if(crater>0.78) c=mix(c,cols[2],(crater-0.78)*2.2);
        if(l.crack){ const k=Math.abs(n2(u*16,v*8)-0.5)+Math.abs(n3(u*32+5,v*16)-0.5)*0.5; if(k<0.035) c=mix(hex(l.crack),c,k/0.035); }
        if(l.cloud){ const cl=n2(u*16+3,v*8+1)*0.42+n3(u*32,v*16)*0.33+n4(u*64,v*32)*0.25; if(cl>0.55) c=mix(c,[246,247,248],Math.min(0.85,(cl-0.55)*5)); }
      }
      const o=(y*TW+x)*3; tex[o]=c[0]; tex[o+1]=c[1]; tex[o+2]=c[2];
    }
    return tex;
  }
  const R=230, D=2*R, LIGHT=(()=>{ const v=[-0.52,-0.42,0.74], n=Math.hypot(...v); return v.map(x=>x/n); })();
  let map=null;   // for each pixel of the disc: where on the sphere it is, and how brightly lit
  function sphere(){
    if(map) return map; const lon=new Float32Array(D*D), lat=new Int16Array(D*D), lit=new Float32Array(D*D), edge=new Float32Array(D*D);
    for(let y=0;y<D;y++) for(let x=0;x<D;x++){ const nx=(x+0.5-R)/R, ny=(y+0.5-R)/R, r2=nx*nx+ny*ny, i=y*D+x;
      if(r2>=1){ lat[i]=-1; continue; } const nz=Math.sqrt(1-r2);
      lon[i]=Math.atan2(nx,nz)/(2*Math.PI); lat[i]=Math.min(TH-1,Math.floor((Math.asin(ny)/Math.PI+0.5)*TH));
      const d=nx*LIGHT[0]+ny*LIGHT[1]+nz*LIGHT[2]; lit[i]=Math.max(0,d)*0.92+0.05; edge[i]=Math.min(1,(1-Math.sqrt(r2))*R/1.6); }
    return map={lon,lat,lit,edge};
  }
  function make(l,seed){
    const tex=texture(l,seed||1), s=sphere(), cv=document.createElement('canvas'); cv.width=cv.height=D;
    const g2=cv.getContext('2d'), img=g2.createImageData(D,D), px=img.data, glow=l.glow?hex(l.glow):null;
    let drawn=null;
    function paint(turn){
      for(let i=0;i<D*D;i++){ const o=i*4; if(s.lat[i]<0){ px[o+3]=0; continue; }
        let u=s.lon[i]+turn; u-=Math.floor(u); const t=(s.lat[i]*TW+Math.floor(u*TW))*3, L=s.lit[i];
        let r=tex[t]*L, gg=tex[t+1]*L, b=tex[t+2]*L;
        if(glow){ const dark=1-Math.min(1,L*1.4); r+=glow[0]*0.34*dark; gg+=glow[1]*0.34*dark; b+=glow[2]*0.34*dark; }   // the night side of a very hot world shines by its own heat
        px[o]=r; px[o+1]=gg; px[o+2]=b; px[o+3]=255*s.edge[i]; }
      g2.putImageData(img,0,0);
    }
    return {look:l, draw(g,x,y,radius,turn){ const step=Math.round((turn||0)*TW*2)/(TW*2); if(step!==drawn){ paint(step); drawn=step; }
      if(radius>3){ const halo=g.createRadialGradient(x,y,radius*0.97,x,y,radius*1.07); const hc=l.c?hex(l.c[1]):[120,170,255];
        halo.addColorStop(0,`rgba(${hc[0]|0},${hc[1]|0},${hc[2]|0},.22)`); halo.addColorStop(1,'rgba(0,0,0,0)'); g.fillStyle=halo; g.beginPath(); g.arc(x,y,radius*1.07,0,6.2832); g.fill(); }
      g.imageSmoothingEnabled=true; g.imageSmoothingQuality='high'; g.drawImage(cv,x-radius,y-radius,radius*2,radius*2); }};
  }
  window.CTEGlobe={look,make,LOOKS};
})();
