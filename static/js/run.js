// run.js — live progress while a run executes; light curve, periodogram and folded plots once it has finished.
(function(){
  const liveRun=document.getElementById('liveRun');
  if(liveRun){
    const steps=[...document.querySelectorAll('#runSteps div')], esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
    const net=window.CTENeural?CTENeural(document.getElementById('neuralLive'),{state:'idle'}):null;
    const clock=s=>s<60?s+' s':Math.floor(s/60)+' min '+(s%60)+' s';
    // what each stage does, for runs whose live commentary is not available (started from the command line)
    const ABOUT={INGEST:'Fetching the star’s brightness measurements from the archive.',NORMALIZATION:'Scaling the measurements so the star’s normal brightness is 1.',
      OUTLIER_REJECTION:'Throwing out glitches: single measurements far from their neighbours.',DETRENDING:'Removing the star’s slow brightness changes so only short dips remain.',
      BLS_SEARCH:'Trying tens of thousands of possible orbital periods, looking for a dip that repeats on schedule.',PHASE_FOLDING:'Stacking every transit on top of the others to build one clean dip.',
      PHYSICS_DERIVATION:'Turning the dip into a companion: its size from the depth, its orbit from the period.',VETTING_DIAGNOSTICS:'Checking for impostors: odd against even transits, a second eclipse, a shift in the star’s image.',
      CNN_INFERENCE:'Showing the folded transit to five neural networks: planet, eclipsing binary, or noise?',REFINED_ANALYSIS:'Fitting a physical transit model, then removing this signal and searching for more planets.',
      ARCHIVE:'Saving the light curve, the results and the audit trail.'};
    const cnnAt=steps.findIndex(el=>el.textContent.trim()==='cnn inference');
    const tick=async()=>{
      try{
        const d=await (await fetch(liveRun.dataset.url)).json();
        if(d.status==='completed'||d.status==='failed'){ location.reload(); return; }
        document.getElementById('runPct').textContent=d.percent+'%';
        document.getElementById('runStage').textContent=d.status==='queued'?'queued':d.stage_label;
        steps.forEach((el,i)=>{ el.className=i<d.stage_index?'done':i===d.stage_index?'now':''; });
        document.getElementById('runNote').textContent=d.note||(d.status==='queued'?'Waiting for a free worker.':ABOUT[d.stage]||'Working…');
        document.getElementById('runElapsed').textContent=clock(d.elapsed_s);
        if(d.events.length) document.getElementById('runFeed').innerHTML=d.events.map(e=>`<li><span class="quiet num">${e.time}</span><span>${esc(e.stage)}</span><span>${esc(e.message)}${e.facts.length?`<span class="facts">${esc(e.facts.join(' · '))}</span>`:''}</span></li>`).join('');
        if(net) net.set(d.stage_index===cnnAt?'thinking':'idle');
      }catch(_){}
      setTimeout(tick,900);
    };
    tick();
    return;
  }
  const plots=document.getElementById('plots'); if(!plots) return;
  fetch(plots.dataset.url).then(r=>{ if(!r.ok) throw new Error('Archived series are missing for this run.'); return r.json(); }).then(d=>{
    const markers=d.transits.map((x,i)=>({x,label:i===0?'t₀':undefined}));
    const lc=view=>CTEPlot(document.getElementById('lcPlot'),view==='norm'?{height:270,zoom:true,xLabel:'Time (days)',yLabel:'Normalized flux',markers,
        series:[{name:'Normalized flux',x:d.time,y:d.flux_norm,mode:'points',tone:'mute',size:1.6,alpha:.7},
                {name:'Savitzky-Golay trend',x:d.time,y:d.trend,mode:'line',tone:'ink',width:1.6},
                {name:'Sigma-clipped',x:d.rejected_time,y:d.rejected_flux,mode:'points',tone:'gap',size:5,hover:false}]}
      :{height:270,zoom:true,xLabel:'Time (days)',yLabel:'Detrended flux',markers,
        series:[{name:'Detrended flux',x:d.time,y:d.flux_flat,mode:'points',tone:'ink',size:1.6,alpha:.6}]});
    const fold=view=>CTEPlot(document.getElementById('foldPlot'),view==='local'?{height:240,xLabel:'Time from mid-transit (hours)',yLabel:'Detrended flux',
        series:[{name:'Folded flux',x:d.local_hours,y:d.local_flux,mode:'points',tone:'mute',size:2,alpha:.55,hover:false},
                {name:'Local tensor',x:d.local_x,y:d.local_view,mode:'line',tone:'ink',width:2}]}
      :{height:240,xLabel:'Orbital phase',yLabel:'Detrended flux',
        series:[{name:'Folded flux',x:d.fold_phase,y:d.fold_flux,mode:'points',tone:'mute',size:1.6,alpha:.5,hover:false},
                {name:'Global tensor',x:d.global_x,y:d.global_view,mode:'line',tone:'ink',width:1.8}]});
    const wire=(id,fn)=>{ const seg=document.getElementById(id);
      seg.addEventListener('click',e=>{ const b=e.target.closest('button'); if(!b) return;
        seg.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));
        const host=document.getElementById(id==='lcSeg'?'lcPlot':'foldPlot'); if(host._legend) host._legend.remove(); fn(b.dataset.view); }); };
    lc('norm'); fold('local'); wire('lcSeg',lc); wire('foldSeg',fold);
    CTEPlot(document.getElementById('pgPlot'),{height:240,zoom:true,xLabel:'Trial period (days)',
      markers:[{x:d.best_period,label:'P = '+d.best_period.toFixed(5)+' d'}],
      series:[{name:'Signal residue',x:d.period,y:d.sr,mode:'line',tone:'ink',width:1.1}]});
    const nd=document.getElementById('neuralDone');
    if(nd && window.CTENeural) CTENeural(nd,{state:'thinking',global:d.global_view,local:d.local_view,
      probs:{planet:+nd.dataset.planet,binary:+nd.dataset.binary,noise:+nd.dataset.noise}});
    const v=d.visuals;
    if(v && document.getElementById('visualModel')){
      const $=id=>document.getElementById(id), ppm=y=>y.toFixed(0);
      CTESystem($('systemView'),v.geometry,{hours:v.phys_hours,flux:v.model_shape||v.model_physical},{hours:v.x_hours,flux:v.observed});
      CTEPlot($('expPlot'),{height:210,yLabel:'Detrended flux',
        series:[{name:'Observed, binned',x:v.x_hours,y:v.observed,mode:'points',tone:'mute',size:5},
                {name:v.geometry.from_shape?'Limb-darkened fit':'Fitted trapezoid',x:v.geometry.from_shape?v.phys_hours:v.x_hours,y:v.geometry.from_shape?v.model_shape:v.model_fit,mode:'line',tone:'ink',width:1.8},
                {name:'Predicted for the catalogued star, b = '+v.geometry.b_assumed,x:v.phys_hours,y:v.model_physical,mode:'line',tone:'gap',width:1.4,dash:true}]});
      CTEPlot($('resPlot'),{height:120,xLabel:'Time from mid-transit (hours)',yLabel:'Residual (ppm)',yFmt:ppm,
        series:[{name:'Observed − fit',x:v.x_hours,y:v.residual_ppm,mode:'points',tone:'ink',size:4},
                {name:'zero',x:[v.x_hours[0],v.x_hours[60]],y:[0,0],mode:'line',tone:'mute',width:1,hover:false,legend:false}]});
      if(v.odd&&v.even) CTEPlot($('oddEvenPlot'),{height:190,xLabel:'Hours from mid-transit',
        series:[{name:'Odd',x:v.x_hours,y:v.odd,mode:'line',tone:'ink',width:1.8},{name:'Even',x:v.x_hours,y:v.even,mode:'line',tone:'gap',width:1.6,dash:true}]});
      else $('oddEvenPlot').innerHTML='<p class="empty">Too few transits to split by parity.</p>';
      if(v.secondary){
        const lo=Math.min(...v.observed.filter(x=>x!=null)), hi=Math.max(...v.observed.filter(x=>x!=null));
        CTEPlot($('secPlot'),{height:190,xLabel:'Hours from phase 0.5',yDomain:[lo-(hi-lo)*.08,hi+(hi-lo)*.08],
          series:[{name:'Flux at phase 0.5',x:v.x_hours,y:v.secondary,mode:'line',tone:'ink',width:1.8},
                  {name:'Primary transit, for scale',x:v.x_hours,y:v.observed,mode:'line',tone:'mute',width:1.2,dash:true}]});
      } else $('secPlot').innerHTML='<p class="empty">No coverage at phase 0.5.</p>';
      if(v.transit_depths.length){
        const xs=v.transit_depths.map(t=>t.epoch), fit=v.geometry.depth*1e6;
        CTEPlot($('depthPlot'),{height:190,xLabel:'Transit number',yFmt:ppm,xFmt:x=>Number.isInteger(x)?String(x):'',
          series:[{name:'Depth (ppm)',x:xs,y:v.transit_depths.map(t=>t.depth_ppm),mode:'points',tone:'ink',size:6},
                  {name:'Fitted depth',x:[Math.min(...xs)-.5,Math.max(...xs)+.5],y:[fit,fit],mode:'line',tone:'gap',width:1.2,dash:true,hover:false}]});
      } else $('depthPlot').innerHTML='<p class="empty">No complete transits.</p>';
      if(v.stack.length) CTEStack($('stackPlot'),v.stack,[v.x_hours[0],v.x_hours[60]]);
      else $('stackPlot').innerHTML='<p class="empty">No complete transits to stack.</p>';
      if(d.explain){
        const imp=y=>(y*100).toFixed(0)+'%';
        CTEPlot($('salLocal'),{height:150,yFmt:imp,xLabel:'Local view · hours from mid-transit',
          series:[{name:'Probability lost',x:d.local_x,y:d.explain.local,mode:'line',tone:'ink',width:1.8}]});
        CTEPlot($('salGlobal'),{height:150,yFmt:imp,xLabel:'Global view · orbital phase',
          series:[{name:'Probability lost',x:d.global_x,y:d.explain.global,mode:'line',tone:'ink',width:1.4}]});
      }
    }
  }).catch(e=>{ document.getElementById('lcPlot').innerHTML=`<p class="empty">${e.message}</p>`; });
})();
