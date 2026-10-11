// report.js — the charts of an analysis, drawn from data embedded in the page (research page and printed report).
(function(){
  const d=window.jsonData&&jsonData('seriesData'); if(!d) return;
  const $=id=>document.getElementById(id), ppm=y=>y.toFixed(0), v=d.visuals;
  const plot=(id,opt)=>{ const el=$(id); if(el) CTEPlot(el,opt); };
  const none=(id,text)=>{ const el=$(id); if(el) el.innerHTML=`<p class="empty">${text}</p>`; };
  const bin=d.local_view.filter(x=>x!=null), blo=Math.min(...bin), bhi=Math.max(...bin), bspan=(bhi-blo)||1e-4;
  plot('foldPlot',{height:250,xLabel:'Time from mid-transit (hours)',yLabel:'Detrended flux',yDomain:[blo-1.6*bspan,bhi+1.6*bspan],
    series:[{name:'Folded measurements',x:d.local_hours,y:d.local_flux,mode:'points',tone:'mute',size:2,alpha:.55,hover:false},
            {name:'Binned',x:d.local_x,y:d.local_view,mode:'line',tone:'ink',width:2}]});
  plot('pgPlot',{height:220,xLabel:'Trial period (days)',markers:[{x:d.best_period,label:'P = '+d.best_period.toFixed(4)+' d'}],
    series:[{name:'Signal residue',x:d.period,y:d.sr,mode:'line',tone:'ink',width:1.1}]});
  plot('lcPlot',{height:220,xLabel:'Time (days)',yLabel:'Detrended flux',markers:d.transits.length<=40?d.transits.map(x=>({x})):[],
    series:[{name:'Detrended flux',x:d.time,y:d.flux_flat,mode:'points',tone:'ink',size:1.6,alpha:.6}]});
  const tt=jsonData('timingData');
  if(tt&&tt.length) plot('ttvPlot',{height:210,xLabel:'Transit number',yLabel:'Minutes',yFmt:y=>y.toFixed(0),xFmt:x=>Number.isInteger(x)?String(x):'',
    series:[{name:'Observed − expected',x:tt.map(v=>v.epoch),y:tt.map(v=>v.o_minus_c_minutes),mode:'points',tone:'ink',size:4.5},
            {name:'On time',x:[tt[0].epoch,tt[tt.length-1].epoch],y:[0,0],mode:'line',tone:'gap',width:1.2,dash:true,hover:false}]});
  if(!v) return;
  plot('expPlot',{height:170,yLabel:'Detrended flux',
    series:[{name:'Observed, binned',x:v.x_hours,y:v.observed,mode:'points',tone:'mute',size:5},
            {name:v.geometry.from_shape?'Limb-darkened fit':'Fitted trapezoid',x:v.geometry.from_shape?v.phys_hours:v.x_hours,y:v.geometry.from_shape?v.model_shape:v.model_fit,mode:'line',tone:'ink',width:1.8}]});
  plot('resPlot',{height:100,xLabel:'Time from mid-transit (hours)',yLabel:'Residual (ppm)',yFmt:ppm,
    series:[{name:'Observed − fit',x:v.x_hours,y:v.residual_ppm,mode:'points',tone:'ink',size:4},
            {name:'zero',x:[v.x_hours[0],v.x_hours[v.x_hours.length-1]],y:[0,0],mode:'line',tone:'mute',width:1,hover:false,legend:false}]});
  if(v.odd&&v.even) plot('oddEvenPlot',{height:180,xLabel:'Hours from mid-transit',
    series:[{name:'Odd',x:v.x_hours,y:v.odd,mode:'line',tone:'ink',width:1.8},{name:'Even',x:v.x_hours,y:v.even,mode:'line',tone:'gap',width:1.6,dash:true}]});
  else none('oddEvenPlot','Too few transits to split by parity.');
  if(v.secondary){
    const seen=v.observed.filter(x=>x!=null), lo=Math.min(...seen), hi=Math.max(...seen);
    plot('secPlot',{height:180,xLabel:'Hours from phase 0.5',yDomain:[lo-(hi-lo)*.08,hi+(hi-lo)*.08],
      series:[{name:'Half an orbit later',x:v.x_hours,y:v.secondary,mode:'line',tone:'ink',width:1.8},
              {name:'The transit, for scale',x:v.x_hours,y:v.observed,mode:'line',tone:'mute',width:1.2,dash:true}]});
  } else none('secPlot','No coverage at phase 0.5.');
  if(v.transit_depths.length){
    const xs=v.transit_depths.map(t=>t.epoch), fit=v.geometry.depth*1e6;
    plot('depthPlot',{height:180,xLabel:'Transit number',yFmt:ppm,xFmt:x=>Number.isInteger(x)?String(x):'',
      series:[{name:'Depth (ppm)',x:xs,y:v.transit_depths.map(t=>t.depth_ppm),mode:'points',tone:'ink',size:5},
              {name:'Fitted depth',x:[Math.min(...xs)-.5,Math.max(...xs)+.5],y:[fit,fit],mode:'line',tone:'gap',width:1.2,dash:true,hover:false}]});
  } else none('depthPlot','No complete transits.');
  if($('stackPlot')){ if(v.stack.length&&window.CTEStack) CTEStack($('stackPlot'),v.stack,[v.x_hours[0],v.x_hours[v.x_hours.length-1]]); else none('stackPlot','No complete transits to stack.'); }
  if(d.explain) plot('salLocal',{height:190,yFmt:y=>(y*100).toFixed(0)+'%',xLabel:'Hours from mid-transit',
    series:[{name:'Probability lost',x:d.local_x,y:d.explain.local,mode:'line',tone:'ink',width:1.8}]});
  document.documentElement.setAttribute('data-charts','ready');
})();
