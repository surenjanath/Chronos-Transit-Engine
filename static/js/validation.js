// validation.js — ROC and training-history plots; live progress text while a job runs.
(function(){
  const d=jsonData('modelData');
  if(d){
    const tones=['ink','mute','gap'];
    CTEPlot(document.getElementById('rocPlot'),{height:240,xLabel:'False positive rate',yLabel:'True positive rate',xDomain:[0,1],yDomain:[0,1.02],
      series:Object.keys(d.roc).map((k,i)=>({name:`${k} (AUC ${d.auc[k].toFixed(3)})`,x:d.roc[k].fpr,y:d.roc[k].tpr,mode:'line',tone:tones[i],width:1.8,dash:i===1}))});
    CTEPlot(document.getElementById('histPlot'),{height:170,xLabel:'Epoch',xFmt:v=>Number.isInteger(v)?String(v):'',
      series:[{name:'Validation AUC',x:d.history.map(h=>h.epoch),y:d.history.map(h=>h.val_auc_planet),mode:'line',tone:'ink',width:1.8}]});
  }
  document.addEventListener('cte:live',e=>{
    const t=e.detail.training, s=document.getElementById('trainStage');
    if(s && t.status==='running'){ s.textContent=t.stage; const p=Math.round(t.progress*100)+'%';
      document.getElementById('trainPct').textContent=p; document.getElementById('trainBar').style.width=p; }
    const v=document.getElementById('valProgress'); if(v) v.textContent=e.detail.validation_progress;
  });
})();
