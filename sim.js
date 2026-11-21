/* Spectator live simulator: the real model (spectator-v1, the frozen model the paper reports), run in the browser.
   Files in data/ are binary float16/float32 tensors for Friends S6E02 and The Wolf of Wall Street (part 8).

   state at the newest frame e:  fast_t = Wf [x_{t-4}; ...; x_t] + bf,   slow_e = bf + scale * sum_l w[:, l] (fast_{e-115+l} - bf)
                                 s_e = [fast_e; slow_e]
   one fMRI frame (1.49 s):      s <- s + L s + d_j        d_j = W std(movie features of frames j-4..j) + c (precomputed per clip)
   parcels:                      y = P s + b               (sub-01's frozen benchmark probe)
   frames not scanned enter the encoder as the model's own forecast, as in the benchmark's Updating task. */
(function(){
"use strict";
const N=2048,NF=1024,P=1000,LAGS=116,HIST=120;

// float16 -> float32
const F16=new Float32Array(65536);
for(let h=0;h<65536;h++){const s=h&0x8000?-1:1,e=(h>>10)&31,f=h&1023;F16[h]=e===0?s*f*Math.pow(2,-24):e===31?(f?NaN:s*Infinity):s*(1+f/1024)*Math.pow(2,e-15);}
function decode(buf,dtype){if(dtype==='float16'){const u=new Uint16Array(buf),o=new Float32Array(u.length);for(let i=0;i<u.length;i++)o[i]=F16[u[i]];return o;}return new Float32Array(buf);}

const unb64=t=>{const bin=atob(t.trim()),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer;};
const stripExt=n=>n.replace(/(\.b64\.txt|\.bin)$/,'');

async function fetchTensor(base,name,info){
  const r=await fetch(base+name);
  if(!r.ok)throw new Error('Failed to fetch '+name+' (HTTP '+r.status+')');
  const b=info.encoding==='base64'?unb64(await r.text()):await r.arrayBuffer();
  return decode(b,info.dtype);
}

async function loadSpectator(base,onProgress,opts){
  opts=opts||{};
  const meta=await (await fetch(base+'sim.json')).json();
  const allNames=Object.keys(meta.files);
  const clipSet=opts.clipIndices?new Set(opts.clipIndices):null;
  const weightNames=allNames.filter(n=>!n.startsWith('clip'));
  const clipFilesByIdx=meta.clips.map((_,k)=>allNames.filter(n=>n.startsWith(`clip${k}_`)));
  const initialNames=clipSet
    ? weightNames.concat(...[...clipSet].map(k=>clipFilesByIdx[k]||[]))
    : allNames;
  const total=initialNames.reduce((a,n)=>a+meta.files[n].bytes,0);
  let done=0;
  const raw={};
  await Promise.all(initialNames.map(async n=>{
    const m=meta.files[n];
    raw[stripExt(n)]=await fetchTensor(base,n,m);
    done+=m.bytes;
    if(onProgress)onProgress(done/total,done,total);
  }));
  const W={L:raw['L.f16'],Wf:raw['Wf.f16'],bf:raw['bf.f32'],lagw:raw['lagw.f32'],P:raw['P.f16'],b:raw['b.f32'],scale:meta.scale};
  const clipPromises={};
  const clips=meta.clips.map((c,k)=>{
    const loaded=!clipSet||clipSet.has(k);
    return Object.assign({},c,{
      loaded,
      me:loaded?raw[`clip${k}_me.f16`]:null,
      other:loaded?raw[`clip${k}_other.f16`]:null,
      drive:loaded?raw[`clip${k}_drive.f16`]:null
    });
  });
  async function loadClip(k){
    if(clips[k].loaded)return clips[k];
    if(clipPromises[k])return clipPromises[k];
    clipPromises[k]=(async()=>{
      const files=clipFilesByIdx[k];
      await Promise.all(files.map(async n=>{
        raw[stripExt(n)]=await fetchTensor(base,n,meta.files[n]);
      }));
      clips[k].me=raw[`clip${k}_me.f16`];
      clips[k].other=raw[`clip${k}_other.f16`];
      clips[k].drive=raw[`clip${k}_drive.f16`];
      clips[k].loaded=true;
      return clips[k];
    })().catch(err=>{delete clipPromises[k];throw err;});
    return clipPromises[k];
  }
  return {meta,W,clips,loadClip};
}

class SpectatorEngine{
  constructor(W){this.W=W;this.s=new Float32Array(N);this.xs=new Float32Array(N);this.tmp=new Float32Array(N);
    this.buf=new Float32Array(HIST*P);this.head=0;
    this.fasts=new Float32Array(LAGS*NF);this.fhead=0;
    this.win=new Float32Array(5*P);}
  frame(i){const k=(this.head+i)%HIST;return this.buf.subarray(k*P,(k+1)*P);}
  fastOf(out){
    for(let o=0;o<5;o++)this.win.set(this.frame(HIST-5+o),o*P);
    const Wf=this.W.Wf,bf=this.W.bf,w=this.win;
    for(let i=0;i<NF;i++){let a=bf[i];const r=i*5*P;for(let j=0;j<5*P;j++)a+=Wf[r+j]*w[j];out[i]=a;}}
  encode(){const s=this.s,bf=this.W.bf,lw=this.W.lagw,sc=this.W.scale,F=this.fasts;
    const newest=(this.fhead+LAGS-1)%LAGS;
    for(let i=0;i<NF;i++)s[i]=F[newest*NF+i];
    for(let i=0;i<NF;i++){let a=0;for(let l=0;l<LAGS;l++){const k=(this.fhead+l)%LAGS;a+=lw[i*LAGS+l]*(F[k*NF+i]-bf[i]);}s[NF+i]=bf[i]+sc*a;}
    this.xs.set(s);}
  loadHistory(frames,first){
    const h=frames.subarray(first*P,(first+HIST)*P),Wf=this.W.Wf,bf=this.W.bf;
    this.buf.set(h);this.head=0;
    for(let e=4;e<HIST;e++){
      const w=h.subarray((e-4)*P,(e+1)*P),o=(e-4)*NF;
      for(let i=0;i<NF;i++){let a=bf[i];const r=i*5*P;for(let j=0;j<5*P;j++)a+=Wf[r+j]*w[j];this.fasts[o+i]=a;}}
    this.fhead=0;this.encode();}
  read(s,out,bias){const Pm=this.W.P,b=this.W.b;for(let p=0;p<P;p++){let a=bias?b[p]:0;const r=p*N;for(let j=0;j<N;j++)a+=Pm[r+j]*s[j];out[p]=a;}}
  advance(x,d){const L=this.W.L,t=this.tmp;for(let i=0;i<N;i++){let a=x[i]+(d?d[i]:0);const r=i*N;for(let j=0;j<N;j++)a+=L[r+j]*x[j];t[i]=a;}x.set(t);}
  step(d,y){
    this.advance(this.s,d);this.advance(this.xs,null);this.read(this.s,y,true);
    this.buf.set(y,this.head*P);this.head=(this.head+1)%HIST;
    const f=new Float32Array(NF);this.fastOf(f);this.fasts.set(f,this.fhead*NF);this.fhead=(this.fhead+1)%LAGS;}
  scan(measured){
    this.buf.set(measured,((this.head+HIST-1)%HIST)*P);
    const f=new Float32Array(NF);this.fastOf(f);this.fasts.set(f,((this.fhead+LAGS-1)%LAGS)*NF);this.encode();}
}
window.SpectatorEngine=SpectatorEngine;window.loadSpectator=loadSpectator;

window.initSpectatorSim=async function(){
const TR=1.49,FUT=40;
const $=id=>document.getElementById(id);
const reduceMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
const COL={ink:'#182226',muted:'#647074',rule:'#e5e4de',obs:'#2f65a8',state:'#14747a',stim:'#cf7722',neg:'#b94438'};
const CLIPCOL=['#c98a3e','#3f7fb5'];
function b64(s,T){const bin=atob(s),buf=new ArrayBuffer(bin.length),u=new Uint8Array(buf);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return new T(buf);}
const M=window.MESH;
const POS=b64(M.pos,Int16Array),FACES=b64(M.faces,Uint16Array),LABEL=b64(M.label,Uint16Array),SULC=b64(M.sulc,Uint8Array);
const NET=M.names.map(n=>n.split('_')[1]);
const REG={'Visual':'Vis','Somatomotor':'SomMot','Dorsal attention':'DorsAttn','Salience':'SalVentAttn','Control':'Cont','Default mode':'Default','Limbic':'Limbic'};
const REG_KEYS=Object.keys(REG);
const KEY_TO_REG={};REG_KEYS.forEach((k,i)=>{KEY_TO_REG[REG[k]]={name:k,idx:i};});

const loadBox=$('simLoading');
function setProgress(frac,done,total){
  const pct=Math.round(100*frac);
  const mb=(total/1e6).toFixed(1);
  const txt=$('simLoadText'),bar=$('simLoadFill');
  if(txt)txt.textContent=`Loading Spectator weights & first clip (${mb} MB)… ${pct}%`;
  else if(loadBox)loadBox.textContent=`Loading Spectator: ${pct}% of ${mb} MB`;
  if(bar)bar.style.width=pct+'%';
}
setProgress(0,0,24500000);

let S;
try{
  S=await loadSpectator('data/',setProgress,{clipIndices:[0]});
}catch(err){
  const isFile=typeof location!=='undefined'&&location.protocol==='file:';
  if(loadBox){
    loadBox.innerHTML='';
    const wrap=document.createElement('div');
    wrap.style.cssText='text-align:center;max-width:520px;padding:10px';
    const msg=document.createElement('p');
    msg.style.margin='0 0 12px';
    msg.textContent=isFile
      ?'Browsers block fetch() for pages opened directly from disk. Serve this folder over HTTP (e.g. python3 -m http.server) and reload.'
      :'Could not load Spectator weights ('+(err&&err.message?err.message:'network error')+'). Check your connection and try again.';
    wrap.appendChild(msg);
    if(!isFile){
      const btn=document.createElement('button');
      btn.type='button';btn.className='copy';btn.style.position='static';btn.textContent='Retry loading';
      btn.onclick=()=>{loadBox.innerHTML='<div class="sim-prog"><span id="simLoadText">Retrying…</span><div class="sim-pbar"><div id="simLoadFill"></div></div></div>';window.initSpectatorSim().catch(()=>{});};
      wrap.appendChild(btn);
    }
    loadBox.appendChild(wrap);
  }
  throw err;
}

if(loadBox)loadBox.remove();
$('simUI').hidden=false;
const {W,clips,loadClip}=S;
const E=new SpectatorEngine(W);
const meas=new Float32Array(P),pred=new Float32Array(P),stP=new Float32Array(P),stimP=new Float32Array(P),zero=new Float32Array(N);
const prev={meas:new Float32Array(P),pred:new Float32Array(P),st:new Float32Array(P),stim:new Float32Array(P)};
const cur={meas,pred,st:stP,stim:stimP};
let clip=0,pos=0,step=0,lastScan=0,swapAt=-99,movie=true,hist=[],fcast=null,lastStepMs=0,autoScan=true;
const drive=(c,j)=>clips[c].drive.subarray(j*N,(j+1)*N);
const measured=(c,j)=>clips[c].me.subarray((HIST+j)*P,(HIST+j+1)*P);
function savePrev(){for(const k in prev)prev[k].set(cur[k]);}
function maps(){
  meas.set(measured(clip,Math.max(0,pos-1)));
  E.read(E.s,pred,true);
  E.read(E.xs,stP,false);
  for(let p=0;p<P;p++)stimP[p]=pred[p]-stP[p];
}
function regionMean(a,r){const key=REG[r||region];let s=0,n=0;for(let p=0;p<P;p++)if(NET[p]===key){s+=a[p];n++;}return s/n;}
function reset(){clip=0;pos=0;step=0;lastScan=0;swapAt=-99;hist=[];fcast=null;movie=true;setTog('simMovie',false);
  E.loadHistory(clips[0].me,0);meas.set(clips[0].me.subarray((HIST-1)*P,HIST*P));E.read(E.s,pred,true);E.read(E.xs,stP,false);
  for(let p=0;p<P;p++)stimP[p]=pred[p]-stP[p];
  savePrev();pressClip(0);updateReadouts();requestForecast();}
function tick(){
  savePrev();
  if(pos>=clips[clip].frames-FUT-5){
    const nxt=(clip+1)%clips.length;
    if(!clips[nxt].loaded)loadClip(nxt).then(()=>{if(clip!==nxt)requestForecast();}).catch(()=>{});
  }
  if(pos>=clips[clip].frames){
    const nxt=(clip+1)%clips.length;
    if(clips[nxt].loaded)cutTo(nxt,true);
    else{pos=0;loadClip(nxt).catch(()=>{});}
  }
  const t0=performance.now();E.step(movie?drive(clip,pos):zero,pred);lastStepMs=performance.now()-t0;
  pos++;step++;
  maps();
  let se=0,ref=0;const mu=clips[clip].mean;for(let p=0;p<P;p++){const e=pred[p]-meas[p],r=meas[p]-mu[p];se+=e*e;ref+=r*r;}
  hist.push({t:step,meas:regionMean(meas),pred:regionMean(pred),clip,scan:false,swap:false,movie,se,ref});if(hist.length>FUT+1)hist.shift();
  if(autoScan&&step-lastScan>=8){scan(false);maps();hist[hist.length-1].scan=true;}
  updateReadouts();requestForecast();}
function scan(manual){E.scan(measured(clip,pos-1));lastScan=step;
  if(manual){maps();savePrev();if(hist.length){const h=hist[hist.length-1];h.pred=regionMean(pred);h.scan=true;}updateReadouts();requestForecast();}
  const fl=$('simFlash');fl.classList.add('on');setTimeout(()=>fl.classList.remove('on'),60);}
function swapMind(){
  E.loadHistory(clips[clip].other,pos);lastScan=step;swapAt=step;maps();savePrev();
  if(hist.length){hist[hist.length-1].swap=true;hist[hist.length-1].pred=regionMean(pred);}
  updateReadouts();requestForecast();}

/* ---------- forecast worker ---------- */
const workerSrc=`let L,N;onmessage=e=>{const m=e.data;if(m.init){L=m.L;N=m.N;return;}
const t0=performance.now();const x=Float32Array.from(m.x),t=new Float32Array(N),out=new Float32Array(m.drives.length/N);
for(let k=0;k<out.length;k++){const o=k*N;for(let i=0;i<N;i++){let a=x[i]+m.drives[o+i];const r=i*N;for(let j=0;j<N;j++)a+=L[r+j]*x[j];t[i]=a;}x.set(t);let s=m.rb;for(let j=0;j<N;j++)s+=m.rv[j]*x[j];out[k]=s;}
postMessage({id:m.id,out,ms:performance.now()-t0});};`;
let worker=null,busy=false,reqId=0,rv=null,rb=0;
function regionReadout(){const key=REG[region],v=new Float32Array(N);let n=0,b=0;for(let p=0;p<P;p++)if(NET[p]===key){n++;b+=W.b[p];for(let j=0;j<N;j++)v[j]+=W.P[p*N+j];}for(let j=0;j<N;j++)v[j]/=n;rv=v;rb=b/n;}
function initWorker(){try{worker=new Worker(URL.createObjectURL(new Blob([workerSrc],{type:'text/javascript'})));
  worker.onmessage=e=>{busy=false;if(e.data.id!==reqId)return;fcast={from:step,vals:Array.from(e.data.out)};$('simFc').textContent=e.data.ms.toFixed(0)+' ms';};
  worker.postMessage({init:true,L:W.L,N});}catch(err){worker=null;$('simFc').textContent='n/a';}}
function requestForecast(){if(!worker||busy)return;const d=new Float32Array(FUT*N);let c=clip,j=pos;
  for(let k=0;k<FUT;k++){
    if(j>=clips[c].frames){const nxt=(c+1)%clips.length;c=clips[nxt].loaded?nxt:c;j=0;}
    if(movie&&clips[c].loaded)d.set(drive(c,j),k*N);
    j++;
  }
  busy=true;reqId++;worker.postMessage({id:reqId,x:E.s,drives:d,rv,rb});}

/* ---------- UI ---------- */
function setTog(id,v){$(id).setAttribute('aria-pressed',v?'true':'false');}
function seg(el,items,init,cb){el.innerHTML='';items.forEach((t,i)=>{const b=document.createElement('button');b.type='button';b.textContent=t;b.setAttribute('aria-pressed',i===init?'true':'false');
  b.onclick=()=>{[...el.children].forEach(c=>c.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');cb(i,t);};el.appendChild(b);});}
function pressSeg(el,i){[...el.children].forEach((c,j)=>c.setAttribute('aria-pressed',j===i?'true':'false'));}
const clipsEl=$('simClips');
const shortClip=s=>s.replace(/,? part.*| \(part.*\)/,'').replace(/^The /,'');
clips.forEach((c,i)=>{const b=document.createElement('button');b.type='button';b.className='sim-clip';b.title=`${c.label} · ${c.genre} · from ${mmss(c.start_seconds)}`;
  b.innerHTML=`<i style="background:${CLIPCOL[i%CLIPCOL.length]}"></i><span>${shortClip(c.label)}</span>`;b.onclick=()=>cutTo(i);clipsEl.appendChild(b);});
function mmss(s){return Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0');}
function pressClip(i){[...clipsEl.children].forEach((b,j)=>b.setAttribute('aria-pressed',i===j?'true':'false'));$('simNow').textContent=clips[i].label;}
let pendingCut=null;
function cutTo(i,auto){
  if(i===clip&&!auto)return;
  if(!clips[i].loaded){
    pendingCut=i;
    const btn=clipsEl.children[i];
    if(btn)btn.dataset.loading='true';
    loadClip(i).then(()=>{
      if(btn)delete btn.dataset.loading;
      if(pendingCut===i){pendingCut=null;clip=i;pos=0;pressClip(i);requestForecast();}
    }).catch(()=>{if(btn)delete btn.dataset.loading;});
    return;
  }
  pendingCut=null;clip=i;pos=0;pressClip(i);if(!auto)requestForecast();
}
function setMode(isAuto){
  autoScan=!!isAuto;
  pressSeg($('simMode'),autoScan?0:1);
  const lbl=$('simScoreLbl');
  if(lbl)lbl.textContent=autoScan?'Updating score (1 min)':'Forecast score (1 min)';
  const note=$('simModeNote');
  if(note)note.textContent=autoScan
    ?'Auto-feeds 1 real fMRI scan every ~12 s (closed-loop)'
    :'No incoming scans — open-loop rollout from movie + initial state';
  updateReadouts();
}
$('simMovie').onclick=()=>{movie=!movie;setTog('simMovie',!movie);updateReadouts();requestForecast();};
$('simScan').onclick=()=>scan(true);
$('simSwap').onclick=swapMind;
seg($('simMode'),['Updating (4/min)','Forecast (open)'],0,i=>setMode(i===0));
let speed=1,paused=false,visible=true;
const speedSeg=$('simSpeed');
seg(speedSeg,['1× real time','4×','10×'],0,i=>{speed=[1,4,10][i];});
function togglePause(){paused=!paused;setTog('simPause',paused);$('simPause').textContent=paused?'Resume':'Pause';}
$('simPause').onclick=togglePause;
$('simReset').onclick=()=>{reset();setMode(true);};
let region='Visual';
const regSeg=$('simRegion');
function selectRegion(t,idx){
  region=t;
  if(idx!==undefined)pressSeg(regSeg,idx);
  regionReadout();hist.forEach(h=>{h.meas=NaN;h.pred=NaN;});fcast=null;requestForecast();
}
seg(regSeg,REG_KEYS,0,(i,t)=>selectRegion(t));
let rowB='pred';
const rowSeg=$('simRow');
function setRowMode(i){
  rowB=['pred','st','stim'][i];
  pressSeg(rowSeg,i);
  $('simRowName').textContent=['Prediction','State term','Stimulus term'][i];
  $('simRowSub').textContent=[
    'Spectator rollout',
    'inferred brain state carried forward (no movie)',
    'movie-driven component since the last scan'
  ][i];
}
seg(rowSeg,['Prediction','State','Stimulus'],0,setRowMode);
let view=0;
seg($('simView'),['Lateral','Medial','Dorsal'],0,i=>{view=i;yaw=0;pitch=0;placeMeshes();});

function updateReadouts(){
  let ma=0,mb=0;for(let p=0;p<P;p++){ma+=meas[p];mb+=pred[p];}ma/=P;mb/=P;let sab=0,saa=0,sbb=0;
  for(let p=0;p<P;p++){const a=meas[p]-ma,b=pred[p]-mb;sab+=a*b;saa+=a*a;sbb+=b*b;}
  $('simR').textContent=(sab/Math.sqrt(saa*sbb+1e-12)).toFixed(2);
  const recent=hist.slice(-FUT);let se=0,ref=0;recent.forEach(h=>{se+=h.se;ref+=h.ref;});
  $('simSkill').textContent=recent.length>=10?(100*(1-se/ref)).toFixed(1)+'%':'–';
  const swapped=swapAt===lastScan&&step-swapAt<40;
  $('simSince').textContent=((step-lastScan)*TR).toFixed(1)+' s'+(swapped?' (swap)':'');
  let vs=0,va=0;for(let p=0;p<P;p++){vs+=stP[p]*stP[p];const a=stimP[p]-W.b[p];va+=a*a;}const sh=Math.round(100*vs/(vs+va+1e-12));
  $('simShare').textContent=sh+'% / '+(100-sh)+'%';$('simShS').style.flexBasis=sh+'%';$('simShA').style.flexBasis=(100-sh)+'%';
  $('simStep').textContent=lastStepMs.toFixed(1)+' ms';
  const badge=$('simStatus');
  if(badge){
    if(swapped){badge.textContent='Swapped state · sub-02 history';badge.dataset.kind='warn';badge.hidden=false;}
    else if(!movie){badge.textContent='Movie input off · state-only rollout';badge.dataset.kind='amber';badge.hidden=false;}
    else if(!autoScan){badge.textContent='Open-loop Forecast · no incoming scans';badge.dataset.kind='teal';badge.hidden=false;}
    else{badge.hidden=true;}
  }
}

/* ---------- keyboard shortcuts (scoped when simulator is hovered/focused) ---------- */
const simRoot=$('sim');
let simHover=false;
if(simRoot){
  simRoot.addEventListener('pointerenter',()=>{simHover=true;});
  simRoot.addEventListener('pointerleave',()=>{simHover=false;});
}
window.addEventListener('keydown',e=>{
  if(!visible)return;
  const tag=(e.target&&e.target.tagName)||'';
  if(tag==='INPUT'||tag==='TEXTAREA'||tag==='SELECT')return;
  const active=simHover||(simRoot&&simRoot.contains(document.activeElement));
  if(!active)return;
  if(e.code==='Space'){e.preventDefault();togglePause();}
  else if(e.key==='s'||e.key==='S'){e.preventDefault();scan(true);}
});

/* ---------- cortex + raycasting ---------- */
const canvas=$('simBrain');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(2,window.devicePixelRatio||1));
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(28,1,10,4000);camera.position.set(0,0,1250);
scene.add(new THREE.AmbientLight(0xffffff,.72));const dl=new THREE.DirectionalLight(0xffffff,.62);dl.position.set(.2,.6,1);scene.add(dl);
const NV=M.nv,NVH=NV/2,sc=M.scale;
function hemiGeom(h){const g=new THREE.BufferGeometry(),pos=new Float32Array(NVH*3);let cx=0,cy=0,cz=0;
  for(let v=0;v<NVH;v++){const i=(v+h*NVH)*3,x=POS[i]*sc,y=POS[i+1]*sc,z=POS[i+2]*sc;pos[v*3]=x;pos[v*3+1]=z;pos[v*3+2]=-y;cx+=x;cy+=z;cz+=-y;}
  cx/=NVH;cy/=NVH;cz/=NVH;for(let v=0;v<NVH;v++){pos[v*3]-=cx;pos[v*3+1]-=cy;pos[v*3+2]-=cz;}
  const nf=FACES.length/6,idx=new Uint16Array(nf*3);for(let i=0;i<nf*3;i++)idx[i]=FACES[h*nf*3+i]-h*NVH;
  g.setAttribute('position',new THREE.BufferAttribute(pos,3));g.setIndex(new THREE.BufferAttribute(idx,1));g.computeVertexNormals();return {g,cx};}
const rows=[],allMeshes=[];
for(let r=0;r<2;r++){const row=[];for(let h=0;h<2;h++){const {g,cx}=hemiGeom(h),col=new Float32Array(NVH*3);g.setAttribute('color',new THREE.BufferAttribute(col,3));
  const m=new THREE.Mesh(g,new THREE.MeshLambertMaterial({vertexColors:true}));m.userData={h,r,cx,col};scene.add(m);row.push(m);allMeshes.push(m);}rows.push(row);}
let yaw=0,pitch=0;
function placeMeshes(){rows.forEach((row,r)=>{const y=r===0?122:-122;row.forEach(m=>{const h=m.userData.h;m.rotation.order='YXZ';
  if(view===0){m.rotation.set(pitch,(h===0?Math.PI/2:-Math.PI/2)+yaw,0);m.position.set(h===0?-120:120,y,0);}
  else if(view===1){m.rotation.set(pitch,(h===0?-Math.PI/2:Math.PI/2)+yaw,0);m.position.set(h===0?-120:120,y,0);}
  else{m.rotation.set(Math.PI/2+pitch,yaw,0,'XYZ');m.position.set(m.userData.cx*.95,y,0);}});});}
placeMeshes();

const raycaster=new THREE.Raycaster(),ptrNdc=new THREE.Vector2();
const tipEl=$('simTip');
let drag=null,hoverParcel=-1,tipPos={x:0,y:0};
function pickParcel(e){
  const rect=canvas.getBoundingClientRect();
  const lx=e.clientX-rect.left,ly=e.clientY-rect.top;
  ptrNdc.x=(lx/rect.width)*2-1;
  ptrNdc.y=-(ly/rect.height)*2+1;
  raycaster.setFromCamera(ptrNdc,camera);
  const hits=raycaster.intersectObjects(allMeshes,false);
  if(!hits.length)return null;
  const hit=hits[0],h=hit.object.userData.h;
  const gv=hit.face.a+h*NVH;
  const lab=LABEL[gv];
  if(!lab)return null;
  return {parcel:lab-1,lx,ly,rect};
}
function fmtZ(v){return (v>=0?'+':'')+v.toFixed(2);}
function renderTip(){
  if(!tipEl)return;
  if(hoverParcel<0){tipEl.hidden=true;return;}
  const rawName=M.names[hoverParcel]||'';
  const parts=rawName.split('_');
  const hemi=parts[0]||'',netKey=parts[1]||'',detail=parts.slice(2).join(' ');
  const regInfo=KEY_TO_REG[netKey];
  const regName=regInfo?regInfo.name:netKey;
  tipEl.innerHTML=`<b>${regName} · ${hemi} ${detail}</b><div class="vals">meas ${fmtZ(meas[hoverParcel])} · pred ${fmtZ(pred[hoverParcel])} z</div><div class="hint">Click to plot ${regName}</div>`;
  tipEl.hidden=false;
  const cw=canvas.clientWidth,ch2=canvas.clientHeight;
  const tw=tipEl.offsetWidth||200,th=tipEl.offsetHeight||56;
  const left=Math.min(Math.max(10,tipPos.x+14),Math.max(10,cw-tw-10));
  const top=tipPos.y-th-12>46?tipPos.y-th-12:Math.min(ch2-th-10,tipPos.y+16);
  tipEl.style.left=left+'px';tipEl.style.top=top+'px';
}
canvas.addEventListener('pointerdown',e=>{
  drag={x:e.clientX,y:e.clientY,yaw,pitch,moved:false};
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove',e=>{
  if(drag){
    const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
    if(Math.abs(dx)>4||Math.abs(dy)>4)drag.moved=true;
    yaw=drag.yaw+dx*.01;
    pitch=Math.max(-1.2,Math.min(1.2,drag.pitch+dy*.01));
    placeMeshes();
    hoverParcel=-1;if(tipEl)tipEl.hidden=true;
    return;
  }
  const hit=pickParcel(e);
  if(hit){
    hoverParcel=hit.parcel;tipPos={x:hit.lx,y:hit.ly};
    canvas.style.cursor='pointer';
    renderTip();
  }else{
    hoverParcel=-1;canvas.style.cursor='grab';
    if(tipEl)tipEl.hidden=true;
  }
});
canvas.addEventListener('pointerup',e=>{
  const wasClick=drag&&!drag.moved;
  drag=null;
  if(wasClick){
    const hit=pickParcel(e);
    if(hit){
      const netKey=NET[hit.parcel],info=KEY_TO_REG[netKey];
      if(info)selectRegion(info.name,info.idx);
    }
  }
});
canvas.addEventListener('pointerleave',()=>{hoverParcel=-1;if(tipEl)tipEl.hidden=true;canvas.style.cursor='grab';});
canvas.addEventListener('dblclick',()=>{yaw=0;pitch=0;placeMeshes();});

const base=new Float32Array(NV);for(let v=0;v<NV;v++)base[v]=.86-.24*(SULC[v]/255);
const NEG2=[.118,.31,.54],NEG1=[.353,.627,.816],POS1=[.949,.643,.278],POS2=[.761,.255,.047];
let cscale=1;
function colorize(row,vals){for(const m of row){const h=m.userData.h,col=m.userData.col;
  for(let v=0;v<NVH;v++){const gv=v+h*NVH,lab=LABEL[gv],b=base[gv];let r=b,g=b,bl=b;
    if(lab>0){const t=Math.max(-1,Math.min(1,vals[lab-1]/cscale)),a=Math.abs(t);
      if(a>.12){const c1=t>0?POS1:NEG1,c2=t>0?POS2:NEG2,u=Math.min(1,(a-.12)/.88),k=Math.min(1,u*1.6),k2=Math.min(1,Math.max(0,u*1.6-.6));
        const cr=c1[0]+(c2[0]-c1[0])*k2,cg=c1[1]+(c2[1]-c1[1])*k2,cb=c1[2]+(c2[2]-c1[2])*k2;r=b+(cr-b)*k;g=b+(cg-b)*k;bl=b+(cb-b)*k;}}
    else{r=g=bl=b*.8;}
    col[v*3]=r;col[v*3+1]=g;col[v*3+2]=bl;}
  m.geometry.attributes.color.needsUpdate=true;}}
{let s=0;const a=clips[0].me;for(let i=0;i<a.length;i++)s+=Math.abs(a[i]);cscale=2*s/a.length;$('simCmapLbl').textContent='± '+cscale.toFixed(2)+' z';}
const lerpA=new Float32Array(P),lerpB=new Float32Array(P);
function resize(){const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;
  const tf=2*Math.tan(camera.fov*Math.PI/360);camera.position.z=Math.max(455/(tf*camera.aspect),470/tf);camera.updateProjectionMatrix();
  $('simLblB').style.top=(h/2+10)+'px';sizeChart();}
window.addEventListener('resize',resize);

/* ---------- screen ---------- */
const scr=$('simScreen'),sx=scr.getContext('2d');
let frac=0;
function drawScreen(){const W2=scr.width,H=scr.height,c=clips[clip],ch=c.feature_change;sx.fillStyle='#13181a';sx.fillRect(0,0,W2,H);
  sx.fillStyle='#f3f2ed';sx.font='600 38px "Newsreader", Georgia, serif';sx.textAlign='left';sx.fillText(c.label.replace(/,? part.*| \(part.*\)/,''),24,56);
  sx.fillStyle='#9ea7aa';sx.font='500 24px "Inter", -apple-system, sans-serif';sx.fillText(`${c.genre} · ${mmss(c.start_seconds+(pos-1+frac)*TR)}`,24,94);
  const x0=24,x1=W2-24,y1=H-60,y0=130,bw=(x1-x0)/ch.length;
  for(let i=0;i<ch.length;i++){const hgt=(y1-y0)*ch[i],on=i<pos;sx.fillStyle=movie?(on?CLIPCOL[clip%CLIPCOL.length]:'#364042'):'#272e30';sx.fillRect(x0+i*bw,y1-hgt,Math.max(1,bw-1),hgt);}
  const px=x0+(pos-1+frac)*bw;sx.fillStyle='#fff';sx.fillRect(px,y0-6,2,y1-y0+12);
  sx.fillStyle='#949ea1';sx.font='500 20px "JetBrains Mono", ui-monospace, monospace';sx.fillText('movie-feature change per frame',24,H-22);
  if(!movie){sx.fillStyle='rgba(19,24,26,.78)';sx.fillRect(0,0,W2,H);sx.fillStyle='#fff';sx.textAlign='center';sx.font='600 30px "Inter", sans-serif';sx.fillText('Movie input off',W2/2,H/2-8);sx.fillStyle='#c9d0d2';sx.font='400 21px "Inter", sans-serif';sx.fillText('the stimulus term is zero',W2/2,H/2+26);}
  $('simTime').textContent=((pos-1+frac)*TR).toFixed(0)+' s';}

/* ---------- chart (with scan, swap, and movie-off markers) ---------- */
const ch=$('simChart'),cx2=ch.getContext('2d');
function sizeChart(){const d=Math.min(2,devicePixelRatio||1);ch.width=ch.clientWidth*d;ch.height=ch.clientHeight*d;cx2.setTransform(d,0,0,d,0,0);}
function drawChart(){const W2=ch.clientWidth,H=ch.clientHeight,pl=34,pr=8,pt=18,pb=24;cx2.clearRect(0,0,W2,H);
  const span=FUT+FUT,X=k=>pl+(W2-pl-pr)*(k+FUT)/span;
  const vals=[...hist.map(h=>h.meas),...hist.map(h=>h.pred),...(fcast?fcast.vals:[])].filter(Number.isFinite);const lim=Math.max(.2,...vals.map(Math.abs))*1.1,Y=v=>pt+(H-pt-pb)*(.5-v/(2*lim));
  cx2.font='11px "JetBrains Mono", ui-monospace, monospace';cx2.lineWidth=1;
  [-lim/1.1,0,lim/1.1].forEach(v=>{cx2.strokeStyle=COL.rule;cx2.beginPath();cx2.moveTo(pl,Y(v));cx2.lineTo(W2-pr,Y(v));cx2.stroke();cx2.fillStyle=COL.muted;cx2.textAlign='right';cx2.fillText(v.toFixed(2),pl-5,Y(v)+4);});
  cx2.textAlign='center';[-40,-20,0,20,40].forEach(k=>cx2.fillText((k>0?'+':'')+Math.round(k*TR)+' s',X(k),H-6));
  const now=X(frac);
  cx2.fillStyle=movie?'rgba(20,116,122,.055)':'rgba(207,119,34,.07)';cx2.fillRect(now,pt,W2-pr-now,H-pt-pb);
  for(const h of hist){
    const k=h.t-step,xL=X(k-1+frac),xR=X(k+frac);
    if(!h.movie){cx2.fillStyle='rgba(207,119,34,.10)';cx2.fillRect(xL,pt,xR-xL+.5,H-pt-pb);}
    cx2.fillStyle=CLIPCOL[h.clip%CLIPCOL.length];cx2.fillRect(xL,H-pb-3,xR-xL+.5,3);
  }
  hist.forEach(h=>{
    const x=X(h.t-step+frac);
    if(h.scan){cx2.strokeStyle=COL.obs;cx2.globalAlpha=.35;cx2.beginPath();cx2.moveTo(x,pt);cx2.lineTo(x,H-pb);cx2.stroke();cx2.globalAlpha=1;}
    if(h.swap){
      cx2.strokeStyle=COL.neg;cx2.lineWidth=1.5;cx2.setLineDash([4,3]);cx2.beginPath();cx2.moveTo(x,pt);cx2.lineTo(x,H-pb);cx2.stroke();cx2.setLineDash([]);cx2.lineWidth=1;
      cx2.fillStyle=COL.neg;cx2.font='600 9.5px "JetBrains Mono", ui-monospace, monospace';cx2.textAlign='center';cx2.fillText('swap',x,pt+10);
      cx2.font='11px "JetBrains Mono", ui-monospace, monospace';
    }
  });
  const line=(pts,color,w,dash)=>{cx2.strokeStyle=color;cx2.lineWidth=w;cx2.setLineDash(dash||[]);cx2.beginPath();let on=false;pts.forEach(([x,y])=>{if(!Number.isFinite(y)){on=false;return;}on?cx2.lineTo(x,y):cx2.moveTo(x,y);on=true;});cx2.stroke();cx2.setLineDash([]);};
  line(hist.map(h=>[X(h.t-step+frac),Y(h.meas)]),COL.obs,2);
  line(hist.map(h=>[X(h.t-step+frac),Y(h.pred)]),COL.ink,1.5);
  if(fcast&&hist.length){const off=fcast.from-step;const last=hist[hist.length-1].pred;line([[X(off+frac),Y(Number.isFinite(last)?last:fcast.vals[0])]].concat(fcast.vals.map((v,i)=>[X(off+i+1+frac),Y(v)])),COL.state,1.8,[5,4]);}
  cx2.strokeStyle=COL.ink;cx2.globalAlpha=.4;cx2.beginPath();cx2.moveTo(now,pt);cx2.lineTo(now,H-pb);cx2.stroke();cx2.globalAlpha=1;
  cx2.fillStyle=COL.muted;cx2.textAlign='left';cx2.fillText(region+' network, mean z',pl+4,11);
  cx2.textAlign='right';cx2.fillText((movie?'past minute  |  next minute':'past minute  |  movie off'),W2-pr,11);}

/* ---------- loop ---------- */
let acc=0,last=performance.now(),fpsN=0,fpsT=0;
function frame(now){const dt=Math.min(.25,(now-last)/1000);last=now;
  if(visible){
    if(!paused){acc+=dt*speed;let n=0;while(acc>=TR&&n<4){acc-=TR;tick();n++;}if(acc>=TR)acc=0;}
    frac=acc/TR;const f=reduceMotion?1:frac,pB=prev[rowB],cB=cur[rowB];
    for(let p=0;p<P;p++){lerpA[p]=prev.meas[p]+(meas[p]-prev.meas[p])*f;lerpB[p]=pB[p]+(cB[p]-pB[p])*f;}
    if(rowB==='stim')for(let p=0;p<P;p++)lerpB[p]-=W.b[p];
    colorize(rows[0],lerpA);colorize(rows[1],lerpB);renderer.render(scene,camera);drawScreen();drawChart();
    if(hoverParcel>=0)renderTip();
    $('simClock').textContent='t = '+((step+frac)*TR).toFixed(1)+' s';
    fpsN++;fpsT+=dt;if(fpsT>.5){$('simFps').textContent=Math.round(fpsN/fpsT)+' fps';fpsN=0;fpsT=0;}}
  requestAnimationFrame(frame);}
const io=new IntersectionObserver(es=>{visible=es[0].isIntersecting;last=performance.now();},{threshold:.05});
io.observe($('sim'));
regionReadout();initWorker();reset();resize();
requestAnimationFrame(t=>{last=t;frame(t);});
};
})();
