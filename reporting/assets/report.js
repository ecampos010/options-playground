(function(){
"use strict";
var D=JSON.parse(document.getElementById('data').textContent);
var $=function(s,r){return (r||document).querySelector(s)};
var $$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s))};
var esc=function(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})};
function usd(v){if(v==null||isNaN(v))return '-';var a=Math.abs(v),s=v<0?'-':'';if(a>=1e9)return s+'$'+(a/1e9).toFixed(2)+'B';if(a>=1e6)return s+'$'+(a/1e6).toFixed(2)+'M';return s+'$'+Math.round(a).toLocaleString('en-US')}
function pct(v,d){if(v==null||isNaN(v))return '-';if(v===Infinity)return 'inf';return (v*100).toFixed(d==null?1:d)+'%'}
var ALLOCS=[5,15,20,50,100],COLORS={5:'var(--s1)',15:'var(--s2)',20:'var(--s3)',50:'var(--s4)',100:'var(--s5)'},DASH={5:'',15:'7 3',20:'2 3',50:'9 3 2 3',100:'1 0'};
function colorFor(a){return COLORS[a]||'var(--s1)'}
function dashFor(a){return DASH[a]||''}
// ---------- series decode
var cache={};
function f32(b64){var b=atob(b64),u=new Uint8Array(b.length);for(var i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer)}
function series(i){if(cache[i])return cache[i];var r=D.rows[i];if(r.s==null)return null;var s=D.series[r.s];return cache[i]={e:f32(s[0]),d:f32(s[1])}}
var axisCache={};
function axis(g){if(axisCache[g])return axisCache[g];var a=D.axes[g];if(!a)return null;return axisCache[g]={iso:a,t:a.map(function(x){return Date.parse(x+'T00:00:00Z')})}}
// ---------- chart
function niceTicks(lo,hi,n){var span=hi-lo,step=Math.pow(10,Math.floor(Math.log10(span/n))),err=span/n/step;step*=err>=7?10:err>=3?5:err>=1.5?2:1;var out=[];for(var v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)out.push(v);return out}
function lineChart(el,o){
  var W=Math.max(el.clientWidth||320,260),H=o.h||(W<520?280:340),wide=W>=620;
  var m={l:o.yfmt==='pct'?46:(W<520?52:62),r:wide?96:14,t:10,b:24},pw=W-m.l-m.r,ph=H-m.t-m.b;
  var xs=o.xs,x0=xs[0],x1=xs[xs.length-1],log=!!o.log;
  var ymin=Infinity,ymax=-Infinity;
  o.lines.forEach(function(L){for(var i=0;i<L.n;i++){var v=L.ys[i];if(v>ymax)ymax=v;if(v<ymin)ymin=v}});
  (o.refs||[]).forEach(function(r){if(r.scale!==false){ymax=Math.max(ymax,r.y);}});
  var lo,hi;
  if(o.fixed){lo=o.fixed[0];hi=o.fixed[1]}
  else if(log){hi=ymax*1.15;lo=Math.max(Math.min(ymin,1e5)*0.8,1);lo=Math.max(lo,hi/1e6);}
  else{lo=Math.min(0,ymin);hi=ymax*1.06}
  if(!(hi>lo))hi=lo+1;
  var Y=function(v){if(log){v=Math.max(v,lo);return m.t+ph*(1-(Math.log(v)-Math.log(lo))/(Math.log(hi)-Math.log(lo)))}return m.t+ph*(1-(v-lo)/(hi-lo))};
  var X=function(t){return m.l+pw*(t-x0)/(x1-x0||1)};
  var fy=o.yfmt==='pct'?function(v){return pct(v,0)}:usd;
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'" role="img" aria-label="'+esc(o.title||'chart')+'">';
  var ticks=log?(function(){var t=[],p=Math.pow(10,Math.floor(Math.log10(lo)));for(var v=p;v<=hi*1.001;v*=10){[1,2,5].forEach(function(k){var u=v*k;if(u>=lo&&u<=hi)t.push(u)})}return t.length>7?t.filter(function(u,i){return String(u)[0]==='1'}):t})():niceTicks(lo,hi,5);
  s+='<g class="axis">';
  ticks.forEach(function(v){var y=Y(v);s+='<line class="gridline" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+y+'" y2="'+y+'"/><text x="'+(m.l-5)+'" y="'+(y+4)+'" text-anchor="end">'+fy(v)+'</text>'});
  var y0=new Date(x0).getUTCFullYear(),y1=new Date(x1).getUTCFullYear(),stepY=W<520?2:1;
  for(var yr=y0;yr<=y1+1;yr++){var t=Date.UTC(yr,0,1);if(t<x0||t>x1||(yr-y0)%stepY)continue;var xx=X(t);s+='<line class="gridline" x1="'+xx+'" x2="'+xx+'" y1="'+m.t+'" y2="'+(m.t+ph)+'" opacity=".5"/><text x="'+xx+'" y="'+(H-6)+'" text-anchor="middle">'+yr+'</text>'}
  s+='</g>';
  (o.refs||[]).forEach(function(r){if(log&&r.y<=0)return;var y=Y(r.y);s+='<line class="refline" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+y+'" y2="'+y+'" style="'+(r.color?'stroke:'+r.color+';':'')+(r.dash!=null?'stroke-dasharray:'+r.dash:'')+'"/><text class="reftxt" x="'+(m.l+4)+'" y="'+(y+(r.below?12:-4))+'">'+esc(r.label)+'</text>'});
  (o.refLines||[]).forEach(function(r){var p='';for(var i=0;i<r.xs.length;i++){if(r.xs[i]<x0||r.xs[i]>x1)continue;p+=(p?'L':'M')+X(r.xs[i]).toFixed(1)+','+Y(r.ys[i]).toFixed(1)}
    s+='<path d="'+p+'" fill="none" stroke="'+r.color+'" stroke-width="1.6" stroke-dasharray="'+r.dash+'"/>'});
  var labels=[];
  o.lines.forEach(function(L){
    var p='';for(var i=0;i<L.n;i++)p+=(i?'L':'M')+X(xs[i]).toFixed(1)+','+Y(L.ys[i]).toFixed(1);
    if(o.area)s+='<path d="'+p+'L'+X(xs[L.n-1]).toFixed(1)+','+Y(0)+'L'+X(xs[0]).toFixed(1)+','+Y(0)+'Z" fill="'+L.color+'" opacity=".18"/>';
    s+='<path d="'+p+'" fill="none" stroke="'+L.color+'" stroke-width="2" stroke-dasharray="'+(L.dash||'')+'" stroke-linejoin="round"/>';
    var ex=X(xs[L.n-1]),ey=Y(L.ys[L.n-1]);
    if(L.broke){s+='<g stroke="'+L.color+'" stroke-width="2.5"><line x1="'+(ex-5)+'" y1="'+(ey-5)+'" x2="'+(ex+5)+'" y2="'+(ey+5)+'"/><line x1="'+(ex-5)+'" y1="'+(ey+5)+'" x2="'+(ex+5)+'" y2="'+(ey-5)+'"/></g>'}
    else if(L.n>1){s+='<circle cx="'+ex+'" cy="'+ey+'" r="3" fill="'+L.color+'" stroke="var(--card)" stroke-width="1.5"/>'}
    labels.push({y:ey,x:ex,text:L.label,color:L.color,broke:L.broke,bd:L.bd});
  });
  if(wide){labels.sort(function(a,b){return a.y-b.y});for(var i=1;i<labels.length;i++)if(labels[i].y-labels[i-1].y<13)labels[i].y=labels[i-1].y+13;
    labels.forEach(function(l){s+='<text class="lbl" x="'+(Math.min(l.x,W-m.r)+8)+'" y="'+(l.y+4)+'" fill="currentColor" style="fill:var(--fg)">'+esc(l.text)+(l.broke?' &#215; '+esc(l.bd):'')+'</text>'})}
  else{labels.forEach(function(l){if(l.broke)s+='<text class="lbl" x="'+(l.x-4)+'" y="'+(l.y-8)+'" text-anchor="end" style="fill:var(--fg)">&#215; '+esc(l.bd)+'</text>'})}
  s+='<line class="xh" x1="0" x2="0" y1="'+m.t+'" y2="'+(m.t+ph)+'" stroke="var(--fg)" stroke-width="1" style="display:none"/>';
  s+='<rect class="hit" x="'+m.l+'" y="'+m.t+'" width="'+pw+'" height="'+ph+'" fill="transparent"/></svg>';
  el.innerHTML=s+'<div class="tip"></div>';
  var tip=$('.tip',el),xh=$('.xh',el),hit=$('.hit',el);
  function show(ev){
    var r=hit.getBoundingClientRect(),px=(ev.clientX-r.left)*(pw/r.width),t=x0+(px/pw)*(x1-x0);
    var lo2=0,hi2=xs.length-1;while(hi2-lo2>1){var mid=(lo2+hi2)>>1;if(xs[mid]<t)lo2=mid;else hi2=mid}
    var i=Math.abs(xs[lo2]-t)<Math.abs(xs[hi2]-t)?lo2:hi2;
    var h='<b>'+esc(o.iso[i])+'</b>';
    o.lines.forEach(function(L){var v=i<L.n?fy(L.ys[i]):(L.broke?'broke':'-');h+='<div><span style="display:inline-block;width:14px;border-top:3px '+(L.dash?'dashed':'solid')+' '+L.color+';vertical-align:middle"></span> '+esc(L.label)+': '+v+'</div>'});
    (o.refLines||[]).forEach(function(rl){var tt=xs[i],lo3=0,hi3=rl.xs.length-1;while(hi3-lo3>1){var m3=(lo3+hi3)>>1;if(rl.xs[m3]<=tt)lo3=m3;else hi3=m3}if(rl.xs[0]<=tt)h+='<div class="muted">'+esc(rl.label)+': '+usd(rl.ys[lo3])+'</div>'});
    tip.innerHTML=h;tip.style.display='block';
    var cx=m.l+pw*(xs[i]-x0)/(x1-x0||1);xh.setAttribute('x1',cx);xh.setAttribute('x2',cx);xh.style.display='';
    var tw=tip.offsetWidth,left=cx*(el.clientWidth/W)+12;if(left+tw>el.clientWidth)left=cx*(el.clientWidth/W)-tw-12;
    tip.style.left=Math.max(0,left)+'px';tip.style.top='8px';
  }
  hit.addEventListener('pointermove',show);hit.addEventListener('pointerdown',show);
  hit.addEventListener('pointerleave',function(){tip.style.display='none';xh.style.display='none'});
}
// ---------- explorer
var rows=D.rows,state={strat:null,variant:null,tickers:null,alloc:15,log:false};
function uniq(a){return a.filter(function(v,i){return a.indexOf(v)===i})}
function fillSelect(sel,vals,cur,fmt){sel.innerHTML=vals.map(function(v){return '<option value="'+esc(v)+'"'+(v===cur?' selected':'')+'>'+esc(fmt?fmt(v):v)+'</option>'}).join('')}
function initExplorer(){
  var strats=uniq(rows.map(function(r){return r.st}));
  // prefer a Green 5-ticker single-strategy default
  var maxnt=Math.max.apply(null,rows.map(function(x){return x.nt}));
  var pref0=rows.filter(function(r){return r.st==='iron_condor'&&r.lvl===1&&r.nt===maxnt&&r.A===15&&!r.br})[0];
  var pref=pref0||rows.filter(function(r){return r.lvl===1&&r.nt===Math.max.apply(null,rows.map(function(x){return x.nt}))&&r.st!==D.mixname}).sort(function(a,b){return (a.rk||1e9)-(b.rk||1e9)})[0]||rows[0];
  state.strat=pref.st;state.variant=pref.pf+'|'+pref.tg;state.tickers=pref.tk;
  $('#sel-strat').addEventListener('change',function(){state.strat=this.value;state.variant=null;state.tickers=null;syncSelects();draw()});
  $('#sel-var').addEventListener('change',function(){state.variant=this.value;state.tickers=null;syncSelects();draw()});
  $('#sel-tk').addEventListener('change',function(){state.tickers=this.value;syncSelects();draw()});
  $('#sel-al').addEventListener('change',function(){state.alloc=+this.value;draw()});
  $('#btn-log').addEventListener('click',function(){state.log=!state.log;this.setAttribute('aria-pressed',state.log);draw()});
  syncSelects();draw();
  var t;window.addEventListener('resize',function(){clearTimeout(t);t=setTimeout(draw,120)});
}
function syncSelects(){
  var strats=uniq(rows.map(function(r){return r.st}));
  fillSelect($('#sel-strat'),strats,state.strat);
  var vs=uniq(rows.filter(function(r){return r.st===state.strat}).map(function(r){return r.pf+'|'+r.tg}));
  if(vs.indexOf(state.variant)<0)state.variant=vs[0];
  fillSelect($('#sel-var'),vs,state.variant,function(v){var r=rows.find(function(x){return x.pf+'|'+x.tg===v});return r.lt+' - '+v.replace('|',' / ')});
  var tks=uniq(rows.filter(function(r){return r.st===state.strat&&r.pf+'|'+r.tg===state.variant}).map(function(r){return r.tk}));
  tks.sort(function(a,b){return b.split('+').length-a.split('+').length||a.localeCompare(b)});
  if(tks.indexOf(state.tickers)<0)state.tickers=tks[0];
  fillSelect($('#sel-tk'),tks,state.tickers);
  var al=uniq(rows.filter(function(r){return r.st===state.strat&&r.pf+'|'+r.tg===state.variant&&r.tk===state.tickers}).map(function(r){return r.A})).sort(function(a,b){return a-b});
  if(al.indexOf(state.alloc)<0)state.alloc=al.indexOf(15)>=0?15:al[0];
  fillSelect($('#sel-al'),al.map(String),String(state.alloc),function(v){return v+'%'});
}
function famRows(){return rows.filter(function(r){return r.st===state.strat&&r.pf+'|'+r.tg===state.variant&&r.tk===state.tickers}).sort(function(a,b){return a.A-b.A})}
function monthsTo(iso,bd){var n=0;for(var i=0;i<iso.length;i++){if(iso[i]<=bd)n=i+1}return Math.max(n,1)}
function draw(){
  var fam=famRows();if(!fam.length)return;
  if(!fam.some(function(r){return r.s!=null})){$('#eq-chart').innerHTML='<p class="muted">Chart data was not embedded for this scenario (file-size cap). See results.csv and equity/*.parquet.</p>';return}
  var g=fam[0].pf+'|'+fam[0].tg+'|'+fam[0].st,ax=axis(g);
  var syn=D.meta.synthetic?'SYNTHETIC - ':'';
  var lines=[],legend='';
  fam.forEach(function(r){
    var ser=series(rows.indexOf(r));if(!ser||!ax)return;
    var n=r.br?monthsTo(ax.iso,r.bd):ax.iso.length;
    var L={label:r.A+'%',color:colorFor(r.A),dash:dashFor(r.A),ys:ser.e,n:Math.min(n,ser.e.length),broke:!!r.br,bd:r.bd};
    lines.push(L);
    legend+='<span><svg width="30" height="10" aria-hidden="true"><line x1="0" y1="5" x2="30" y2="5" stroke="'+L.color+'" stroke-width="2.5" stroke-dasharray="'+(L.dash||'')+'"/></svg><b>'+r.A+'%</b> '+(r.br?'broke '+esc(r.bd)+' ('+usd(r.fe)+')':usd(r.fe)+' ('+r.zn+')')+'</span>';
  });
  var refs=[{y:D.meta.start,label:'Start $100,000'}];
  if(!state.log)refs.push({y:0,label:'$0 "broke" floor',below:false,color:'var(--s5)'});
  var refLines=[];
  var t0=ax.t[0],start=D.meta.start,apy=D.meta.apy;
  refLines.push({label:'Cash @ '+(apy*100).toFixed(1)+'% APY',color:'var(--cash)',dash:'1 4',xs:ax.t,ys:ax.t.map(function(t){return start*Math.pow(1+apy,(t-t0)/(365.25*864e5))})});
  var spyNote='';
  if(D.meta.spy){var sp=D.meta.spy,st=sp.t,sv=sp.v,j0=0;while(j0<st.length-1&&st[j0+1]<=t0)j0++;var base=sv[j0];
    refLines.push({label:'SPY buy & hold',color:'var(--spy)',dash:'9 2 2 2',xs:ax.t,ys:ax.t.map(function(t){var j=0,lo=0,hi=st.length-1;while(hi-lo>1){var mid=(lo+hi)>>1;if(st[mid]<=t)lo=mid;else hi=mid}j=st[hi]<=t?hi:lo;return start*sv[j]/base})})}
  else spyNote=' SPY buy-and-hold line omitted: no --spy-csv supplied.';
  var lg=$('#eq-legend');lg.innerHTML=legend+'<span><svg width="30" height="10" aria-hidden="true"><line x1="0" y1="5" x2="30" y2="5" stroke="var(--cash)" stroke-width="2" stroke-dasharray="1 4"/></svg>Cash @ '+(apy*100).toFixed(1)+'%</span>'+(D.meta.spy?'<span><svg width="30" height="10" aria-hidden="true"><line x1="0" y1="5" x2="30" y2="5" stroke="var(--spy)" stroke-width="2" stroke-dasharray="9 2 2 2"/></svg>SPY buy &amp; hold</span>':'');
  $('#eq-title').textContent=syn+state.strat+' | '+state.variant.replace('|',' / ')+' | '+state.tickers;
  $('#eq-note').textContent=(state.log?'Log scale; the $0 floor cannot be drawn on a log axis, broke lines end at the bottom with an x and date. ':'')+'Month-end equity; x marks the month the account went broke.'+spyNote;
  lineChart($('#eq-chart'),{xs:ax.t,iso:ax.iso,lines:lines,refs:refs,refLines:refLines,log:state.log,title:'Equity by allocation'});
  // selected scenario
  var sel=fam.find(function(r){return r.A===state.alloc})||fam[0];
  drawDD(sel,ax);drawThermo(sel);
  $$('#rank-body tr').forEach(function(tr){tr.classList.toggle('sel',tr.dataset.i==rows.indexOf(sel))});
}
function drawDD(r,ax){
  var ser=series(rows.indexOf(r));if(!ser)return;
  var n=r.br?monthsTo(ax.iso,r.bd):ax.iso.length;n=Math.min(n,ser.d.length);
  var ys=new Float32Array(n);for(var i=0;i<n;i++)ys[i]=-ser.d[i];
  var mx=0;for(i=0;i<n;i++)mx=Math.max(mx,ser.d[i]);
  var z=D.zones,refs=[];z.forEach(function(k){if(k.dd<=Math.max(mx*1.25,0.2))refs.push({y:-k.dd,label:k.name+' limit -'+Math.round(k.dd*100)+'%',scale:false})});
  $('#dd-title').textContent=(D.meta.synthetic?'SYNTHETIC - ':'')+'Drawdown: '+r.st+' | '+r.pf+' / '+r.tg+' | '+r.tk+' | '+r.A+'%';
  lineChart($('#dd-chart'),{xs:ax.t,iso:ax.iso,lines:[{label:'Drawdown',color:colorFor(r.A),dash:dashFor(r.A),ys:ys,n:n,broke:!!r.br,bd:r.bd}],refs:refs,fixed:[-Math.min(1,Math.max(mx*1.2,0.1)),0.005],yfmt:'pct',area:true,title:'Drawdown',h:230});
}
function zoneCls(z){return 'z-'+(z==='Breaking point'?'Breaking':z)}
function drawThermo(r){
  var names=D.zone_names,idx=Math.max(names.indexOf(r.zn),0),h='<div class="gauge" role="img" aria-label="Zone: '+esc(r.zn)+'">';
  names.forEach(function(n,i){h+='<div class="'+zoneCls(n)+(i===idx?' on':'')+'">'+(i===idx?'&#9654; ':'')+esc(n)+(i===idx?' &#9664;':'')+'</div>'});
  h+='</div><div class="pointer"><b>Zone: '+esc(r.zn)+'</b> &mdash; '+esc(r.lt)+'</div>';
  var haveMC=!!r.hm;
  var kv=[['Allocation',r.A+'%'],['Delta setting',(r.dl!=null?'short delta ~'+(r.dl*100).toFixed(0)+' ('+esc(r.tg)+')':esc(r.tg))],['Typical VIX regime',esc(D.vixnames[r.vx]||r.vx||'-')],
    ['Max drawdown',pct(r.dd)],['Worst month',pct(r.wm)],['Ruin odds ('+D.horizon+'m)',haveMC&&r.ruin!=null?pct(r.ruin,2):'n/a (Monte Carlo not run)']];
  if(haveMC)kv.push(['P&amp;L p10 / p50 / p90 ('+D.horizon+'m)',usd(r.p10-D.meta.start)+' / '+usd(r.p50-D.meta.start)+' / '+usd(r.p90-D.meta.start)]);
  h+='<div class="kv">'+kv.map(function(k){return '<div><span class="muted">'+k[0]+'</span><b>'+k[1]+'</b></div>'}).join('')+'</div>';
  if(!haveMC)h+='<p class="small flag">Monte Carlo was not run: this zone uses max drawdown and the broke flag only.</p>';
  h+='<p class="small muted">Zone limits (editable in config/thermometer.yaml): '+D.zones.map(function(k){return esc(k.name)+' = ruin &lt; '+pct(k.ruin,0)+' &amp; DD &lt; '+pct(k.dd,0)}).join('; ')+'; anything worse, or broke = Breaking point.</p>';
  $('#thermo').innerHTML=h;
}
// ---------- ranked table
var cols=[['rk','Rank','n'],['lt','Level'],['st','Strategy'],['tk','Tickers'],['A','Alloc','n'],['zn','Zone'],['fe','Final equity','n'],['cagr','CAGR','n'],['dd','Max DD','n'],['wm','Worst month','n'],['wt','Worst trade % eq','n'],['cal','Return / maxDD','n'],['n','Trades','n'],['v','Verdict']];
var tstate={key:'rk',dir:1,lvl:'',nt:'',st:'',zn:'',q:'',view:'ok',limit:150};
function initTable(){
  var lv=uniq(rows.map(function(r){return r.lt})).sort(),sts=uniq(rows.map(function(r){return r.st})).sort(),nts=uniq(rows.map(function(r){return r.nt})).sort();
  $('#f-lvl').innerHTML='<option value="">All levels</option>'+lv.map(function(v){return '<option>'+esc(v)+'</option>'}).join('');
  $('#f-st').innerHTML='<option value="">All strategies</option>'+sts.map(function(v){return '<option>'+esc(v)+'</option>'}).join('');
  $('#f-nt').innerHTML='<option value="">Any # tickers</option>'+nts.map(function(v){return '<option>'+v+'</option>'}).join('');
  $('#f-zn').innerHTML='<option value="">All zones</option>'+D.zone_names.map(function(v){return '<option>'+esc(v)+'</option>'}).join('');
  ['lvl','st','nt','zn'].forEach(function(k){$('#f-'+k).addEventListener('change',function(){tstate[k]=this.value;tstate.limit=150;renderTable()})});
  $('#f-q').addEventListener('input',function(){tstate.q=this.value.toLowerCase();tstate.limit=150;renderTable()});
  $('#f-view').addEventListener('change',function(){tstate.view=this.value;tstate.key=tstate.view==='ok'?'rk':'bd';tstate.dir=tstate.view==='ok'?1:-1;tstate.limit=150;renderTable()});
  $('#more').addEventListener('click',function(){tstate.limit+=500;renderTable()});
  $('#rank-head').addEventListener('click',function(e){var th=e.target.closest('th');if(!th)return;var k=th.dataset.k;if(tstate.key===k)tstate.dir*=-1;else{tstate.key=k;tstate.dir=1}renderTable()});
  $('#rank-body').addEventListener('click',function(e){var tr=e.target.closest('tr');if(!tr)return;var r=rows[+tr.dataset.i];state.strat=r.st;state.variant=r.pf+'|'+r.tg;state.tickers=r.tk;state.alloc=r.A;syncSelects();draw();$('#explorer').scrollIntoView({behavior:'smooth'})});
  renderTable();
}
function renderTable(){
  var want=tstate.view==='ok'?0:1;
  var list=[];rows.forEach(function(r,i){if(!!r.br!==!!want)return;if(tstate.lvl&&r.lt!==tstate.lvl)return;if(tstate.st&&r.st!==tstate.st)return;if(tstate.nt&&String(r.nt)!==tstate.nt)return;if(tstate.zn&&r.zn!==tstate.zn)return;
    if(tstate.q&&(r.tk+' '+r.st+' '+r.pf+' '+r.tg+' '+r.v).toLowerCase().indexOf(tstate.q)<0)return;list.push(i)});
  var k=tstate.key;list.sort(function(a,b){var x=rows[a][k],y=rows[b][k];if(x==null)x=-1e18;if(y==null)y=-1e18;if(typeof x==='string')return x.localeCompare(y)*tstate.dir;return (x-y)*tstate.dir});
  var h='';list.slice(0,tstate.limit).forEach(function(i){var r=rows[i];
    h+='<tr class="clickable" data-i="'+i+'"><td class="n">'+r.rk+'</td><td><span class="tag">'+esc(r.lt)+'</span></td><td>'+esc(r.st)+'<div class="muted small">'+esc(r.pf+' / '+r.tg)+'</div></td><td>'+esc(r.tk)+'</td><td class="n">'+r.A+'%</td><td><span class="zone '+zoneCls(r.zn)+'">'+esc(r.zn)+'</span></td><td class="n">'+usd(r.fe)+'</td><td class="n">'+pct(r.cagr)+'</td><td class="n">'+pct(r.dd)+'</td><td class="n">'+pct(r.wm)+'</td><td class="n">'+pct(r.wt)+'</td><td class="n">'+(r.br?'broke '+esc(r.bd):(r.cal==null?'-':r.cal.toFixed(2)))+'</td><td class="n">'+r.n+(r.few?' <span class="flag">TOO FEW</span>':'')+'</td><td class="verdict">'+esc(r.v)+'</td></tr>'});
  $('#rank-body').innerHTML=h||'<tr><td colspan="14">No scenarios match the filters.</td></tr>';
  $('#rank-count').textContent=list.length+' scenarios'+(list.length>tstate.limit?' (showing '+tstate.limit+')':'');
  $('#more').style.display=list.length>tstate.limit?'':'none';
  $$('#rank-head th').forEach(function(th){th.setAttribute('aria-sort',th.dataset.k===tstate.key?(tstate.dir>0?'ascending':'descending'):'none')});
}
function buildHead(){
  $('#rank-head').innerHTML='<tr>'+cols.map(function(c){var k=c[0]==='lt'?'lvl':c[0];return '<th scope="col" data-k="'+(c[0]==='lt'?'lvl':c[0])+'" class="'+(c[2]||'')+'" tabindex="0">'+c[1]+'</th>'}).join('')+'</tr>';
}
// generic sortable static tables
function initStatic(){
  $$('table.sortable').forEach(function(t){
    t.addEventListener('click',function(e){var th=e.target.closest('th');if(!th||!t.tHead.contains(th))return;
      var i=th.cellIndex,dir=th.getAttribute('aria-sort')==='ascending'?-1:1;
      $$('th',t).forEach(function(x){x.setAttribute('aria-sort','none')});th.setAttribute('aria-sort',dir>0?'ascending':'descending');
      var body=t.tBodies[0],rs=Array.prototype.slice.call(body.rows);
      rs.sort(function(a,b){var x=a.cells[i],y=b.cells[i],xv=x.dataset.v!=null?parseFloat(x.dataset.v):x.textContent,yv=y.dataset.v!=null?parseFloat(y.dataset.v):y.textContent;
        if(typeof xv==='number'){if(isNaN(xv))xv=-1e18;if(isNaN(yv))yv=-1e18;return (xv-yv)*dir}return String(xv).localeCompare(String(yv))*dir});
      rs.forEach(function(r){body.appendChild(r)})})});
}
function initTheme(){var b=$('#btn-theme');b.addEventListener('click',function(){var r=document.documentElement,cur=r.getAttribute('data-theme');
  var dark=cur?cur==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;r.setAttribute('data-theme',dark?'light':'dark');draw()})}
document.addEventListener('DOMContentLoaded',function(){
  window.__errors=window.__errors||[];
  try{buildHead();initExplorer();initTable();initStatic();initTheme();document.body.dataset.ready='1'}catch(e){window.__errors.push(String(e&&e.stack||e));document.body.dataset.ready='error'}
});
window.addEventListener('error',function(e){(window.__errors=window.__errors||[]).push(e.message)});
})();
