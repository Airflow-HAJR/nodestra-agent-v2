"""
Airflow API — main backend server.

Endpoints:
    GET  /                          Health check
    POST /update/{airport_id}       Refresh map cache from Supabase
    GET  /map/{airport_id}          Interactive map viewer (browser)
    GET  /find-poi                  Fuzzy POI search — returns ID or ambiguous notice
    POST /resolve-poi               Disambiguate duplicate POIs using nearby landmarks
    POST /find-nearest              Nearest POI of a given type to a source POI
    GET  /route                     Shortest path API (JSON)
    GET  /get-nodes                 List airport map nodes
    POST /subscribe-to-flight       Subscribe a passenger to gate-change alerts

Run locally:
    uvicorn api.index:app --reload
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from supabase import create_client, Client

from agent.db import invalidate_map_cache, load_map_levels
from agent.map_engine import (
    dijkstra_multilevel as _dijkstra_multilevel,
    fmt_poi as _fmt_poi,
    matching_poi_types as _matching_poi_types,
    search_pois as _search_pois,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
AIRPORT_ID   = os.environ.get("AIRPORT_ID", "")

SESSION_TTL_DAYS = 30

supabase: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_supabase() -> Client:
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase is not configured.")
    return supabase


def _cleanup_old_sessions() -> None:
    if supabase is None:
        return
    cutoff = (datetime.now(timezone.utc) - timedelta(days=SESSION_TTL_DAYS)).isoformat()
    supabase.table("agent_sessions").delete().lt("created_at", cutoff).execute()


def _start_session_cleanup_thread() -> None:
    def _loop():
        while True:
            try:
                _cleanup_old_sessions()
            except Exception:
                pass
            time.sleep(3600)
    threading.Thread(target=_loop, daemon=True).start()


def _print_map_data():
    if not supabase or not AIRPORT_ID:
        return
    print("\n" + "=" * 60)
    print(f"  AIRFLOW — MAP DATA FOR {AIRPORT_ID}")
    print("=" * 60)
    airport_resp = supabase.table("airports").select("id, name").eq("id", AIRPORT_ID).limit(1).execute()
    airport = (airport_resp.data or [None])[0]
    if not airport:
        print(f"  No airport found with id='{AIRPORT_ID}'")
    else:
        print(f"\n  {airport['id']}  {airport.get('name', '')}")
        maps = load_map_levels(AIRPORT_ID)
        if maps:
            print(f"\n  Levels ({len(maps)}):")
            for m in maps:
                print(f"    {m['name']:35s}  pois={len(m['pois'])}  waypoints={len(m['waypoints'])}  edges={len(m['edges'])}")
        else:
            print("  No levels found in maps table.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _print_map_data()
    _start_session_cleanup_thread()
    yield


app = FastAPI(title="Airflow API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Cache update endpoint
# ---------------------------------------------------------------------------

@app.post("/update/{airport_id}")
def update_map_cache(airport_id: str):
    """Refresh map cache from Supabase."""
    invalidate_map_cache(airport_id)
    levels = load_map_levels(airport_id)
    return {
        "airport_id": airport_id,
        "status": "updated",
        "levels": len(levels),
    }


# ---------------------------------------------------------------------------
# Interactive map viewer
# ---------------------------------------------------------------------------

_MAP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__AIRPORT_NAME__ Map</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#080d14;color:#c9d1d9;font-family:ui-monospace,'SF Mono',monospace;font-size:13px;overflow:hidden}
#app{display:flex;flex-direction:column;height:100vh}
#topbar{display:flex;align-items:center;gap:14px;padding:10px 18px;background:#0d1117;border-bottom:1px solid #1f2937;flex-shrink:0;min-height:46px}
#topbar h1{font-size:14px;color:#e2e8f0;font-weight:600;white-space:nowrap}
.level-tab{padding:4px 14px;border-radius:20px;cursor:pointer;background:#161b22;border:1px solid #30363d;color:#8b949e;font-family:inherit;font-size:12px;transition:all .15s}
.level-tab:hover{border-color:#58a6ff;color:#c9d1d9}
.level-tab.active{background:#1f6feb;border-color:#1f6feb;color:#fff}
#main{display:flex;flex:1;overflow:hidden}
#map-wrap{flex:1;overflow:hidden;cursor:grab;background:#080d14;position:relative}
#map-wrap.grabbing{cursor:grabbing}
svg#map{width:100%;height:100%}
#sidebar{width:310px;flex-shrink:0;background:#0d1117;border-left:1px solid #1f2937;display:flex;flex-direction:column;overflow:hidden}
.panel{padding:14px 16px;border-bottom:1px solid #1f2937;flex-shrink:0}
.panel-title{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#484f58;margin-bottom:12px;font-weight:600}
.row{display:flex;align-items:center;gap:8px;margin-bottom:9px}
.row-label{font-size:11px;color:#484f58;width:44px;flex-shrink:0}
.dot-s,.dot-e{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dot-s{background:#22c55e}
.dot-e{background:#ef4444}
select{flex:1;background:#161b22;border:1px solid #30363d;color:#c9d1d9;padding:5px 8px;border-radius:6px;font-family:inherit;font-size:12px;cursor:pointer}
select:focus{outline:none;border-color:#1f6feb}
select.sel-start{border-color:#22c55e55}
select.sel-end{border-color:#ef444455}
.btn{width:100%;padding:7px 10px;border-radius:6px;border:none;cursor:pointer;font-family:inherit;font-size:12px;font-weight:600;transition:background .15s;margin-top:6px}
.btn-primary{background:#1f6feb;color:#fff}
.btn-primary:hover{background:#388bfd}
.btn-primary:disabled{background:#1f2937;color:#484f58;cursor:default}
.btn-secondary{background:#161b22;color:#8b949e;border:1px solid #30363d}
.btn-secondary:hover{background:#1f2937;color:#c9d1d9}
.btn-secondary:disabled{opacity:.4;cursor:default}
#steps-panel{flex:1;overflow-y:auto;padding:14px 16px}
.step{display:flex;gap:10px;margin-bottom:14px;align-items:flex-start}
.step-badge{width:22px;height:22px;border-radius:50%;background:#161b22;border:1px solid #30363d;font-size:10px;color:#8b949e;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.step-badge.s{background:#052e16;border-color:#22c55e;color:#4ade80}
.step-badge.e{background:#450a0a;border-color:#ef4444;color:#f87171}
.step-name{font-size:12px;color:#e2e8f0;line-height:1.5;font-weight:500}
.step-type{font-size:11px;color:#484f58;margin-top:1px}
.step-verb{font-size:11px;color:#6b7280;margin-bottom:2px}
.hint{color:#484f58;font-size:12px;line-height:1.6}
#route-meta{margin-top:14px;padding-top:12px;border-top:1px solid #1f2937;color:#484f58;font-size:11px}
.no-path{color:#f87171;font-size:12px}
#tooltip{position:fixed;background:#1f2937;border:1px solid #374151;color:#e2e8f0;font-size:12px;padding:5px 10px;border-radius:6px;pointer-events:none;display:none;z-index:100;white-space:nowrap;}
</style>
</head>
<body>
<div id="tooltip"></div>
<div id="app">
  <div id="topbar">
    <h1>__AIRPORT_NAME__</h1>
    <div id="tabs"></div>
  </div>
  <div id="main">
    <div id="map-wrap">
      <svg id="map" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <filter id="glow-sm" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>
        <g id="edges-layer"></g>
        <g id="route-layer"></g>
        <g id="nodes-layer"></g>
        <circle id="anim-dot" r="9" fill="white" opacity="0" filter="url(#glow)"/>
      </svg>
    </div>
    <div id="sidebar">
      <div class="panel">
        <div class="panel-title">Pathfinding</div>
        <div class="row">
          <span class="row-label">Start</span>
          <div class="dot-s"></div>
          <select id="sel-start" class="sel-start">
            <option value="">click map or select…</option>
          </select>
        </div>
        <div class="row">
          <span class="row-label">End</span>
          <div class="dot-e"></div>
          <select id="sel-end" class="sel-end">
            <option value="">click map or select…</option>
          </select>
        </div>
        <button class="btn btn-primary" id="btn-find" onclick="findRoute()" disabled>Find Shortest Path</button>
        <button class="btn btn-secondary" id="btn-anim" onclick="animateRoute()" disabled>▶  Animate Route</button>
        <button class="btn btn-secondary" onclick="clearAll()" style="margin-top:4px">✕  Clear</button>
      </div>
      <div id="steps-panel">
        <div class="hint" id="hint">Select a start and end node — click directly on the map, or use the dropdowns above.</div>
        <div id="steps" style="display:none"></div>
        <div id="route-meta" style="display:none"></div>
      </div>
    </div>
  </div>
</div>
<script>
const LEVELS = __LEVELS_JSON__;

const TYPE_COLOR = {
  gate:'#4ade80',restroom:'#60a5fa',cafe:'#c084fc',coffee:'#c084fc',
  restaurant:'#fb923c',food:'#fb923c',bar:'#fb923c',shop:'#fbbf24',
  elevator:'#22d3ee',escalator:'#67e8f9',stairs:'#67e8f9',
  security:'#f87171',lounge:'#818cf8',info:'#94a3b8',atm:'#fcd34d',
  baggage:'#6b7280',telephone:'#94a3b8',charging:'#86efac',airline:'#38bdf8',
};
function nodeColor(poi){
  const t=(poi.type||'').toLowerCase(), n=(poi.name||'').toLowerCase();
  for(const [k,c] of Object.entries(TYPE_COLOR)) if(t.includes(k)||n.includes(k)) return c;
  return '#94a3b8';
}

// ── state ──────────────────────────────────────────────────────────────────
let curLevel=0, curPath=null, animFrame=null;

// ── viewBox zoom/pan ────────────────────────────────────────────────────────
const VW=1200, VH=700, PAD=70;
let vb={x:0,y:0,w:VW,h:VH};
const svgEl=document.getElementById('map');
const mapWrap=document.getElementById('map-wrap');

function applyVB(){svgEl.setAttribute('viewBox',`${vb.x} ${vb.y} ${vb.w} ${vb.h}`);}
applyVB();

svgEl.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=svgEl.getBoundingClientRect();
  const mx=vb.x+(e.clientX-r.left)/r.width*vb.w;
  const my=vb.y+(e.clientY-r.top)/r.height*vb.h;
  const f=e.deltaY<0?0.88:1/0.88;
  vb.w*=f; vb.h*=f;
  vb.x=mx-(e.clientX-r.left)/r.width*vb.w;
  vb.y=my-(e.clientY-r.top)/r.height*vb.h;
  applyVB();
},{passive:false});

let dragging=false,panStart,vbStart;
svgEl.addEventListener('mousedown',e=>{
  if(e.target.closest('.poi-g')) return;
  dragging=true; mapWrap.classList.add('grabbing');
  const r=svgEl.getBoundingClientRect();
  panStart={x:e.clientX,y:e.clientY}; vbStart={...vb};
});
window.addEventListener('mousemove',e=>{
  if(!dragging) return;
  const r=svgEl.getBoundingClientRect();
  vb.x=vbStart.x-(e.clientX-panStart.x)/r.width*vbStart.w;
  vb.y=vbStart.y-(e.clientY-panStart.y)/r.height*vbStart.h;
  applyVB();
});
window.addEventListener('mouseup',()=>{dragging=false;mapWrap.classList.remove('grabbing');});

// ── coord helpers ───────────────────────────────────────────────────────────
function sx(x){return PAD+x*(VW-PAD*2);}
function sy(y){return PAD+y*(VH-PAD*2);}

// ── render level ────────────────────────────────────────────────────────────
function render(idx){
  curLevel=idx; curPath=null;
  stopAnim();
  document.getElementById('anim-dot').setAttribute('opacity','0');

  const lvl=LEVELS[idx];
  const nodesById={};
  [...lvl.pois,...lvl.waypoints].forEach(n=>nodesById[n.id]=n);

  // populate selects
  ['sel-start','sel-end'].forEach(id=>{
    const sel=document.getElementById(id);
    const prev=sel.value;
    sel.innerHTML='<option value="">click map or select…</option>';
    lvl.pois.filter(p=>p.name).sort((a,b)=>a.name.localeCompare(b.name)).forEach(p=>{
      const o=document.createElement('option');
      o.value=p.id; o.textContent=p.name; sel.appendChild(o);
    });
    if(prev && lvl.pois.find(p=>p.id===prev)) sel.value=prev;
  });

  // edges
  const el=document.getElementById('edges-layer');
  el.innerHTML='';
  for(const e of lvl.edges){
    const a=nodesById[e.from],b=nodesById[e.to];
    if(!a||!b) continue;
    const line=svgNS('line',{x1:sx(a.x),y1:sy(a.y),x2:sx(b.x),y2:sy(b.y),stroke:'#1e2d40','stroke-width':2});
    el.appendChild(line);
  }

  // waypoints
  const nl=document.getElementById('nodes-layer');
  nl.innerHTML='';
  for(const wp of lvl.waypoints){
    nl.appendChild(svgNS('circle',{cx:sx(wp.x),cy:sy(wp.y),r:4,fill:'#1e2d40',stroke:'#2d4a6a','stroke-width':1}));
  }

  // POIs
  for(const poi of lvl.pois){
    const g=document.createElementNS('http://www.w3.org/2000/svg','g');
    g.classList.add('poi-g');
    g.setAttribute('data-id',poi.id);
    g.style.cursor='pointer';

    const c=svgNS('circle',{cx:sx(poi.x),cy:sy(poi.y),r:9,fill:nodeColor(poi),stroke:'#0d1117','stroke-width':2});
    c.classList.add('poi-c');

    g.append(c);
    g.addEventListener('click',()=>handleClick(poi.id));
    g.addEventListener('mouseenter',e=>showTooltip(e,poi));
    g.addEventListener('mousemove',e=>moveTooltip(e));
    g.addEventListener('mouseleave',hideTooltip);
    nl.appendChild(g);
  }

  // tabs highlight
  document.querySelectorAll('.level-tab').forEach((t,i)=>t.classList.toggle('active',i===idx));
  clearRoute();
  updateFindBtn();
}

function svgNS(tag,attrs){
  const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [k,v] of Object.entries(attrs)) el.setAttribute(k,v);
  return el;
}

// ── node click ──────────────────────────────────────────────────────────────
function handleClick(id){
  const ss=document.getElementById('sel-start');
  const se=document.getElementById('sel-end');
  if(!ss.value||id===se.value){ss.value=id; se.value='';}
  else if(!se.value&&id!==ss.value){se.value=id;}
  else{ss.value=id; se.value='';}
  highlightSelected();
  updateFindBtn();
}

function highlightSelected(){
  const s=document.getElementById('sel-start').value;
  const e=document.getElementById('sel-end').value;
  document.querySelectorAll('.poi-c').forEach(c=>{
    const id=c.parentElement.getAttribute('data-id');
    if(id===s){c.setAttribute('stroke','#22c55e');c.setAttribute('stroke-width',3);c.setAttribute('filter','url(#glow-sm)');}
    else if(id===e){c.setAttribute('stroke','#ef4444');c.setAttribute('stroke-width',3);c.setAttribute('filter','url(#glow-sm)');}
    else{c.setAttribute('stroke','#0d1117');c.setAttribute('stroke-width',2);c.removeAttribute('filter');}
  });
}

document.getElementById('sel-start').addEventListener('change',()=>{highlightSelected();updateFindBtn();});
document.getElementById('sel-end').addEventListener('change',()=>{highlightSelected();updateFindBtn();});

function updateFindBtn(){
  const ok=document.getElementById('sel-start').value&&document.getElementById('sel-end').value;
  document.getElementById('btn-find').disabled=!ok;
}

// ── Dijkstra on waypoint graph ───────────────────────────────────────────────
function dijkstra(wpById, edges, startWpId, endWpId){
  const adj={};
  for(const id of Object.keys(wpById)) adj[id]=[];
  for(const e of edges){
    const w=e.weight||1;
    if(adj[e.from]!==undefined) adj[e.from].push({to:e.to,w});
    if(adj[e.to]!==undefined)   adj[e.to].push({to:e.from,w});
  }
  const dist={[startWpId]:0}, prev={}, visited=new Set();
  const pq=[[0,startWpId]];
  while(pq.length){
    pq.sort((a,b)=>a[0]-b[0]);
    const [d,u]=pq.shift();
    if(visited.has(u)) continue;
    visited.add(u);
    if(u===endWpId) break;
    for(const {to,w} of (adj[u]||[])){
      const nd=d+w;
      if(nd<(dist[to]??Infinity)){dist[to]=nd;prev[to]=u;pq.push([nd,to]);}
    }
  }
  if(prev[endWpId]===undefined&&startWpId!==endWpId) return null;
  const path=[]; let cur=endWpId;
  while(cur!==undefined){path.unshift(cur);cur=prev[cur];}
  return {path,dist:dist[endWpId]||0};
}

// ── find route ──────────────────────────────────────────────────────────────
function findRoute(){
  const startPoiId=document.getElementById('sel-start').value;
  const endPoiId=document.getElementById('sel-end').value;
  if(!startPoiId||!endPoiId) return;

  const lvl=LEVELS[curLevel];
  const poiById={}, wpById={};
  lvl.pois.forEach(p=>poiById[p.id]=p);
  lvl.waypoints.forEach(w=>wpById[w.id]=w);

  // Type priority for picking best POI when multiple share a waypoint
  const TYPE_PRI={gate:100,security:90,elevator:80,escalator:75,stairs:70,
    restroom:60,restaurant:50,cafe:50,coffee:50,food:50,bar:45,lounge:40,
    shop:30,bookstore:30,atm:20,info:20,charging:20,baggage:15,telephone:10,aed:5};
  function poiPri(p){return TYPE_PRI[(p.type||'').toLowerCase()]||25;}

  // Build waypoint->best POI map
  const wpToPois={};
  lvl.pois.forEach(p=>{
    if(!p.waypointId||!p.name) return;
    if(!wpToPois[p.waypointId]||poiPri(p)>poiPri(wpToPois[p.waypointId]))
      wpToPois[p.waypointId]=p;
  });

  const startWp=poiById[startPoiId]?.waypointId;
  const endWp=poiById[endPoiId]?.waypointId;

  document.getElementById('hint').style.display='none';
  const stepsEl=document.getElementById('steps');
  const metaEl=document.getElementById('route-meta');

  if(!startWp||!endWp){
    stepsEl.innerHTML='<div class="no-path">One or more selected POIs has no waypoint — cannot route.</div>';
    stepsEl.style.display=''; metaEl.style.display='none';
    document.getElementById('btn-anim').disabled=true;
    return;
  }

  const result=dijkstra(wpById,lvl.edges,startWp,endWp);

  if(!result){
    stepsEl.innerHTML='<div class="no-path">No path found between these two points.</div>';
    stepsEl.style.display=''; metaEl.style.display='none';
    document.getElementById('btn-anim').disabled=true;
    return;
  }

  // Collect best POI per waypoint along the path
  const stops=[];
  const seenWps=new Set();
  for(const wpId of result.path){
    if(seenWps.has(wpId)) continue;
    seenWps.add(wpId);
    if(wpToPois[wpId]) stops.push(wpToPois[wpId]);
  }

  // Animate along waypoints
  curPath=result.path.map(id=>wpById[id]);
  drawPath(result.path,wpById);
  renderSteps(stops,result.dist);
  document.getElementById('btn-anim').disabled=false;
}

function drawPath(wpPath,wpById){
  const rl=document.getElementById('route-layer');
  rl.innerHTML='';
  for(let i=0;i<wpPath.length-1;i++){
    const a=wpById[wpPath[i]],b=wpById[wpPath[i+1]];
    if(!a||!b) continue;
    rl.appendChild(svgNS('line',{
      x1:sx(a.x),y1:sy(a.y),x2:sx(b.x),y2:sy(b.y),
      stroke:'#f59e0b','stroke-width':5,'stroke-linecap':'round',opacity:.9
    }));
  }
}

function renderSteps(stops,totalDist){
  const stepsEl=document.getElementById('steps');
  stepsEl.innerHTML='';

  stops.forEach((n,i)=>{
    const isFirst=i===0, isLast=i===stops.length-1;
    const verb=isFirst?'Start at':isLast?'Arrive at':'Pass through';
    const div=document.createElement('div');
    div.className='step';
    div.innerHTML=
      '<div class="step-badge'+(isFirst?' s':isLast?' e':'')+'">'+(i+1)+'</div>'+
      '<div>'+
        '<div class="step-verb">'+verb+'</div>'+
        '<div class="step-name">'+n.name+'</div>'+
        '<div class="step-type">'+(n.type||'')+'</div>'+
      '</div>';
    stepsEl.appendChild(div);
  });

  if(!stops.length){
    stepsEl.innerHTML='<div class="hint">Route found but no named stops along the way.</div>';
  }

  stepsEl.style.display='';
  const meta=document.getElementById('route-meta');
  const estMin=Math.max(1,Math.round(totalDist*10));
  meta.textContent='~'+estMin+' min walk · '+stops.length+' stops';
  meta.style.display='';
}

// ── animation ───────────────────────────────────────────────────────────────
function animateRoute(){
  if(!curPath||curPath.length<2) return;
  stopAnim();
  const dot=document.getElementById('anim-dot');
  dot.setAttribute('opacity','1');

  const segs=[];
  let total=0;
  for(let i=0;i<curPath.length-1;i++){
    const a=curPath[i],b=curPath[i+1];
    const dx=sx(b.x)-sx(a.x), dy=sy(b.y)-sy(a.y);
    const len=Math.sqrt(dx*dx+dy*dy);
    segs.push({x0:sx(a.x),y0:sy(a.y),x1:sx(b.x),y1:sy(b.y),len});
    total+=len;
  }

  const SPEED=100; // svg-units per second
  const dur=total/SPEED*1000;
  const t0=performance.now();

  function frame(now){
    const t=Math.min((now-t0)/dur,1);
    const target=t*total;
    let acc=0;
    for(const seg of segs){
      if(acc+seg.len>=target){
        const f=(target-acc)/seg.len;
        dot.setAttribute('cx',seg.x0+(seg.x1-seg.x0)*f);
        dot.setAttribute('cy',seg.y0+(seg.y1-seg.y0)*f);
        break;
      }
      acc+=seg.len;
    }
    if(t<1) animFrame=requestAnimationFrame(frame);
    else{ const last=curPath[curPath.length-1]; dot.setAttribute('cx',sx(last.x)); dot.setAttribute('cy',sy(last.y)); }
  }
  animFrame=requestAnimationFrame(frame);
}

function stopAnim(){
  if(animFrame){cancelAnimationFrame(animFrame);animFrame=null;}
}

// ── clear ────────────────────────────────────────────────────────────────────
function clearRoute(){
  document.getElementById('route-layer').innerHTML='';
  document.getElementById('steps').innerHTML='';
  document.getElementById('steps').style.display='none';
  document.getElementById('route-meta').style.display='none';
  document.getElementById('hint').style.display='';
  document.getElementById('btn-anim').disabled=true;
  stopAnim();
  document.getElementById('anim-dot').setAttribute('opacity','0');
}

function clearAll(){
  document.getElementById('sel-start').value='';
  document.getElementById('sel-end').value='';
  highlightSelected();
  updateFindBtn();
  clearRoute();
  curPath=null;
}

// ── tooltip ──────────────────────────────────────────────────────────────────
const tip=document.getElementById('tooltip');
function showTooltip(e,poi){
  tip.textContent=(poi.name||'?')+(poi.type?' · '+poi.type:'');
  tip.style.display='block';
  moveTooltip(e);
}
function moveTooltip(e){
  tip.style.left=(e.clientX+14)+'px';
  tip.style.top=(e.clientY-28)+'px';
}
function hideTooltip(){ tip.style.display='none'; }

// ── tabs ─────────────────────────────────────────────────────────────────────
function buildTabs(){
  const tabs=document.getElementById('tabs');
  LEVELS.forEach((lvl,i)=>{
    const btn=document.createElement('button');
    btn.className='level-tab'+(i===0?' active':'');
    btn.textContent=lvl.name;
    btn.onclick=()=>render(i);
    tabs.appendChild(btn);
  });
}

buildTabs();
render(0);
</script>
</body>
</html>"""


def _build_map_html(airport_name: str, levels: list[dict]) -> str:
    return (
        _MAP_HTML
        .replace("__AIRPORT_NAME__", airport_name)
        .replace("__LEVELS_JSON__", json.dumps(levels))
    )


@app.get("/map/{airport_id}", response_class=HTMLResponse)
def map_view(airport_id: str):
    """Interactive map viewer — open in browser."""
    db = require_supabase()
    airport_resp = db.table("airports").select("name").eq("id", airport_id).limit(1).execute()
    airport_name = airport_resp.data[0]["name"] if airport_resp.data else airport_id
    levels = load_map_levels(airport_id)
    if not levels:
        raise HTTPException(status_code=404, detail=f"No map data found for '{airport_id}'.")
    return _build_map_html(airport_name, levels)


# ---------------------------------------------------------------------------
# Route API
# ---------------------------------------------------------------------------

@app.get("/find-poi")
def find_poi(airport_id: str, q: str):
    """
    Fuzzy POI search. Returns the matching POI's ID, or an ambiguous notice
    if multiple locations share the same name (e.g. two Starbucks on different floors).

    Numbers in the query are matched exactly — 'Gate 5' will never resolve to 'Gate 6'.

    Responses:
        Unique match  → { found: true, id, name, type, level }
        Ambiguous     → { found: false, ambiguous: true, message, matches: [...] }
        Not found     → { found: false, message }
    """
    levels = load_map_levels(airport_id)
    results = _search_pois(levels, q)

    if not results:
        return {"found": False, "message": f"No POI found matching '{q}'."}

    best_score = results[0]["_score"]
    top = [r for r in results if r["_score"] == best_score]

    # Ambiguous: multiple POIs with the same name (case-insensitive) at best score
    unique_names = {r["name"].lower().strip() for r in top}
    if len(top) > 1 and len(unique_names) == 1:
        return {
            "found": False,
            "ambiguous": True,
            "message": (
                f"Found {len(top)} locations named '{top[0]['name']}' on different floors. "
                f"Call /resolve-poi with the primary name and any nearby landmarks to identify which one."
            ),
            "matches": [_fmt_poi(r) for r in top],
        }

    return {"found": True, **_fmt_poi(top[0])}


class ResolvePOIRequest(BaseModel):
    airport_id: str
    name: str          # primary POI name to search for
    landmarks: list[str]  # nearby landmark names the user mentioned


@app.post("/resolve-poi")
def resolve_poi(body: ResolvePOIRequest):
    """
    Disambiguate a POI when multiple share the same name.

    Finds all candidates matching `name`, then fuzzy-searches each landmark,
    runs Dijkstra from every candidate to every landmark, and returns the
    candidate with the smallest total graph distance to those landmarks.

    Body:
        { "airport_id": "SJC", "name": "Starbucks", "landmarks": ["Gate 10", "Restroom"] }

    Response:
        { found: true, id, name, type, level }
    """
    levels = load_map_levels(body.airport_id)

    # Find all candidates for the primary name
    candidates = _search_pois(levels, body.name)
    if not candidates:
        return {"found": False, "message": f"No POI found matching '{body.name}'."}

    best_score = candidates[0]["_score"]
    candidates = [c for c in candidates if c["_score"] == best_score]

    if len(candidates) == 1:
        return {"found": True, **_fmt_poi(candidates[0])}

    # Resolve each landmark to a POI
    landmark_pois = []
    unresolved = []
    for lm in body.landmarks:
        lm_results = _search_pois(levels, lm)
        if lm_results:
            landmark_pois.append(lm_results[0])
        else:
            unresolved.append(lm)

    if not landmark_pois:
        return {
            "found": False,
            "message": (
                f"Could not identify any of the provided landmarks. "
                f"Try different nearby landmark names. "
                f"Candidates: {[c['name'] + ' (' + c['_level_name'] + ')' for c in candidates]}"
            ),
        }

    # Score each candidate by total Dijkstra distance to all resolved landmarks
    best_candidate = None
    best_total = float("inf")

    for candidate in candidates:
        total = 0.0
        for lm_poi in landmark_pois:
            result = _dijkstra_multilevel(levels, candidate["id"], lm_poi["id"])
            if result is None:
                total = float("inf")
                break
            total += result["distance"]
        if total < best_total:
            best_total = total
            best_candidate = candidate

    if best_candidate is None:
        return {"found": False, "message": "Could not determine location from the provided landmarks."}

    return {"found": True, **_fmt_poi(best_candidate)}


class FindNearestRequest(BaseModel):
    airport_id: str
    source_id: str
    poi_type: str


@app.post("/find-nearest")
def find_nearest(body: FindNearestRequest):
    """
    Find the POI of a given type that is closest (by walking distance) to a source POI.

    Body:
        { "airport_id": "SJC", "source_id": "<poi_id>", "poi_type": "restroom" }

    Returns:
        Found     → { found: true, id, name, type, level, distance, estimated_minutes }
        Not found → { found: false, message }
    """
    airport_id, source_id, poi_type = body.airport_id, body.source_id, body.poi_type
    levels = load_map_levels(airport_id)
    if not levels:
        raise HTTPException(status_code=404, detail=f"No map data for '{airport_id}'.")

    source_found = any(p["id"] == source_id for lvl in levels for p in lvl["pois"])
    if not source_found:
        return {"found": False, "message": f"Source POI '{source_id}' not found."}

    matched_types = _matching_poi_types(poi_type, levels)
    if not matched_types:
        return {"found": False, "message": f"No POIs of type '{poi_type}' found."}

    candidates = [
        {**p, "_level_name": lvl["name"]}
        for lvl in levels
        for p in lvl["pois"]
        if p["id"] != source_id and (p.get("type") or "").lower().strip() in matched_types
    ]

    if not candidates:
        return {"found": False, "message": f"No POIs of type '{poi_type}' found."}

    best_candidate = None
    best_distance = float("inf")
    for c in candidates:
        result = _dijkstra_multilevel(levels, source_id, c["id"])
        if result is None:
            continue
        if result["distance"] < best_distance:
            best_distance = result["distance"]
            best_candidate = c

    if best_candidate is None:
        return {"found": False, "message": f"No reachable POI of type '{poi_type}'."}

    poi_type_str = best_candidate.get("type", "")
    name = best_candidate.get("name") or poi_type_str.capitalize() or "Unknown"
    return {
        "found": True,
        "id": best_candidate["id"],
        "name": name,
        "type": poi_type_str,
        "level": best_candidate["_level_name"],
        "distance": round(best_distance, 4),
        "estimated_minutes": max(1, round(best_distance * 10)),
    }


@app.get("/route")
def route(airport_id: str, start_id: str, end_id: str):
    """
    Shortest path between two POIs — handles single and multi-level automatically.

    Params:
        airport_id  — IATA code, e.g. SJC
        start_id    — POI id of the start location
        end_id      — POI id of the destination

    Returns:
        { found, stops, level_changes, distance, estimated_minutes }
        stops        — ordered named POIs along the route (no waypoints), including node type
        level_changes — [{from_level, to_level}] each time the floor changes
    """
    levels = load_map_levels(airport_id)
    if not levels:
        raise HTTPException(status_code=404, detail=f"No map data for '{airport_id}'.")

    result = _dijkstra_multilevel(levels, start_id, end_id)

    if not result:
        return {"found": False, "stops": [], "level_changes": [], "distance": 0, "estimated_minutes": 0}

    return {
        "found": True,
        "stops": [
            {"step": i + 1, "id": p["id"], "name": p["name"], "type": p.get("type", ""), "level": p.get("level_name", "")}
            for i, p in enumerate(result["poi_stops"])
        ],
        "level_changes": result["level_changes"],
        "distance": round(result["distance"], 4),
        "estimated_minutes": max(1, round(result["distance"] * 10)),
    }


# ---------------------------------------------------------------------------
# Get nodes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "Airflow API is running."}



# ---------------------------------------------------------------------------
# Get nodes
# ---------------------------------------------------------------------------

@app.get("/get-nodes")
def get_nodes(airport_id: str, floor_name: Optional[str] = None):
    """Return POI nodes for an airport, optionally filtered by floor name."""
    levels = load_map_levels(airport_id)
    nodes = []
    for lvl in levels:
        if floor_name and lvl["name"] != floor_name:
            continue
        nodes.extend(lvl["pois"])
    return nodes


# ---------------------------------------------------------------------------
# Agent session REST API
# ---------------------------------------------------------------------------

class SessionUpdate(BaseModel):
    current_poi_id:       Optional[str] = None
    current_poi_name:     Optional[str] = None
    destination_poi_id:   Optional[str] = None
    destination_poi_name: Optional[str] = None
    route_stops:          Optional[list] = None
    route_step_index:     Optional[int]  = None


@app.get("/sessions/history/{phone_number}")
def get_session_history(phone_number: str, limit: int = 5):
    """Return the most recent sessions for a phone number, newest first."""
    db = require_supabase()
    resp = (
        db.table("agent_sessions")
        .select("*")
        .eq("phone_number", phone_number)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []


@app.get("/sessions/{call_id}")
def get_session_api(call_id: str):
    """Return the full session row for a call ID."""
    db = require_supabase()
    resp = db.table("agent_sessions").select("*").eq("call_id", call_id).limit(1).execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail=f"Session '{call_id}' not found.")
    return resp.data[0]


@app.patch("/sessions/{call_id}")
def patch_session_api(call_id: str, body: SessionUpdate):
    """Partial update of any session field."""
    db = require_supabase()
    patch = {k: v for k, v in body.dict().items() if v is not None}
    if not patch:
        raise HTTPException(status_code=400, detail="No fields to update.")
    resp = db.table("agent_sessions").update(patch).eq("call_id", call_id).execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail=f"Session '{call_id}' not found.")
    return resp.data[0]


@app.delete("/sessions/{call_id}")
def delete_session_api(call_id: str):
    """Delete a session after the call ends."""
    db = require_supabase()
    db.table("agent_sessions").delete().eq("call_id", call_id).execute()
    return {"deleted": call_id}


# ---------------------------------------------------------------------------
# Flight subscription
# ---------------------------------------------------------------------------

class FlightSubscription(BaseModel):
    flight_number: str
    phone_number: str
    airport_id: str


@app.post("/subscribe-to-flight")
def subscribe_to_flight(body: FlightSubscription):
    db = require_supabase()

    flight = db.table("flights").select("flight_number").eq("flight_number", body.flight_number).limit(1).execute()
    if not flight.data:
        db.table("flights").insert({"flight_number": body.flight_number, "airport_departing_id": body.airport_id}).execute()

    passenger = db.table("passengers").select("id").eq("phone_number", body.phone_number).limit(1).execute()
    if passenger.data:
        passenger_id = passenger.data[0]["id"]
    else:
        new = db.table("passengers").insert({"phone_number": body.phone_number}).execute()
        if not new.data:
            raise HTTPException(status_code=500, detail="Failed to create passenger.")
        passenger_id = new.data[0]["id"]

    existing = (
        db.table("flight_passengers")
        .select("id")
        .eq("flight_id", body.flight_number)
        .eq("passenger_id", passenger_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return {"status": "already_subscribed", "flight": body.flight_number}

    db.table("flight_passengers").insert({
        "flight_id": body.flight_number,
        "passenger_id": passenger_id,
        "airport_id": body.airport_id,
    }).execute()
    return {"status": "subscribed", "flight": body.flight_number}

