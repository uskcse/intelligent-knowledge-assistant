from pathlib import Path
from html import escape
W,H=1920,1320
parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '''<defs><pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#dfe4eb"/></pattern><filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#172b4d" flood-opacity=".08"/></filter><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#55657c" stroke-width="1.5"/></marker></defs>''','<rect width="1920" height="1320" fill="#f8fafc"/><rect width="1920" height="1320" fill="url(#dots)"/>']
def text(x,y,s,size=18,color='#25354b',weight=400):
 parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(s)}</text>')
def rect(x,y,w,h,fill,stroke='#dce3ed',r=16):
 parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
def card(x,y,w,h,title,lines,fill='#fff',accent='#5274b9'):
 parts.append(f'<g filter="url(#shadow)">');rect(x,y,w,h,fill);parts.append('</g>')
 parts.append(f'<rect x="{x}" y="{y+16}" width="5" height="{h-32}" rx="2" fill="{accent}"/>')
 text(x+20,y+32,title,20,weight=700)
 for i,line in enumerate(lines):text(x+20,y+59+i*23,line,16,color='#59677a')
def arrow(points,label=None,lx=0,ly=0,dash=False):
 p=' '.join(f'{x},{y}' for x,y in points)
 parts.append(f'<polyline points="{p}" fill="none" stroke="#55657c" stroke-width="2" marker-end="url(#arrow)"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
 if label:text(lx,ly,label,15,color='#59677a',weight=600)
text(52,63,'INTELLIGENT KNOWLEDGE ASSISTANT',36,weight=700)
text(52,98,'Architecture & demo flow  /  Existing assessment design',21,color='#66758a')
rect(1570,35,295,55,'#fff6d9','#e9d99b',10);text(1590,69,'DESIGN • NOT YET IMPLEMENTED',15,weight=700)
# ingestion
rect(35,130,1850,265,'#edf4ff','#d2e1f5');text(60,165,'01  DOCUMENT INGESTION',21,'#315f9b',700);text(440,165,'Separate process • ingest once, reuse at query time',18,'#526f95')
xs=[60,330,600,870,1140,1410]
cards=[('Source PDFs',['RAG patterns • Vector DBs','Agentic AI frameworks']),('Manifest & extraction',['Hash / skip unchanged','Page-aware text + tables']),('Clean & chunk',['Headings + token boundaries','Page, section and chunk IDs']),('Batch embeddings',['Embedding provider adapter','Same model at query time']),('Validate & publish',['Stage corpus version','Activate atomically']),('Knowledge base',['PostgreSQL + pgvector','Vectors + text + metadata'])]
for x,(title,lines) in zip(xs,cards):card(x,190,245,110,title,lines,accent='#578bd2')
for i in range(5):arrow([(xs[i]+245,245),(xs[i+1],245)])
rect(60,325,760,43,'#fff','#d2e1f5',8);text(78,352,'Persistent source volume: original PDFs + ingestion manifests',17)
text(850,352,'Failed ingestion keeps the last working corpus available.',17,'#526f95')
# serving
rect(35,415,1850,650,'#f2f0ff','#ded8f3');text(60,452,'02  QUESTION → EVIDENCE → CITED ANSWER',21,'#64519e',700)
card(60,490,220,115,'Streamlit UI',['Ask a question','Show answer + sources'],accent='#8b79bf')
card(320,490,230,115,'FastAPI',['Validate request','Request ID + deadline'],accent='#8b79bf')
card(590,490,230,115,'Query router',['Structured classification','Simple / multi-step / unclear'],accent='#8b79bf')
arrow([(280,548),(320,548)]);arrow([(550,548),(590,548)])
card(890,485,265,100,'Simple RAG',['One retrieval workflow'], '#effafa','#349b9a')
card(890,625,265,130,'LangGraph agent',['Plan → retrieve → assess','Rewrite / retry within budget','Read-only search tool'], '#fff4df','#c99431')
card(590,660,230,95,'Clarification',['Resolve material ambiguity'],'#fff','#8b79bf')
arrow([(820,528),(890,528)],'simple',835,513)
arrow([(820,560),(850,560),(850,682),(890,682)],'multi-step',834,615)
arrow([(705,605),(705,660)],'unclear',716,638)
card(1215,535,300,145,'Shared retrieval service',['Dense + lexical retrieval','Reciprocal rank fusion','Authorised corpus version'], '#effafa','#349b9a')
arrow([(1155,535),(1180,535),(1180,578),(1215,578)])
arrow([(1155,686),(1180,686),(1180,635),(1215,635)])
card(1575,535,270,145,'Evidence builder',['Deduplicate retrieved chunks','Preserve source IDs','Enforce context token budget'], '#effafa','#349b9a')
arrow([(1515,608),(1575,608)])
arrow([(1530,300),(1530,470),(1370,470),(1370,535)],'query pinned to active corpus',1400,455)
card(890,790,265,95,'Workflow checkpoints',['Separate PostgreSQL schema'],'#fff4df','#c99431')
arrow([(1020,755),(1020,790)],'persist state',1034,779,True)
# output row right to left
card(1575,805,270,110,'Answer generation',['LLM synthesis from evidence','Claims reference source IDs'],'#fff','#8b79bf')
card(1215,805,300,110,'Grounding & citations',['Validate source identifiers','Support check / one repair'],'#fff','#8b79bf')
card(590,805,230,110,'API response',['Answer + citations + route','Status + request ID'],'#eaf6eb','#54966a')
card(60,805,490,110,'Answer in the UI',['Answered • Partial • Insufficient evidence','Clarification when needed'],'#eaf6eb','#54966a')
arrow([(1710,680),(1710,805)])
arrow([(1575,860),(1515,860)])
arrow([(1215,860),(1185,860),(1185,940),(850,940),(850,860),(820,860)])
text(900,966,'Unsupported claims → repair once or abstain',16,'#6e5c99')
arrow([(590,860),(550,860)])
arrow([(590,705),(570,705),(570,835),(590,835)])
rect(60,985,1785,53,'#fff','#ded8f3',9)
text(80,1018,'MODEL PROVIDER ADAPTER',16,'#64519e',700)
text(375,1018,'Embeddings • Routing • Planning • Answer generation • Support check  |  Timeouts, token budgets, capped retries',17)
# operations
rect(35,1085,1850,180,'#fff9e8','#ebdfbc');text(60,1122,'03  SHARED ENGINEERING CONTROLS',21,'#8a702a',700)
card(60,1140,410,95,'Security & configuration',['Untrusted evidence • allowlisted tools','Secret configuration • safe errors'],'#fff','#b69a45')
card(510,1140,410,95,'Observability',['Stage timings • tokens • route • errors','Redacted logs + trace spans'],'#fff','#b69a45')
card(960,1140,410,95,'Evaluation & testing',['12 labelled questions • grounding checks','Unit + API + end-to-end tests'],'#fff','#b69a45')
card(1410,1140,435,95,'Docker Compose',['Database → ingestion → API → UI','Persistent volumes + readiness checks'],'#fff','#b69a45')
text(52,1300,'Demo: 1. Explain ingestion   2. Ask a simple question   3. Show multi-step retrieval   4. Show an unanswerable question',18,color='#66758a')
parts.append('</svg>')
Path(__file__).with_name('architecture-demo.svg').write_text('\n'.join(parts))
