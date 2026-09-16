
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='Football Analyzer', page_icon='⚽', layout='wide')
st.markdown('<style>.block-container{padding-top:1.5rem}.positive{color:#087f23;font-weight:700}</style>', unsafe_allow_html=True)

FIXTURES_URL='https://www.football-data.co.uk/matches/resources/fixtures.csv'
BASE_URL='https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'

LEAGUES={
'England':{'Premier League':'E0','Championship':'E1','League One':'E2','League Two':'E3'},
'Italy':{'Serie A':'I1','Serie B':'I2'},
'Spain':{'La Liga':'SP1','La Liga 2':'SP2'},
'Germany':{'Bundesliga':'D1','2. Bundesliga':'D2'},
'France':{'Ligue 1':'F1','Ligue 2':'F2'},
'Netherlands':{'Eredivisie':'N1'},'Belgium':{'Jupiler League':'B1'},
'Portugal':{'Primeira Liga':'P1'},'Turkey':{'Super Lig':'T1'},'Greece':{'Super League':'G1'},
'Scotland':{'Premiership':'SC0','Championship':'SC1','League One':'SC2','League Two':'SC3'},
'Poland':{'Ekstraklasa':'POL'},'Romania':{'Liga 1':'ROM'},'Russia':{'Premier League':'RUS'},
'USA':{'MLS':'USA'},'Japan':{'J1 League':'JPN'},'Brazil':{'Serie A':'BRA'},
'Argentina':{'Liga Profesional':'ARG'},'Mexico':{'Liga MX':'MEX'},
'Austria':{'Bundesliga':'AUT'},'Denmark':{'Superliga':'DNK'},'Norway':{'Eliteserien':'NOR'},
'Sweden':{'Allsvenskan':'SWE'},'Switzerland':{'Super League':'SWZ'},
'Finland':{'Veikkausliiga':'FIN'},'Ireland':{'Premier Division':'IRL'},'China':{'Super League':'CHN'}
}


def season_code(y): return f'{y%100:02d}{(y+1)%100:02d}'
def current_season():
    d=date.today(); return d.year if d.month>=8 else d.year-1

def num(x):
    try: return float(str(x).replace(',','.'))
    except: return np.nan

def fair(p): return 1/p if p>0 else np.nan
def value(p,o): return (p*o-1)*100 if p>0 and o>1 else np.nan

def poisson(k,l): return math.exp(-l)*l**k/math.factorial(k) if l>0 else (1 if k==0 else 0)

@st.cache_data(ttl=1800, show_spinner=False)
def load_league(year, code):
    url=BASE_URL.format(season=season_code(year),league=code)
    try:
        r=requests.get(url,timeout=25,headers={'User-Agent':'Mozilla/5.0 FootballAnalyzer'})
        if r.status_code!=200: return pd.DataFrame(),f'HTTP {r.status_code}',url
        df=pd.read_csv(BytesIO(r.content),encoding='latin1',on_bad_lines='skip')
        df.columns=df.columns.astype(str).str.strip()
        if 'Date' in df: df['Date']=pd.to_datetime(df['Date'],dayfirst=True,errors='coerce')
        for c in ['FTHG','FTAG','HTHG','HTAG','HS','AS','HST','AST','HC','AC','HY','AY','HR','AR']:
            if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
        return df,'OK',url
    except Exception as e: return pd.DataFrame(),str(e),url

@st.cache_data(ttl=900, show_spinner=False)
def load_fixtures():
    try:
        r=requests.get(FIXTURES_URL,timeout=25,headers={'User-Agent':'Mozilla/5.0 FootballAnalyzer'})
        if r.status_code!=200: return pd.DataFrame(),f'HTTP {r.status_code}'
        df=pd.read_csv(BytesIO(r.content),encoding='latin1',on_bad_lines='skip')
        df.columns=df.columns.astype(str).str.strip()
        if 'Date' in df: df['Date']=pd.to_datetime(df['Date'],dayfirst=True,errors='coerce')
        return df,'OK'
    except Exception as e: return pd.DataFrame(),str(e)

def completed(df):
    if df.empty or not {'FTHG','FTAG'}.issubset(df.columns): return pd.DataFrame()
    d=df.copy(); d['FTHG']=pd.to_numeric(d['FTHG'],errors='coerce'); d['FTAG']=pd.to_numeric(d['FTAG'],errors='coerce')
    return d[d.FTHG.notna() & d.FTAG.notna()].sort_values('Date')

def team_hist(df,team,before=None,n=12):
    d=completed(df)
    if before is not None and 'Date' in d: d=d[d.Date<before]
    d=d[(d.HomeTeam==team)|(d.AwayTeam==team)].tail(n)
    return d

def strengths(df,team,before=None):
    h=team_hist(df,team,before)
    if h.empty:return 1,1,0
    gf=[];ga=[]
    for _,r in h.iterrows():
        if r.HomeTeam==team: gf.append(r.FTHG);ga.append(r.FTAG)
        else: gf.append(r.FTAG);ga.append(r.FTHG)
    lg=max((completed(df).FTHG.mean()+completed(df).FTAG.mean())/2,0.8)
    att=np.nanmean(gf)/lg; de=np.nanmean(ga)/lg; w=min(1,len(h)/8)
    return float(np.clip(1+(att-1)*w,.55,1.7)),float(np.clip(1+(de-1)*w,.55,1.7)),len(h)

def xg(df,home,away,before=None):
    d=completed(df)
    if before is not None:d=d[d.Date<before]
    lh=float(d.FTHG.mean()) if not d.empty else 1.45; la=float(d.FTAG.mean()) if not d.empty else 1.15
    ha,hd,hn=strengths(df,home,before); aa,ad,an=strengths(df,away,before)
    return float(np.clip(lh*ha*ad,.15,4.5)),float(np.clip(la*aa*hd,.15,4.5)),hn,an

def markets(xh,xa):
    m=np.array([[poisson(i,xh)*poisson(j,xa) for j in range(11)] for i in range(11)]);m/=m.sum()
    one=np.tril(m,-1).sum(); draw=np.trace(m); two=np.triu(m,1).sum()
    total={}
    for i in range(21): total[i]=sum(m[h,a] for h in range(11) for a in range(11) if h+a==i)
    out={'1':one,'X':draw,'2':two,'1X':one+draw,'X2':draw+two,'12':one+two}
    for line in [.5,1.5,2.5,3.5,4.5]:
        out[f'Over {line}']=sum(p for g,p in total.items() if g>line)
        out[f'Under {line}']=sum(p for g,p in total.items() if g<line)
    btts=m[1:,1:].sum();out['BTTS']=btts;out['No BTTS']=1-btts
    out['Home to score']=1-m[:,0].sum();out['Away to score']=1-m[0,:].sum()
    out['Home clean sheet']=m[:,0].sum();out['Away clean sheet']=m[0,:].sum()
    return out

ODDS={'1':['B365H','MaxH','AvgH'],'X':['B365D','MaxD','AvgD'],'2':['B365A','MaxA','AvgA'],
      'Over 2.5':['B365>2.5','P>2.5','Max>2.5','Avg>2.5'],'Under 2.5':['B365<2.5','P<2.5','Max<2.5','Avg<2.5'],
      'BTTS':['B365BTTS','AvgBTTS']}

def odds(row,market):
    for c in ODDS.get(market,[]):
        if c in row.index:
            v=num(row[c])
            if np.isfinite(v) and v>1:return v,c
    return np.nan,''

def top_scores(xh,xa,n=8):
    rows=[]
    for h in range(11):
        for a in range(11): rows.append((h,a,poisson(h,xh)*poisson(a,xa)))
    return sorted(rows,key=lambda z:z[2],reverse=True)[:n]


def scan_future(fixtures,min_value,min_prob,codes,days):
    if fixtures.empty:return pd.DataFrame()
    f=fixtures.copy()
    if not {'Date','HomeTeam','AwayTeam'}.issubset(f.columns):return pd.DataFrame()
    today=pd.Timestamp(date.today());end=today+pd.Timedelta(days=int(days))
    f=f[f.Date.notna()&(f.Date>=today)&(f.Date<=end)]
    if 'Div' in f:f=f[f.Div.isin(codes)]
    cache={};rows=[]
    for _,r in f.iterrows():
        code=r.get('Div',''); home=r.HomeTeam;away=r.AwayTeam;dt=r.Date
        if code not in cache:
            d,_,_=load_league(current_season(),code)
            if d.empty:d,_,_=load_league(current_season()-1,code)
            cache[code]=d
        d=cache[code]
        if d.empty:continue
        hist=completed(d);hist=hist[hist.Date<dt]
        if len(hist)<8:continue
        xh,xa,hn,an=xg(hist,home,away)
        for market,p in markets(xh,xa).items():
            if p*100<min_prob:continue
            o,src=odds(r,market)
            if not np.isfinite(o):continue
            v=value(p,o)
            if not np.isfinite(v) or v<min_value:continue
            rows.append({'Data':dt.strftime('%d/%m/%Y'),'Ora':r.get('Time',''),'Campionato':code,'Partita':f'{home} - {away}',
                         'Mercato':market,'Prob. modello %':round(p*100,2),'Quota':round(o,2),'Quota equa':round(fair(p),2),
                         'Value %':round(v,2),'xG Casa':round(xh,2),'xG Trasferta':round(xa,2),'Campione':f'{hn}/{an}','Fonte quota':src})
    return pd.DataFrame(rows).sort_values(['Value %','Prob. modello %'],ascending=False) if rows else pd.DataFrame()


def backtest(df,market,min_prob,min_value):
    d=completed(df);rows=[]
    for dt in d.Date.dropna().unique():
        before=d[d.Date<dt]
        if len(before)<10:continue
        matches=d[d.Date==dt]
        for _,r in matches.iterrows():
            xh,xa,_,_=xg(before,r.HomeTeam,r.AwayTeam);p=markets(xh,xa).get(market)
            o,_=odds(r,market)
            if p is None or not np.isfinite(o) or p*100<min_prob:continue
            v=value(p,o)
            if not np.isfinite(v) or v<min_value:continue
            h=num(r.FTHG);a=num(r.FTAG);g=h+a
            if market.startswith('Over '): won=g>float(market.split()[1])
            elif market.startswith('Under '): won=g<float(market.split()[1])
            elif market=='BTTS':won=h>0 and a>0
            elif market=='No BTTS':won=not(h>0 and a>0)
            elif market=='1':won=h>a
            elif market=='X':won=h==a
            elif market=='2':won=h<a
            elif market=='1X':won=h>=a
            elif market=='X2':won=a>=h
            elif market=='12':won=h!=a
            else:continue
            rows.append({'Data':r.Date,'Partita':f'{r.HomeTeam} - {r.AwayTeam}','Mercato':market,'Probabilità %':round(p*100,2),'Quota':o,'Value %':v,'Vinta':won,'Profitto':o-1 if won else -1})
    return pd.DataFrame(rows)


def dashboard():
    st.title('⚽ Football Analyzer')
    st.write('Modello pre-partita basato su dati Football-Data, xG semplificati e distribuzione di Poisson.')
    c1,c2,c3=st.columns(3);c1.metric('Dati','Football-Data');c2.metric('Modello','Poisson + forma');c3.metric('Scanner','Fixture future')
    st.info('Il Value Scanner calcola Value = probabilità modello × quota − 1. È una misura teorica, non una garanzia di profitto.')


def analysis():
    st.title('🔎 Analisi partita')
    country=st.selectbox('Nazione',list(LEAGUES));name=st.selectbox('Campionato',list(LEAGUES[country]));code=LEAGUES[country][name]
    year=st.selectbox('Stagione',[current_season(),current_season()-1,current_season()-2])
    df,status,url=load_league(year,code)
    if df.empty:st.error(status);return
    d=completed(df);teams=sorted(set(d.HomeTeam.dropna())|set(d.AwayTeam.dropna()))
    h=st.selectbox('🏠 Casa',teams);a=st.selectbox('✈️ Trasferta',[x for x in teams if x!=h])
    if st.button('🚀 Analizza',type='primary',use_container_width=True):
        xh,xa,hn,an=xg(d,h,a);m=markets(xh,xa)
        c1,c2,c3=st.columns(3);c1.metric('xG Casa',f'{xh:.2f}');c2.metric('xG Trasferta',f'{xa:.2f}');c3.metric('xG Totali',f'{xh+xa:.2f}')
        tab=pd.DataFrame([{'Mercato':k,'Probabilità %':round(v*100,2),'Quota equa':round(fair(v),2)} for k,v in m.items()])
        st.dataframe(tab,use_container_width=True,hide_index=True)
        st.subheader('🎯 Risultati esatti');st.dataframe(pd.DataFrame([{'Risultato':f'{h}-{a}','Probabilità %':round(p*100,2)} for h,a,p in top_scores(xh,xa)]),use_container_width=True,hide_index=True)
        st.caption(f'Campione recente utilizzato: {h}={hn} partite, {a}={an} partite.')


def scanner():
    st.title('💰 Value Scanner')
    c1,c2,c3=st.columns(3)
    minv=c1.number_input('Value minimo %',-50.0,100.0,3.0,.5)
    minp=c2.number_input('Probabilità minima %',1.0,99.0,55.0,1.0)
    days=c3.number_input('Giorni da analizzare',1,60,14,1)
    labels={code:f'{country} - {name}' for country,ls in LEAGUES.items() for name,code in ls.items()}
    selected=st.multiselect('Campionati',list(labels.values()),default=list(labels.values()))
    codes=[c for c,l in labels.items() if l in selected]
    c1,c2=st.columns(2)
    if c1.button('🔄 Aggiorna dati',use_container_width=True):st.cache_data.clear();st.rerun()
    run=c2.button('🔎 Avvia scansione',type='primary',use_container_width=True)
    if run:
        with st.spinner('Scarico fixture e calcolo il modello...'):f,status=load_fixtures();res=scan_future(f,minv,minp,codes,days) if not f.empty else pd.DataFrame()
        if res.empty:st.warning('Nessuna opportunità trovata con i filtri impostati.');return
        st.success(f'Trovate {len(res)} opportunità teoriche.')
        st.dataframe(res,use_container_width=True,hide_index=True)
        st.download_button('📥 Scarica CSV',res.to_csv(index=False).encode('utf-8-sig'),'value_scanner.csv','text/csv')
        st.caption('Le quote delle fixture sono quelle pubblicate da Football-Data e non necessariamente sono quote live.')


def backtest_page():
    st.title('🧪 Backtest walk-forward')
    country=st.selectbox('Nazione',list(LEAGUES),key='bc');name=st.selectbox('Campionato',list(LEAGUES[country]),key='bl');code=LEAGUES[country][name]
    year=st.selectbox('Stagione',[current_season()-1,current_season()-2,current_season()-3],key='by')
    market=st.selectbox('Mercato',['Over 0.5','Over 1.5','Over 2.5','Over 3.5','Under 2.5','BTTS','1','X','2','1X','X2','12'])
    c1,c2=st.columns(2);mp=c1.number_input('Probabilità minima %',1.0,99.0,55.0,1.0);mv=c2.number_input('Value minimo %',-50.0,100.0,3.0,.5)
    if st.button('▶️ Avvia backtest',type='primary',use_container_width=True):
        df,status,_=load_league(year,code)
        if df.empty:st.error(status);return
        with st.spinner('Calcolo...'):r=backtest(df,market,mp,mv)
        if r.empty:st.warning('Nessuna operazione trovata.');return
        bets=len(r);wins=int(r.Vinta.sum());profit=r.Profitto.sum();roi=profit/bets*100
        a,b,c,d=st.columns(4);a.metric('Scommesse',bets);b.metric('Vinte',wins);c.metric('Perse',bets-wins);d.metric('ROI',f'{roi:.2f}%')
        st.dataframe(r,use_container_width=True,hide_index=True)


def debug():
    st.title('🛠️ Debug dati')
    if st.button('🔄 Aggiorna'):st.cache_data.clear();st.rerun()
    f,status=load_fixtures();st.write('Fixture:',status)
    if not f.empty:st.dataframe(f.head(30),use_container_width=True,hide_index=True)
    st.divider()
    for country,ls in list(LEAGUES.items())[:6]:
        for name,code in list(ls.items())[:2]:
            df,s,_=load_league(current_season(),code);st.write(f'{country} - {name} ({code}): {s}, {len(df)} righe')

st.sidebar.title('⚽ Football Analyzer')
menu=st.sidebar.radio('Menu',['Dashboard','Analisi partita','💰 Value Scanner','🧪 Backtest','🛠️ Debug dati'])
if menu=='Dashboard':dashboard()
elif menu=='Analisi partita':analysis()
elif menu=='💰 Value Scanner':scanner()
elif menu=='🧪 Backtest':backtest_page()
elif menu=='🛠️ Debug dati':debug()
st.divider();st.caption('Fonte dati: Football-Data.co.uk. Modello sperimentale: le probabilità sono stime statistiche e non garanzie di risultato.')
