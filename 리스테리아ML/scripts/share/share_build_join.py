# -*- coding: utf-8 -*-
"""[작업 1-B·2] LIMS(listeria_body.parquet) + DB수집 데이터셋 시공간 조인 → 공유용/데이터/
- LIMS조인_데이터셋.parquet : 시료 1행 = 1행(676,537행). 값은 DB 원자료 그대로(채우기·단위 변환·집계 없음)
- LIMS조인_토양pH_필지단위.parquet : 토양검정은 같은 동·연도에 필지가 여러 개라 집계 없이 시료×필지로 펼친 별도 파일(DB수집에 soil_ph_info 가 있을 때만)
- 컬럼사전.xlsx / _품질진단.json : 한글명·명세서 설명·연결 방법·값 있음 비율 (데이터품질진단.md 는 share_quality_md.py 가 _품질진단.json 에서 생성)
연결 규칙(값을 바꾸지 않는 '어느 행을 붙일지' 규칙):
  · 국내 기상·대기 = 국내 시료 좌표(DB 전처리 시료표 위도·경도, 없으면 시료 파일 좌표)와 기준일(mnfctur_de) 당일 자료 행이 있는 관측소 중 가장 가까운 1곳(하버사인 거리).
    관측소 좌표는 관측소표의 현재(최신 이력) 좌표를 씀(옛 기준일에도 당시 위치가 아니라 최신 위치). 그날 행이 있으면 항목이 전부 빈 행이라도 그 관측소가 뽑힘.
  · 수입 시료에는 국내 관측소를 붙이지 않음(빈 값)
  · 품목제조대장 = 품목제조번호 원문 일치 → 없으면 숫자만 일치; 같은 번호 여러 행이면 삭제 안 된 행 우선 → 최신 수정일시 → 품목 보고 대장 번호 큰 행. 삭제 안 된 행이 없으면 삭제 행이 붙음(ledg_del_yn='Y'로 구분)
  · 수입 검사 원본 = 수입접수번호 일치; 접수번호당 여러 행이면 리스테리아 행 우선, 그다음 일련번호 작은 행
  · 식품공전 분류·수분활성도 표 = DB 전처리 품목코드(mapped_prdlst_code, 없으면 시료 품목코드) 정확 일치
  · 수분 문헌값 = DB 전처리 시료표 품목명(prdlst_nm) 정확 일치(같은 이름 여러 행이면 첫 행)
"""
import sys, os, json, re, datetime as dt
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
SHARE = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\공유용\데이터"; DBD = os.path.join(SHARE, 'DB수집'); T = r"C:\Users\NT751\.claude\jobs\d9b9b418\tmp"
BODY = r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\listeria_body.parquet"
STUDY0, STUDY1 = pd.Timestamp('2015-04-07'), pd.Timestamp('2026-06-30')
STAMP = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
def P(name): return pd.read_parquet(os.path.join(DBD, name + '.parquet'))
def HAVE(name): return os.path.exists(os.path.join(DBD, name + '.parquet'))
MISSING = []
def has(s):
    if s.dtype == object or str(s.dtype) in ('string', 'category', 'str'):
        return s.notna() & (s.astype('string').str.strip().fillna('') != '')
    return s.notna()
def pct(m): return round(float(np.mean(m) * 100), 1) if len(m) else None
def pct2(m): return round(float(np.mean(m) * 100), 2) if len(m) else None
Q = {'생성': STAMP, '연결': {}, '기상': {}, '컬럼': {}}
DICT = []
def reg(col, table, src, how, note=''): DICT.append({'컬럼': col, '원표': table, '원컬럼': src, '연결 방법': how, '비고': note})

# ================= 0. 시료 파일 =================
body = pd.read_parquet(BODY)
body['lims_data_key'] = body['lims_data_key'].astype(str)
N = len(body); KEY = body['lims_data_key'].copy()
DOM = (body['src_route'].astype(str) == 'domestic').to_numpy(); OVS = ~DOM
DET = (body['judge_result'].astype(str) == '검출').to_numpy()
base_date = pd.to_datetime(body['mnfctur_de']).dt.normalize()
for c in body.columns: reg(c, 'listeria_body.parquet(시료 파일)', c, '원본 그대로')
X = body.copy()
def merge_left(X, right, left_on, right_on, prefix, cols, how_txt, table, note='', keep_key=False):
    r = right[[right_on] + cols].copy()
    assert not r[right_on].duplicated().any(), f'{table} {right_on} 중복'
    r = r.rename(columns={c: f'{prefix}{c}' for c in cols})
    m = X.merge(r, left_on=left_on, right_on=right_on, how='left')
    assert len(m) == N and (m['lims_data_key'].to_numpy() == KEY.to_numpy()).all(), f'{table} 조인 후 행 불일치'
    if right_on != left_on and not keep_key: m = m.drop(columns=[right_on])
    for c in cols: reg(f'{prefix}{c}', table, c, how_txt, note)
    return m

# ================= 1. DB 전처리 시료표(연결 키·참고 컬럼) =================
hp = P('hazard_preprocessed_data'); hp['lims_data_key'] = hp['lims_data_key'].astype(str)
HPC = ['lat', 'lon', 'prdlst_nm', 'mapped_prdlst_code', 'statistic_prdlst_nm_1', 'statistic_prdlst_nm_2', 'statistic_prdlst_nm_3', 'statistic_prdlst_nm_4', 'prdlst_mnfctur_sttemnt_no', 'cntc_sys_key2']
X = merge_left(X, hp, 'lims_data_key', 'lims_data_key', 'db_', HPC, '시료 고유키(lims_data_key) 1:1', 'ready_data.hazard_preprocessed_data', 'lat·lon·prdlst_nm·품목제조번호·수입접수번호는 다른 표를 붙이는 연결 키')
Q['연결']['hazard_preprocessed_data'] = {'키 매칭 %': pct(X['db_mapped_prdlst_code'].notna() | X['db_lat'].notna()), '설명': '우리 시료 키 전부 DB에 존재'}
lat_use = X['db_lat'].fillna(body['lat'].astype(float)).to_numpy(float); lon_use = X['db_lon'].fillna(body['lon'].astype(float)).to_numpy(float)
Q['기상']['좌표'] = {'DB 좌표 없어 시료 파일 좌표 사용(행)': int(X['db_lat'].isna().sum()), '좌표 없음(행)': int(np.isnan(lat_use).sum())}

# ================= 2. 식약처 검사 원본 =================
mo = P('m_org_lims_data_info'); mo['lims_data_key'] = mo['lims_data_key'].astype(str)
MOC = [c for c in ['prdlst_nm', 'upper_prdlst_code', 'upper_prdlst_nm', 'cnvync_sttus_tp', 'cnvync_sttus_tp_unit_code', 'cnvync_mth_code', 'splore_sttus_code', 'distb_de', 'valid_tmlmt'] if c in mo.columns]
if 'prdlst_nm' not in mo.columns: MISSING.append('m_org_lims_data_info.prdlst_nm(검사 원본 품목 명, 컬럼 1개): 2026-09-14 추출본에 없어 빠짐')
X = merge_left(X, mo.drop_duplicates('lims_data_key'), 'lims_data_key', 'lims_data_key', 'morg_', MOC, '시료 고유키(lims_data_key) 1:1', 'ba_data.m_org_lims_data_info')
Q['연결']['m_org_lims_data_info'] = {'키 매칭 % 국내': pct(X.loc[DOM, 'lims_data_key'].isin(set(mo['lims_data_key']))), '키 매칭 % 수입': pct(X.loc[OVS, 'lims_data_key'].isin(set(mo['lims_data_key']))), '표 행': int(len(mo))}
cm = P('t_cm_cmmn_code'); cm['cmmn_code'] = cm['cmmn_code'].astype(str)
for cls, src, new in [('IM08', 'morg_cnvync_mth_code', 'morg_cnvync_mth_nm'), ('IM11', 'morg_splore_sttus_code', 'morg_splore_sttus_nm'), ('IM40', 'morg_cnvync_sttus_tp_unit_code', 'morg_cnvync_sttus_tp_unit_nm')]:
    d = cm[cm['cmmn_cl_code'] == cls].drop_duplicates('cmmn_code').set_index('cmmn_code')['cmmn_code_nm']
    X[new] = X[src].astype('string').map(d).astype('string'); reg(new, 'ba_data.t_cm_cmmn_code', 'cmmn_code_nm', f'코드 {cls} 이름 찾기(코드값 = cmmn_code)', '코드 이름(참고 열)')

# ================= 3. 식품공전 분류 · 수분활성도 표 =================
code_use = X['db_mapped_prdlst_code'].astype('string').fillna(body['prdlst_code'].astype('string'))
X['_code_use'] = code_use
cl = P('t_prdlst_cl'); CLC = ['kor_nm', 'lv', 'hrnk_prdlst_cd', 'htrk_prdlst_cd', 'fdgrp_yn', 'artcl_gun_code', 'food_hf_ls_cl_cd', 'dfn']
X = merge_left(X, cl.drop_duplicates('prdlst_cd'), '_code_use', 'prdlst_cd', 'cl_', CLC, 'DB 전처리 품목코드(mapped_prdlst_code, 없으면 시료 품목코드) = prdlst_cd 정확 일치', 'ba_data.t_prdlst_cl')
clset = set(cl['prdlst_cd'].astype(str)); cl_hit = X['_code_use'].isin(clset)
un = X.loc[~cl_hit, '_code_use'].astype(str)
Q['연결']['t_prdlst_cl'] = {'키 매칭 % 국내': pct(cl_hit[DOM]), '키 매칭 % 수입': pct(cl_hit[OVS]), '표 행': int(len(cl)), '표 중복 코드': int(cl['prdlst_cd'].duplicated().sum()),
                          '연결 안 된 코드 상위': {k: int(v) for k, v in un.value_counts().head(3).items()}, '연결 안 된 코드 자릿수': {int(k): int(v) for k, v in un.str.len().value_counts().items()}}
aw = P('t_aw_cl'); AWC = ['aw', 'ph', 'na', 'mc', 'aw_source']
X = merge_left(X, aw.drop_duplicates('prdlst_cd'), '_code_use', 'prdlst_cd', 'aw_', AWC, '위와 같은 품목코드 = prdlst_cd 정확 일치(상위 품목으로 올라가 찾지 않음)', 'ba_data.t_aw_cl', '리서치팀 파일 적재본, DB에서 채운 값 섞임(결과보고 3-6). aw_source 로는 채운 값 구분 안 됨')
Q['연결']['t_aw_cl'] = {'키 매칭 % 국내': pct(X.loc[DOM, '_code_use'].isin(set(aw['prdlst_cd'].astype(str)))), '키 매칭 % 수입': pct(X.loc[OVS, '_code_use'].isin(set(aw['prdlst_cd'].astype(str)))), '표 행': int(len(aw)), '표 중복 코드': int(aw['prdlst_cd'].duplicated().sum())}
X = X.drop(columns=['_code_use'])

# ================= 4. 수분 문헌값(품목명 정확 일치) =================
if HAVE('food_moisture_info'):
    fm = P('food_moisture_info')
    fm_dup = int(fm['prdlst_nm'].duplicated().sum())
    fm1 = fm.sort_values(['prdlst_nm']).drop_duplicates('prdlst_nm', keep='first')
    FMC = [c for c in ['water_activate', 'water_activate_source', 'moisture_content', 'food_type'] if c in fm.columns]
    X['_nm'] = X['db_prdlst_nm'].astype('string')
    X = merge_left(X, fm1, '_nm', 'prdlst_nm', 'fmi_', FMC, 'DB 전처리 시료표 품목명(prdlst_nm) = prdlst_nm 글자 정확 일치', 'ba_data_info.food_moisture_info', f'품목코드 없음. 같은 품목명 행 {fm_dup}개는 첫 행 사용' if fm_dup else '품목코드 없음')
    Q['연결']['food_moisture_info'] = {'품목명 일치 % 국내': pct(X.loc[DOM, '_nm'].isin(set(fm['prdlst_nm'].astype(str)))), '품목명 일치 % 수입': pct(X.loc[OVS, '_nm'].isin(set(fm['prdlst_nm'].astype(str)))), '표 행': int(len(fm)), '표 중복 품목명': fm_dup}
    X = X.drop(columns=['_nm'])
else:
    MISSING.append('food_moisture_info(수분 문헌값 표, 컬럼 5개): 추출본 없어 빠짐'); Q['연결']['food_moisture_info'] = {'상태': '추출본 없음(DB 복구 후 재수집)'}

# ================= 5. 품목제조대장 =================
lg = P('t_prdlst_mnf_ledg')
lg['_live'] = lg['del_yn'].astype('string').fillna('N') != 'Y'
lg1 = lg.sort_values(['prdlst_report_no', '_live', 'last_updt_dtm', 'prdlst_report_ledg_no'], ascending=[True, False, False, False]).drop_duplicates('prdlst_report_no')
tie = lg[lg['_live']].groupby(['prdlst_report_no', 'last_updt_dtm']).size(); tie_nos = set(tie[tie > 1].reset_index()['prdlst_report_no'])
tie2 = lg[lg['_live']].groupby(['prdlst_report_no', 'last_updt_dtm', 'prdlst_report_ledg_no']).size(); tie2_nos = set(tie2[tie2 > 1].reset_index()['prdlst_report_no'])  # 세 기준이 모두 같은 완전 동점(파일 순서 앞 행이 붙음)
LGC = ['mnf_cnd_cn', 'prsrv_distb_stdr_cn', 'pog_daycnt', 'qlity_mntnc_tmlmt_daycnt', 'frmlc_mtrqlt', 'prdlst_report_ledg_no', 'del_yn', 'last_updt_dtm']
raw_no = X['db_prdlst_mnfctur_sttemnt_no'].astype('string'); dig_no = raw_no.str.replace(r'\D', '', regex=True)
lgset = set(lg1['prdlst_report_no'].astype(str))
raw_hit = raw_no.isin(lgset).fillna(False); dig_hit = (~raw_hit) & dig_no.isin(lgset).fillna(False) & (dig_no != '')
X['_ledg_key'] = raw_no.where(raw_hit, dig_no.where(dig_hit, pd.NA))
X = merge_left(X, lg1, '_ledg_key', 'prdlst_report_no', 'ledg_', LGC, '품목제조번호 원문 = prdlst_report_no, 없으면 기호를 뺀 숫자만 일치; 같은 번호 여러 행이면 삭제 안 된 행 우선 → 최신 수정일시 → 대장 번호 큰 행(삭제 안 된 행이 없으면 삭제 행이 붙음)', 'ba_data.t_prdlst_mnf_ledg')
X['ledg_match_type'] = np.select([raw_hit.to_numpy(), dig_hit.to_numpy()], ['원문 일치', '숫자만 일치'], default=None); X['ledg_match_type'] = X['ledg_match_type'].astype('string')
reg('ledg_match_type', '(연결 정보)', '', '품목제조대장이 어떤 번호 형태로 붙었는지', '원문 일치 / 숫자만 일치 / 빈 값=연결 안 됨')
Q['연결']['t_prdlst_mnf_ledg'] = {'번호 있음 % 국내': pct(has(X.loc[DOM, 'db_prdlst_mnfctur_sttemnt_no'])), '키 매칭 % 국내': pct(X.loc[DOM, 'ledg_match_type'].notna()), '키 매칭 % 수입': pct(X.loc[OVS, 'ledg_match_type'].notna()),
                                  '키 매칭 % 국내 검출': pct(X.loc[DOM & DET, 'ledg_match_type'].notna()), '원문 일치(행)': int(raw_hit.sum()), '숫자만 일치(행)': int(dig_hit.sum()), '표 행': int(len(lg)), '삭제 행만 있는 번호': int((~lg.groupby('prdlst_report_no')['_live'].any()).sum()),
                                  '삭제 행이 붙은 시료 행': int((X['ledg_del_yn'].astype('string') == 'Y').sum()), '삭제 안 된 행끼리 수정일시가 같은 번호': int(len(tie_nos)), '그 번호가 붙은 시료 행': int(X['_ledg_key'].isin(tie_nos).sum()),
                                  '수정일시·대장번호까지 같은 완전 동점 번호': int(len(tie2_nos)), '완전 동점 번호가 붙은 시료 행': int(X['_ledg_key'].isin(tie2_nos).sum())}
X = X.drop(columns=['_ledg_key'])
# 원표 자체의 빈 값 비율(연결과 무관하게 표가 비어 있는지 보기 위함)
Q['원표 빈 값 %'] = {f'm_org_lims_data_info.{c}': round(float((~has(mo[c])).mean() * 100), 1) for c in ['cnvync_sttus_tp', 'cnvync_sttus_tp_unit_code', 'splore_sttus_code', 'valid_tmlmt', 'distb_de'] if c in mo.columns}
Q['원표 빈 값 %'].update({f't_prdlst_cl.{c}': round(float((~has(cl[c])).mean() * 100), 1) for c in ['artcl_gun_code', 'dfn']})

# ================= 6. 수입 검사 원본 =================
so = P('s_org_lims_data_info')
so1 = so.assign(_lis=(so['label'].astype('string') == '리스테리아')).sort_values(['rcept_no', '_lis', 'lims_seq'], ascending=[True, False, True]).drop_duplicates('rcept_no')
SOC = ['prdlst_nm', 'prduct_dvs_cd', 'dclr_prduct_dvs', 'hsk_no', 'distb_bgn_de', 'distb_end_de', 'label']
X['_rc'] = X['db_cntc_sys_key2'].astype('string')
X = merge_left(X, so1, '_rc', 'rcept_no', 'sorg_', SOC, '수입접수번호(cntc_sys_key2) = rcept_no; 접수번호당 여러 행이면 리스테리아 행 우선', 'ba_data.s_org_lims_data_info', '수입 시료 전용(국내 시료는 수입접수번호가 수입 검사 원본 표에 있는 몇 행 빼고 빈 값)')
so_hit = X['_rc'].isin(set(so['rcept_no'].astype(str)))
Q['연결']['s_org_lims_data_info'] = {'키 매칭 % 수입': pct(so_hit[OVS]), '키 매칭 % 국내': pct(so_hit[DOM]), '국내 연결 행': int(so_hit[DOM].sum()), '표 행': int(len(so)),
                                     '붙은 행이 리스테리아 행인 % (붙은 행 기준)': pct((X.loc[so_hit, 'sorg_label'].astype('string') == '리스테리아').fillna(False))}
X = X.drop(columns=['_rc'])

# ================= 7. 국내 기상·대기: 기준일 당일 자료가 있는 관측소 중 가장 가까운 곳 =================
def hav(la1, lo1, la2, lo2):
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
def nearest_join(X, st, daily, sid, dcol, VARS, prefix, netname, st_table, name_col=None):
    daily = daily.copy(); daily['_d'] = pd.to_datetime(daily[dcol]).dt.normalize()
    n_dup = int(daily.duplicated([sid, '_d']).sum()); daily = daily.drop_duplicates([sid, '_d'], keep='first')
    st = st.dropna(subset=['lat', 'lon']).drop_duplicates(sid).reset_index(drop=True)
    no_coord = sorted(set(daily[sid].astype(str)) - set(st[sid].astype(str)))
    n_no_coord_rows = int(daily[sid].astype(str).isin(no_coord).sum())
    daily = daily[daily[sid].isin(set(st[sid]))]
    pos = {s: i for i, s in enumerate(st[sid].tolist())}
    SLA, SLO = np.radians(st['lat'].to_numpy(float)), np.radians(st['lon'].to_numpy(float))
    d_sid = daily[sid].to_numpy()
    day_rows = pd.Series(np.arange(len(daily))).groupby(daily['_d'].to_numpy()).apply(lambda s: s.to_numpy()).to_dict()
    out_sid = np.full(N, None, dtype=object); out_dist = np.full(N, np.nan)
    ok = DOM & ~np.isnan(lat_use)
    idx_by_day = pd.Series(np.arange(N)[ok]).groupby(base_date.to_numpy()[ok]).apply(lambda s: s.to_numpy()).to_dict()
    no_day = {}
    for d, rows in idx_by_day.items():
        if d not in day_rows: no_day[str(pd.Timestamp(d).date())] = int(len(rows)); continue
        cs = np.unique(d_sid[day_rows[d]]); ci = np.array([pos[s] for s in cs])
        la, lo = np.radians(lat_use[rows]), np.radians(lon_use[rows])
        D = hav(la[:, None], lo[:, None], SLA[None, ci], SLO[None, ci])
        j = D.argmin(1); out_sid[rows] = cs[j]; out_dist[rows] = D[np.arange(len(rows)), j]
    X[f'{prefix}{sid}'] = pd.Series(out_sid, index=X.index).astype('string'); X[f'{prefix}dist_km'] = np.round(out_dist, 3)
    reg(f'{prefix}{sid}', st_table, sid, '연결된 관측소(기준일 당일 자료가 있는 관측소 중 가장 가까운 곳)', '연결 정보'); reg(f'{prefix}dist_km', '(연결 정보)', '', '시료 좌표에서 연결된 관측소(현재 좌표)까지 지표면 거리(km, 하버사인, 소수 3자리)', '연결 정보')
    if name_col:
        nm = st.set_index(sid)[name_col].astype('string'); X[f'{prefix}{name_col}'] = X[f'{prefix}{sid}'].map(nm).astype('string'); reg(f'{prefix}{name_col}', st_table, name_col, '연결된 관측소 이름(관측소표)', '연결 정보')
    key = pd.DataFrame({sid: X[f'{prefix}{sid}'].astype(object), '_d': base_date})
    vals = daily[[sid, '_d'] + VARS].copy(); vals[sid] = vals[sid].astype(object)
    m = key.merge(vals, on=[sid, '_d'], how='left'); assert len(m) == N
    for v in VARS:
        X[f'{prefix}{v}'] = m[v].to_numpy(); reg(f'{prefix}{v}', netname[1], v, '연결된 관측소의 기준일 당일 값(원값 그대로)')
    # 빈 값 원인 분해(국내): 관측소 없음(그날 자료 없는 날짜) / 그 관측소가 시료 기간에 그 항목을 한 번도 기록 안 함 / 행은 있으나 그 항목만 빈 값
    assigned = pd.Series(out_sid).notna().to_numpy()
    stud = daily[(daily['_d'] >= STUDY0) & (daily['_d'] <= STUDY1)]
    never = {v: set(stud.groupby(sid)[v].apply(lambda s: not has(s).any()).loc[lambda s: s].index.astype(str)) for v in VARS}
    allnull = ~np.column_stack([has(X[f'{prefix}{v}']).to_numpy() for v in VARS]).any(axis=1)
    why = {}
    for v in VARS:
        h = has(X[f'{prefix}{v}']).to_numpy(); nv = X[f'{prefix}{sid}'].astype(str).isin(never[v]).to_numpy()
        why[v] = {'값 있음': pct(h[DOM]), '관측소 없음(그날 자료 없는 날짜)': pct(~assigned[DOM]), '그 항목을 기록하지 않는 관측소': pct((assigned & ~h & nv)[DOM]), '그날 그 항목만 빈 값': pct((assigned & ~h & ~nv)[DOM]),
                  '값 있음(국내 검출)': pct(h[DOM & DET]), '기록하지 않는 관측소 수(시료 기간)': len(never[v])}
    Q['기상'][netname[0]] = {'관측소 후보(좌표 있고 일자료 있음)': int(daily[sid].nunique()), '일자료에만 있고 관측소표에 좌표 없는 관측소': {'수': len(no_coord), '일자료 행': n_no_coord_rows, '이름': no_coord[:25]},
                              '일자료 중복(관측소·날짜) 행': n_dup, '국내 시료 연결 % (관측소 정해짐)': pct2(assigned[DOM]), '관측소 못 정한 국내 행': int((~assigned[DOM]).sum()), '자료가 없는 날짜(국내 시료 행 수)': no_day,
                              '관측소는 정해졌으나 그날 항목이 전부 빈 국내 행': int((assigned & allnull)[DOM].sum()),
                              '거리 km 중앙값': round(float(np.nanmedian(out_dist[DOM])), 1), '거리 km 90%': round(float(np.nanquantile(out_dist[DOM], .9)), 1), '거리 km 최대': round(float(np.nanmax(out_dist[DOM])), 1), '거리 50km 초과 국내 행': int((out_dist[DOM] > 50).sum()),
                              '연결 관측소 수': int(pd.Series(out_sid[DOM]).nunique()), '항목별': why}
    return X
ast = P('kma_ground_obs_station_info')
ast['chg_dt'] = pd.to_datetime(ast['chg_dt'], errors='coerce'); ast['reg_dt'] = pd.to_datetime(ast['reg_dt'], errors='coerce')
sA = ast.sort_values(['chg_dt', 'reg_dt'], ascending=False, na_position='last').drop_duplicates('stn_id')[['stn_id', 'stn_ko', 'lat', 'lon']].astype({'lat': float, 'lon': float})
# 관측소 이전(이력 행 사이 좌표 1km 초과 차이) 수
g = ast.astype({'lat': float, 'lon': float}).groupby('stn_id')
moved = [s for s, d in g if len(d) > 1 and hav(np.radians(d['lat'].min()), np.radians(d['lon'].min()), np.radians(d['lat'].max()), np.radians(d['lon'].max())) > 1.0]
ASOS_V = ['avg_ta', 'max_ta', 'min_ta', 'sum_rn', 'avg_ws', 'max_ws', 'max_ins_ws', 'max_ws_wd', 'max_ins_ws_wd', 'max_ws1', 'sum_sshr', 'sum_gsr', 'avg_cm5_te', 'avg_cm10_te', 'avg_cm20_tes', 'avg_cm30_te', 'avg_ts', 'min_tg']
X = nearest_join(X, sA, P('kma_asos_daily_info'), 'stn_id', 'meas_date', ASOS_V, 'asos_', ('종관기상(ASOS)', 'api_data_info.kma_asos_daily_info'), 'api_data_info.kma_ground_obs_station_info', 'stn_ko')
Q['기상']['종관기상(ASOS)']['이력 행 사이 좌표가 1km 넘게 다른 관측소'] = {'수': len(moved), '그 관측소가 붙은 국내 행': int(X.loc[DOM, 'asos_stn_id'].astype(str).isin(set(map(str, moved))).sum())}
nst = P('nas_agri_weather_obs_station'); nst['chg_dt'] = pd.to_datetime(nst['chg_dt'], errors='coerce')
sN = nst.sort_values('chg_dt', ascending=False).drop_duplicates('obsr_spot_code')[['obsr_spot_code', 'obsr_spot_nm', 'instl_la', 'instl_lo']].rename(columns={'instl_la': 'lat', 'instl_lo': 'lon'}).astype({'lat': float, 'lon': float})
NAS_V = ['temp', 'hghst_artmp', 'lowst_artmp', 'hum', 'rn', 'wind', 'max_wind', 'widdir', 'sun_time', 'srqty', 'soil_temp', 'gr_temp', 'soil_wt']
X = nearest_join(X, sN, P('nas_agri_weather_daily_info'), 'obsr_spot_code', 'meas_date', NAS_V, 'nas_', ('농업기상(NAS)', 'api_data_info.nas_agri_weather_daily_info'), 'api_data_info.nas_agri_weather_obs_station', 'obsr_spot_nm')
kst = P('keco_airkorea_obs_station'); kst['chg_dt'] = pd.to_datetime(kst['chg_dt'], errors='coerce'); kst['_u'] = (kst['use_yn'].astype('string') == '1').astype(int)
sK = kst.sort_values(['_u', 'chg_dt'], ascending=False).drop_duplicates('station_nm')[['station_nm', 'dm_x', 'dm_y']].rename(columns={'dm_x': 'lat', 'dm_y': 'lon'}).astype({'lat': float, 'lon': float})
AIR_V = ['no2_value', 'co_value', 'so2_value', 'pm10_value']
X = nearest_join(X, sK, P('keco_airkorea_fine_dust_info'), 'station_nm', 'msur_dt', AIR_V, 'air_', ('에어코리아', 'api_data_info.keco_airkorea_fine_dust_info'), 'api_data_info.keco_airkorea_obs_station')
Q['기상']['방식'] = '국내 시료만. 기준일(mnfctur_de) 당일 자료 행이 있는 관측소 가운데 시료 좌표에서 가장 가까운 1곳의 그날 값. 빈 값은 채우지 않음. 관측소 좌표는 관측소표의 현재(최신 이력) 좌표: 종관기상 = 관측소표 최신 이력 행, 농업기상 = 설치 위도·경도, 에어코리아 = dm_x(위도)·dm_y(경도), 같은 이름 여러 행이면 운영 중 행 우선. 그날 행이 있으면 항목이 전부 빈 행이라도 그 관측소가 뽑힘'
Q['기상']['시험용 관측소'] = {'동방로거테스트(561034A001)로 연결된 국내 행': int((X.loc[DOM, 'nas_obsr_spot_code'].astype('string') == '561034A001').sum())}
if not HAVE('water_pollution'): MISSING.append('water_pollution(수질오염원 설치시설 표, 컬럼 8개): 추출본 없어 DB수집 폴더에 없음(원래도 좌표계 미확인이라 시료에 붙이지 않는 표)')

# ================= 8. 한글 매핑·타입 정리·저장 =================
spec = json.load(open(os.path.join(T, 'spec_desc.json'), encoding='utf-8'))['modelteam']
rep = json.load(open(os.path.join(T, 'report_sheet1.json'), encoding='utf-8'))
dbc = pd.read_csv(os.path.join(DBD, '_DB컬럼설명.csv'))
dbc_map = {(r.tbl, r.col): (r.db_comment if isinstance(r.db_comment, str) else '') for r in dbc.itertuples()}
dbt_map = {(r.tbl, r.col): r.db_type for r in dbc.itertuples()}
BODY_SPEC = {'mnfctur_de': 'mnfctur_de_or_fallback', 'src_route': 'geo_route'}  # 시료 파일 열 이름 ↔ 정의서 열 이름이 다른 경우
MANUAL = {'lims_data_key': '시료 고유키(검사기관코드_시료의뢰번호_시료순번_시험항목순번)', 'hazard_label': '위해요소 이름', 'judge_result': '판정 결과(검출/불검출)', 'src_route': '국내/수입 경로(domestic=국내, overseas=수입)', 'data_source': '데이터 출처',
          'expr_iem_nm': '시험 항목 명', 'result_value': '결과 값 원문', 'result_value_num': '결과 값(숫자 변환)', 'judge_stdr': '판정 기준 원문', 'judge_stdr_num': '판정 기준(숫자 변환)', 'inspct_instt_code': '검사 기관 코드',
          'prdlst_code': '품목 코드', 'train_prdlst_code': '학습용 품목코드(전처리)', 'train_prdlst_nm': '학습용 품목명(전처리)', 'prduct_nm': '제품 명', 'mnfctur_cmpny_nm': '제조 회사 명', 'mnfctur_de': '기준일(제조일, 없으면 선적일·의뢰일)',
          'date_basis': '기준일 출처(제조일/선적일/의뢰일)', 'addr': '주소', 'addr_source': '주소 출처', 'legal_dong_code': '법정동 코드(10자리)', 'lat': '위도', 'lon': '경도',
          'db_lat': '전처리 시료표 위도(관측소 찾기에 사용)', 'db_lon': '전처리 시료표 경도(관측소 찾기에 사용)', 'db_prdlst_nm': '전처리 시료표 품목 명(수분 문헌값 연결 키)', 'db_mapped_prdlst_code': '전처리 시료표 품목코드(옛 코드를 새 코드로 바꾼 값, 분류·수분활성도 표 연결 키)',
          'db_statistic_prdlst_nm_1': '전처리 시료표 통계용 품목분류 1단계', 'db_statistic_prdlst_nm_2': '전처리 시료표 통계용 품목분류 2단계', 'db_statistic_prdlst_nm_3': '전처리 시료표 통계용 품목분류 3단계', 'db_statistic_prdlst_nm_4': '전처리 시료표 통계용 품목분류 4단계',
          'db_prdlst_mnfctur_sttemnt_no': '전처리 시료표 품목제조번호(품목제조대장 연결 키)', 'db_cntc_sys_key2': '전처리 시료표 수입접수번호(수입 검사 원본 연결 키)',
          'asos_stn_id': '종관기상 연결 관측소 번호', 'asos_stn_ko': '종관기상 연결 관측소 이름', 'asos_dist_km': '종관기상 연결 관측소까지 거리(km)', 'nas_obsr_spot_code': '농업기상 연결 관측소 코드', 'nas_obsr_spot_nm': '농업기상 연결 관측소 이름', 'nas_dist_km': '농업기상 연결 관측소까지 거리(km)',
          'air_station_nm': '에어코리아 연결 측정소 이름', 'air_dist_km': '에어코리아 연결 측정소까지 거리(km)', 'ledg_match_type': '품목제조대장 연결 방식(원문 일치/숫자만 일치)',
          'morg_cnvync_mth_nm': '검사 원본 운반 방법 이름', 'morg_splore_sttus_nm': '검사 원본 시료 상태 이름(냉장/냉동/실온)', 'morg_cnvync_sttus_tp_unit_nm': '검사 원본 운반 상태 온도 단위 이름', 'morg_distb_de': '검사 원본 유통일',
          'cl_lv': '식품공전 분류 단계', 'aw_aw': '수분활성도 표 수분활성도(aw)', 'aw_ph': '수분활성도 표 식품 pH', 'aw_na': '수분활성도 표 염도', 'aw_mc': '수분활성도 표 수분함량', 'aw_aw_source': '수분활성도 표 값 출처 표시', 'sorg_label': '수입 검사 원본 위해요소 이름(label)',
          'nas_temp': '농업기상 평균기온', 'cl_kor_nm': '식품공전 분류 품목 한글명', 'db_prdlst_nm': '전처리 시료표 품목 명(수분 문헌값 연결 키, DB 복구 후 사용)'}
NOTE_ADD = {'mnfctur_de': '정의서 설명과 달리 실제 값은 제조일/의뢰일/선적일', 'date_basis': '정의서 설명과 달리 실제 값은 제조일/의뢰일/선적일', 'asos_max_ins_ws_wd': "정의서 설명의 '충속'은 풍속의 오타", 'asos_sum_gsr': "정의서 설명의 '일사랑'은 일사량의 오타",
            'nas_hum': '결과보고 1-1 표에는 없고 테이블정의서(20260901) 색 표시로 추가한 열', 'air_pm10_value': '결과보고 1-1 표에는 없고 테이블정의서(20260901) 색 표시로 추가한 열',
            'ledg_pog_daycnt': '값은 문장 텍스트(일수 숫자 아님)', 'ledg_qlity_mntnc_tmlmt_daycnt': '값은 문장 텍스트(일수 숫자 아님)', 'cl_artcl_gun_code': '분류표 자체에서 절반 넘게 빈 글자', 'cl_dfn': '분류표 자체에서 절반 넘게 빈 글자'}
JUDGE_ADD = {'nas_hum': '정의서 색 표시(추가)', 'air_pm10_value': '정의서 색 표시(추가)'}
NET = {'asos_': '종관기상 ', 'nas_': '농업기상 ', 'air_': '에어코리아 ', 'morg_': '검사 원본 ', 'cl_': '식품공전 분류 ', 'aw_': '수분활성도 표 ', 'fmi_': '수분 문헌값 ', 'ledg_': '품목제조대장 ', 'sorg_': '수입 검사 원본 ', 'db_': '전처리 시료표 '}
def korean(col, table, src):
    is_body = 'listeria_body' in table
    tb = 'hazard_preprocessed_data' if is_body else (table.split('.')[-1] if '.' in table else table)
    spec_src = BODY_SPEC.get(src, src) if is_body else src
    rk = rep.get(tb, {}).get(src, {}).get('한글') if src else None
    judge = (rep.get(tb, {}).get(src, {}) or (rep.get(tb, {}).get('statistic_prdlst_nm_1~4', {}) if str(src).startswith('statistic_prdlst_nm_') else {})).get('판정', '') if src else ''
    sp = spec.get(tb, {}).get(spec_src) if src else None
    sp = sp if sp and str(sp).lower() not in ('nan', 'none') else ''
    if is_body and sp and spec_src != src: sp = f'{sp} (정의서 열 이름 {spec_src})'
    db = dbc_map.get((tb, spec_src), '') if src else ''
    pre = next((NET[p] for p in NET if col.startswith(p)), '')
    if MANUAL.get(col): base, pre = MANUAL[col], ''
    elif is_body and MANUAL.get(src): base = MANUAL[src]
    elif rk: base = rk
    else:
        raw = sp or db or col
        base = str(raw).split('.')[0].split('(')[0].strip().replace('_', ' ')
    return pre + base, sp, db, judge
# 정수형 DB 열은 왼쪽 조인으로 float 가 된 것을 정수(빈 값 허용)로 되돌림 — 값은 그대로, 표시만 999.0 → 999
INT_RE = re.compile(r'^(integer|bigint|smallint|numeric\(\d+,0\))$')
restored = []
for d in DICT:
    col, tb, src = d['컬럼'], d['원표'].split('.')[-1], d['원컬럼']
    dtp = dbt_map.get((tb, src), '') if src else ''
    if col in X.columns and INT_RE.match(str(dtp)) and str(X[col].dtype).startswith('float'):
        v = X[col]; nn = v.dropna()
        if len(nn) and (nn == np.floor(nn)).all(): X[col] = v.astype('Int64'); restored.append(col)
for col in X.columns:  # 전부 빈 열은 문자열 타입을 명시(arrow null 타입 방지)
    if X[col].isna().all() and str(X[col].dtype) in ('object', 'null'): X[col] = X[col].astype('string')
rows = []
for d in DICT:
    col = d['컬럼']; s = X[col]
    kor, sp, db, judge = korean(col, d['원표'], d['원컬럼'])
    tb = d['원표'].split('.')[-1]
    h_all = has(s); note = ' · '.join(x for x in [d['비고'], NOTE_ADD.get(col, ''), 'DB 정수형(빈 값 허용 정수로 저장)' if col in restored else ''] if x)
    judge = judge or JUDGE_ADD.get(col, '')
    rows.append({**d, '비고': note, '한글명': kor, '명세서_설명': sp, 'DB_설명': db, '결과보고 판정': judge, 'DB 타입': dbt_map.get((tb, d['원컬럼']), '') if d['원컬럼'] else '', '데이터타입': str(s.dtype),
                 '값 있음 %(전체)': pct(h_all), '값 있음 %(국내)': pct(h_all[DOM]), '값 있음 %(수입)': pct(h_all[OVS]), '값 있음 %(검출)': pct(h_all[DET]), '결측 %(전체)': round(100 - pct(h_all), 1),
                 '고유값 수': int(s.nunique(dropna=True)), '최빈값 3개': ' | '.join(f'{k}({v:,})' for k, v in s.value_counts(dropna=True).head(3).items())})
dic = pd.DataFrame(rows); assert list(dic['컬럼']) == list(X.columns), '사전과 컬럼 순서 불일치'
Q['컬럼'] = {r['컬럼']: {'한글명': r['한글명'], '값 있음 %(전체)': r['값 있음 %(전체)'], '값 있음 %(국내)': r['값 있음 %(국내)'], '값 있음 %(수입)': r['값 있음 %(수입)'], '값 있음 %(검출)': r['값 있음 %(검출)']} for r in rows}
Q['정수형 복원 열'] = restored
tbl = pa.Table.from_pandas(X, preserve_index=False)
fields = []
for f in tbl.schema:
    r = dic[dic['컬럼'] == f.name].iloc[0]
    fields.append(f.with_metadata({'한글명': r['한글명'], '명세서_설명': r['명세서_설명'] or '', 'DB_설명': r['DB_설명'] or '', '원표': r['원표'], '원컬럼': r['원컬럼'] or '', '연결 방법': r['연결 방법']}))
meta = dict(tbl.schema.metadata or {}); meta[b'column_dictionary'] = json.dumps(dic[['컬럼', '한글명', '명세서_설명', 'DB_설명', '원표', '원컬럼', '연결 방법', '비고']].to_dict('records'), ensure_ascii=False).encode('utf-8')
meta[b'build_note'] = json.dumps({'생성': STAMP, '기준': 'listeria_body.parquet 676,537행 전부 유지(left join)', '원칙': '값은 DB 원자료 그대로. 채우기·단위 변환·집계 없음', '기상 연결': Q['기상']['방식'], '빠진 것': MISSING}, ensure_ascii=False).encode('utf-8')
tbl = pa.Table.from_arrays([tbl.column(i) for i in range(tbl.num_columns)], schema=pa.schema(fields, metadata=meta))
pq.write_table(tbl, os.path.join(SHARE, 'LIMS조인_데이터셋.parquet'), compression='snappy')
print('LIMS조인_데이터셋', X.shape, '정수형 복원', restored)

# ================= 9. 토양 pH 필지 단위(시료×필지, 집계 없음) =================
sk = pd.DataFrame({'lims_data_key': KEY, 'legal_dong_code': body['legal_dong_code'].astype('string'), 'any_year': base_date.dt.year.astype('string')})[DOM]
if HAVE('soil_ph_info'):
    soil = P('soil_ph_info'); soil['legal_dong_code'] = soil['legal_dong_code'].astype('string'); soil['_y'] = soil['any_year'].astype('string').str.strip()
    sj = sk.merge(soil, left_on=['legal_dong_code', 'any_year'], right_on=['legal_dong_code', '_y'], how='inner').drop(columns=['_y', 'any_year_x'], errors='ignore')
    sj = sj.rename(columns={'any_year_y': 'any_year'}).rename(columns={c: f'soil_{c}' for c in soil.columns if c not in ('legal_dong_code', '_y')}).rename(columns={'legal_dong_code': 'soil_legal_dong_code'})
    sj.to_parquet(os.path.join(SHARE, 'LIMS조인_토양pH_필지단위.parquet'), index=False)
    Q['연결']['soil_ph_info(필지 단위 별도 파일)'] = {'국내 시료 중 같은 동·연도 필지 있음 %': pct(sk['lims_data_key'].isin(set(sj['lims_data_key']))), '국내 검출 시료 중 %': pct(sk.loc[DET[DOM], 'lims_data_key'].isin(set(sj['lims_data_key']))),
                                                     '시료×필지 행': int(len(sj)), '시료당 필지 수 중앙값': float(sj.groupby('lims_data_key').size().median()), '시료당 필지 수 최대': int(sj.groupby('lims_data_key').size().max()), '표 행': int(len(soil)),
                                                     '시군구 단위 법정동코드(끝 00000) 국내 행': int(sk['legal_dong_code'].str.endswith('00000').fillna(False).sum())}
    print('토양 필지 단위', sj.shape)
else:
    cnt = pd.read_parquet(r"C:\Users\NT751\Desktop\천영진\리스테리아ML\data\collected\raw\raw_soil_ph_info_dong_year_counts.parquet")
    cnt['legal_dong_code'] = cnt['legal_dong_code'].astype('string'); cnt['any_year'] = cnt['any_year'].astype('string').str.strip()
    m = sk.merge(cnt, on=['legal_dong_code', 'any_year'], how='left')
    Q['연결']['soil_ph_info(필지 단위 별도 파일)'] = {'상태': '필지 행 추출본 없음(DB 복구 후 재수집). 수치는 동·연도별 필지 수 집계 파일(raw_soil_ph_info_dong_year_counts.parquet, 180,948행)로 계산', '국내 시료 중 같은 동·연도 필지 있음 %': pct(m['n_parcel'].notna()), '국내 검출 시료 중 %': pct(m.loc[DET[DOM], 'n_parcel'].notna()),
                                                     '시료×필지 행(예상)': int(m['n_parcel'].fillna(0).sum()), '시료당 필지 수 중앙값': float(m['n_parcel'].dropna().median()), '시료당 필지 수 최대': int(m['n_parcel'].max()),
                                                     '시군구 단위 법정동코드(끝 00000) 국내 행': int(sk['legal_dong_code'].str.endswith('00000').fillna(False).sum())}
    sj = pd.DataFrame(columns=['lims_data_key']); MISSING.append('soil_ph_info(토양검정 필지 표, 컬럼 6개): 추출본 없어 DB수집 폴더에도 없고 LIMS조인_토양pH_필지단위.parquet 도 못 만듦')
Q['빠진 것'] = MISSING

# ================= 10. 컬럼사전·품질 JSON 저장 =================
Q['시료'] = {'행': N, '국내': int(DOM.sum()), '수입': int(OVS.sum()), '검출': int(DET.sum()), '국내 검출': int((DOM & DET).sum()), '수입 검출': int((OVS & DET).sum()), '조인 후 행': int(len(X)), '조인 후 열': int(X.shape[1])}
with pd.ExcelWriter(os.path.join(SHARE, '컬럼사전.xlsx')) as w:
    dic.to_excel(w, sheet_name='LIMS조인_데이터셋', index=False)
    if len(sj.columns) > 1:
        soil_dic = pd.DataFrame([{'컬럼': c, '원표': 'api_data_info.soil_ph_info' if c.startswith('soil_') else 'listeria_body.parquet', '한글명': ('토양검정 ' + (rep.get('soil_ph_info', {}).get(c[5:], {}).get('한글') or spec.get('soil_ph_info', {}).get(c[5:]) or c[5:])) if c.startswith('soil_') else MANUAL.get(c, c),
                                  '연결 방법': '시료 법정동코드 + 기준일 연도 = 토양 법정동코드 + 조사 연도 (필지 여러 행 그대로)'} for c in sj.columns])
        soil_dic.to_excel(w, sheet_name='LIMS조인_토양pH_필지단위', index=False)
    pd.read_csv(os.path.join(DBD, '_수집목록.csv')).to_excel(w, sheet_name='DB수집 목록', index=False)
json.dump(Q, open(os.path.join(SHARE, '_품질진단.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
json.dump(Q, open(os.path.join(T, 'share_quality.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in Q.items() if k not in ('컬럼',)}, ensure_ascii=False, indent=1, default=str)[:6000])
print('DONE')
