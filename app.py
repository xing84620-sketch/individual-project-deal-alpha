from pathlib import Path
import json
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Deal Alpha · 并购公告效应", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    results = pd.read_csv(ROOT / 'outputs/tables/event_study_main.csv')
    features = pd.read_csv(ROOT / 'data/processed/case_features.csv')
    paths = pd.read_csv(ROOT / 'outputs/tables/car_series.csv')
    return results, features, paths

results, features, paths = load_data()
names = features.drop_duplicates('case_id').set_index('case_id')['case_name'].to_dict()
roles = {'target': '被收购方', 'acquirer': '收购方'}
st.title('Deal Alpha')
st.caption('并购公告效应 · 事件研究、历史比较与样本外回测')
st.write('选择并购案例，观察公告前后相对于市场模型的超额收益，并比较同类历史交易。')
case = st.sidebar.selectbox('并购案例', list(names), format_func=lambda key: names[key])
window = st.sidebar.selectbox('事件窗口', ['m1p1', '0p5', 'm5p5'], format_func=lambda x: {'m1p1':'公告前后 [−1,+1]', '0p5':'公告后一周 [0,+5]', 'm5p5':'扩展窗口 [−5,+5]'}[x])
lo, hi = {'m1p1':(-1,1), '0p5':(0,5), 'm5p5':(-5,5)}[window]
selected = results[results.case_id.eq(case)]
meta = features[features.case_id.eq(case)].iloc[0]
st.sidebar.caption('数据沿用原项目已采集的静态快照。交易事实与行情来源见数据说明。')
st.sidebar.download_button('下载案例特征', features.to_csv(index=False).encode('utf-8-sig'), 'case_features.csv', 'text/csv')
tabs = st.tabs(['公告效应', '历史对照', '预测与回测', '结果解读', '数据与方法'])

with tabs[0]:
    st.subheader(names[case])
    st.write(f"公告日期 {meta['announce_date']} · 交易规模 {meta['deal_value_usd_bn']:g} 十亿美元")
    if selected.empty:
        st.warning('该案例行情覆盖不足，当前无法计算事件窗口收益。')
    else:
        columns = st.columns(len(selected))
        for column, (_, row) in zip(columns, selected.iterrows()):
            with column:
                st.metric(f"{roles[row.perspective]} · {row.ticker}", f"{row['car_'+window]:+.2%}")
                p = row.get('p_'+window, np.nan)
                st.caption(f"CAR · p = {p:.3f}" if pd.notna(p) else 'CAR · 该窗口未提供 p 值')
        chart = paths[paths.case_id.eq(case) & paths.rel.between(lo, hi)].copy()
        chart['累计超额收益（%）'] = chart.groupby(['perspective','ticker'])['ar'].cumsum() * 100
        chart['公司'] = chart.ticker + ' · ' + chart.perspective.map(roles)
        st.line_chart(chart.pivot(index='rel', columns='公司', values='累计超额收益（%）'), x_label='相对公告日的交易日', y_label='CAR（%）')
        st.caption('每个窗口从零重新累加 AR，图中第一个点包含窗口首日的超额收益。')
        st.bar_chart(chart.pivot(index='rel', columns='公司', values='ar') * 100, x_label='相对交易日', y_label='每日 AR（%）')
        if selected.est_contaminated_by.notna().any():
            st.warning('该案例估计窗口存在其它事件影响，需结合稳健性结果解释。')
        st.dataframe(selected[['ticker','event_date','n_est','alpha','beta','r2','car_'+window]], hide_index=True)

with tabs[1]:
    view = results.copy()
    view['案例'] = view.case_id.map(names)
    view['视角'] = view.perspective.map(roles)
    st.subheader('各案例公告窗口收益')
    st.bar_chart(view.pivot(index='案例', columns='视角', values='car_'+window) * 100, y_label='CAR（%）')
    group = st.selectbox('比较交易特征', ['cash_pct','competing_bid','antitrust_challenge'], format_func=lambda x: {'cash_pct':'现金占比','competing_bid':'是否竞购','antitrust_challenge':'事后反垄断争议'}[x])
    st.caption('监管争议字段包含公告后信息，仅用于事后描述。')
    st.dataframe(view.groupby([group,'视角'])['car_'+window].agg(['count','mean','median']).rename(columns={'count':'观测数','mean':'平均 CAR','median':'中位 CAR'}), width='stretch')
    st.download_button('下载事件研究结果', results.to_csv(index=False).encode('utf-8-sig'), 'event_study.csv', 'text/csv')

@st.cache_data
def predict():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    # Exclude realized review duration and later regulatory outcomes from prediction.
    columns = ['ln_deal_value','cash_ratio','competing_bid']
    dataset = features[features.usable.eq(1) & features.car_m1_p1.notna()].copy()
    rows, holdouts = [], []
    for perspective, part in dataset.groupby('perspective'):
        history = part[part.is_main_case.eq(0)]
        target = part[part.is_main_case.eq(1)]
        for name in ['历史均值','Ridge','随机森林']:
            def estimate(train, test):
                if name == '历史均值':
                    return np.repeat(train.car_m1_p1.mean(), len(test))
                regressor = Ridge(alpha=10) if name == 'Ridge' else RandomForestRegressor(n_estimators=150, max_depth=2, min_samples_leaf=2, random_state=42)
                model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), regressor)
                model.fit(train[columns], train.car_m1_p1)
                return model.predict(test[columns])
            for deal in history.case_id.unique():
                train = history[history.case_id.ne(deal)]
                test = history[history.case_id.eq(deal)]
                for (_, observation), value in zip(test.iterrows(), estimate(train, test)):
                    rows.append({'case_id':deal,'视角':roles[perspective],'模型':name,'实际':observation.car_m1_p1,'预测':value})
            if len(target):
                for (_, observation), value in zip(target.iterrows(), estimate(history, target)):
                    holdouts.append({'视角':roles[perspective],'模型':name,'实际':observation.car_m1_p1,'预测':value,'训练交易数':history.case_id.nunique()})
    predictions = pd.DataFrame(rows)
    metrics = []
    for (role, model), part in predictions.groupby(['视角','模型']):
        metrics.append({'视角':role,'模型':model,'测试观测数':len(part),'MAE（百分点）':mean_absolute_error(part['实际'],part['预测'])*100,'RMSE（百分点）':np.sqrt(mean_squared_error(part['实际'],part['预测']))*100,'方向准确率':np.mean(np.sign(part['实际'])==np.sign(part['预测']))})
    return predictions, pd.DataFrame(metrics), pd.DataFrame(holdouts)

with tabs[2]:
    st.subheader('留一交易交叉验证')
    st.write('目标方和收购方分别建模。每轮留出一笔完整交易，华纳案始终独立留作样本外检验。缺失值填补与标准化仅在每轮训练集拟合。')
    st.caption('固定使用对数交易规模、现金占比、竞购三个特征。事后审查时长和监管结果不进入模型。Ridge α=10，随机森林深度=2，参数未根据华纳案调优。')
    predictions, metrics, holdouts = predict()
    st.dataframe(metrics, hide_index=True, width='stretch')
    st.subheader('华纳案样本外回测')
    display = holdouts.copy()
    for col in ['实际','预测']:
        display[col+'（%）'] = display.pop(col)*100
    display['预测误差（百分点）'] = display['预测（%）']-display['实际（%）']
    st.dataframe(display, hide_index=True, width='stretch')
    st.caption('样本较小，误差由交叉验证衡量。当前模型输出点估计，尚未校准预测区间。原数据的华纳案日期与交易信息仍需持续核验。')
    st.scatter_chart(predictions, x='实际', y='预测', color='模型')
    st.download_button('下载交叉验证预测', predictions.to_csv(index=False).encode('utf-8-sig'), 'cross_validation.csv')

with tabs[3]:
    st.subheader('量化结果解读')
    facts = []
    for _, row in selected.iterrows():
        p = row.get('p_'+window, np.nan)
        inference = '在 5% 水平达到统计显著' if pd.notna(p) and p<0.05 else '现有检验未达到 5% 显著性水平'
        facts.append(f"{roles[row.perspective]} {row.ticker} 的 CAR 为 {row['car_'+window]:+.2%}，{inference}。")
    summary = ' '.join(facts)
    st.write(summary or '当前案例数据不足。')
    st.caption('以上为根据计算结果生成的规则解读。CAR 为实际收益减去市场模型预期收益的累加，因果归因还受同期新闻和估计窗口影响。')
    prompt = f'用普通学生能理解的中文解释以下并购事件研究结果。保留数字、显著性和局限，不编造交易事实，不给买卖建议。案例：{names[case]}，窗口 [{lo},{hi}]。结果：{summary}。说明小样本、估计窗口污染及同期消息风险。'
    with st.expander('查看 AI 解读输入'):
        st.code(prompt, language=None)
    if st.button('生成 AI 解读', disabled=selected.empty):
        try:
            api_key = st.secrets.get('OPENAI_API_KEY', '')
        except FileNotFoundError:
            api_key = ''
        if not api_key:
            st.info('在 Streamlit Secrets 中配置 OPENAI_API_KEY 后可生成 AI 解读。当前规则解读可直接使用。')
        else:
            try:
                payload = json.dumps({'model':st.secrets.get('OPENAI_MODEL','gpt-4.1-mini'),'messages':[{'role':'user','content':prompt}]}).encode()
                request = urllib.request.Request('https://api.openai.com/v1/chat/completions',data=payload,headers={'Authorization':'Bearer '+st.secrets['OPENAI_API_KEY'],'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=60) as response:
                    st.write(json.load(response)['choices'][0]['message']['content'])
            except Exception:
                st.error('AI 服务请求失败，请检查密钥、模型访问权限与网络连接。')

with tabs[4]:
    st.subheader('研究方法与数据来源')
    st.write('市场模型使用 [−120,−11] 交易日估计正常收益，事件窗与估计窗不重叠。AR = 实际收益 − 正常收益，CAR 为窗口内 AR 的总和。公告日休市时顺延到交易日。')
    st.warning('本应用沿用原项目的数据快照及结果。已识别的局限包括部分字段待核、估计窗口污染、分红调整口径不一致、同期消息干扰及小样本。报告附录中的问卷为模拟数据。')
    st.dataframe(features[['case_name','perspective','ticker','announce_date','source_note','feature_status','usable']], hide_index=True)
    for filename in ['数据源评估报告.md','阶段2-事件研究完成报告.md']:
        with st.expander(filename.removesuffix('.md')):
            st.markdown((ROOT/'docs'/filename).read_text())
    with st.expander('已有分析图'):
        for image in sorted((ROOT/'outputs/figures').glob('*.png')):
            st.image(str(image))
