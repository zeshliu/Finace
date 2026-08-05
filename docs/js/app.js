(() => {
  'use strict';

  const page = document.body.dataset.page;
  const state = { all: [], filtered: [], payload: null, chart: null, metadataStamp: null };
  const q = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
  const value = (number, suffix = '', digits = 2) => Number.isFinite(Number(number)) ? `${Number(number).toFixed(digits)}${suffix}` : '—';
  const signed = (number, suffix = '%') => Number.isFinite(Number(number)) ? `${Number(number) > 0 ? '+' : ''}${Number(number).toFixed(2)}${suffix}` : '—';
  const formatAmount = (number) => {
    const amount = Number(number);
    if (!Number.isFinite(amount)) return '—';
    if (amount >= 100000000) return `${(amount / 100000000).toFixed(2)}亿`;
    if (amount >= 10000) return `${(amount / 10000).toFixed(0)}万`;
    return amount.toFixed(0);
  };
  const directionClass = (number) => Number(number) > 0 ? 'up' : Number(number) < 0 ? 'down' : '';
  const formatTime = (iso) => {
    if (!iso) return '等待首次运行';
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? iso : date.toLocaleString('zh-CN', { hour12: false });
  };

  function initTheme() {
    const preferred = localStorage.getItem('a-stock-theme');
    if (preferred) document.documentElement.dataset.theme = preferred;
    q('#themeToggle')?.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      localStorage.setItem('a-stock-theme', next);
      if (state.chart) setTimeout(() => state.chart.resize(), 0);
    });
  }

  async function getJson(path) {
    const response = await fetch(`${path}?v=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`数据读取失败：${response.status}`);
    return response.json();
  }

  async function initHome() {
    try {
      const [metadata, oversold, overnight, t0Etf] = await Promise.all([
        getJson('data/metadata.json'), getJson('data/oversold_latest.json'), getJson('data/overnight_latest.json'), getJson('data/t0_etf_latest.json')
      ]);
      q('#latestTradeDate').textContent = metadata.latest_trade_date || '尚未生成';
      q('#lastUpdated').textContent = formatTime(metadata.last_updated);
      q('#dataStatus').textContent = metadata.success ? '更新成功' : '等待数据';
      q('#oversoldCount').textContent = oversold.candidates?.length ?? 0;
      q('#overnightCount').textContent = overnight.candidates?.length ?? 0;
      q('#t0EtfCount').textContent = t0Etf.candidates?.length ?? 0;
    } catch (error) {
      q('#dataStatus').textContent = '读取失败';
    }
  }

  function renderOversoldRow(item) {
    const days = item.selected_days || 1;
    return `<tr data-code="${escapeHtml(item.code)}">
      <td class="stock-name"><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.code)}</span></td>
      <td><span class="score-badge" style="background:var(--accent-subtle, rgba(39,149,170,0.15));color:var(--accent,#2795aa);">${days}天</span></td>
      <td><span class="industry-pill">${escapeHtml(item.industry || '未分类')}</span></td>
      <td><strong>${value(item.price, '', 2)}</strong></td><td class="${directionClass(item.change_pct)}">${signed(item.change_pct)}</td>
      <td><span class="score-badge">${value(item.score, '', 1)}</span></td>
      <td>${escapeHtml(item.ma_state)}</td>
      <td>${value(item.volume_ratio_5, '×', 2)}</td><td class="${directionClass(item.return_20_pct)}">${signed(item.return_20_pct)}</td><td>${value(item.rsi6, '', 1)}</td>
      <td>${(item.reasons || []).slice(0, 3).map((text) => `<span class="tag">${escapeHtml(text)}</span>`).join('')}</td>
      <td><span class="risk-text">${escapeHtml((item.risks || []).join('；'))}</span></td></tr>`;
  }

  function renderOvernightRow(item) {
    const days = item.selected_days || 1;
    return `<tr data-code="${escapeHtml(item.code)}">
      <td class="stock-name"><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.code)}</span></td>
      <td><span class="score-badge" style="background:var(--accent-subtle, rgba(39,149,170,0.15));color:var(--accent,#2795aa);">${days}天</span></td>
      <td><strong>${value(item.price, '', 2)}</strong></td><td class="${directionClass(item.change_pct)}">${signed(item.change_pct)}</td>
      <td>${value(item.high_open_rate_pct, '%', 1)}</td><td class="${directionClass(item.average_open_return_pct)}">${signed(item.average_open_return_pct)}</td>
      <td>${value(item.below_minus_1_probability_pct, '%', 1)}</td><td class="${directionClass(item.last_30_change_pct)}">${signed(item.last_30_change_pct)}</td>
      <td>${value(item.range_position_pct, '%', 1)}</td><td>${value(item.tail_score, '', 1)}</td><td><span class="score-badge">${value(item.score, '', 1)}</span></td>
      <td><span class="risk-text">${escapeHtml((item.risks || []).join('；'))}</span></td><td>${escapeHtml(formatTime(item.updated_at))}</td></tr>`;
  }

  function renderT0EtfRow(item) {
    const days = item.selected_days || 1;
    return `<tr data-code="${escapeHtml(item.code)}">
      <td class="stock-name"><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.code)} · ${escapeHtml(item.category || 'T+0 ETF')}</span></td>
      <td><span class="score-badge" style="background:var(--cool-soft);color:var(--cool);">${days}天</span></td>
      <td><strong>${value(item.price, '', 3)}</strong></td><td class="${directionClass(item.change_pct)}">${signed(item.change_pct)}</td>
      <td><span class="score-badge">${value(item.score, '', 1)}</span></td>
      <td>${escapeHtml(item.macd_state)}</td><td>${escapeHtml(item.kdj_state)}</td><td>${escapeHtml(item.ma_state)}</td>
      <td>${value(item.rsi6, '', 1)}</td><td>${value(item.atr14, '', 4)}<small class="cell-sub">${value(item.atr_pct, '%', 2)}</small></td>
      <td>${formatAmount(item.amount)}</td><td>${value(item.avg_amplitude_20, '%', 2)}</td><td>${value(item.volume_ratio, '×', 2)}</td>
      <td>${(item.reasons || []).slice(0, 3).map((text) => `<span class="tag">${escapeHtml(text)}</span>`).join('')}</td>
      <td><span class="risk-text">${escapeHtml((item.risks || []).join('；') || '暂无额外风险标记')}</span></td></tr>`;
  }

  function renderCard(item) {
    const days = item.selected_days || 1;
    const metrics = page === 'oversold'
      ? [['上榜天数', `${days}天`], ['所属板块', item.industry || '未分类'], ['5日量比', value(item.volume_ratio_5, '×', 2)], ['RSI6', value(item.rsi6, '', 1)]]
      : page === 'overnight'
        ? [['上榜天数', `${days}天`], ['历史高开率', value(item.high_open_rate_pct, '%', 1)], ['尾盘30分钟', signed(item.last_30_change_pct)], ['区间位置', value(item.range_position_pct, '%', 1)]]
        : [['上榜天数', `${days}天`], ['ETF类型', item.category || 'T+0 ETF'], ['ATR', value(item.atr14, '', 4)], ['20日振幅', value(item.avg_amplitude_20, '%', 2)]];
    const notes = page === 'overnight' ? (item.risks || []).join('；') : (item.reasons || []).join(' · ');
    return `<article class="result-card" data-code="${escapeHtml(item.code)}">
      <div class="result-card-head"><div class="stock-name"><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.code)}</span></div><span class="score-badge">${value(item.score, '', 1)}</span></div>
      <div class="result-card-price"><strong>${value(item.price, '', 2)}</strong><span class="${directionClass(item.change_pct)}">${signed(item.change_pct)}</span></div>
      <div class="mobile-metrics">${metrics.map(([label, val]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(val)}</strong></div>`).join('')}</div>
      <div class="mobile-notes">${escapeHtml(notes || '点击查看完整技术图表')}</div></article>`;
  }

  function renderResults() {
    const table = q('#resultTable');
    const cards = q('#resultCards');
    const empty = q('#emptyState');
    const items = state.filtered;
    const rowRenderer = page === 'oversold' ? renderOversoldRow : page === 'overnight' ? renderOvernightRow : renderT0EtfRow;
    table.innerHTML = items.map(rowRenderer).join('');
    cards.innerHTML = items.map(renderCard).join('');
    empty.hidden = items.length > 0;
    q('.table-wrap').hidden = items.length === 0;
    q('#candidateCount').textContent = items.length;
    if (page !== 't0_etf') {
      document.querySelectorAll('[data-code]').forEach((element) => element.addEventListener('click', () => openDetail(element.dataset.code)));
    }
  }

  function applyFilters() {
    const term = (q('#searchInput').value || '').trim().toLowerCase();
    const min = Number(q('#minPrice').value || -Infinity);
    const max = Number(q('#maxPrice').value || Infinity);
    const sort = q('#sortSelect').value;
    state.filtered = state.all.filter((item) => (!term || item.code.includes(term) || item.name.toLowerCase().includes(term)) && Number(item.price) >= min && Number(item.price) <= max);
    state.filtered.sort((a, b) => {
      if (sort === 'days-desc') {
        const diff = (Number(b.selected_days) || 1) - (Number(a.selected_days) || 1);
        return diff !== 0 ? diff : Number(b.score) - Number(a.score);
      }
      if (sort === 'price-asc') return Number(a.price) - Number(b.price);
      if (sort === 'change-desc') return Number(b.change_pct) - Number(a.change_pct);
      return Number(b.score) - Number(a.score);
    });
    renderResults();
  }

  async function loadHistoryOptions() {
    const select = q('#historyDateSelect');
    if (!select) return;
    try {
      const indexData = await getJson('data/history_index.json');
      const items = indexData[page] || [];
      select.innerHTML = '<option value="latest">最新数据</option>' + items.map((item) => {
        const dateStr = item.trade_date || item.generated_at?.slice(0, 10) || '未知日期';
        const timeStr = item.generated_at ? formatTime(item.generated_at) : '';
        return `<option value="${escapeHtml(item.filename)}">${dateStr} (${timeStr}) [${item.candidate_count || 0}只]</option>`;
      }).join('');
    } catch (_) {
      select.innerHTML = '<option value="latest">最新数据</option>';
    }
  }

  async function loadList(manual = false) {
    const button = q('#refreshButton');
    const selectedFile = q('#historyDateSelect')?.value || 'latest';
    if (manual) button?.classList.add('loading');
    try {
      const targetPath = selectedFile === 'latest' ? `data/${page}_latest.json` : `data/history/${selectedFile}`;
      const payload = await getJson(targetPath);
      state.payload = payload;
      state.all = Array.isArray(payload.candidates) ? payload.candidates : [];
      state.metadataStamp = payload.generated_at;
      const isHistory = selectedFile !== 'latest';
      q('#updateText').textContent = payload.generated_at
        ? `${isHistory ? '【往期记录】' : ''}数据时间 ${formatTime(payload.generated_at)} · 交易日 ${payload.trade_date || '—'} · 扫描 ${payload.scanned_stocks || 0} 只`
        : payload.disclaimer;
      applyFilters();
    } catch (error) {
      q('#updateText').textContent = `暂时无法读取数据：${error.message}`;
      state.all = [];
      applyFilters();
    } finally {
      button?.classList.remove('loading');
    }
  }

  function detailHeader(item) {
    const reasons = page === 'overnight' ? ['历史开盘统计与尾盘量价质量综合入选'] : (item.reasons || []);
    const gap = item.detail?.gap_stats;
    const industry = page === 'oversold'
      ? ` · ${escapeHtml(item.industry || '未分类')}`
      : page === 't0_etf' ? ` · ${escapeHtml(item.category || 'T+0 ETF')}` : '';
    const days = item.selected_days || 1;
    const datesList = (item.history_dates || []).slice(0, 6).join('、');
    return `<div class="detail-head"><div><h2 id="detailTitle">${escapeHtml(item.name)}</h2><span class="detail-code">${escapeHtml(item.code)}${industry} · ${value(item.price, '', 2)}元</span></div><div class="detail-score">${value(item.score, '', 1)}<small>综合评分</small></div></div>
      <div class="detail-notes">
        <div><strong>上榜统计</strong>已累计在筛选中上榜 <strong>${days}</strong> 天${datesList ? `（包含：${escapeHtml(datesList)} 等）` : ''}</div>
        <div><strong>入选原因</strong>${escapeHtml(reasons.join('；'))}</div>
        <div><strong>风险提示</strong>${escapeHtml((item.risks || []).join('；'))}</div>
      </div>
      ${gap ? `<div class="gap-summary"><div><span>近60日高开率</span><strong>${value(gap.high_open_rate * 100, '%', 1)}</strong></div><div><span>平均开盘收益</span><strong>${signed(gap.average_open_return * 100)}</strong></div><div><span>最大低开</span><strong>${signed(gap.max_low_open * 100)}</strong></div><div><span>有效样本</span><strong>${gap.sample_count || 0}</strong></div></div><div class="recent-gaps" title="最近20次次日开盘结果">${(gap.recent_results || []).map((entry) => `<span class="${entry.high_open ? 'positive' : 'negative'}" title="${escapeHtml(entry.next_date)} ${signed(entry.return_pct)}">${entry.high_open ? '↑' : '↓'}</span>`).join('')}</div>` : ''}`;
  }

  function renderChart(item) {
    if (!window.echarts) {
      q('#stockChart').innerHTML = '<div class="empty-state"><p>图表组件加载失败，请检查网络后刷新。</p></div>';
      return;
    }
    const rows = item.detail?.chart || [];
    const chartElement = q('#stockChart');
    state.chart?.dispose();
    state.chart = echarts.init(chartElement);
    const dates = rows.map((row) => row.date);
    const series = (key) => rows.map((row) => row[key]);
    const candlestick = rows.map((row) => [row.open, row.close, row.low, row.high]);
    const colors = getComputedStyle(document.documentElement);
    const ink = colors.getPropertyValue('--ink').trim();
    const muted = colors.getPropertyValue('--muted').trim();
    const line = colors.getPropertyValue('--line').trim();
    const up = colors.getPropertyValue('--positive').trim();
    const down = colors.getPropertyValue('--negative').trim();
    state.chart.setOption({
      animation: false,
      backgroundColor: 'transparent',
      textStyle: { color: muted, fontFamily: 'system-ui' },
      legend: [
        { top: 4, data: ['K线', 'MA5', 'MA10', 'MA20', 'MA60', '布林上轨', '布林下轨'], textStyle: { color: muted, fontSize: 10 } },
        { top: 424, data: ['MACD', 'DIF', 'DEA'], textStyle: { color: muted, fontSize: 10 } },
        { top: 522, data: ['K', 'D', 'J', 'RSI6'], textStyle: { color: muted, fontSize: 10 } }
      ],
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        borderWidth: 1,
        textStyle: { fontSize: 11 },
        formatter: (params) => {
          if (!Array.isArray(params) || !params.length) return '';
          const allowed = ['K线', 'MA5', 'MA10', 'MA20', 'MA60', '成交量'];
          const filtered = params.filter((p) => allowed.includes(p.seriesName));
          if (!filtered.length) return '';
          const date = filtered[0].name || filtered[0].axisValue || '';
          let lines = [`<div style="font-weight:bold;margin-bottom:2px;">${escapeHtml(date)}</div>`];
          filtered.forEach((p) => {
            if (p.seriesName === 'K线') {
              const d = p.data;
              if (Array.isArray(d) && d.length >= 5) {
                lines.push(`${p.marker} <strong>K线</strong> 开:${value(d[1])} 收:${value(d[2])} 低:${value(d[3])} 高:${value(d[4])}`);
              }
            } else if (p.seriesName === '成交量') {
              lines.push(`${p.marker} <strong>成交量</strong>: ${Number(p.value).toLocaleString()}`);
            } else {
              lines.push(`${p.marker} <strong>${escapeHtml(p.seriesName)}</strong>: ${value(p.value)}`);
            }
          });
          return lines.join('');
        }
      },
      axisPointer: { link: [{ xAxisIndex: 'all' }] },
      dataZoom: [{ type: 'inside', xAxisIndex: [0, 1, 2, 3], start: 45, end: 100 }, { show: true, xAxisIndex: [0, 1, 2, 3], bottom: 3, height: 18, start: 45, end: 100 }],
      grid: [{ left: 52, right: 18, top: 38, height: 285 }, { left: 52, right: 18, top: 342, height: 78 }, { left: 52, right: 18, top: 440, height: 78 }, { left: 52, right: 18, top: 538, height: 90 }],
      xAxis: [0, 1, 2, 3].map((index) => ({ type: 'category', gridIndex: index, data: dates, boundaryGap: index === 0, axisLine: { lineStyle: { color: line } }, axisLabel: { show: index === 3, color: muted }, axisTick: { show: false } })),
      yAxis: [0, 1, 2, 3].map((index) => ({ scale: true, gridIndex: index, splitNumber: 3, axisLine: { show: false }, axisLabel: { color: muted, fontSize: 9 }, splitLine: { lineStyle: { color: line, type: 'dashed' } } })),
      series: [
        { name: 'K线', type: 'candlestick', data: candlestick, xAxisIndex: 0, yAxisIndex: 0, itemStyle: { color: up, color0: down, borderColor: up, borderColor0: down } },
        ...[['MA5', 'ma5', '#e3a72f'], ['MA10', 'ma10', '#7c75d8'], ['MA20', 'ma20', '#2795aa'], ['MA60', 'ma60', '#9a7162'], ['布林上轨', 'boll_upper', '#839099'], ['布林下轨', 'boll_lower', '#839099']].map(([name, key, color]) => ({ name, type: 'line', data: series(key), xAxisIndex: 0, yAxisIndex: 0, showSymbol: false, lineStyle: { width: name.startsWith('布林') ? 1 : 1.3, color, type: name.startsWith('布林') ? 'dashed' : 'solid' } })),
        { name: '成交量', type: 'bar', data: series('volume'), xAxisIndex: 1, yAxisIndex: 1, itemStyle: { color: '#719884' } },
        { name: 'MACD', type: 'bar', data: series('macd_hist'), xAxisIndex: 2, yAxisIndex: 2, itemStyle: { color: (params) => Number(params.value) >= 0 ? up : down } },
        { name: 'DIF', type: 'line', data: series('dif'), xAxisIndex: 2, yAxisIndex: 2, showSymbol: false, lineStyle: { width: 1.2, color: '#e3a72f' } },
        { name: 'DEA', type: 'line', data: series('dea'), xAxisIndex: 2, yAxisIndex: 2, showSymbol: false, lineStyle: { width: 1.2, color: '#7c75d8' } },
        { name: 'K', type: 'line', data: series('k'), xAxisIndex: 3, yAxisIndex: 3, showSymbol: false, lineStyle: { width: 1, color: '#e3a72f' } },
        { name: 'D', type: 'line', data: series('d'), xAxisIndex: 3, yAxisIndex: 3, showSymbol: false, lineStyle: { width: 1, color: '#2795aa' } },
        { name: 'J', type: 'line', data: series('j'), xAxisIndex: 3, yAxisIndex: 3, showSymbol: false, lineStyle: { width: 1, color: '#cc27bb' } },
        { name: 'RSI6', type: 'line', data: series('rsi6'), xAxisIndex: 3, yAxisIndex: 3, showSymbol: false, lineStyle: { width: 1.4, color: ink } }
      ]
    });
  }

  function openDetail(code) {
    const item = state.all.find((candidate) => candidate.code === code);
    if (!item) return;
    q('#detailContent').innerHTML = detailHeader(item);
    q('#detailModal').hidden = false;
    document.body.style.overflow = 'hidden';
    requestAnimationFrame(() => renderChart(item));
  }

  function closeDetail() {
    q('#detailModal').hidden = true;
    document.body.style.overflow = '';
    state.chart?.dispose();
    state.chart = null;
  }

  async function checkForUpdates() {
    if (document.hidden) return;
    try {
      const metadata = await getJson('data/metadata.json');
      const section = metadata.sections?.[page];
      if (section?.last_updated && section.last_updated !== state.metadataStamp) await loadList(false);
    } catch (_) { /* 下一轮继续检查 */ }
  }

  function initList() {
    ['#searchInput', '#minPrice', '#maxPrice'].forEach((selector) => q(selector)?.addEventListener('input', applyFilters));
    q('#sortSelect')?.addEventListener('change', applyFilters);
    q('#historyDateSelect')?.addEventListener('change', () => loadList(false));
    q('#refreshButton')?.addEventListener('click', () => loadList(true));
    document.querySelectorAll('[data-close-modal]').forEach((element) => element.addEventListener('click', closeDetail));
    document.addEventListener('keydown', (event) => {
      const modal = q('#detailModal');
      if (event.key === 'Escape' && modal && !modal.hidden) closeDetail();
    });
    window.addEventListener('resize', () => state.chart?.resize());
    loadHistoryOptions().then(() => loadList());
    setInterval(checkForUpdates, 60000);
  }

  initTheme();
  if (page === 'home') initHome();
  if (page === 'oversold' || page === 'overnight' || page === 't0_etf') initList();
})();
