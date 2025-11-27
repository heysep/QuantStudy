import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DraggableText:
    """드래그 가능한 텍스트 박스 클래스 (figure 좌표 사용 - 차트 밖으로 이동 가능)"""
    def __init__(self, text_obj):
        self.text = text_obj
        self.press = None
        
    def connect(self):
        """이벤트 연결"""
        self.cidpress = self.text.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.text.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.text.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
    
    def on_press(self, event):
        """마우스 버튼을 누를 때"""
        # 텍스트 박스 영역 내에 마우스가 있는지 확인 (figure 전체 영역에서)
        contains, attrd = self.text.contains(event)
        if not contains:
            return
        
        # 현재 텍스트 위치 (figure 좌표)
        pos = self.text.get_position()
        x_pos, y_pos = pos
        
        # 화면 좌표를 figure 좌표로 변환
        trans = self.text.figure.transFigure.inverted()
        x_fig, y_fig = trans.transform((event.x, event.y))
        
        # 오프셋 저장 (마우스 위치 - 텍스트 위치)
        self.press = (x_fig - x_pos, y_fig - y_pos)
        self.text.set_alpha(0.7)  # 드래그 중 투명도 변경
        self.text.figure.canvas.draw()
    
    def on_motion(self, event):
        """마우스 이동 시"""
        if self.press is None:
            return
        
        # 화면 좌표를 figure 좌표로 변환
        trans = self.text.figure.transFigure.inverted()
        x_fig, y_fig = trans.transform((event.x, event.y))
        
        # 오프셋을 빼서 새로운 위치 계산
        x_new = x_fig - self.press[0]
        y_new = y_fig - self.press[1]
        
        # figure 좌표 범위 내로 제한 (약간의 여유 공간)
        x_new = max(-0.1, min(1.1, x_new))  # 차트 밖으로도 이동 가능하도록 여유 공간 확대
        y_new = max(-0.1, min(1.1, y_new))
        
        # 위치 업데이트
        self.text.set_position((x_new, y_new))
        self.text.figure.canvas.draw()
    
    def on_release(self, event):
        """마우스 버튼을 놓을 때"""
        if self.press is None:
            return
        
        self.press = None
        self.text.set_alpha(0.85)  # 원래 투명도로 복원
        self.text.figure.canvas.draw()
    
    def disconnect(self):
        """이벤트 연결 해제"""
        self.text.figure.canvas.mpl_disconnect(self.cidpress)
        self.text.figure.canvas.mpl_disconnect(self.cidrelease)
        self.text.figure.canvas.mpl_disconnect(self.cidmotion)

# 한글 폰트 설정 (Windows) - 강화된 버전
import matplotlib.font_manager as fm
import os

# matplotlib 폰트 캐시 클리어 (필요시)
try:
    # 폰트 캐시 파일 경로
    cache_dir = fm.get_cachedir()
    cache_file = os.path.join(cache_dir, 'fontlist-v330.json')
    if os.path.exists(cache_file):
        # 캐시는 삭제하지 않고, 폰트 매니저를 다시 빌드
        pass
except:
    pass

# Windows에서 사용 가능한 한글 폰트 목록 (우선순위 순)
korean_fonts = [
    'Malgun Gothic',      # 맑은 고딕 (Windows 기본)
    'NanumGothic',        # 나눔고딕
    'NanumBarunGothic',   # 나눔바른고딕
    'Gulim',              # 굴림
    'Gungsuh',            # 궁서
    'Batang',             # 바탕
    'Dotum',              # 돋움
    'AppleGothic'         # 맥용
]

font_found = False
selected_font = None

# 방법 1: 폰트 이름으로 직접 찾기
for font_name in korean_fonts:
    try:
        # 폰트를 직접 찾아서 설정
        font_path = None
        for font in fm.fontManager.ttflist:
            if font.name == font_name:
                font_path = font.fname
                break
        
        if font_path and os.path.exists(font_path):
            # 폰트를 직접 등록
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_name
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            font_found = True
            selected_font = font_name
            print(f"한글 폰트 설정 완료: {font_name} (경로: {font_path})")
            break
    except Exception as e:
        continue

# 방법 2: 폰트 이름이 정확히 일치하지 않는 경우 부분 매칭
if not font_found:
    available_font_names = [f.name for f in fm.fontManager.ttflist]
    for korean_font in korean_fonts:
        # 부분 매칭 시도
        matching_fonts = [f for f in available_font_names if korean_font.lower() in f.lower() or f.lower() in korean_font.lower()]
        if matching_fonts:
            try:
                plt.rcParams['font.family'] = matching_fonts[0]
                plt.rcParams['font.sans-serif'] = [matching_fonts[0]] + plt.rcParams['font.sans-serif']
                font_found = True
                selected_font = matching_fonts[0]
                print(f"한글 폰트 설정 완료 (부분 매칭): {matching_fonts[0]}")
                break
            except:
                continue

if not font_found:
    # 폰트를 찾지 못한 경우 기본 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("경고: 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    print("설치된 폰트 목록 샘플:", [f.name for f in fm.fontManager.ttflist[:20]])

# 한글 표시를 위한 추가 설정
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# 폰트가 제대로 설정되었는지 테스트
if font_found and selected_font:
    try:
        # 테스트용 더미 플롯 생성 (실제로는 표시하지 않음)
        fig_test = plt.figure(figsize=(1, 1))
        fig_test.patch.set_visible(False)
        ax_test = fig_test.add_subplot(111)
        ax_test.text(0.5, 0.5, '테스트', fontsize=10)
        plt.close(fig_test)
    except Exception as e:
        print(f"폰트 테스트 중 오류: {e}")


def get_sector_data(ticker, period='max'):
    """
    yfinance를 사용하여 섹터 ETF 데이터 가져오기 (월봉)
    
    Parameters:
    -----------
    ticker : str
        티커 심볼 (예: 'XLK', 'SPY')
    period : str
        데이터 기간 (기본값: 'max' - 가능한 최대 기간)
    
    Returns:
    --------
    pd.DataFrame
        월봉 주가 데이터프레임
    """
    stock = yf.Ticker(ticker)
    # 최대 기간 데이터 가져오기 (일반적으로 20년 이상 가능)
    df = stock.history(period=period)
    
    if df.empty:
        print(f"경고: '{ticker}'의 데이터를 가져올 수 없습니다.")
        return None
    
    # 일봉 데이터를 월봉으로 변환
    # 월봉: Open=월 첫날, High=월 최고가, Low=월 최저가, Close=월 마지막날, Volume=합계
    df_monthly = df.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return df_monthly


def calculate_relative_strength(sector_df, sp500_df, window=60):
    """
    S&P 500 대비 상대 강도(Ratio) 및 평균 회귀 분석 (Z-Score)
    
    Parameters:
    -----------
    sector_df : pd.DataFrame
        섹터 주가 데이터프레임
    sp500_df : pd.DataFrame
        S&P 500 주가 데이터프레임
    window : int
        이동평균 산출 기간 (월). 기본 60개월(5년)
    
    Returns:
    --------
    dict
        상대 강도 비율 및 Z-Score 정보
    """
    if len(sector_df) == 0 or len(sp500_df) == 0:
        return None
    
    # 공통 날짜 맞추기
    common_dates = sector_df.index.intersection(sp500_df.index)
    
    if len(common_dates) == 0:
        return None
    
    sector = sector_df.loc[common_dates]['Close']
    sp500 = sp500_df.loc[common_dates]['Close']
    
    # 1. 상대 강도 비율 (Ratio) 계산
    # 이 값이 오르면 섹터가 시장을 이기고 있는 것
    ratio = sector / sp500
    
    # 2. 비율의 이동평균 (Trend)
    ratio_ma = ratio.rolling(window=window).mean()
    
    # 3. 비율의 표준편차 (Volatility)
    ratio_std = ratio.rolling(window=window).std()
    
    # 4. Z-Score (현재 비율이 평균에서 얼마나 벗어났는가?)
    # -2.0 이하라면: 역사적 평균 대비 2표준편차 이상 저평가 (강력한 매수 신호 가능성)
    # +2.0 이상이라면: 역사적 평균 대비 고평가
    z_score = (ratio - ratio_ma) / ratio_std
    
    # 5. 추가 통계 정보
    current_z_score = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else None
    current_ratio = ratio.iloc[-1]
    current_ratio_ma = ratio_ma.iloc[-1] if not pd.isna(ratio_ma.iloc[-1]) else None
    
    # 6. 평균 회귀 정도 계산 (현재 비율이 평균 대비 얼마나 떨어져 있는가?)
    if current_ratio_ma is not None and not pd.isna(current_ratio_ma):
        deviation_from_mean = ((current_ratio / current_ratio_ma) - 1) * 100
    else:
        deviation_from_mean = None
    
    return {
        'ratio': ratio,
        'ratio_ma': ratio_ma,
        'ratio_std': ratio_std,
        'z_score': z_score,
        'current_z_score': current_z_score,
        'current_ratio': current_ratio,
        'current_ratio_ma': current_ratio_ma,
        'deviation_from_mean_pct': deviation_from_mean,
        'window': window
    }


def analyze_all_sectors():
    """
    S&P 500과 모든 섹터를 분석하고 인터랙티브 차트로 표시
    체크박스로 섹터를 켜고 끌 수 있음
    """
    # S&P 500 및 섹터 ETF 정의
    sectors = {
        'Technology (기술)': 'XLK',
        'Healthcare (헬스케어)': 'XLV',
        'Financials (금융)': 'XLF',
        'Consumer Discretionary (소비재)': 'XLY',
        'Communication Services (통신)': 'XLC',
        'Industrials (산업)': 'XLI',
        'Consumer Staples (필수소비재)': 'XLP',
        'Energy (에너지)': 'XLE',
        'Utilities (유틸리티)': 'XLU',
        'Real Estate (부동산)': 'XLRE',
        'Materials (소재)': 'XLB'
    }
    
    print("=" * 80)
    print("저평가 섹터 분석 시스템 (S&P 500 대비 이격도 분석)")
    print("=" * 80)
    print("\n데이터 수집 중...")
    
    # S&P 500 데이터 수집
    print(f"  - S&P 500 (^GSPC) 데이터 수집 중...")
    sp500_df = get_sector_data('^GSPC')
    if sp500_df is None or sp500_df.empty:
        print("오류: S&P 500 데이터를 가져올 수 없습니다.")
        return
    
    # 섹터 데이터 수집
    sector_data = {}
    for sector_name, ticker in sectors.items():
        print(f"  - {sector_name} ({ticker}) 데이터 수집 중...")
        df = get_sector_data(ticker)
        if df is not None and not df.empty:
            sector_data[sector_name] = {'ticker': ticker, 'data': df}
        else:
            print(f"    경고: {sector_name} 데이터를 가져올 수 없습니다.")
    
    if not sector_data:
        print("오류: 수집된 섹터 데이터가 없습니다.")
        return
    
    print(f"\n총 {len(sector_data)}개 섹터 데이터 수집 완료\n")
    
    # 상대 강도 비율 및 Z-Score 계산 및 출력
    print("=" * 80)
    print("S&P 500 대비 상대 강도 비율 및 평균 회귀 분석 (Z-Score)")
    print("=" * 80)
    
    relative_strength_data = {}
    z_score_summary = []
    
    for sector_name, info in sector_data.items():
        df = info['data']
        rs_data = calculate_relative_strength(df, sp500_df, window=60)
        
        if rs_data:
            relative_strength_data[sector_name] = rs_data
            
            current_z = rs_data['current_z_score']
            current_ratio = rs_data['current_ratio']
            deviation_pct = rs_data['deviation_from_mean_pct']
            
            z_score_summary.append({
                'sector': sector_name,
                'ticker': info['ticker'],
                'z_score': current_z if current_z is not None else float('inf'),
                'current_ratio': current_ratio,
                'deviation_from_mean_pct': deviation_pct
            })
            
            print(f"\n{sector_name} ({info['ticker']}):")
            print(f"  현재 상대 강도 비율: {current_ratio:.4f}")
            if rs_data['current_ratio_ma'] is not None:
                print(f"  5년 평균 비율: {rs_data['current_ratio_ma']:.4f}")
            if current_z is not None:
                print(f"  Z-Score: {current_z:.2f} (음수=저평가, 양수=고평가)")
            if deviation_pct is not None:
                print(f"  평균 대비 이격도: {deviation_pct:+.2f}%")
    
    # Z-Score 기준 저평가 순위 (낮은 Z-Score = 더 저평가)
    if z_score_summary:
        z_score_summary.sort(key=lambda x: x['z_score'] if x['z_score'] != float('inf') else 999)
        
        print("\n" + "=" * 80)
        print("Z-Score 기준 저평가 순위 (낮은 Z-Score = 더 저평가)")
        print("=" * 80)
        for i, item in enumerate(z_score_summary, 1):
            z_val = item['z_score']
            if z_val == float('inf'):
                z_str = "N/A"
            else:
                z_str = f"{z_val:.2f}"
            print(f"{i}. {item['sector']} ({item['ticker']}): Z-Score={z_str}, 비율={item['current_ratio']:.4f}")
    
    # 인터랙티브 차트 생성
    print("\n인터랙티브 차트 생성 중...")
    
    # 차트 및 체크박스 레이아웃 설정 (더 큰 크기)
    fig = plt.figure(figsize=(22, 14))
    ax = plt.subplot(111)
    # 왼쪽: 체크박스 공간, 오른쪽: 범례 공간 확보 (더 넓게)
    plt.subplots_adjust(left=0.22, bottom=0.08, right=0.75, top=0.92)
    
    # 20년치 월봉 데이터 범위 결정 (약 240개월 = 20년)
    months_20y = 12 * 20  # 20년 = 약 240개월
    
    # 모든 섹터와 S&P 500의 공통 날짜 범위 찾기
    all_dates = set(sp500_df.index)
    for sector_name, info in sector_data.items():
        all_dates = all_dates.intersection(set(info['data'].index))
    
    if not all_dates:
        print("오류: 공통 날짜가 없습니다.")
        return
    
    # 최근 20년 데이터의 시작 날짜 찾기
    all_dates_sorted = sorted(all_dates)
    if len(all_dates_sorted) >= months_20y:
        chart_start_date = all_dates_sorted[-months_20y]
        chart_dates = [d for d in all_dates_sorted if d >= chart_start_date]
    else:
        chart_start_date = all_dates_sorted[0]
        chart_dates = all_dates_sorted
    
    chart_dates_index = pd.DatetimeIndex(chart_dates)
    
    # 데이터 기간 정보 출력
    if len(chart_dates) > 0:
        print(f"\n차트 데이터 기간: {chart_dates[0].strftime('%Y-%m')} ~ {chart_dates[-1].strftime('%Y-%m')} ({len(chart_dates)}개월)")
        print(f"정규화 기준일: {chart_dates[0].strftime('%Y-%m-%d')}")
    
    # 상대 강도 비율(Ratio) 차트 데이터 준비
    sp500_chart_data = sp500_df.loc[chart_dates_index]
    
    # 각 섹터별 상대 강도 비율 라인 생성 (초기에는 모두 숨김)
    sector_lines = {}
    sector_colors = plt.cm.tab10(np.linspace(0, 1, len(sector_data)))
    
    # 모든 Ratio 값을 수집하여 Y축 범위 결정
    all_ratios = []
    
    for idx, (sector_name, info) in enumerate(sector_data.items()):
        sector_chart_data = info['data'].loc[chart_dates_index]
        
        if len(sector_chart_data) > 0 and not sector_chart_data['Close'].isna().iloc[0]:
            # 공통 날짜로 필터링
            common_chart_dates = sector_chart_data.index.intersection(sp500_chart_data.index)
            if len(common_chart_dates) > 0:
                sector_aligned = sector_chart_data.loc[common_chart_dates]['Close']
                sp500_aligned = sp500_chart_data.loc[common_chart_dates]['Close']
                
                # 상대 강도 비율 계산
                ratio = sector_aligned / sp500_aligned
                all_ratios.extend(ratio.dropna().values)
                
                line, = ax.plot(ratio.index, ratio.values,
                               label=f"{sector_name} ({info['ticker']})",
                               linewidth=2.5, color=sector_colors[idx], alpha=0.8, visible=False)
                sector_lines[sector_name] = line
    
    # Y축 범위 자동 조정 (데이터에 맞게)
    if all_ratios:
        ratio_min = min(all_ratios)
        ratio_max = max(all_ratios)
        # 여유 공간을 위해 5% 여백 추가
        y_margin = (ratio_max - ratio_min) * 0.05
        if y_margin == 0:
            y_margin = ratio_max * 0.1 if ratio_max > 0 else 0.1
        ax.set_ylim(max(0, ratio_min - y_margin), ratio_max + y_margin)
    
    # 기준선 (Ratio = 1.0) 추가
    if len(chart_dates_index) > 0:
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.6, label='기준선 (Ratio=1.0)')
    
    # 차트 설정
    # 차트 제목에 실제 데이터 기간 표시
    data_years = len(chart_dates) / 12
    ax.set_title(f'S&P 500 대비 섹터별 상대 강도 비율 (Ratio) - 월봉, 약 {data_years:.1f}년', 
                fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel('날짜', fontsize=14, fontweight='bold')
    # Y축 레이블 제거 (사용자 요청)
    ax.set_ylabel('', fontsize=14, fontweight='bold')
    # 범례를 오른쪽 중앙에 배치하여 차트와 겹치지 않도록 (폰트 크기 증가)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    
    # 축 레이블 폰트 크기 증가
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 이격도 정보 텍스트 박스 - figure 좌표 사용 (차트 밖으로도 이동 가능)
    # monospace 제거 - 한글 지원을 위해 기본 폰트 사용
    # 크기 축소: 폰트 크기와 패딩 감소
    # 초기 위치: 차트 오른쪽 하단 (figure 좌표 기준)
    info_text = fig.text(0.98, 0.02, '', 
                       fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, pad=4))
    
    # 드래그 가능하게 만들기
    draggable_info = DraggableText(info_text)
    draggable_info.connect()
    
    # 체크박스 생성
    sector_names = list(sector_data.keys())
    sector_labels = [f"{name} ({sector_data[name]['ticker']})" for name in sector_names]
    
    # 체크박스 위치 설정 (왼쪽에 배치)
    ax_check = plt.axes([0.02, 0.1, 0.18, 0.8])
    ax_check.set_xticks([])
    ax_check.set_yticks([])
    ax_check.axis('off')
    
    # 체크박스 초기 상태 (모두 체크 해제)
    check_states = [False] * len(sector_names)
    check_buttons = CheckButtons(ax_check, sector_labels, check_states)
    
    # 체크박스 라벨 폰트 크기 설정 및 한글 폰트 명시적 설정
    for label in check_buttons.labels:
        label.set_fontsize(10)
        # 한글 폰트가 설정되어 있으면 적용
        if font_found and selected_font:
            try:
                label.set_fontfamily(selected_font)
            except:
                pass
    
    # 체크박스 이벤트 핸들러
    def update_chart(label):
        """체크박스 상태 변경 시 차트 업데이트"""
        # 라벨에서 섹터 이름 추출
        sector_name = None
        for name in sector_names:
            if name in label or sector_data[name]['ticker'] in label:
                sector_name = name
                break
        
        if sector_name and sector_name in sector_lines:
            line = sector_lines[sector_name]
            line.set_visible(not line.get_visible())
            
            # 이격도 정보 업데이트
            update_info_text()
            
            plt.draw()
    
    def update_info_text():
        """표시된 섹터들의 상대 강도 비율 및 Z-Score 정보 업데이트"""
        visible_sectors = []
        for sector_name, line in sector_lines.items():
            if line.get_visible():
                if sector_name in relative_strength_data:
                    rs_info = relative_strength_data[sector_name]
                    ticker = sector_data[sector_name]['ticker']
                    current_ratio = rs_info['current_ratio']
                    z_score = rs_info['current_z_score']
                    deviation_pct = rs_info['deviation_from_mean_pct']
                    
                    # 섹터 이름에서 영어 부분만 추출 (한글 깨짐 방지)
                    # 예: "Technology (기술)" -> "Technology"
                    sector_display = sector_name.split(' (')[0] if ' (' in sector_name else sector_name
                    
                    # Z-Score 정보 포맷팅
                    if z_score is not None and not pd.isna(z_score):
                        z_str = f"Z={z_score:.2f}"
                    else:
                        z_str = "Z=N/A"
                    
                    # 평균 대비 이격도 정보 포맷팅
                    if deviation_pct is not None and not pd.isna(deviation_pct):
                        dev_str = f", Dev={deviation_pct:+.1f}%"
                    else:
                        dev_str = ""
                    
                    # 더 간결한 형식으로 표시
                    visible_sectors.append(f"{ticker}: R={current_ratio:.4f} {z_str}{dev_str}")
        
        if visible_sectors:
            info_str = "Z-Score:\n" + "\n".join(visible_sectors)
        else:
            info_str = "Z-Score:\n(Select)"
        
        info_text.set_text(info_str)
    
    # 체크박스 이벤트 연결
    check_buttons.on_clicked(update_chart)
    
    # 초기 정보 텍스트 설정
    update_info_text()
    
    plt.show()
    
    print("\n분석 완료!")
    print("\n사용 방법:")
    print("- 왼쪽 체크박스를 클릭하여 섹터를 켜고 끌 수 있습니다.")
    print("- 상대 강도 비율(Ratio) 및 Z-Score는 차트 오른쪽 상단에 표시됩니다.")
    print("- Ratio > 1.0: 섹터가 S&P 500보다 강함")
    print("- Ratio < 1.0: 섹터가 S&P 500보다 약함")
    print("- Z-Score < -2.0: 역사적 평균 대비 2표준편차 이상 저평가 (매수 신호 가능성)")
    print("- Z-Score > +2.0: 역사적 평균 대비 고평가")


if __name__ == "__main__":
    analyze_all_sectors()

# snp500 섹터에서 계속 오르는 예들만 오른다는 사실을 확인함 
# 따라서 좋지 않은 섹터는 갑자기 올랐을때 이게 계속 미래에도 오를지 생각을 잘해봐야 함함