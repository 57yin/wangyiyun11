import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st
from collections import Counter
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import subprocess
import sys
from dateutil.relativedelta import relativedelta

# 新增机器学习相关库
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 依赖安装相关函数（保持不变）
def install_deps():
    required_packages = [
        'streamlit>=1.28.0', 'pandas', 'plotly', 'openpyxl', 'numpy', 
        'jieba', 'scikit-learn', 'wordcloud', 'matplotlib', 'seaborn',
        'statsmodels'
    ]
    try:
        import pkg_resources
        installed = {p.key for p in pkg_resources.working_set}
        print(f"正在检查并安装/升级依赖库: {', '.join(required_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *required_packages])
        print("依赖库安装/升级完成。")
    except Exception as e:
        print(f"自动安装依赖失败: {e}")
        print("请手动安装以下库: " + ", ".join(required_packages))

# 首次运行依赖检查（保持不变）
try:
    from importlib.metadata import version
    st_version = version('streamlit')
    print(f"当前 Streamlit 版本: {st_version}")
    if tuple(map(int, st_version.split('.'))) < (1, 28, 0):
        print("Streamlit 版本过低，需要升级...")
        raise ImportError("Streamlit version too old")
    # 检查scikit-learn是否安装
    import sklearn
except (ImportError, Exception):
    print("检测到缺失依赖或版本不兼容，正在尝试自动安装...")
    install_deps()
    # 重新导入所有库
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import streamlit as st
    from collections import Counter
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import re

# ---------------------- 全局配置 ----------------------
st.set_page_config(
    page_title="网易云歌单+榜单评论综合数据分析工具",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"  
)

# 自定义样式（优化数据概览样式）
custom_style = """
    <style>
        /* 全局重置与基础样式 */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* 页面背景渐变 */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        /* 隐藏默认菜单和页脚 */
        #MainMenu {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        header {visibility: hidden !important;}
        
        /* 数据概览容器样式 */
        .overview-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        /* 指标卡片样式 */
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid #1DB954;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .metric-card h4 {
            color: #6c757d;
            font-size: 14px;
            margin: 0 0 8px 0;
        }
        
        .metric-card p {
            color: #1DB954;
            font-size: 24px;
            font-weight: bold;
            margin: 0;
        }
        
        /* 推荐卡片样式 */
        .recommendation-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #1DB954;
        }
        
        .recommendation-card h4 {
            color: #1DB954;
            margin-bottom: 10px;
        }
        
        .recommendation-card p {
            margin: 5px 0;
            color: #333333;
        }
        
        .recommendation-card .match-score {
            background-color: #1DB954;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            display: inline-block;
            margin-top: 10px;
        }
        
        /* 页面标题样式 */
        .page-title {
            font-size: 28px;
            font-weight: bold;
            color: #1DB954;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .sub-title {
            font-size: 20px;
            font-weight: 600;
            color: #2d3436;
            margin: 20px 0 10px 0;
        }
        
        /* 模型评估卡片 */
        .model-metric {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 10px;
            text-align: center;
        }
        
        .model-metric h4 {
            color: #1DB954;
            margin-bottom: 5px;
        }
        
        .model-metric p {
            font-size: 24px;
            font-weight: bold;
            color: #2d3436;
        }
    </style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# 颜色配置（保持不变）
COLOR_PALETTE = {
    'primary': '#1DB954',      
    'primary_light': '#1ed760',
    'primary_dark': '#1a9e48', 
    'secondary': '#FF6B6B',    
    'accent': '#4ECDC4',       
    'background': '#F8F9FA',   
    'text': '#2d3436',         
    'light_text': '#6c757d',   
    'card_bg': '#FFFFFF',      
    'success': '#28a745',      
    'warning': '#ffc107',      
    'danger': '#dc3545',       
    'info': '#17a2b8'          
}

# 情感分析阈值（保持不变）
NEGATIVE_THRESHOLD = 0.4  
POSITIVE_THRESHOLD = 0.6  

# 数据源配置（保持不变）
TYPE_LIST_STYLE = ['流行', '热血', '00后', '华语', '伤感', '夜晚', '治愈', '放松', '感动', '安静', '民谣', '孤独', '浪漫']
TYPE_LIST_RANK = ['热歌榜', '新歌榜', '飙升榜', '原创榜']
DATA_DIR = Path(__file__).parent  
RANK_DATA_ROOT = "multi_playlist_results"  

# ---------------------- 数据加载与预处理模块（保持不变） ----------------------
def load_style_playlist_data():
    all_data = []
    found_files = []
    skipped_files = []
    for cat in TYPE_LIST_STYLE:
        file_path = DATA_DIR / f"{cat}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, index_col=0, on_bad_lines='skip')
                if df.empty:
                    skipped_files.append(f"{cat}.csv (文件为空)")
                    continue
                required_columns = ['名称', '创建日期', '播放次数', '收藏量', '转发量', '评论数', '歌单长度', 'tag1']
                if not all(col in df.columns for col in required_columns):
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    skipped_files.append(f"{cat}.csv (缺少列: {', '.join(missing_cols)})")
                    continue
                df['分类'] = cat.strip()
                all_data.append(df)
                found_files.append(cat)
            except Exception as e:
                skipped_files.append(f"{cat}.csv (读取错误: {str(e)})")
        else:
            skipped_files.append(f"{cat}.csv (文件不存在)")
    if not all_data:
        return pd.DataFrame(), found_files, skipped_files, 0
    combined_df = pd.concat(all_data, ignore_index=True)
    duplicate_cols = ['名称', '分类', '创建日期']
    before_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=duplicate_cols, keep='first')
    after_count = len(combined_df)
    dup_count = before_count - after_count
    combined_df['创建日期'] = pd.to_datetime(combined_df['创建日期'], errors='coerce')
    numeric_cols = ['播放次数', '收藏量', '转发量', '评论数', '歌单长度']
    for col in numeric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0).astype(int)
    combined_df['tag1'] = combined_df['tag1'].str.replace('nan', '').str.strip()
    combined_df['收藏播放比'] = (combined_df['收藏量'] / combined_df['播放次数'] * 100).round(4)
    combined_df['评论播放比'] = (combined_df['评论数'] / combined_df['播放次数'] * 100).round(4)
    combined_df['创建月份'] = combined_df['创建日期'].dt.to_period('M')
    
    # 为推荐系统添加的预处理
    # 1. 创建歌单特征文本（名称+分类+标签）
    combined_df['特征文本'] = combined_df['名称'] + ' ' + combined_df['分类'] + ' ' + combined_df['tag1']
    # 2. 处理缺失值
    combined_df['特征文本'] = combined_df['特征文本'].fillna('')
    
    return combined_df, found_files, skipped_files, dup_count

def load_rank_comment_data():
    all_rank_data = []
    found_ranks = []
    skipped_ranks = []
    for rank_name in TYPE_LIST_RANK:
        rank_dir = DATA_DIR / RANK_DATA_ROOT / rank_name
        dataset_path = rank_dir / f"{rank_name}_dataset.csv"
        comment_dir = rank_dir / "detailed_comments"  
        if dataset_path.exists():
            try:
                df = pd.read_csv(dataset_path, on_bad_lines='skip', encoding='utf-8-sig')
                if df.empty:
                    skipped_ranks.append(f"{rank_name} (文件为空)")
                    continue
                required_columns = ['歌曲ID', '歌曲名称', '歌手', '评论总数', '积极评论数', '消极评论数', '中立评论数', '积极评论占比', '消极评论占比', '中立评论占比', '高频字眼']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    skipped_ranks.append(f"{rank_name} (缺少列: {', '.join(missing_cols)})")
                    continue
                df['评论文件路径'] = df['歌曲ID'].apply(
                    lambda song_id: str(comment_dir / f"comments_{song_id}.csv") if (comment_dir / f"comments_{song_id}.csv").exists() else ""
                )
                df['榜单类型'] = rank_name.strip()
                all_rank_data.append(df)
                found_ranks.append(rank_name)
            except Exception as e:
                skipped_ranks.append(f"{rank_name} (读取错误: {str(e)})")
        else:
            skipped_ranks.append(f"{rank_name} (数据集文件不存在)")
    if not all_rank_data:
        return pd.DataFrame(), found_ranks, skipped_ranks
    combined_df = pd.concat(all_rank_data, ignore_index=True)
    numeric_cols = ['评论总数', '积极评论数', '消极评论数', '中立评论数', '积极评论占比', '消极评论占比', '中立评论占比']
    for col in numeric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
    combined_df['情感倾向'] = combined_df.apply(
        lambda x: '积极' if x['积极评论占比'] > x['消极评论占比'] and x['积极评论占比'] > 0.3 
                  else '消极' if x['消极评论占比'] > x['积极评论占比'] and x['消极评论占比'] > 0.3
                  else '中立', axis=1
    )
    
    # 为推荐系统添加的预处理
    # 1. 创建歌曲特征文本（名称+歌手+榜单类型+高频字眼+情感倾向）
    combined_df['特征文本'] = combined_df['歌曲名称'] + ' ' + combined_df['歌手'] + ' ' + combined_df['榜单类型'] + ' ' + combined_df['高频字眼'].fillna('') + ' ' + combined_df['情感倾向']
    # 2. 处理缺失值
    combined_df['特征文本'] = combined_df['特征文本'].fillna('')
    
    return combined_df, found_ranks, skipped_ranks

def load_all_data(selected_data_source):
    if selected_data_source == "13类风格歌单数据":
        df, found, skipped, dup_count = load_style_playlist_data()
        load_summary = {
            "data_type": "风格歌单",
            "found_count": len(found),
            "total_count": len(TYPE_LIST_STYLE),
            "found_items": found,
            "skipped_items": skipped,
            "dup_count": dup_count
        }
    else:
        df, found, skipped = load_rank_comment_data()
        load_summary = {
            "data_type": "榜单评论",
            "found_count": len(found),
            "total_count": len(TYPE_LIST_RANK),
            "found_items": found,
            "skipped_items": skipped,
            "dup_count": 0
        }
    return df, load_summary

# ---------------------- 优化后的数据概览卡片 ----------------------
def display_data_overview(df, data_source):
    st.markdown('<div class="sub-title">📈 数据概览</div>', unsafe_allow_html=True)
    
    # 根据屏幕宽度自动调整列数
    cols_per_row = 3  # 默认每行3列
    if data_source == "13类风格歌单数据":
        metrics = [
            ("📊 总歌单数量", len(df), COLOR_PALETTE['primary']),
            ("▶️ 总播放次数", f"{df['播放次数'].sum():,}", COLOR_PALETTE['secondary']),
            ("❤️ 总收藏量", f"{df['收藏量'].sum():,}", COLOR_PALETTE['accent']),
            ("🎵 平均歌单长度", f"{df['歌单长度'].mean():.1f}", COLOR_PALETTE['primary_dark']),
            ("💬 总评论数", f"{df['评论数'].sum():,}", COLOR_PALETTE['warning']),
            ("🔄 总转发量", f"{df['转发量'].sum():,}", COLOR_PALETTE['danger']),
            ("📈 平均收藏播放比(%)", f"{df['收藏播放比'].mean():.2f}", COLOR_PALETTE['info'])
        ]
    else:
        metrics = [
            ("🎵 总歌曲数量", len(df), COLOR_PALETTE['primary']),
            ("💬 总评论数", f"{df['评论总数'].sum():,}", COLOR_PALETTE['secondary']),
            ("😊 平均积极评论占比(%)", f"{df['积极评论占比'].mean() * 100:.2f}", COLOR_PALETTE['accent']),
            ("👍 积极情感歌曲数", len(df[df['情感倾向'] == '积极']), COLOR_PALETTE['primary_dark']),
            ("👎 消极情感歌曲数", len(df[df['情感倾向'] == '消极']), COLOR_PALETTE['warning']),
            ("😐 中立情感歌曲数", len(df[df['情感倾向'] == '中立']), COLOR_PALETTE['danger']),
            ("📊 平均单首歌曲评论数", f"{df['评论总数'].mean():.1f}", COLOR_PALETTE['info'])
        ]
    
    # 分批次显示卡片
    for i in range(0, len(metrics), cols_per_row):
        row_metrics = metrics[i:i+cols_per_row]
        cols = st.columns(len(row_metrics))
        
        for col, (title, value, color) in zip(cols, row_metrics):
            with col:
                col.markdown(f"""
                <div style="background-color: #ffffff; border-radius: 10px; padding: 20px; border-left: 5px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <div style="color: #666; font-size: 14px; margin-bottom: 8px;">{title}</div>
                    <div style="color: {color}; font-size: 28px; font-weight: 600;">{value}</div>
                </div>
                """, unsafe_allow_html=True)

# ---------------------- 新增：聚类分析模块 ----------------------
def perform_clustering_analysis(df, data_type):
    """执行聚类分析"""
    st.markdown('<div class="sub-title">🧩 聚类分析结果</div>', unsafe_allow_html=True)
    
    if data_type == "风格歌单":
        # 选择聚类特征
        features = ['播放次数', '收藏量', '评论数', '歌单长度', '收藏播放比', '评论播放比']
        cluster_df = df[features].copy()
        
        # 数据标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)
        
        # 选择聚类数量
        n_clusters = st.slider("选择聚类数量", min_value=2, max_value=10, value=4)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['聚类标签'] = kmeans.fit_predict(scaled_data)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(scaled_data, df['聚类标签'])
        
        # 显示聚类统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="model-metric">
                <h4>聚类数量</h4>
                <p>{n_clusters}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="model-metric">
                <h4>轮廓系数</h4>
                <p>{silhouette_avg:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="model-metric">
                <h4>总数据量</h4>
                <p>{len(df)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 聚类分布饼图
        cluster_counts = df['聚类标签'].value_counts().sort_index()
        fig = px.pie(
            values=cluster_counts.values,
            names=[f'聚类 {i}' for i in cluster_counts.index],
            title='各聚类数据分布',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, width='stretch')
        
        # PCA降维可视化
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        fig = px.scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            color=df['聚类标签'].astype(str),
            title=f'PCA降维聚类可视化 (解释方差: {pca.explained_variance_ratio_.sum():.2%})',
            labels={'x': 'PCA维度1', 'y': 'PCA维度2', 'color': '聚类标签'},
            hover_data={
                '名称': df['名称'],
                '分类': df['分类'],
                '播放次数': df['播放次数'],
                '收藏量': df['收藏量']
            },
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig, width='stretch')
        
        # 各聚类特征分析
        st.markdown("### 各聚类特征分析")
        cluster_analysis = df.groupby('聚类标签')[features].mean().round(2)
        
        # 热力图展示各聚类特征
        fig = px.imshow(
            cluster_analysis.T,
            title='各聚类特征热力图',
            labels=dict(x="聚类标签", y="特征", color="平均值"),
            x=[f'聚类 {i}' for i in cluster_analysis.index],
            y=features,
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig, width='stretch')
        
        # 显示各聚类详细信息
        for cluster_id in sorted(df['聚类标签'].unique()):
            cluster_data = df[df['聚类标签'] == cluster_id]
            with st.expander(f"聚类 {cluster_id} 详情 (共{len(cluster_data)}个歌单)", expanded=False):
                # 显示聚类特征统计
                st.dataframe(cluster_data[features].describe().round(2))
                
                # 显示聚类中的歌单示例
                st.markdown("#### 歌单示例")
                sample_data = cluster_data[['名称', '分类', '播放次数', '收藏量', '评论数']].head(10)
                st.dataframe(sample_data)
        
    else:  # 榜单评论数据聚类
        # 选择聚类特征
        features = ['评论总数', '积极评论数', '消极评论数', '中立评论数', '积极评论占比', '消极评论占比']
        cluster_df = df[features].copy()
        
        # 数据标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)
        
        # 选择聚类数量
        n_clusters = st.slider("选择聚类数量", min_value=2, max_value=8, value=3)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['聚类标签'] = kmeans.fit_predict(scaled_data)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(scaled_data, df['聚类标签'])
        
        # 显示聚类统计信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="model-metric">
                <h4>聚类数量</h4>
                <p>{n_clusters}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="model-metric">
                <h4>轮廓系数</h4>
                <p>{silhouette_avg:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="model-metric">
                <h4>总数据量</h4>
                <p>{len(df)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 聚类与情感倾向交叉分析
        cluster_sentiment = pd.crosstab(df['聚类标签'], df['情感倾向'])
        fig = px.imshow(
            cluster_sentiment,
            title='聚类与情感倾向交叉分析',
            labels=dict(x="情感倾向", y="聚类标签", color="歌曲数量"),
            x=cluster_sentiment.columns,
            y=[f'聚类 {i}' for i in cluster_sentiment.index],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, width='stretch')
        
        # PCA降维可视化
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        fig = px.scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            color=df['聚类标签'].astype(str),
            symbol=df['情感倾向'],
            title=f'PCA降维聚类可视化 (解释方差: {pca.explained_variance_ratio_.sum():.2%})',
            labels={'x': 'PCA维度1', 'y': 'PCA维度2', 'color': '聚类标签', 'symbol': '情感倾向'},
            hover_data={
                '歌曲名称': df['歌曲名称'],
                '歌手': df['歌手'],
                '评论总数': df['评论总数'],
                '积极评论占比': df['积极评论占比']
            },
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig, width='stretch')
        
        # 各聚类特征分析
        st.markdown("### 各聚类特征分析")
        cluster_analysis = df.groupby('聚类标签')[features].mean().round(2)
        
        # 雷达图展示各聚类特征
        for cluster_id in sorted(df['聚类标签'].unique()):
            cluster_mean = cluster_analysis.loc[cluster_id]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=cluster_mean.values,
                theta=features,
                fill='toself',
                name=f'聚类 {cluster_id}'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title=f'聚类 {cluster_id} 特征雷达图',
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')

# ---------------------- 新增：随机森林预测模块 ----------------------
def perform_random_forest_prediction(df, data_type):
    """执行随机森林预测分析"""
    st.markdown('<div class="sub-title">🌳 随机森林预测分析</div>', unsafe_allow_html=True)
    
    if data_type == "风格歌单":
        # 定义预测任务
        prediction_task = st.selectbox(
            "选择预测任务",
            ["预测歌单受欢迎程度", "预测歌单分类"]
        )
        
        if prediction_task == "预测歌单受欢迎程度":
            # 定义受欢迎程度标签（基于播放次数分位数）
            df['受欢迎程度'] = pd.qcut(
                df['播放次数'], 
                q=3, 
                labels=['低', '中', '高']
            )
            
            # 特征选择
            features = ['收藏量', '评论数', '转发量', '歌单长度', '收藏播放比', '评论播放比']
            X = df[features].fillna(0)
            y = df['受欢迎程度']
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 训练随机森林分类器
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            
            # 预测与评估
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(rf, X, y, cv=5)
            
            # 显示模型评估指标
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="model-metric">
                    <h4>测试集准确率</h4>
                    <p>{accuracy:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="model-metric">
                    <h4>5折交叉验证均值</h4>
                    <p>{cv_scores.mean():.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="model-metric">
                    <h4>特征数量</h4>
                    <p>{len(features)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                '特征': features,
                '重要性': rf.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='重要性',
                y='特征',
                orientation='h',
                title='特征重要性排名',
                color='重要性',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, width='stretch')
            
            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm,
                title='混淆矩阵',
                labels=dict(x="预测标签", y="真实标签", color="数量"),
                x=['低', '中', '高'],
                y=['低', '中', '高'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, width='stretch')
            
            # 分类报告
            st.markdown("### 分类报告")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df)
            
            # 预测示例
            st.markdown("### 预测示例")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                input_fav = st.number_input("收藏量", value=10000)
            with col2:
                input_comment = st.number_input("评论数", value=500)
            with col3:
                input_share = st.number_input("转发量", value=100)
            with col4:
                input_length = st.number_input("歌单长度", value=50)
            
            if st.button("预测受欢迎程度"):
                input_data = pd.DataFrame({
                    '收藏量': [input_fav],
                    '评论数': [input_comment],
                    '转发量': [input_share],
                    '歌单长度': [input_length],
                    '收藏播放比': [input_fav / 100000 * 100],  # 假设播放次数为10万
                    '评论播放比': [input_comment / 100000 * 100]
                })
                
                prediction = rf.predict(input_data)[0]
                prediction_proba = rf.predict_proba(input_data)[0]
                
                st.success(f"预测结果：{prediction}受欢迎程度")
                st.markdown("预测概率：")
                prob_df = pd.DataFrame({
                    '受欢迎程度': ['低', '中', '高'],
                    '概率': prediction_proba
                })
                fig = px.bar(prob_df, x='受欢迎程度', y='概率', title='预测概率分布')
                st.plotly_chart(fig, width='stretch')
        
        elif prediction_task == "预测歌单分类":
            # 特征选择
            features = ['播放次数', '收藏量', '评论数', '转发量', '歌单长度', '收藏播放比', '评论播放比']
            X = df[features].fillna(0)
            y = df['分类']
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 训练随机森林分类器
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            
            # 预测与评估
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # 显示模型评估指标
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="model-metric">
                    <h4>测试集准确率</h4>
                    <p>{accuracy:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="model-metric">
                    <h4>类别数量</h4>
                    <p>{df['分类'].nunique()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                '特征': features,
                '重要性': rf.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='重要性',
                y='特征',
                orientation='h',
                title='特征重要性排名',
                color='重要性',
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig, width='stretch')
    
    else:  # 榜单评论数据预测
        # 定义预测任务
        prediction_task = st.selectbox(
            "选择预测任务",
            ["预测歌曲情感倾向", "预测积极评论占比"]
        )
        
        if prediction_task == "预测歌曲情感倾向":
            # 特征选择
            features = ['评论总数', '积极评论数', '消极评论数', '中立评论数', '消极评论占比', '中立评论占比']
            X = df[features].fillna(0)
            y = df['情感倾向']
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 训练随机森林分类器
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            
            # 预测与评估
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # 显示模型评估指标
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="model-metric">
                    <h4>测试集准确率</h4>
                    <p>{accuracy:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="model-metric">
                    <h4>情感类别数</h4>
                    <p>{df['情感倾向'].nunique()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                '特征': features,
                '重要性': rf.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='重要性',
                y='特征',
                orientation='h',
                title='特征重要性排名',
                color='重要性',
                color_continuous_scale='cividis'
            )
            st.plotly_chart(fig, width='stretch')
            
            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm,
                title='混淆矩阵',
                labels=dict(x="预测标签", y="真实标签", color="数量"),
                x=df['情感倾向'].unique(),
                y=df['情感倾向'].unique(),
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, width='stretch')
        
        else:  # 预测积极评论占比
            # 特征选择
            features = ['评论总数', '消极评论数', '中立评论数', '消极评论占比', '中立评论占比']
            X = df[features].fillna(0)
            y = df['积极评论占比'].fillna(0)
            
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 训练随机森林回归器
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            
            # 预测与评估
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            # 显示模型评估指标
            st.markdown(f"""
            <div class="model-metric">
                <h4>R²分数</h4>
                <p>{r2:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                '特征': features,
                '重要性': rf.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='重要性',
                y='特征',
                orientation='h',
                title='特征重要性排名',
                color='重要性',
                color_continuous_scale='inferno'
            )
            st.plotly_chart(fig, width='stretch')

# ---------------------- 高级可视化模块（完整代码） ----------------------
def plot_style_playlist_visualizations(df):
    """13类风格歌单可视化"""
    if df.empty:
        st.warning("没有可供可视化的风格歌单数据")
        return
    
    st.markdown('<div class="sub-title">🎯 风格歌单深度分析</div>', unsafe_allow_html=True)
    
    # 创建标签页（新增机器学习标签页）
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        '分类分析', '时间趋势', '相关性分析', '高级洞察', '智能推荐',
        '聚类分析', '预测分析'  # 新增标签页
    ])
    
    # Tab 1: 分类分析
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # 各分类歌单数量
            cat_counts = df['分类'].value_counts()
            fig = px.bar(
                x=cat_counts.index,
                y=cat_counts.values,
                title='各分类歌单数量分布',
                labels={'x': '分类', 'y': '歌单数量'},
                color=cat_counts.values,
                color_continuous_scale='Reds',
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
    
        with col2:
            # 各分类平均播放量
            avg_play = df.groupby('分类')['播放次数'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=avg_play.index,
                y=avg_play.values,
                title='各分类平均播放量',
                labels={'x': '分类', 'y': '平均播放次数'},
                color=avg_play.values,
                color_continuous_scale='Blues',
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        # 各分类综合指标雷达图
        st.markdown("### 各分类综合表现对比")
        # 修改点：获取所有分类，不再限制前6类
        all_categories = df['分类'].unique()  
        cat_metrics = df[df['分类'].isin(all_categories)].groupby('分类').agg({
            '播放次数': 'mean',
            '收藏量': 'mean',
            '评论数': 'mean',
            '歌单长度': 'mean'
        }).reset_index()
    
        # 数据标准化
        for col in ['播放次数', '收藏量', '评论数', '歌单长度']:
            cat_metrics[col] = (cat_metrics[col] - cat_metrics[col].min()) / (cat_metrics[col].max() - cat_metrics[col].min())
    
        fig = go.Figure()
        for _, row in cat_metrics.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['播放次数'], row['收藏量'], row['评论数'], row['歌单长度']],
                theta=['播放次数', '收藏量', '评论数', '歌单长度'],
                name=row['分类']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, width='stretch')

    
    # Tab 2: 时间趋势
    with tab2:
        # 按月份统计筛选后歌单的创建数量（整体趋势）
        monthly_trend = df.groupby('创建月份').size().reset_index(name='歌单数量')
        monthly_trend['创建月份'] = monthly_trend['创建月份'].astype(str)
    
        fig = px.line(
            monthly_trend,
            x='创建月份',
            y='歌单数量',
            title='筛选后歌单创建时间趋势',
            labels={'创建月份': '月份', '歌单数量': '新增歌单数量'},
            template='plotly_white',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
        
        # 近6个月各分类歌单增长情况（基于筛选后的数据）
        if not df.empty and not df['创建日期'].isna().all():
            # 1. 从筛选后的数据中获取最新月份（Period类型）
            latest_month_period = df['创建日期'].dt.to_period('M').max()
            latest_month_dt = latest_month_period.to_timestamp()  # 转为datetime用于计算
            
            # 2. 计算筛选后数据的"近6个月"起始时间
            six_months_ago_dt = latest_month_dt - relativedelta(months=6)
            six_months_ago_period = six_months_ago_dt.to_period('M')  # 转回Period用于筛选
            
            # 3. 从筛选后的数据中，再筛选近6个月的记录
            recent_data = df[df['创建月份'].between(six_months_ago_period, latest_month_period)]
        
            if len(recent_data) > 0:
                monthly_cat = recent_data.groupby(['创建月份', '分类']).size().reset_index(name='歌单数量')
                monthly_cat['创建月份'] = monthly_cat['创建月份'].astype(str)
                
                fig = px.area(
                    monthly_cat,
                    x='创建月份',
                    y='歌单数量',
                    color='分类',
                    title='筛选后近6个月各分类歌单增长趋势',  # 标题明确标注"筛选后"
                    labels={'创建月份': '月份', '歌单数量': '歌单数量'},
                    template='plotly_white'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("筛选后的数据中，近6个月内没有找到歌单数据")
        else:
            st.info("筛选后的数据中没有有效日期数据，无法展示近6个月趋势")
    
    # Tab 3: 相关性分析
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # 播放量vs收藏量散点图
            fig = px.scatter(
                df,
                x='播放次数',
                y='收藏量',
                color='分类',
                size='歌单长度',
                hover_data=['名称', '创建日期'],
                title='播放量 vs 收藏量',
                labels={'播放次数': '播放次数', '收藏量': '收藏量'},
                opacity=0.7,
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # 播放量vs评论数散点图
            fig = px.scatter(
                df,
                x='播放次数',
                y='评论数',
                color='分类',
                size='收藏量',
                hover_data=['名称', '创建日期'],
                title='播放量 vs 评论数',
                labels={'播放次数': '播放次数', '评论数': '评论数'},
                opacity=0.7,
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        # 数值特征相关性热力图
        numeric_features = ['播放次数', '收藏量', '转发量', '评论数', '歌单长度', '收藏播放比', '评论播放比']
        corr_matrix = df[numeric_features].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='特征相关性热力图',
            labels=dict(color='相关系数'),
            x=numeric_features,
            y=numeric_features,
            color_continuous_scale='RdBu_r',
            template='plotly_white'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    # Tab 4: 高级洞察
    with tab4:
        # Top 10 高收藏播放比歌单
        st.markdown("### Top 10 高收藏率歌单")
        # 过滤掉播放次数为0的歌单，避免除以零错误
        high_fav_ratio_df = (
            df[df['播放次数'] > 1000]
            .sort_values('收藏播放比', ascending=False)
            .drop_duplicates(subset='名称', keep='first')
            .nlargest(10, '收藏播放比')
            [['名称', '分类', '播放次数', '收藏量', '收藏播放比', '创建日期']]
        )
        
        fig = px.bar(
            high_fav_ratio_df,
            x='名称',
            y='收藏播放比',
            color='分类',
            title='收藏率最高的10个歌单 (收藏量/播放量%)',
            labels={'名称': '歌单名称', '收藏播放比': '收藏率(%)'},
            template='plotly_white',
            hover_data=['播放次数', '收藏量', '创建日期'],
            category_orders={"名称": high_fav_ratio_df.sort_values('收藏播放比', ascending=False)['名称'].tolist()}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
        
        # 歌单长度分布
        st.markdown("### 歌单长度分布")
        # 计算合适的 nbins 值，这里假设歌单长度最大可能到 10000，你可根据实际数据调整
        max_playlist_length = df['歌单长度'].max() if not df.empty else 10000
        nbins = int(max_playlist_length / 10)  
        fig = px.histogram(
            df,
            x='歌单长度',
            nbins=nbins,
            title='歌单长度分布',
            labels={'歌单长度': '歌曲数量', 'count': '歌单数量'},
            color_discrete_sequence=['#4ECDC4'],
            template='plotly_white'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
        
        # 标签云（使用Plotly的条形图模拟）
        st.markdown("### 热门标签分析")
        if 'tag1' in df.columns:
            # 过滤掉空标签
            tag_counts = df['tag1'].replace('', pd.NA).dropna().value_counts().head(15)
            if not tag_counts.empty:
                fig = px.bar(
                    x=tag_counts.values,
                    y=tag_counts.index,
                    orientation='h',
                    title='热门标签 Top 15',
                    labels={'x': '出现次数', 'y': '标签'},
                    color=tag_counts.values,
                    color_continuous_scale='Oranges',
                    template='plotly_white'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("没有找到有效的标签数据。")
        else:
            st.warning("数据中缺少 'tag1' 列，无法进行热门标签分析。")
    
    # Tab 5: 智能推荐
    with tab5:
        st.markdown("### 🎯 歌单智能推荐系统")
        
        # 创建推荐模型
        @st.cache_resource
        def create_playlist_recommendation_model(df):
            """创建歌单推荐模型"""
            # 准备文本数据
            texts = df['特征文本'].tolist()
            
            # 创建TF-IDF向量izer
            vectorizer = TfidfVectorizer(
                tokenizer=jieba.cut,
                stop_words=['的', '了', '是', '我', '在', '和', '也', '都', '很', '就', '还', '有'],
                max_features=5000
            )
            
            # 转换文本为TF-IDF矩阵
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            return vectorizer, tfidf_matrix
        
        # 获取推荐模型
        if not df.empty and '特征文本' in df.columns:
            with st.spinner("正在初始化推荐模型..."):
                vectorizer, tfidf_matrix = create_playlist_recommendation_model(df)
            
            # 用户输入
            st.markdown("#### 请输入你的需求")
            user_query = st.text_input("例如：我想听伤感的华语歌曲，适合夜晚听的", "")
            
            # 推荐参数设置
            col1, col2 = st.columns(2)
            with col1:
                min_play_count = st.number_input("最低播放次数", min_value=0, value=10000)
            with col2:
                recommendation_count = st.number_input("推荐数量", min_value=1, max_value=20, value=5)
            
            # 执行推荐
            if st.button("获取推荐"):
                if not user_query:
                    st.warning("请输入你的音乐需求")
                else:
                    with st.spinner("正在为你推荐歌单..."):
                        # 处理用户查询
                        query_vector = vectorizer.transform([user_query])
                        
                        # 计算相似度
                        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
                        
                        # 创建相似度DataFrame
                        similarity_df = pd.DataFrame({
                            'index': range(len(similarities)),
                            'similarity': similarities
                        })
                        
                        # 筛选相似度高的歌单
                        similarity_df = similarity_df[similarity_df['similarity'] > 0.1].sort_values('similarity', ascending=False)
                        
                        # 获取推荐结果
                        recommendations = []
                        for _, row in similarity_df.iterrows():
                            if len(recommendations) >= recommendation_count:
                                break
                                
                            playlist_idx = int(row['index'])
                            playlist = df.iloc[playlist_idx]
                            
                            # 过滤条件
                            if playlist['播放次数'] >= min_play_count:
                                recommendations.append({
                                    'index': playlist_idx,
                                    'similarity': row['similarity'],
                                    'playlist': playlist
                                })
                        
                        # 显示推荐结果
                        if recommendations:
                            st.markdown(f"#### 为你找到 {len(recommendations)} 个符合条件的歌单：")
                            
                            for rec in recommendations:
                                playlist = rec['playlist']
                                similarity_score = rec['similarity']
                                
                                # 生成匹配理由
                                match_reasons = []
                                query_words = set(jieba.cut(user_query))
                                playlist_words = set(jieba.cut(playlist['特征文本']))
                                common_words = query_words.intersection(playlist_words)
                                
                                if common_words:
                                    match_reasons.append(f"包含关键词：{', '.join(common_words)}")
                                if playlist['收藏播放比'] > df['收藏播放比'].mean():
                                    match_reasons.append("收藏率高于平均水平")
                                if playlist['评论播放比'] > df['评论播放比'].mean():
                                    match_reasons.append("互动率较高")
                                
                                # 显示推荐卡片
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>{playlist['名称']}</h4>
                                    <p><strong>分类：</strong>{playlist['分类']}</p>
                                    <p><strong>播放次数：</strong>{playlist['播放次数']:,}</p>
                                    <p><strong>收藏量：</strong>{playlist['收藏量']:,}</p>
                                    <p><strong>歌单长度：</strong>{playlist['歌单长度']}首歌曲</p>
                                    <p><strong>匹配理由：</strong>{' | '.join(match_reasons) if match_reasons else '综合特征匹配'}</p>
                                    <span class="match-score">匹配度：{similarity_score:.2%}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("没有找到完全匹配的歌单，建议尝试调整搜索关键词或降低播放次数要求")
        else:
            st.warning("数据不足，无法创建推荐模型")
    
    # Tab 6: 聚类分析（新增）
    with tab6:
        perform_clustering_analysis(df, "风格歌单")
    
    # Tab 7: 预测分析（新增）
    with tab7:
        perform_random_forest_prediction(df, "风格歌单")

def plot_rank_comment_visualizations(df):
    """4类榜单歌曲评论可视化"""
    if df.empty:
        st.warning("没有可供可视化的榜单评论数据")
        return
    
    st.markdown('<div class="sub-title">🎯 榜单歌曲评论深度分析</div>', unsafe_allow_html=True)
    
    # 创建标签页（新增机器学习标签页）
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        '情感分析', '评论量分析', '高频词分析', '高级洞察', '智能推荐',
        '聚类分析', '预测分析'  # 新增标签页
    ])
    
    # Tab 1: 情感分析
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # 各榜单情感倾向分布
            sentiment_counts = df.groupby(['榜单类型', '情感倾向']).size().reset_index(name='歌曲数量')
            fig = px.bar(
                sentiment_counts,
                x='榜单类型',
                y='歌曲数量',
                color='情感倾向',
                barmode='group',
                title='各榜单歌曲情感倾向分布',
                labels={'歌曲数量': '歌曲数量', '榜单类型': '榜单类型'},
                color_discrete_map={'积极': '#2ECC40', '消极': '#FF4136', '中立': '#AAAAAA'},
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # 各榜单平均情感占比
            avg_sentiment = df.groupby('榜单类型').agg({
                '积极评论占比': 'mean',
                '消极评论占比': 'mean',
                '中立评论占比': 'mean'
            }).reset_index()
            
            fig = px.line(
                avg_sentiment,
                x='榜单类型',
                y=['积极评论占比', '消极评论占比', '中立评论占比'],
                title='各榜单平均情感占比趋势',
                labels={'value': '平均占比', 'variable': '情感类型'},
                template='plotly_white',
                markers=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        # 情感得分分布箱线图
        st.markdown("### 各榜单情感得分分布")
        fig = px.box(
            df,
            x='榜单类型',
            y=['积极评论占比', '消极评论占比'],
            title='各榜单情感得分分布箱线图',
            labels={'value': '情感占比', 'variable': '情感类型'},
            color_discrete_map={'积极评论占比': '#2ECC40', '消极评论占比': '#FF4136'},
            template='plotly_white'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    # Tab 2: 评论量分析
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # 各榜单评论总数分布
            fig = px.histogram(
                df,
                x='评论总数',
                color='榜单类型',
                title='各榜单歌曲评论总数分布',
                labels={'评论总数': '评论总数', 'count': '歌曲数量'},
                template='plotly_white',
                opacity=0.7
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # 各榜单平均评论数
            avg_comments = df.groupby('榜单类型')['评论总数'].agg(['mean', 'median', 'max']).reset_index()
            fig = px.bar(
                avg_comments,
                x='榜单类型',
                y=['mean', 'median', 'max'],
                title='各榜单歌曲评论数统计',
                labels={'value': '评论数', 'variable': '统计类型'},
                barmode='group',
                template='plotly_white'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        # 评论数与情感倾向关系
        st.markdown("### 评论数与情感倾向关系")
        fig = px.scatter(
            df,
            x='评论总数',
            y='积极评论占比',
            color='榜单类型',
            size='消极评论占比',
            hover_data=['歌曲名称', '歌手'],
            title='评论总数 vs 积极评论占比',
            labels={'评论总数': '评论总数', '积极评论占比': '积极评论占比'},
            template='plotly_white',
            opacity=0.7
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    # Tab 3: 高频词分析
    with tab3:
        # 合并所有高频词
        all_keywords = []
        for keywords in df['高频字眼'].dropna():
            if keywords and keywords != '':
                all_keywords.extend([kw.strip() for kw in keywords.split(',') if kw.strip()])
        
        if all_keywords:
            # 新增：将高频词列表转换为文本字符串
            keywords_text = ' '.join(all_keywords)  # 用空格连接高频词，供词云使用
            # 1. 定义项目内字体路径（fonts文件夹下的simsun.ttc）
            font_dir = Path(__file__).parent / "fonts"
            font_path = font_dir / "STZHONGS.TTF"  # 确保字体文件名正确
            
            # 2. 验证字体文件是否存在，不存在则尝试系统字体，最后fallback
            if not font_path.exists():
                st.warning("项目内字体文件未找到，尝试加载系统字体...")
                # 尝试系统字体（兼容不同环境）
                system_fonts = [
                    "C:/Windows/Fonts/STZHONGS.TTF"          # Windows
                ]
                for sys_font in system_fonts:
                    if Path(sys_font).exists():
                        font_path = Path(sys_font)
                        break
                else:
                    # 所有尝试失败，用默认字体（可能无法显示中文，但不报错）
                    font_path = None
                    st.warning("系统字体也未找到，词云可能无法显示中文！")

            # 生成词云
            wordcloud = WordCloud(
                font_path=str(font_path) if font_path else None,  # 路径转字符串（WordCloud需要str类型）
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100,
                max_font_size=100,
                contour_width=3,
                contour_color=COLOR_PALETTE['primary']
            ).generate(keywords_text)  # 现在 keywords_text 已定义
                
            # 显示词云
            st.markdown("### 高频词云图")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
                        # 原来的高频词条形图和榜单对比
            keyword_counts = Counter(all_keywords).most_common(20)
            keywords_df = pd.DataFrame(keyword_counts, columns=['关键词', '出现次数'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 高频词词云（条形图模拟）
                fig = px.bar(
                    keywords_df,
                    x='出现次数',
                    y='关键词',
                    orientation='h',
                    title='所有歌曲高频关键词 Top 20',
                    labels={'出现次数': '出现次数', '关键词': '关键词'},
                    color='出现次数',
                    color_continuous_scale='Viridis',
                    template='plotly_white'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # 各榜单高频词对比（取前5个）
                st.markdown("### 各榜单Top5高频词")
                rank_keywords = {}
                
                for rank in df['榜单类型'].unique():
                    rank_df = df[df['榜单类型'] == rank]
                    rank_keywords_list = []
                    
                    for keywords in rank_df['高频字眼'].dropna():
                        if keywords and keywords != '':
                            rank_keywords_list.extend([kw.strip() for kw in keywords.split(',') if kw.strip()])
                    
                    if rank_keywords_list:
                        rank_keywords[rank] = Counter(rank_keywords_list).most_common(5)
                
                # 创建表格显示
                for rank, keywords in rank_keywords.items():
                    st.subheader(f"{rank}")
                    kw_df = pd.DataFrame(keywords, columns=['关键词', '出现次数'])
                    st.dataframe(kw_df, width='stretch')
        else:
            st.info("没有找到有效的高频词数据")
  
    # Tab 4: 高级洞察
    with tab4:
        # Top 10 积极评论占比最高的歌曲
        st.markdown("### Top 10 积极评论占比最高的歌曲")
        top_positive = df.nlargest(10, '积极评论占比')[['歌曲名称', '歌手', '榜单类型', '积极评论占比', '评论总数', '高频字眼']]
        
        fig = px.bar(
            top_positive,
            x='歌曲名称',
            y='积极评论占比',
            color='榜单类型',
            title='积极评论占比最高的10首歌曲',
            labels={'歌曲名称': '歌曲名称', '积极评论占比': '积极评论占比'},
            template='plotly_white',
            hover_data=['歌手', '评论总数', '高频字眼']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
            
        # Top 10 消极评论占比最高的歌曲
        st.markdown("### Top 10 消极评论占比最高的歌曲")
        top_negative = df.nlargest(10, '消极评论占比')[['歌曲名称', '歌手', '榜单类型', '消极评论占比', '评论总数', '高频字眼']]
            
        fig = px.bar(
            top_negative,
            x='歌曲名称',
            y='消极评论占比',
            color='榜单类型',
            title='消极评论占比最高的10首歌曲',
            labels={'歌曲名称': '歌曲名称', '消极评论占比': '消极评论占比'},
            template='plotly_white',
            hover_data=['歌手', '评论总数', '高频字眼'],
            color_discrete_map={'热歌榜': '#FF4136', '新歌榜': '#FF851B', '飙升榜': '#FFDC00', '原创榜': '#B10DC9'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
            
        # 各榜单歌曲情感特征雷达图
        st.markdown("### 各榜单情感特征对比")
        rank_sentiment = df.groupby('榜单类型').agg({
            '积极评论占比': 'mean',
            '消极评论占比': 'mean',
            '中立评论占比': 'mean',
            '评论总数': 'mean'
        }).reset_index()
            
        # 数据标准化
        for col in ['积极评论占比', '消极评论占比', '中立评论占比', '评论总数']:
            rank_sentiment[col] = (rank_sentiment[col] - rank_sentiment[col].min()) / (rank_sentiment[col].max() - rank_sentiment[col].min())
            
        fig = go.Figure()
        for _, row in rank_sentiment.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['积极评论占比'], row['消极评论占比'], row['中立评论占比'], row['评论总数']],
                theta=['积极评论占比', '消极评论占比', '中立评论占比', '平均评论数'],
                name=row['榜单类型']
            ))
            
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, width='stretch')
    
    # Tab 5: 智能推荐
    with tab5:
        st.markdown("### 🎯 歌曲智能推荐系统")
        
        # 创建推荐模型
        @st.cache_resource
        def create_song_recommendation_model(df):
            """创建歌曲推荐模型"""
            # 准备文本数据
            texts = df['特征文本'].tolist()
            
            # 创建TF-IDF向量izer
            vectorizer = TfidfVectorizer(
                tokenizer=jieba.cut,
                stop_words=['的', '了', '是', '我', '在', '和', '也', '都', '很', '就', '还', '有'],
                max_features=5000
            )
            
            # 转换文本为TF-IDF矩阵
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            return vectorizer, tfidf_matrix
        
        # 获取推荐模型
        if not df.empty and '特征文本' in df.columns:
            with st.spinner("正在初始化推荐模型..."):
                vectorizer, tfidf_matrix = create_song_recommendation_model(df)
            
            # 用户输入
            st.markdown("#### 请输入你的需求")
            user_query = st.text_input("例如：我想听积极向上的流行歌曲，歌词要有梦想和希望", "")
            
            # 推荐参数设置
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_preference = st.selectbox("情感倾向", ["不限", "积极", "消极", "中立"])
            with col2:
                min_comment_count = st.number_input("最低评论数", min_value=0, value=100)
            with col3:
                recommendation_count = st.number_input("推荐数量", min_value=1, max_value=20, value=5)
            
            # 执行推荐
            if st.button("获取推荐"):
                if not user_query:
                    st.warning("请输入你的音乐需求")
                else:
                    with st.spinner("正在为你推荐歌曲..."):
                        # 处理用户查询
                        query_vector = vectorizer.transform([user_query])
                        
                        # 计算相似度
                        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
                        
                        # 创建相似度DataFrame
                        similarity_df = pd.DataFrame({
                            'index': range(len(similarities)),
                            'similarity': similarities
                        })
                        
                        # 筛选相似度高的歌曲
                        similarity_df = similarity_df[similarity_df['similarity'] > 0.05].sort_values('similarity', ascending=False)
                        
                        # 获取推荐结果
                        recommendations = []
                        for _, row in similarity_df.iterrows():
                            if len(recommendations) >= recommendation_count:
                                break
                                
                            song_idx = int(row['index'])
                            song = df.iloc[song_idx]
                            
                            # 过滤条件
                            if song['评论总数'] >= min_comment_count:
                                if sentiment_preference == "不限" or song['情感倾向'] == sentiment_preference:
                                    recommendations.append({
                                        'index': song_idx,
                                        'similarity': row['similarity'],
                                        'song': song
                                    })
                        
                        # 显示推荐结果
                        if recommendations:
                            st.markdown(f"#### 为你找到 {len(recommendations)} 首符合条件的歌曲：")
                            
                            for rec in recommendations:
                                song = rec['song']
                                similarity_score = rec['similarity']
                                
                                # 生成匹配理由
                                match_reasons = []
                                query_words = set(jieba.cut(user_query))
                                song_words = set(jieba.cut(song['特征文本']))
                                common_words = query_words.intersection(song_words)
                                
                                if common_words:
                                    match_reasons.append(f"包含关键词：{', '.join(common_words)}")
                                if song['情感倾向'] == '积极' and song['积极评论占比'] > df['积极评论占比'].mean():
                                    match_reasons.append("积极评论占比较高")
                                if song['评论总数'] > df['评论总数'].mean():
                                    match_reasons.append("人气较高")
                                
                                # 显示推荐卡片
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>{song['歌曲名称']} - {song['歌手']}</h4>
                                    <p><strong>榜单：</strong>{song['榜单类型']}</p>
                                    <p><strong>评论数：</strong>{song['评论总数']:,}</p>
                                    <p><strong>情感倾向：</strong>{song['情感倾向']} (积极: {song['积极评论占比']:.1%}, 消极: {song['消极评论占比']:.1%})</p>
                                    <p><strong>高频关键词：</strong>{song['高频字眼'] if pd.notna(song['高频字眼']) else '无'}</p>
                                    <p><strong>匹配理由：</strong>{' | '.join(match_reasons) if match_reasons else '综合特征匹配'}</p>
                                    <span class="match-score">匹配度：{similarity_score:.2%}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("没有找到完全匹配的歌曲，建议尝试调整搜索关键词或降低筛选条件")
        else:
            st.warning("数据不足，无法创建推荐模型")
    
    # Tab 6: 聚类分析（新增）
    with tab6:
        perform_clustering_analysis(df, "榜单评论")
    
    # Tab 7: 预测分析（新增）
    with tab7:
        perform_random_forest_prediction(df, "榜单评论")

# ---------------------- 主界面布局与逻辑 ----------------------
def main():
    # 页面标题
    st.markdown('<div class="page-title">🎵 网易云歌单+榜单评论综合数据分析工具</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 数据源选择
    selected_data_source = st.selectbox(
        "请选择要分析的数据源",
        ["13类风格歌单数据", "4类榜单歌曲评论数据"]
    )
    
    # 加载数据（使用st.spinner显示加载状态）
    with st.spinner("正在加载数据，请稍候..."):
        df, load_summary = load_all_data(selected_data_source)
    
    # 显示加载状态
    if not df.empty:
        st.success(f"✅ 成功加载 {load_summary['found_count']} / {load_summary['total_count']} 个{load_summary['data_type']}数据")
        if load_summary['dup_count'] > 0:
            st.info(f"🔍 数据去重完成：共移除 {load_summary['dup_count']} 条重复数据")
    else:
        st.warning("⚠️ 数据加载失败或没有找到有效数据")
    
    # 显示跳过的文件
    if load_summary['skipped_items']:
        with st.expander("⚠️ 查看被跳过的文件", expanded=False):
            for item in load_summary['skipped_items']:
                st.write(item)
    
    st.markdown("---")
    
    # 显示数据概览
    if not df.empty:
        display_data_overview(df, selected_data_source)
    st.markdown("---")
    
    # --- 核心修改：将筛选条件从侧边栏移至主页面 ---
    filtered_df = pd.DataFrame()
    if not df.empty:
        st.markdown('<div class="sub-title">🔍 筛选条件</div>', unsafe_allow_html=True)
        
        # 使用expander组件来容纳所有筛选器，保持页面整洁
        with st.expander("展开/折叠筛选器", expanded=True):
            if selected_data_source == "13类风格歌单数据":
                # 创建一个2列的布局来放置筛选器
                col1, col2 = st.columns(2)
                
                with col1:
                    # 歌单分类筛选
                    selected_cats = st.multiselect(
                        "歌单分类", 
                        options=df['分类'].unique(), 
                        default=df['分类'].unique()
                    )
                    
                    # 播放次数筛选
                    play_min, play_max = st.slider(
                        "播放次数范围",
                        min_value=int(df['播放次数'].min()),
                        max_value=int(df['播放次数'].max()),
                        value=(int(df['播放次数'].min()), int(df['播放次数'].max()))
                    )
                    
                    # 收藏量筛选
                    fav_min = st.number_input(
                        "最小收藏量", 
                        min_value=0, 
                        max_value=int(df['收藏量'].max()), 
                        value=0
                    )
                
                with col2:
                    # 日期筛选
                    has_dates = not df['创建日期'].isna().all()
                    date_min_ts, date_max_ts = None, None
                    if has_dates:
                        date_min, date_max = st.date_input(
                            "创建日期范围",
                            value=(df['创建日期'].min(), df['创建日期'].max()),
                            min_value=df['创建日期'].min(),
                            max_value=df['创建日期'].max()
                        )
                        date_min_ts = pd.to_datetime(date_min)
                        date_max_ts = pd.to_datetime(date_max)
                    
                    # 歌单长度筛选
                    len_min, len_max = st.slider(
                        "歌单歌曲数量",
                        min_value=1,
                        max_value=int(df['歌单长度'].max()),
                        value=(1, int(df['歌单长度'].max()))
                    )
                
                # 应用筛选
                filtered_df = df[
                    (df['分类'].isin(selected_cats)) &
                    (df['播放次数'] >= play_min) &
                    (df['播放次数'] <= play_max) &
                    (df['收藏量'] >= fav_min) &
                    (df['歌单长度'] >= len_min) &
                    (df['歌单长度'] <= len_max)
                ].copy()
                
                if has_dates and date_min_ts and date_max_ts:
                    filtered_df = filtered_df[
                        (filtered_df['创建日期'] >= date_min_ts) &
                        (filtered_df['创建日期'] <= date_max_ts)
                    ]
            
            else: # 4类榜单歌曲评论数据
                # 创建一个2列的布局来放置筛选器
                col1, col2 = st.columns(2)
                
                with col1:
                    # 榜单类型筛选
                    selected_ranks = st.multiselect(
                        "榜单类型",
                        options=df['榜单类型'].unique(),
                        default=df['榜单类型'].unique()
                    )
                    
                    # 评论数筛选
                    comment_min, comment_max = st.slider(
                        "评论总数范围",
                        min_value=int(df['评论总数'].min()),
                        max_value=int(df['评论总数'].max()),
                        value=(int(df['评论总数'].min()), int(df['评论总数'].max()))
                    )
                
                with col2:
                    # 情感倾向筛选
                    selected_sentiments = st.multiselect(
                        "情感倾向",
                        options=['积极', '消极', '中立'],
                        default=['积极', '消极', '中立']
                    )
                    
                    # 积极评论占比筛选
                    pos_ratio_min, pos_ratio_max = st.slider(
                        "积极评论占比范围",
                        min_value=0.0,
                        max_value=1.0,
                        value=(0.0, 1.0),
                        step=0.01
                    )
                
                # 应用筛选
                filtered_df = df[
                    (df['榜单类型'].isin(selected_ranks)) &
                    (df['评论总数'] >= comment_min) &
                    (df['评论总数'] <= comment_max) &
                    (df['情感倾向'].isin(selected_sentiments)) &
                    (df['积极评论占比'] >= pos_ratio_min) &
                    (df['积极评论占比'] <= pos_ratio_max)
                ].copy()

    # 显示筛选结果
    if not df.empty:
        st.markdown('<div class="sub-title">📋 筛选结果</div>', unsafe_allow_html=True)
        st.markdown(f"**符合条件的记录数量：{len(filtered_df)}**")
        
        # 显示数据表格
        with st.expander("查看详细数据", expanded=False):
            st.dataframe(filtered_df, width='stretch')
        
        # 可视化分析
        st.markdown("---")
        if selected_data_source == "13类风格歌单数据":
            plot_style_playlist_visualizations(filtered_df)
        else:
            plot_rank_comment_visualizations(filtered_df)

if __name__ == "__main__":
    main()