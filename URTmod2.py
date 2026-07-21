"""
Enhanced Unconventional Rock Typing Workflow - Streamlit App
Based on URTeC 2026 Paper ID4459370
Interactive Web Application for Unconventional Reservoir Characterization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Unconventional Rock Typing Workflow",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        padding: 0.5rem 0;
        border-bottom: 2px solid #ecf0f1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    .instruction-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #17a2b8;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# 1. TOC ESTIMATION METHODS (ENHANCED WITH MANUAL INPUTS)
# ============================================================================

def estimate_toc_passey(df, method='passey', LOM=10, baseline_RT=None, baseline_GR=None):
    """Estimate TOC using Passey's ΔlogR method with manual inputs"""
    try:
        # Use manual baselines if provided, otherwise calculate from data
        if baseline_RT is None:
            baseline_RT = df['RT'].quantile(0.05)
        if baseline_GR is None:
            baseline_GR = df['GR'].quantile(0.05)
        
        # Calculate ΔlogR
        RT_ratio = df['RT'] / baseline_RT
        df['LOG_RT'] = np.log10(RT_ratio)
        GR_diff = df['GR'] - baseline_GR
        df['DLOG_R'] = df['LOG_RT'] + 0.02 * GR_diff
        
        # Apply LOM factor
        if method == 'passey':
            LOM_factor = 10 ** (2.297 - 0.1688 * LOM)
            df['TOC'] = df['DLOG_R'] * LOM_factor
        elif method == 'passey_modified':
            if 'PHIE' in df.columns:
                VSH_clay = df.get('VSH', 0)
                clay_correction = 1 - VSH_clay
                df['TOC'] = df['DLOG_R'] * 4.06 * (1 + 0.5 * df['PHIE']) * clay_correction
        
        df['TOC'] = df['TOC'].clip(0, 20)
        
        # Store parameters for display
        df.attrs['baseline_RT'] = baseline_RT
        df.attrs['baseline_GR'] = baseline_GR
        df.attrs['LOM'] = LOM
        
        return df
    except Exception as e:
        st.error(f"Error in Passey TOC estimation: {str(e)}")
        return df

def estimate_toc_schmoker(df, method='schmoker'):
    """Estimate TOC using Schmoker's method"""
    try:
        RHOB_matrix = 2.65
        RHOB_kerogen = 1.0
        
        if method == 'schmoker':
            df['TOC'] = (df['RHOB'] - RHOB_matrix) / (RHOB_kerogen - RHOB_matrix)
            GR_norm = (df['GR'] - df['GR'].min()) / (df['GR'].max() - df['GR'].min())
            df['TOC'] = df['TOC'] * (0.5 + 0.5 * GR_norm)
        elif method == 'schmoker_modified':
            df['TOC'] = (df['RHOB'] - RHOB_matrix) / (RHOB_kerogen - RHOB_matrix)
            if 'VSH' in df.columns and 'PHIE' in df.columns:
                clay_correction = 1 - df['VSH'] * 0.5
                porosity_correction = 1 + df['PHIE'] * 0.3
                df['TOC'] = df['TOC'] * clay_correction * porosity_correction
        
        df['TOC'] = df['TOC'].clip(0, 20)
        return df
    except Exception as e:
        st.error(f"Error in Schmoker TOC estimation: {str(e)}")
        return df

def estimate_toc_combined(df, method='passey', LOM=10, baseline_RT=None, baseline_GR=None):
    """Combined TOC estimation with manual inputs"""
    try:
        if method == 'passey':
            df = estimate_toc_passey(df, LOM=LOM, baseline_RT=baseline_RT, baseline_GR=baseline_GR)
        elif method == 'schmoker':
            df = estimate_toc_schmoker(df)
        elif method == 'ensemble':
            df_passey = estimate_toc_passey(df.copy(), LOM=LOM, baseline_RT=baseline_RT, baseline_GR=baseline_GR)
            df_schmoker = estimate_toc_schmoker(df.copy())
            
            valid_passey = ~df_passey['TOC'].isna()
            valid_schmoker = ~df_schmoker['TOC'].isna()
            df['TOC'] = 0
            
            if valid_passey.any() and valid_schmoker.any():
                corr_coef = np.corrcoef(df_passey.loc[valid_passey & valid_schmoker, 'TOC'],
                                       df_schmoker.loc[valid_passey & valid_schmoker, 'TOC'])[0, 1]
                if corr_coef > 0.5:
                    df['TOC'] = 0.5 * df_passey['TOC'] + 0.5 * df_schmoker['TOC']
                else:
                    df['TOC'] = 0.4 * df_passey['TOC'] + 0.6 * df_schmoker['TOC']
            elif valid_passey.any():
                df['TOC'] = df_passey['TOC']
            elif valid_schmoker.any():
                df['TOC'] = df_schmoker['TOC']
        
        return df
    except Exception as e:
        st.error(f"Error in combined TOC estimation: {str(e)}")
        return df

def calibrate_toc_with_lab_data(df, lab_data):
    """Calibrate TOC with laboratory measurements and store lab data for plotting"""
    try:
        if lab_data is not None and len(lab_data) > 0:
            if 'Depth' in df.columns and 'Depth' in lab_data.columns:
                estimated_at_lab = []
                lab_measured = []
                lab_depths = []
                
                for idx, row in lab_data.iterrows():
                    depth_lab = row['Depth']
                    nearest_idx = (df['Depth'] - depth_lab).abs().idxmin()
                    estimated_at_lab.append(df.loc[nearest_idx, 'TOC'])
                    lab_measured.append(row['TOC_lab'])
                    lab_depths.append(depth_lab)
                
                estimated_at_lab = np.array(estimated_at_lab)
                lab_measured = np.array(lab_measured)
                
                # Store lab data for visualization
                df.attrs['lab_data'] = {
                    'depths': lab_depths,
                    'estimated': estimated_at_lab.tolist(),
                    'measured': lab_measured.tolist()
                }
                
                if len(estimated_at_lab) > 3:
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                    model.fit(estimated_at_lab.reshape(-1, 1), lab_measured)
                    df['TOC'] = model.predict(df[['TOC']].values)
                    df['TOC'] = df['TOC'].clip(0, 20)
        
        return df
    except Exception as e:
        st.warning(f"Could not calibrate TOC with lab data: {str(e)}")
        return df

# ============================================================================
# 2. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(df, estimate_toc=True, toc_method='passey', 
                            LOM=10, baseline_RT=None, baseline_GR=None):
    """Load and preprocess well log data with manual TOC parameters"""
    try:
        df = df.dropna()
        
        if 'TOC' not in df.columns and estimate_toc:
            df = estimate_toc_combined(df, method=toc_method, LOM=LOM, 
                                      baseline_RT=baseline_RT, baseline_GR=baseline_GR)
            
            if df['TOC'].isna().all():
                st.error("TOC estimation failed. Please ensure required columns are present.")
                return None
        
        # Calculate derived properties
        df['VPVS_RATIO'] = df['VP'] / df['VS']
        df['AI'] = df['VP'] * df['RHOB']
        
        df['E_DYN'] = (df['RHOB'] * df['VS']**2 *
                       (3*df['VP']**2 - 4*df['VS']**2) /
                       (df['VP']**2 - df['VS']**2))
        
        df['NU_DYN'] = (df['VP']**2 - 2*df['VS']**2) / (2*(df['VP']**2 - df['VS']**2))
        
        if df['PHIE'].max() < 0.1:
            df['PHIE_PCT'] = df['PHIE'] * 100
        else:
            df['PHIE_PCT'] = df['PHIE']
        
        df['SHC'] = 1 - (df['RT'] / (df['RT'].max() * 0.1))
        df['SHC'] = df['SHC'].clip(0, 1)
        
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

# ============================================================================
# 3. BRITTLENESS INDEX CALCULATION
# ============================================================================

def calculate_brittleness_index(df, method='rickman'):
    """Calculate Brittleness Index"""
    try:
        if method == 'rickman':
            E_min = df['E_DYN'].min()
            E_max = df['E_DYN'].max()
            nu_min = df['NU_DYN'].min()
            nu_max = df['NU_DYN'].max()
            
            E_norm = (df['E_DYN'] - E_min) / (E_max - E_min) if E_max > E_min else np.zeros(len(df))
            nu_norm = (df['NU_DYN'] - nu_max) / (nu_min - nu_max) if nu_min < nu_max else np.zeros(len(df))
            df['BRIT'] = ((E_norm + nu_norm) / 2) * 100
            
        elif method == 'wang':
            df['BRIT'] = 100 * (1 - df['VPVS_RATIO'] / df['VPVS_RATIO'].max())
            
        elif method == 'jarvie':
            df['BRIT'] = 100 * (1 - df['VSH'])
            
        elif method == 'comprehensive':
            E_min = df['E_DYN'].min()
            E_max = df['E_DYN'].max()
            nu_min = df['NU_DYN'].min()
            nu_max = df['NU_DYN'].max()
            
            if E_max > E_min and nu_min < nu_max:
                E_norm = (df['E_DYN'] - E_min) / (E_max - E_min)
                nu_norm = (df['NU_DYN'] - nu_max) / (nu_min - nu_max)
                brit_rickman = ((E_norm + nu_norm) / 2) * 100
            else:
                brit_rickman = 50
            
            brit_mineral = 100 * (1 - df['VSH'])
            brit_acoustic = 100 * (1 - df['VPVS_RATIO'] / df['VPVS_RATIO'].max())
            df['BRIT'] = 0.4 * brit_rickman + 0.3 * brit_mineral + 0.3 * brit_acoustic
        
        df['BRIT'] = df['BRIT'].clip(0, 100)
        return df
    except Exception as e:
        st.error(f"Error calculating brittleness: {str(e)}")
        return df

# ============================================================================
# 4. TUNNEL WALL COLLAPSE (TWC) CALCULATION
# ============================================================================

def calculate_twc(df):
    """Calculate Tunnel Wall Collapse parameter"""
    try:
        E_min = df['E_DYN'].min()
        E_max = df['E_DYN'].max()
        nu_min = df['NU_DYN'].min()
        nu_max = df['NU_DYN'].max()
        
        E_norm = (df['E_DYN'] - E_min) / (E_max - E_min) if E_max > E_min else 0.5
        nu_norm = (df['NU_DYN'] - nu_min) / (nu_max - nu_min) if nu_max > nu_min else 0.5
        
        df['TWC'] = (E_norm * 0.6 + (1 - nu_norm) * 0.4) * 100
        return df
    except Exception as e:
        st.error(f"Error calculating TWC: {str(e)}")
        return df

# ============================================================================
# 5. URT_INDEX CALCULATION
# ============================================================================

def calculate_urt_index(df, weights=None):
    """Calculate Unconventional Rock Type Index"""
    try:
        if weights is None:
            weights = {'W1': 0.25, 'W2': 0.25, 'W3': 0.25, 'W4': 0.25}
        
        scaler = MinMaxScaler()
        df['TOC_norm'] = scaler.fit_transform(df[['TOC']])
        df['BRIT_norm'] = scaler.fit_transform(df[['BRIT']])
        df['PHIT_norm'] = scaler.fit_transform(df[['PHIE_PCT']])
        df['SHC_norm'] = scaler.fit_transform(df[['SHC']])
        
        df['URT_Index'] = (weights['W1'] * df['TOC_norm'] +
                           weights['W2'] * df['BRIT_norm'] +
                           weights['W3'] * df['PHIT_norm'] +
                           weights['W4'] * df['SHC_norm'])
        
        df['URT_Index'] = scaler.fit_transform(df[['URT_Index']])
        return df
    except Exception as e:
        st.error(f"Error calculating URT_Index: {str(e)}")
        return df

# ============================================================================
# 6. FHZI CALCULATION
# ============================================================================

def calculate_fhzi(df):
    """Calculate Hydraulic Fracture Zone Index"""
    try:
        df['BRIT_norm_fhzi'] = (df['BRIT'] - df['BRIT'].min()) / (df['BRIT'].max() - df['BRIT'].min())
        df['TWC_norm_fhzi'] = (df['TWC'] - df['TWC'].min()) / (df['TWC'].max() - df['TWC'].min())
        df['FHZI'] = 1 - (df['BRIT_norm_fhzi'] + df['TWC_norm_fhzi']) / 2
        df['FHZI'] = df['FHZI'].clip(0, 1)
        return df
    except Exception as e:
        st.error(f"Error calculating FHZI: {str(e)}")
        return df

# ============================================================================
# 7. ROCK TYPE CLASSIFICATION
# ============================================================================

def classify_rock_types(df):
    """Classify rocks into 4 types based on URT_Index"""
    try:
        quartiles = df['URT_Index'].quantile([0.25, 0.5, 0.75])
        
        conditions = [
            df['URT_Index'] >= quartiles[0.75],
            (df['URT_Index'] >= quartiles[0.5]) & (df['URT_Index'] < quartiles[0.75]),
            (df['URT_Index'] >= quartiles[0.25]) & (df['URT_Index'] < quartiles[0.5]),
            df['URT_Index'] < quartiles[0.25]
        ]
        
        choices = ['RT1 (Best)', 'RT2 (Good)', 'RT3 (Fair)', 'RT4 (Poor)']
        df['Rock_Type'] = np.select(conditions, choices, default='RT4 (Poor)')
        
        rock_type_map = {'RT1 (Best)': 1, 'RT2 (Good)': 2, 'RT3 (Fair)': 3, 'RT4 (Poor)': 4}
        df['Rock_Type_Num'] = df['Rock_Type'].map(rock_type_map)
        
        return df
    except Exception as e:
        st.error(f"Error classifying rock types: {str(e)}")
        return df

# ============================================================================
# 8. SWEET SPOT IDENTIFICATION
# ============================================================================

def identify_sweet_spots(df, urt_threshold=0.6, fhzi_threshold=0.4):
    """Identify sweet spots for hydraulic fracturing"""
    try:
        df['URT_norm'] = (df['URT_Index'] - df['URT_Index'].min()) / (df['URT_Index'].max() - df['URT_Index'].min())
        df['FHZI_norm'] = (df['FHZI'] - df['FHZI'].min()) / (df['FHZI'].max() - df['FHZI'].min())
        df['BRIT_norm'] = (df['BRIT'] - df['BRIT'].min()) / (df['BRIT'].max() - df['BRIT'].min())
        
        df['Sweet_Spot'] = ((df['URT_Index'] >= urt_threshold) &
                            (df['FHZI'] >= fhzi_threshold) &
                            (df['BRIT'] >= 50))
        
        df['Sweet_Spot_Quality'] = (df['URT_norm'] * 0.4 + 
                                   df['FHZI_norm'] * 0.3 + 
                                   df['BRIT_norm'] * 0.2 + 
                                   df['TOC_norm'] * 0.1)
        
        conditions = [
            df['Sweet_Spot_Quality'] >= df['Sweet_Spot_Quality'].quantile(0.75),
            (df['Sweet_Spot_Quality'] >= df['Sweet_Spot_Quality'].quantile(0.5)) &
            (df['Sweet_Spot_Quality'] < df['Sweet_Spot_Quality'].quantile(0.75)),
            df['Sweet_Spot_Quality'] < df['Sweet_Spot_Quality'].quantile(0.5)
        ]
        
        choices = ['Excellent', 'Good', 'Poor']
        df['Sweet_Spot_Class'] = np.select(conditions, choices, default='Poor')
        
        return df
    except Exception as e:
        st.error(f"Error identifying sweet spots: {str(e)}")
        return df

# ============================================================================
# 9. STAGE OPTIMIZATION
# ============================================================================

def optimize_fracture_stages(df, max_stages=10, min_stages=3):
    """Optimize number and placement of fracture stages"""
    try:
        df['Stage_Score'] = (df['URT_norm'] * 0.4 +
                             df['FHZI_norm'] * 0.3 +
                             df['BRIT_norm'] * 0.2 +
                             df['TOC_norm'] * 0.1)
        
        score_std = df['Stage_Score'].std()
        
        if score_std > 0.3:
            optimal_stages = max_stages
        elif score_std > 0.2:
            optimal_stages = (max_stages + min_stages) // 2
        else:
            optimal_stages = min_stages
        
        df['Recommended_Stages'] = optimal_stages
        
        high_threshold = df['Stage_Score'].quantile(0.7)
        med_threshold = df['Stage_Score'].quantile(0.4)
        
        df['Stage_Recommendation'] = np.where(
            df['Stage_Score'] >= high_threshold,
            'Stimulate (High Priority)',
            np.where(
                df['Stage_Score'] >= med_threshold,
                'Stimulate (Medium Priority)',
                'Avoid Stimulation'
            )
        )
        
        return df
    except Exception as e:
        st.error(f"Error optimizing fracture stages: {str(e)}")
        return df

# ============================================================================
# 10. STREAMLIT VISUALIZATION FUNCTIONS
# ============================================================================

def create_toc_plot(df, lab_data=None):
    """Create TOC visualization with optional lab data points"""
    try:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('TOC Distribution', 'TOC vs Depth'),
                            specs=[[{'type': 'histogram'}, {'type': 'scatter'}]])
        
        # TOC Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=df['TOC'],
                nbinsx=20,
                name='TOC Distribution',
                marker_color='blue',
                opacity=0.7,
                hovertemplate='TOC: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_vline(x=df['TOC'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {df['TOC'].mean():.2f}%",
                      annotation_position="top", row=1, col=1)
        fig.add_vline(x=df['TOC'].median(), line_dash="dash", line_color="green",
                      annotation_text=f"Median: {df['TOC'].median():.2f}%",
                      annotation_position="bottom", row=1, col=1)
        
        # TOC vs Depth plot
        if 'Depth' in df.columns:
            # Main TOC profile
            fig.add_trace(
                go.Scatter(
                    x=df['TOC'],
                    y=df['Depth'],
                    mode='lines+markers',
                    name='TOC Profile',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4, color='blue'),
                    hovertemplate='Depth: %{y:.1f}<br>TOC: %{x:.2f}%<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add lab data points if available
            if lab_data is not None and len(lab_data) > 0:
                # Extract lab data from dataframe attributes
                lab_info = df.attrs.get('lab_data', None)
                if lab_info is not None:
                    lab_depths = lab_info['depths']
                    lab_measured = lab_info['measured']
                    
                    # Add red dots for lab measurements
                    fig.add_trace(
                        go.Scatter(
                            x=lab_measured,
                            y=lab_depths,
                            mode='markers',
                            name='Lab TOC Measurements',
                            marker=dict(
                                size=12,
                                color='red',
                                symbol='circle',
                                line=dict(width=2, color='darkred')
                            ),
                            hovertemplate='Lab Depth: %{y:.1f}<br>Lab TOC: %{x:.2f}%<extra></extra>'
                        ),
                        row=1, col=2
                    )
                    
                    # Add estimated TOC at lab depths
                    lab_estimated = lab_info['estimated']
                    fig.add_trace(
                        go.Scatter(
                            x=lab_estimated,
                            y=lab_depths,
                            mode='markers',
                            name='Estimated at Lab Depths',
                            marker=dict(
                                size=8,
                                color='orange',
                                symbol='diamond',
                                line=dict(width=1, color='darkorange')
                            ),
                            hovertemplate='Depth: %{y:.1f}<br>Estimated TOC: %{x:.2f}%<extra></extra>'
                        ),
                        row=1, col=2
                    )
            
            fig.add_vline(x=2, line_dash="dash", line_color="red",
                          annotation_text="Threshold (2%)",
                          annotation_position="top", row=1, col=2)
            fig.update_yaxes(autorange="reversed", row=1, col=2)
        
        fig.update_layout(height=500, showlegend=True, hovermode='closest')
        fig.update_xaxes(title_text="TOC (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="TOC (%)", row=1, col=2)
        fig.update_yaxes(title_text="Depth", row=1, col=2)
        
        return fig
    except Exception as e:
        st.error(f"Error creating TOC plot: {str(e)}")
        return go.Figure()

def create_urt_distribution_plot(df):
    """Create URT_Index distribution visualization"""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('URT_Index Distribution', 'FHZI Distribution',
                           'Rock Type Distribution', 'Sweet Spot Classification'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # URT_Index histogram
        fig.add_trace(
            go.Histogram(
                x=df['URT_Index'],
                nbinsx=20,
                name='URT_Index',
                marker_color='blue',
                opacity=0.7,
                hovertemplate='URT_Index: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        fig.add_vline(x=df['URT_Index'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {df['URT_Index'].mean():.3f}",
                      annotation_position="top", row=1, col=1)
        
        # FHZI histogram
        fig.add_trace(
            go.Histogram(
                x=df['FHZI'],
                nbinsx=20,
                name='FHZI',
                marker_color='green',
                opacity=0.7,
                hovertemplate='FHZI: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        fig.add_vline(x=df['FHZI'].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {df['FHZI'].mean():.3f}",
                      annotation_position="top", row=1, col=2)
        
        # Rock Type distribution
        rock_counts = df['Rock_Type'].value_counts()
        colors = ['green', 'yellowgreen', 'gold', 'red']
        fig.add_trace(
            go.Bar(
                x=rock_counts.index,
                y=rock_counts.values,
                name='Rock Types',
                marker_color=colors[:len(rock_counts)],
                hovertemplate='Rock Type: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Sweet Spot classification
        sweet_counts = df['Sweet_Spot_Class'].value_counts()
        color_map = {'Excellent': 'green', 'Good': 'orange', 'Poor': 'red'}
        colors_pie = [color_map.get(x, 'gray') for x in sweet_counts.index]
        fig.add_trace(
            go.Pie(
                labels=sweet_counts.index,
                values=sweet_counts.values,
                name='Sweet Spots',
                marker=dict(colors=colors_pie),
                hole=0.3,
                hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, hovermode='closest')
        fig.update_xaxes(title_text="URT_Index", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="FHZI", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Rock Type", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        return fig
    except Exception as e:
        st.error(f"Error creating URT distribution plot: {str(e)}")
        return go.Figure()

def create_well_log_plot(df):
    """Create well log visualization"""
    try:
        if 'Depth' not in df.columns:
            df['Depth'] = np.arange(len(df))
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('URT_Index vs Depth', 'FHZI vs Depth', 'Brittleness vs Depth',
                           'URT_Index vs Brittleness', 'TOC vs URT_Index', 'Stage Optimization'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # URT_Index with depth
        fig.add_trace(
            go.Scatter(
                x=df['URT_Index'],
                y=df['Depth'],
                mode='lines+markers',
                name='URT_Index',
                line=dict(color='blue', width=2),
                marker=dict(size=3),
                hovertemplate='Depth: %{y:.1f}<br>URT_Index: %{x:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        fig.add_vline(x=0.6, line_dash="dash", line_color="red",
                      annotation_text="Threshold (0.6)",
                      annotation_position="top", row=1, col=1)
        
        # FHZI with depth
        fig.add_trace(
            go.Scatter(
                x=df['FHZI'],
                y=df['Depth'],
                mode='lines+markers',
                name='FHZI',
                line=dict(color='green', width=2),
                marker=dict(size=3),
                hovertemplate='Depth: %{y:.1f}<br>FHZI: %{x:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        fig.add_vline(x=0.4, line_dash="dash", line_color="red",
                      annotation_text="Threshold (0.4)",
                      annotation_position="top", row=1, col=2)
        
        # Brittleness with depth
        fig.add_trace(
            go.Scatter(
                x=df['BRIT'],
                y=df['Depth'],
                mode='lines+markers',
                name='Brittleness',
                line=dict(color='red', width=2),
                marker=dict(size=3),
                hovertemplate='Depth: %{y:.1f}<br>Brittleness: %{x:.1f}%<extra></extra>'
            ),
            row=1, col=3
        )
        fig.add_vline(x=50, line_dash="dash", line_color="black",
                      annotation_text="Threshold (50%)",
                      annotation_position="top", row=1, col=3)
        
        # URT_Index vs Brittleness
        fig.add_trace(
            go.Scatter(
                x=df['URT_Index'],
                y=df['BRIT'],
                mode='markers',
                name='URT vs Brittleness',
                marker=dict(
                    size=8,
                    color=df['FHZI'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="FHZI", x=1.02)
                ),
                hovertemplate='URT_Index: %{x:.3f}<br>Brittleness: %{y:.1f}%<br>FHZI: %{marker.color:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # TOC vs URT_Index
        fig.add_trace(
            go.Scatter(
                x=df['TOC'],
                y=df['URT_Index'],
                mode='markers',
                name='TOC vs URT',
                marker=dict(
                    size=8,
                    color=df['BRIT'],
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Brittleness (%)", x=1.02)
                ),
                hovertemplate='TOC: %{x:.2f}%<br>URT_Index: %{y:.3f}<br>Brittleness: %{marker.color:.1f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Stage optimization
        stage_counts = df['Stage_Recommendation'].value_counts()
        color_map = {'Stimulate (High Priority)': 'green',
                     'Stimulate (Medium Priority)': 'orange',
                     'Avoid Stimulation': 'red'}
        fig.add_trace(
            go.Bar(
                x=stage_counts.index,
                y=stage_counts.values,
                name='Stage Recommendations',
                marker_color=[color_map.get(x, 'gray') for x in stage_counts.index],
                hovertemplate='Recommendation: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=3
        )
        
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=3)
        
        fig.update_layout(height=900, showlegend=False, hovermode='closest')
        
        fig.update_xaxes(title_text="URT_Index", row=1, col=1)
        fig.update_yaxes(title_text="Depth", row=1, col=1)
        fig.update_xaxes(title_text="FHZI", row=1, col=2)
        fig.update_yaxes(title_text="Depth", row=1, col=2)
        fig.update_xaxes(title_text="Brittleness (%)", row=1, col=3)
        fig.update_yaxes(title_text="Depth", row=1, col=3)
        fig.update_xaxes(title_text="URT_Index", row=2, col=1)
        fig.update_yaxes(title_text="Brittleness (%)", row=2, col=1)
        fig.update_xaxes(title_text="TOC (%)", row=2, col=2)
        fig.update_yaxes(title_text="URT_Index", row=2, col=2)
        
        return fig
    except Exception as e:
        st.error(f"Error creating well log plot: {str(e)}")
        return go.Figure()

def create_3d_plot(df, lab_data=None):
    """Create 3D scatter plot with lab data points highlighted in red"""
    try:
        # Main 3D plot
        fig = go.Figure()
        
        # Add main data points
        fig.add_trace(go.Scatter3d(
            x=df['URT_Index'],
            y=df['BRIT'],
            z=df['TOC'],
            mode='markers',
            name='Well Log Data',
            marker=dict(
                size=6,
                color=df['FHZI'],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="FHZI")
            ),
            hovertemplate='URT_Index: %{x:.3f}<br>Brittleness: %{y:.1f}%<br>TOC: %{z:.2f}%<br>FHZI: %{marker.color:.3f}<extra></extra>'
        ))
        
        # Add lab data points if available
        if lab_data is not None and len(lab_data) > 0:
            lab_info = df.attrs.get('lab_data', None)
            if lab_info is not None:
                # Get lab data
                lab_measured = lab_info['measured']
                
                # Find corresponding URT_Index and BRIT values for lab depths
                lab_urt = []
                lab_brit = []
                lab_depths = lab_info['depths']
                
                for depth in lab_depths:
                    nearest_idx = (df['Depth'] - depth).abs().idxmin()
                    lab_urt.append(df.loc[nearest_idx, 'URT_Index'])
                    lab_brit.append(df.loc[nearest_idx, 'BRIT'])
                
                # Add lab data points as red markers
                fig.add_trace(go.Scatter3d(
                    x=lab_urt,
                    y=lab_brit,
                    z=lab_measured,
                    mode='markers',
                    name='Lab TOC Measurements',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='circle',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate='Lab URT_Index: %{x:.3f}<br>Lab Brittleness: %{y:.1f}%<br>Lab TOC: %{z:.2f}%<extra></extra>'
                ))
        
        fig.update_layout(
            title='3D Visualization of Key Parameters',
            scene=dict(
                xaxis_title='URT_Index',
                yaxis_title='Brittleness (%)',
                zaxis_title='TOC (%)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700,
            legend=dict(x=0.05, y=0.95)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating 3D plot: {str(e)}")
        return go.Figure()

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    try:
        cols = ['TOC', 'PHIE_PCT', 'BRIT', 'URT_Index', 'FHZI', 'VPVS_RATIO', 'SHC']
        if 'TWC' in df.columns:
            cols.append('TWC')
        
        corr_matrix = df[cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Correlation Matrix of Key Parameters',
            height=600,
            width=700
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        return go.Figure()

def create_toc_validation_plot(df, lab_data):
    """Create TOC validation plot"""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, r2_score
        
        estimated_values = []
        lab_values = []
        
        for idx, row in lab_data.iterrows():
            depth_lab = row['Depth']
            nearest_idx = (df['Depth'] - depth_lab).abs().idxmin()
            estimated_values.append(df.loc[nearest_idx, 'TOC'])
            lab_values.append(row['TOC_lab'])
        
        estimated_values = np.array(estimated_values)
        lab_values = np.array(lab_values)
        
        mae = mean_absolute_error(lab_values, estimated_values)
        r2 = r2_score(lab_values, estimated_values)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=lab_values,
            y=estimated_values,
            mode='markers',
            name='Validation Points',
            marker=dict(
                size=12,
                color='blue',
                opacity=0.7,
                line=dict(width=1, color='darkblue')
            ),
            hovertemplate='Lab TOC: %{x:.2f}%<br>Estimated TOC: %{y:.2f}%<extra></extra>'
        ))
        
        min_val = min(lab_values.min(), estimated_values.min())
        max_val = max(lab_values.max(), estimated_values.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='1:1 Line',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='%{x:.2f}%<extra></extra>'
        ))
        
        model = LinearRegression()
        model.fit(lab_values.reshape(-1, 1), estimated_values)
        x_range = np.linspace(min_val, max_val, 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred,
            mode='lines',
            name=f'Regression (R²={r2:.3f})',
            line=dict(color='green', width=2),
            hovertemplate='%{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'TOC Estimation Validation<br>R² = {r2:.3f}, MAE = {mae:.2f}%',
            xaxis_title='Laboratory TOC (%)',
            yaxis_title='Estimated TOC (%)',
            height=600,
            hovermode='closest',
            legend=dict(x=0.05, y=0.95)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating TOC validation plot: {str(e)}")
        return go.Figure()

# ============================================================================
# 11. GENERATE SAMPLE DATA
# ============================================================================

def generate_sample_data():
    """Generate sample well log data for demonstration"""
    try:
        np.random.seed(42)
        n_samples = 500
        
        # Create depth
        depth = np.linspace(1000, 1500, n_samples)
        
        # Generate synthetic logs with realistic relationships
        # Porosity with some variation
        phie = 0.08 + 0.05 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + 0.02 * np.random.randn(n_samples)
        phie = np.clip(phie, 0.03, 0.18)
        
        # Density inversely related to porosity
        rhob = 2.65 - 0.5 * phie + 0.05 * np.random.randn(n_samples)
        rhob = np.clip(rhob, 2.2, 2.75)
        
        # VP related to porosity and density
        vp = 4500 - 2000 * phie - 200 * (rhob - 2.4) + 200 * np.random.randn(n_samples)
        vp = np.clip(vp, 2800, 4800)
        
        # VS related to VP
        vs = vp * (0.5 + 0.05 * np.random.randn(n_samples))
        vs = np.clip(vs, 1400, 2800)
        
        # VSH with some variation
        vsh = 0.3 + 0.2 * np.sin(np.linspace(0, 3*np.pi, n_samples)) + 0.1 * np.random.randn(n_samples)
        vsh = np.clip(vsh, 0.1, 0.7)
        
        # Resistivity
        rt = 20 + 60 * np.exp(-phie * 20) + 20 * np.random.randn(n_samples)
        rt = np.clip(rt, 5, 150)
        
        # GR
        gr = 60 + 80 * vsh + 20 * np.random.randn(n_samples)
        gr = np.clip(gr, 40, 180)
        
        # TOC with zones of higher organic content
        toc = 2 + 3 * np.sin(np.linspace(0, 3*np.pi, n_samples)) + 0.5 * np.random.randn(n_samples)
        toc = np.clip(toc, 0, 10)
        
        df = pd.DataFrame({
            'Depth': depth,
            'PHIE': phie,
            'VP': vp,
            'VS': vs,
            'RHOB': rhob,
            'VSH': vsh,
            'RT': rt,
            'GR': gr,
            'TOC': toc
        })
        
        return df
    except Exception as e:
        st.error(f"Error generating sample data: {str(e)}")
        return None

# ============================================================================
# 12. MAIN STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🪨 Enhanced Unconventional Rock Typing Workflow</div>', unsafe_allow_html=True)
    st.markdown('*Based on URTeC 2026 Paper ID4459370*')
    st.markdown('---')
    
    # Instructions Section
    with st.expander("📋 How to Use This Application", expanded=True):
        st.markdown("""
        <div class="instruction-box">
            <h4>📖 Getting Started Guide</h4>
            
            <h5>🔍 Required Well Log Data</h5>
            <p>Your CSV file must contain the following columns:</p>
            <ul>
                <li><b>PHIE</b> - Effective Porosity (fraction or %)</li>
                <li><b>VP</b> - Compressional Wave Velocity (m/s or ft/s)</li>
                <li><b>VS</b> - Shear Wave Velocity (m/s or ft/s)</li>
                <li><b>RHOB</b> - Bulk Density (g/cm³)</li>
                <li><b>VSH</b> - Volume of Shale (fraction)</li>
                <li><b>RT</b> - Deep Resistivity (ohm.m)</li>
                <li><b>GR</b> - Gamma Ray (API)</li>
            </ul>
            
            <h5>📊 Optional Data</h5>
            <ul>
                <li><b>Depth</b> - Depth values for visualization</li>
                <li><b>TOC</b> - Total Organic Carbon (if available, otherwise estimated)</li>
            </ul>
            
            <h5>🔬 Calibration Data (Optional)</h5>
            <p>Upload a CSV with columns:</p>
            <ul>
                <li><b>Depth</b> - Depth of core samples</li>
                <li><b>TOC_lab</b> - Laboratory measured TOC values</li>
            </ul>
            <p><b>Note:</b> Lab data points will appear as <span style="color:red;">red dots</span> in the TOC vs Depth plot and 3D visualization.</p>
            
            <h5>⚙️ TOC Estimation Parameters</h5>
            <p>Configure the following parameters in the sidebar:</p>
            <ul>
                <li><b>LOM (Level of Organic Maturity)</b>: 5-15 (default: 10)</li>
                <li><b>Baseline Resistivity (Rt baseline)</b>: Manual or automatic</li>
                <li><b>Baseline Gamma Ray (GR baseline)</b>: Manual or automatic</li>
            </ul>
            
            <h5>🚀 Workflow Steps</h5>
            <ol>
                <li>Upload your well log data (or use sample data)</li>
                <li>Configure TOC estimation parameters (LOM, baselines)</li>
                <li>Select methods and thresholds</li>
                <li>Click "Run Workflow" to process data</li>
                <li>Explore interactive visualizations</li>
                <li>Download processed results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Configuration")
        
        # Data upload
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file with well log data",
            type=['csv'],
            key="uploaded_file",
            help="Required columns: PHIE, VP, VS, RHOB, VSH, RT, GR"
        )
        
        # Sample data option
        use_sample = st.checkbox("Use sample data", value=False, key="use_sample_checkbox")
        
        if use_sample:
            st.info("Using generated sample data for demonstration")
        
        st.markdown("---")
        
        # TOC Parameters Section
        st.markdown("### 🧪 TOC Estimation Parameters")
        st.markdown("*Configure Passey ΔlogR method parameters*")
        
        # LOM Input
        LOM = st.number_input(
            "LOM (Level of Organic Maturity)",
            min_value=5,
            max_value=15,
            value=10,
            step=1,
            key="lom_input",
            help="Level of Organic Maturity (typically 8-12 for most formations)"
        )
        
        # Baseline RT Input
        st.markdown("**Baseline Resistivity (Rt baseline)**")
        baseline_rt_option = st.radio(
            "Select baseline method:",
            ["Auto-calculate", "Manual input"],
            key="baseline_rt_radio",
            help="Auto: uses 5th percentile of data | Manual: enter specific value"
        )
        
        if baseline_rt_option == "Manual input":
            baseline_RT = st.number_input(
                "Enter Rt baseline value (ohm.m):",
                min_value=0.1,
                value=10.0,
                step=0.5,
                key="baseline_rt_input",
                help="Manual baseline resistivity value for ΔlogR calculation"
            )
        else:
            baseline_RT = None
        
        # Baseline GR Input
        st.markdown("**Baseline Gamma Ray (GR baseline)**")
        baseline_gr_option = st.radio(
            "Select baseline method:",
            ["Auto-calculate", "Manual input"],
            key="baseline_gr_radio",
            help="Auto: uses 5th percentile of data | Manual: enter specific value"
        )
        
        if baseline_gr_option == "Manual input":
            baseline_GR = st.number_input(
                "Enter GR baseline value (API):",
                min_value=0.0,
                value=50.0,
                step=5.0,
                key="baseline_gr_input",
                help="Manual baseline GR value for ΔlogR calculation"
            )
        else:
            baseline_GR = None
        
        st.markdown("---")
        
        # Method selection
        st.markdown("### ⚙️ Methods")
        
        toc_method = st.selectbox(
            "TOC Estimation Method",
            ['passey', 'schmoker', 'ensemble'],
            key="toc_method_select",
            help="Method for estimating Total Organic Carbon"
        )
        
        brittleness_method = st.selectbox(
            "Brittleness Calculation Method",
            ['rickman', 'wang', 'jarvie', 'comprehensive'],
            key="brittleness_method_select",
            help="Method for calculating Brittleness Index"
        )
        
        st.markdown("---")
        
        # Thresholds
        st.markdown("### 🎯 Thresholds")
        
        urt_threshold = st.slider(
            "URT_Index Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            key="urt_threshold_slider",
            help="Threshold for sweet spot identification"
        )
        
        fhzi_threshold = st.slider(
            "FHZI Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            key="fhzi_threshold_slider",
            help="Threshold for sweet spot identification"
        )
        
        st.markdown("---")
        
        # Lab data upload
        st.markdown("### 🔬 Calibration Data")
        lab_file = st.file_uploader(
            "Upload lab TOC data (optional)",
            type=['csv'],
            key="lab_file_uploader",
            help="Columns: Depth, TOC_lab"
        )
        
        st.markdown("---")
        
        # Run button
        run_workflow = st.button("🚀 Run Workflow", type="primary", use_container_width=True, key="run_workflow_button")
    
    # Main content
    if use_sample:
        # Generate sample data
        df_original = generate_sample_data()
        if df_original is not None:
            st.session_state['df_original'] = df_original
            st.session_state['data_source'] = 'sample'
            
            if run_workflow or 'df_processed' not in st.session_state:
                with st.spinner('Processing data...'):
                    df = load_and_preprocess_data(df_original.copy(), toc_method=toc_method,
                                                 LOM=LOM, baseline_RT=baseline_RT, baseline_GR=baseline_GR)
                    
                    if df is not None:
                        df = calculate_brittleness_index(df, method=brittleness_method)
                        df = calculate_twc(df)
                        df = calculate_urt_index(df)
                        df = calculate_fhzi(df)
                        df = classify_rock_types(df)
                        df = identify_sweet_spots(df, urt_threshold, fhzi_threshold)
                        df = optimize_fracture_stages(df)
                        
                        # Load lab data if available
                        lab_data = None
                        if lab_file is not None:
                            try:
                                lab_data = pd.read_csv(lab_file)
                                if 'Depth' in lab_data.columns and 'TOC_lab' in lab_data.columns:
                                    df = calibrate_toc_with_lab_data(df, lab_data)
                                    st.session_state['lab_data'] = lab_data
                                    st.success(f"✅ Loaded {len(lab_data)} lab calibration points")
                            except Exception as e:
                                st.warning(f"Could not load lab data: {str(e)}")
                        
                        st.session_state['df_processed'] = df
                        st.session_state['data_loaded'] = True
                        st.rerun()
    
    elif uploaded_file is not None:
        try:
            df_original = pd.read_csv(uploaded_file)
            st.session_state['df_original'] = df_original
            st.session_state['data_source'] = 'upload'
            
            if run_workflow or 'df_processed' not in st.session_state:
                with st.spinner('Processing data...'):
                    # Check required columns
                    required_cols = ['PHIE', 'VP', 'VS', 'RHOB', 'VSH', 'RT', 'GR']
                    missing_cols = [col for col in required_cols if col not in df_original.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                        st.info("Please ensure your CSV file contains all required columns.")
                        st.session_state['data_loaded'] = False
                    else:
                        df = load_and_preprocess_data(df_original.copy(), toc_method=toc_method,
                                                     LOM=LOM, baseline_RT=baseline_RT, baseline_GR=baseline_GR)
                        
                        if df is not None:
                            df = calculate_brittleness_index(df, method=brittleness_method)
                            df = calculate_twc(df)
                            df = calculate_urt_index(df)
                            df = calculate_fhzi(df)
                            df = classify_rock_types(df)
                            df = identify_sweet_spots(df, urt_threshold, fhzi_threshold)
                            df = optimize_fracture_stages(df)
                            
                            # Load lab data if available
                            lab_data = None
                            if lab_file is not None:
                                try:
                                    lab_data = pd.read_csv(lab_file)
                                    if 'Depth' in lab_data.columns and 'TOC_lab' in lab_data.columns:
                                        df = calibrate_toc_with_lab_data(df, lab_data)
                                        st.session_state['lab_data'] = lab_data
                                        st.success(f"✅ Loaded {len(lab_data)} lab calibration points")
                                except Exception as e:
                                    st.warning(f"Could not load lab data: {str(e)}")
                            
                            st.session_state['df_processed'] = df
                            st.session_state['data_loaded'] = True
                            st.rerun()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state['data_loaded'] = False
    
    # Display results if data is processed
    if st.session_state.get('data_loaded', False) and 'df_processed' in st.session_state:
        df = st.session_state['df_processed']
        lab_data = st.session_state.get('lab_data', None)
        
        # Display TOC parameters used
        st.markdown('<div class="sub-header">⚙️ TOC Estimation Parameters Used</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>LOM Value</h4>
                    <h3>{LOM}</h3>
                    <small>Level of Organic Maturity</small>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            actual_RT = df.attrs.get('baseline_RT', 'Auto-calculated')
            if isinstance(actual_RT, (int, float)):
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Baseline Resistivity</h4>
                        <h3>{actual_RT:.2f} ohm.m</h3>
                        <small>{'Manual' if baseline_RT is not None else 'Auto-calculated'}</small>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Baseline Resistivity</h4>
                        <h3>{actual_RT}</h3>
                        <small>Auto-calculated from data</small>
                    </div>
                """, unsafe_allow_html=True)
        
        with col3:
            actual_GR = df.attrs.get('baseline_GR', 'Auto-calculated')
            if isinstance(actual_GR, (int, float)):
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Baseline Gamma Ray</h4>
                        <h3>{actual_GR:.2f} API</h3>
                        <small>{'Manual' if baseline_GR is not None else 'Auto-calculated'}</small>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4>Baseline Gamma Ray</h4>
                        <h3>{actual_GR}</h3>
                        <small>Auto-calculated from data</small>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Summary metrics
        st.markdown('<div class="sub-header">📊 Summary Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>TOC</h4>
                    <h3>{df['TOC'].mean():.2f}%</h3>
                    <small>Range: {df['TOC'].min():.2f} - {df['TOC'].max():.2f}</small>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Brittleness</h4>
                    <h3>{df['BRIT'].mean():.1f}%</h3>
                    <small>Range: {df['BRIT'].min():.1f} - {df['BRIT'].max():.1f}</small>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>URT_Index</h4>
                    <h3>{df['URT_Index'].mean():.3f}</h3>
                    <small>Range: {df['URT_Index'].min():.3f} - {df['URT_Index'].max():.3f}</small>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <h4>FHZI</h4>
                    <h3>{df['FHZI'].mean():.3f}</h3>
                    <small>Range: {df['FHZI'].min():.3f} - {df['FHZI'].max():.3f}</small>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            n_sweet = df[df['Sweet_Spot']].shape[0]
            sweet_pct = (n_sweet / len(df)) * 100
            st.markdown(f"""
                <div class="metric-card">
                    <h4>Sweet Spots</h4>
                    <h3>{n_sweet}</h3>
                    <small>{sweet_pct:.1f}% of samples</small>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Rock Type Distribution
        st.markdown('<div class="sub-header">🎯 Rock Type Distribution</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            rock_counts = df['Rock_Type'].value_counts()
            fig_rock = px.pie(
                values=rock_counts.values,
                names=rock_counts.index,
                title='Rock Type Distribution',
                color=rock_counts.index,
                color_discrete_map={
                    'RT1 (Best)': 'green',
                    'RT2 (Good)': 'yellowgreen',
                    'RT3 (Fair)': 'gold',
                    'RT4 (Poor)': 'red'
                }
            )
            fig_rock.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_rock, use_container_width=True)
        
        with col2:
            sweet_counts = df['Sweet_Spot_Class'].value_counts()
            fig_sweet = px.pie(
                values=sweet_counts.values,
                names=sweet_counts.index,
                title='Sweet Spot Classification',
                color=sweet_counts.index,
                color_discrete_map={
                    'Excellent': 'green',
                    'Good': 'orange',
                    'Poor': 'red'
                }
            )
            fig_sweet.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_sweet, use_container_width=True)
        
        st.markdown("---")
        
        # Visualizations
        st.markdown('<div class="sub-header">📈 Visualizations</div>', unsafe_allow_html=True)
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 TOC Analysis",
            "📈 URT & FHZI Distribution",
            "🪨 Well Log Visualization",
            "🌐 3D Visualization",
            "🔬 Correlation & Validation"
        ])
        
        with tab1:
            # Pass lab_data to TOC plot function
            st.plotly_chart(create_toc_plot(df, lab_data), use_container_width=True)
            
            # TOC statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean TOC", f"{df['TOC'].mean():.2f}%")
            with col2:
                st.metric("Median TOC", f"{df['TOC'].median():.2f}%")
            with col3:
                st.metric("Std Dev", f"{df['TOC'].std():.2f}%")
            
            # Show lab data info if available
            if lab_data is not None:
                lab_info = df.attrs.get('lab_data', None)
                if lab_info is not None:
                    st.info(f"📊 {len(lab_info['depths'])} lab calibration points shown as red dots on the TOC vs Depth plot")
        
        with tab2:
            st.plotly_chart(create_urt_distribution_plot(df), use_container_width=True)
        
        with tab3:
            st.plotly_chart(create_well_log_plot(df), use_container_width=True)
        
        with tab4:
            # Pass lab_data to 3D plot function
            st.plotly_chart(create_3d_plot(df, lab_data), use_container_width=True)
        
        with tab5:
            # Correlation heatmap
            st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)
            
            # TOC validation with lab data
            if lab_data is not None and 'Depth' in df.columns:
                st.plotly_chart(create_toc_validation_plot(df, lab_data), use_container_width=True)
            else:
                st.info("Upload lab calibration data to see TOC validation plot")
            
            # Stage optimization summary
            st.markdown("### 🎯 Stage Optimization Summary")
            stage_counts = df['Stage_Recommendation'].value_counts()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Priority", stage_counts.get('Stimulate (High Priority)', 0))
            with col2:
                st.metric("Medium Priority", stage_counts.get('Stimulate (Medium Priority)', 0))
            with col3:
                st.metric("Avoid Stimulation", stage_counts.get('Avoid Stimulation', 0))
            
            st.info(f"Recommended number of fracture stages: **{df['Recommended_Stages'].iloc[0]}**")
        
        # Download results
        st.markdown("---")
        st.markdown('<div class="sub-header">💾 Download Results</div>', unsafe_allow_html=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data as CSV",
            data=csv,
            file_name="urt_index_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_button"
        )
        
        # Success message
        st.markdown("""
            <div class="success-box">
                ✅ Workflow completed successfully! Explore the interactive visualizations above.
            </div>
        """, unsafe_allow_html=True)
    
    else:
        # Instructions when no data is loaded
        st.markdown("""
            <div class="info-box">
                <h4>🚀 Getting Started</h4>
                <p>Upload your well log data or use the sample data to get started.</p>
                <ul>
                    <li><strong>Required columns:</strong> PHIE, VP, VS, RHOB, VSH, RT, GR</li>
                    <li><strong>Optional columns:</strong> Depth, TOC (will be estimated if missing)</li>
                    <li><strong>Optional calibration:</strong> Upload lab TOC data for validation</li>
                </ul>
                <p>Configure the methods and thresholds in the sidebar, then click "Run Workflow".</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
