import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('outputs', exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_data(filepath):
    """Load and prepare dataset"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Price change features
    df['price_change_pct'] = df['Close'].pct_change() * 100
    df['price_change_abs'] = df['Close'] - df['Close'].shift(1)
    df['price_lag1'] = df['Close'].shift(1)
    df['price_lag2'] = df['Close'].shift(2)
    df['price_lag3'] = df['Close'].shift(3)
    
    # Weather lag features
    df['temp_lag1'] = df['temperature_2m_max'].shift(1)
    df['temp_lag2'] = df['temperature_2m_max'].shift(2)
    df['precip_lag1'] = df['precipitation_sum'].shift(1)
    df['precip_lag2'] = df['precipitation_sum'].shift(2)
    
    # Interaction: pollination period
    if 'cal_is_pollination' in df.columns:
        df['pollination'] = df['cal_is_pollination'].astype(int)
    else:
        df['pollination'] = 0
    
    # Drop rows with NaN (due to lags)
    df = df.dropna()
    return df

def analyze_weather_price_relationship(df, commodity_name):
    """Main analysis: weather vs price relationship"""
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {commodity_name.upper()}")
    print(f"{'='*60}")
    print(f"Records: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # ---- 1. Correlation matrix ----
    print("\n--- Correlation with Price Change ---")
    core_cols = ['temperature_2m_max', 'precipitation_sum', 'et0_fao_evapotranspiration',
                 'price_change_pct', 'price_change_abs']
    if 'pollination' in df.columns:
        core_cols.append('pollination')
    
    corr_matrix = df[core_cols].corr(method='spearman')
    print("\nSpearman correlation with price_change_pct:")
    print(corr_matrix['price_change_pct'].sort_values(ascending=False))
    
    # ---- 2. Categorize weather variables ----
    # Precipitation categories
    df['precip_cat'] = pd.cut(
        df['precipitation_sum'],
        bins=[-np.inf, 5, 25, 45, np.inf],
        labels=['Very Dry (<5)', 'Dry (5-25)', 'Normal (25-45)', 'Wet (>45)']
    )
    
    # Temperature categories
    df['temp_cat'] = pd.cut(
        df['temperature_2m_max'],
        bins=[-np.inf, 15, 25, 32, np.inf],
        labels=['Cold (<15°C)', 'Mild (15-25°C)', 'Warm (25-32°C)', 'Hot (>32°C)']
    )
    
    # ---- 3. Summary by categories ----
    print("\n--- Average Price Change by Precipitation Category ---")
    print(df.groupby('precip_cat', observed=True)['price_change_pct'].agg(['mean', 'std', 'count']))
    
    print("\n--- Average Price Change by Temperature Category ---")
    print(df.groupby('temp_cat', observed=True)['price_change_pct'].agg(['mean', 'std', 'count']))
    
    # ---- 4. Interaction with pollination ----
    if 'pollination' in df.columns:
        print("\n--- Price Change During Pollination Period ---")
        print(df.groupby('pollination')['price_change_pct'].agg(['mean', 'std', 'count']))
        
        # Combined: temp + pollination
        print("\n--- Price Change: Hot (>32°C) & Pollination ---")
        hot_poll = df[(df['temp_cat'] == 'Hot (>32°C)') & (df['pollination'] == 1)]
        if len(hot_poll) > 0:
            print(f"Records: {len(hot_poll)}")
            print(f"Mean price change: {hot_poll['price_change_pct'].mean():.2f}%")
            print(f"Std: {hot_poll['price_change_pct'].std():.2f}")
        else:
            print("No records for Hot + Pollination")
    
    # ---- 5. Lag analysis ----
    print("\n--- Lag Correlation with Current Price Change ---")
    lag_cols = ['temp_lag1', 'temp_lag2', 'precip_lag1', 'precip_lag2']
    for col in lag_cols:
        if col in df.columns:
            corr, pval = spearmanr(df[col], df['price_change_pct'])
            print(f"{col}: r = {corr:.3f} (p={pval:.3f})")
    
    # ---- 6. Scenario testing ----
    print("\n--- Scenario: Extreme Weather Events ---")
    
    # Drought: precipitation < 10 mm and dry_spell > 2
    if 'dry_spell' in df.columns:
        drought = df[(df['precipitation_sum'] < 10) & (df['dry_spell'] > 2)]
        print(f"Drought (precip<10, dry_spell>2): n={len(drought)}, mean price change = {drought['price_change_pct'].mean():.2f}%")
    else:
        drought = df[df['precipitation_sum'] < 10]
        print(f"Low precip (<10mm): n={len(drought)}, mean price change = {drought['price_change_pct'].mean():.2f}%")
    
    # Flood: precipitation > 60 mm
    flood = df[df['precipitation_sum'] > 60]
    print(f"Heavy rain (>60mm): n={len(flood)}, mean price change = {flood['price_change_pct'].mean():.2f}%")
    
    # Ideal: temp 20-28 and precip 20-40
    ideal = df[(df['temperature_2m_max'] >= 20) & (df['temperature_2m_max'] <= 28) &
               (df['precipitation_sum'] >= 20) & (df['precipitation_sum'] <= 40)]
    print(f"Ideal conditions (temp 20-28, rain 20-40): n={len(ideal)}, mean price change = {ideal['price_change_pct'].mean():.2f}%")
    
    # ---- 7. Plotting ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{commodity_name.upper()} - Weather vs Price Change', fontsize=16)
    
    # Scatter: temperature vs price change (colored by pollination)
    ax1 = axes[0, 0]
    if 'pollination' in df.columns:
        scatter = ax1.scatter(df['temperature_2m_max'], df['price_change_pct'],
                              c=df['pollination'], cmap='coolwarm', alpha=0.6, edgecolors='k')
        ax1.set_title('Temperature vs Price Change (color=pollination)')
        plt.colorbar(scatter, ax=ax1, label='Pollination')
    else:
        ax1.scatter(df['temperature_2m_max'], df['price_change_pct'], alpha=0.6)
        ax1.set_title('Temperature vs Price Change')
    ax1.set_xlabel('Max Temperature (°C)')
    ax1.set_ylabel('Price Change (%)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Scatter: precipitation vs price change
    ax2 = axes[0, 1]
    ax2.scatter(df['precipitation_sum'], df['price_change_pct'], alpha=0.6, color='steelblue')
    ax2.set_title('Precipitation vs Price Change')
    ax2.set_xlabel('Precipitation (mm)')
    ax2.set_ylabel('Price Change (%)')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Boxplot: price change by temperature category
    ax3 = axes[1, 0]
    df_temp = df.dropna(subset=['temp_cat', 'price_change_pct'])
    if len(df_temp) > 0:
        sns.boxplot(x='temp_cat', y='price_change_pct', data=df_temp, ax=ax3)
        ax3.set_title('Price Change by Temperature Category')
        ax3.set_xlabel('Temperature Category')
        ax3.set_ylabel('Price Change (%)')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Boxplot: price change by precipitation category
    ax4 = axes[1, 1]
    df_precip = df.dropna(subset=['precip_cat', 'price_change_pct'])
    if len(df_precip) > 0:
        sns.boxplot(x='precip_cat', y='price_change_pct', data=df_precip, ax=ax4)
        ax4.set_title('Price Change by Precipitation Category')
        ax4.set_xlabel('Precipitation Category')
        ax4.set_ylabel('Price Change (%)')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.setp(ax4.get_xticklabels(), rotation=15)
    
    plt.tight_layout()
    plt.savefig(f'outputs/{commodity_name}_weather_price_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Chart saved: outputs/{commodity_name}_weather_price_analysis.png")
    
    return df

# ---- MAIN ----
if __name__ == "__main__":
    print("="*70)
    print("WEATHER-PRICE PATTERN EXPLORATION")
    print("="*70)
    
    # Process Corn
    try:
        df_corn = load_data('dataset/integrated_corn_weekly.csv')
        analyze_weather_price_relationship(df_corn, 'corn')
    except FileNotFoundError:
        print("⚠️ Corn dataset not found at dataset/integrated_corn_weekly.csv")
    
    # Process Coffee
    try:
        df_coffee = load_data('dataset/integrated_coffee_weekly.csv')
        analyze_weather_price_relationship(df_coffee, 'coffee')
    except FileNotFoundError:
        print("⚠️ Coffee dataset not found at dataset/integrated_coffee_weekly.csv")
    
    print("\n" + "="*70)
    print("✅ Analysis complete. Check 'outputs/' directory for charts.")
    print("="*70)
