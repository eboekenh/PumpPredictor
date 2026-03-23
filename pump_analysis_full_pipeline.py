# %%

## 3. Detailed Exploratory Data Analysis (EDA - Exploratory Data Analysis)

### 3.1 Dataset Inspection

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load datasets
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

train_values = pd.read_csv(os.path.join(DATA_DIR, 'training_set_values.csv'))
train_labels = pd.read_csv(os.path.join(DATA_DIR, 'training_set_labels.csv'))
test_values = pd.read_csv(os.path.join(DATA_DIR, 'training_set_values.csv'))

# %%
# Merge training data
# train_values: Features
# train_labels: Target variable
train_df = train_values.merge(train_labels, on='id')

print(f"Training set size: {train_df.shape}")
print(f"Test set size: {test_values.shape}")
print(f"\nFirst 5 rows:")
print(train_df.head())

# %%

### 3.2 Data Structure and Types

# Inspect data types and missing values
def analyze_data_structure(df, name='Dataset'):
    """
    Detailed analysis of dataset structure
    
    Parameters:
    -----------
    df : DataFrame
        Dataset to analyze
    name : str
        Dataset name (displayed in output)
    """
    print(f"\n{'='*60}")
    print(f"{name} - General Information")
    print(f"{'='*60}\n")
    
    # Basic information
    print(f"Number of rows: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Total cell count: {df.shape[0] * df.shape[1]:,}")
    
    # Veri tipleri
    print(f"\n{'='*60}")
    print("Data Type Distribution:")
    print(f"{'='*60}")
    print(df.dtypes.value_counts())
    
    # Missing values
    print(f"\n{'='*60}")
    print("Missing Value Analysis:")
    print(f"{'='*60}")
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing Percentage (%)': missing_pct.values
    })
    
    # Show only columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing Count', ascending=False
    )
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")
    
    return missing_df

# Run analysis
missing_analysis = analyze_data_structure(train_df, 'Training Set')

# %%

### 3.3 Target Variable Distribution

def plot_target_distribution(df, target_col='status_group'):
    """
    Visualize target variable distribution
    
    Parameters:
    -----------
    df : DataFrame
        Veri seti
    target_col : str
        Target variable column name
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Class distribution - Count
    target_counts = df[target_col].value_counts()
    axes[0].bar(target_counts.index, target_counts.values, 
                color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[0].set_title('Pump Status Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Durum', fontsize=12)
    axes[0].set_ylabel('Pump Count', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Write values on bars
    for i, v in enumerate(target_counts.values):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
    
    # Class distribution - Percentage
    target_pct = 100 * target_counts / len(df)
    axes[1].pie(target_pct.values, labels=target_pct.index, autopct='%1.1f%%',
                colors=['#2ecc71', '#f39c12', '#e74c3c'], startangle=90)
    axes[1].set_title('Pump Status Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical summary
    print("\n" + "="*60)
    print("Target Variable Statistics:")
    print("="*60)
    summary_df = pd.DataFrame({
        'Durum': target_counts.index,
        'Count': target_counts.values,
        'Percentage (%)': target_pct.values
    })
    print(summary_df.to_string(index=False))
    
    # Imbalance ratio (imbalance ratio)
    max_class = target_counts.max()
    min_class = target_counts.min()
    imbalance_ratio = max_class / min_class
    
    print(f"\n⚠️  Class Imbalance Ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 2:
        print("   → Data is imbalanced! Sampling techniques may be needed.")
    else:
        print("   → Data appears balanced.")

# Run visualization
plot_target_distribution(train_df)

# %%

### 3.4 Categorical Variable Analysis

def analyze_categorical_features(df, target_col='status_group', top_n=5):
    """
    Analyzes categorical variables and their relationship with the target
    
    Parameters:
    -----------
    df : DataFrame
        Veri seti
    target_col : str
        Target variable
    top_n : int
        Number of top values to show per category
    """
    # Find categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Target variablei exclude
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # Exclude 'id' column
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    
    print(f"\n{'='*60}")
    print(f"Toplam {len(categorical_cols)} categorical variables found")
    print(f"{'='*60}\n")
    
    for col in categorical_cols[:10]:  # Inspect first 10 categorical variables
        print(f"\n{'─'*60}")
        print(f"📊 Variable: {col}")
        print(f"{'─'*60}")
        
        # Unique value count
        n_unique = df[col].nunique()
        print(f"Unique value count: {n_unique}")
        
        # Top valueler
        top_values = df[col].value_counts().head(top_n)
        print(f"\nTop {top_n} value:")
        for val, count in top_values.items():
            pct = 100 * count / len(df)
            print(f"  • {val}: {count:,} ({pct:.2f}%)")
        
        # Target variablele cross table
        if n_unique <= 10:  # Sadece az kategorili variableler for
            print(f"\n{col} - {target_col} Relationship:")
            ct = pd.crosstab(df[col], df[target_col], normalize='index') * 100
            print(ct.round(2))

# Run analysis
analyze_categorical_features(train_df)

# %%
### 3.5 Numerical Variable Analysis

def analyze_numerical_features(df, target_col='status_group'):
    """
    Countsal variableleri analysis eder
    
    Parameters:
    -----------
    df : DataFrame
        Veri seti
    target_col : str
        Target variable
    """
    # Find numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Exclude 'id' column
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    
    print(f"\n{'='*60}")
    print(f"Toplam {len(numerical_cols)} numerical variables found")
    print(f"{'='*60}\n")
    
    # Basic statistics
    stats_df = df[numerical_cols].describe().T
    stats_df['missing'] = df[numerical_cols].isnull().sum()
    stats_df['missing_pct'] = 100 * stats_df['missing'] / len(df)
    
    print("Basic Statistics:")
    print(stats_df.round(2))
    
    # Visualization: Box plot
    fig, axes = plt.subplots(
        nrows=(len(numerical_cols) + 2) // 3, 
        ncols=3, 
        figsize=(18, 4 * ((len(numerical_cols) + 2) // 3))
    )
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        # Box plot for outlier inspection
        df.boxplot(column=col, by=target_col, ax=axes[idx])
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xlabel('Pump Status')
        axes[idx].set_ylabel(col)
        plt.sca(axes[idx])
        plt.xticks(rotation=45)
    
    # Hide unused subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('numerical_features_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run analysis
analyze_numerical_features(train_df)

# %%
### 3.6 Geographical Analysis

def plot_geographical_distribution(df):
    """
    Visualizes geographical distribution of pumps
    
    Parameters:
    -----------
    df : DataFrame
        Dataset (must contain latitude, longitude, status_group)
    """
    # Filter valid coordinates (non-zero)
    geo_df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
    
    print(f"Valid coordinate count: {len(geo_df):,} / {len(df):,}")
    
    # Status encoding (for coloring)
    status_colors = {
        'functional': '#2ecc71',
        'functional needs repair': '#f39c12',
        'non functional': '#e74c3c'
    }
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for status, color in status_colors.items():
        mask = geo_df['status_group'] == status
        ax.scatter(
            geo_df[mask]['longitude'], 
            geo_df[mask]['latitude'],
            c=color, 
            label=status,
            alpha=0.5,
            s=10
        )
    
    ax.set_xlabel('Longitude (Longitude)', fontsize=12)
    ax.set_ylabel('Latitude (Latitude)', fontsize=12)
    ax.set_title('Tanzania Water Pumps - Geographical Distribution', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('geographical_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Regionsel statistics
    print("\nRegional Distribution (Region):")
    region_stats = pd.crosstab(
        df['region'], 
        df['status_group'], 
        normalize='index'
    ) * 100
    print(region_stats.round(2))

# Run visualization
plot_geographical_distribution(train_df)

# %%
## 4. Data Preprocessing (Data Preprocessing)

### 4.1 Missing Value Handling

def handle_missing_values(df):
    """
    Missing valuesi processr
    
    Parameters:
    -----------
    df : DataFrame
        Dataset to process
    
    Returns:
    --------
    DataFrame
        Dataset with processed missing values
    """
    df_clean = df.copy()
    
    # 1. Countsal variablelerde eksik valueleri medyan with doldur
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"✓ {col}: Missing values {median_val} with filled")
    
    # 2. Kategorik variablelerde eksik valueleri mod (en common value) with doldur
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"✓ {col}: Missing values '{mode_val}' with filled")
    
    # 3. Special cases
    # In some numerical columns, 0 may represent missing values
    # For example: population, construction_year, gps_height
    
    # construction_year = 0 ise (bilinmiyor), medyan with doldur
    if 'construction_year' in df_clean.columns:
        mask = df_clean['construction_year'] == 0
        if mask.sum() > 0:
            valid_years = df_clean[df_clean['construction_year'] > 0]['construction_year']
            median_year = valid_years.median()
            df_clean.loc[mask, 'construction_year'] = median_year
            print(f"✓ construction_year: 0 values replaced {median_year} with replaced")
    
    # gps_height = 0 ise (deniz seviyesinde veya bilinmiyor)
    # Bu durumda average with doldurmak daha mantıklı
    if 'gps_height' in df_clean.columns:
        mask = df_clean['gps_height'] == 0
        if mask.sum() > 0:
            mean_height = df_clean[df_clean['gps_height'] != 0]['gps_height'].mean()
            df_clean.loc[mask, 'gps_height'] = mean_height
            print(f"✓ gps_height: 0 valueleri {mean_height:.2f} with replaced")
    
    # longitude/latitude = 0 ise (konum bilinmiyor), bölge averagesı with doldur
    if 'latitude' in df_clean.columns and 'longitude' in df_clean.columns:
        mask = (df_clean['latitude'] == 0) | (df_clean['longitude'] == 0)
        if mask.sum() > 0:
            # Region bazında average koordinatlar
            if 'region' in df_clean.columns:
                for region in df_clean['region'].unique():
                    region_mask = (df_clean['region'] == region) & mask
                    if region_mask.sum() > 0:
                        region_coords = df_clean[
                            (df_clean['region'] == region) & 
                            (df_clean['latitude'] != 0)
                        ]
                        if len(region_coords) > 0:
                            mean_lat = region_coords['latitude'].mean()
                            mean_lon = region_coords['longitude'].mean()
                            df_clean.loc[region_mask, 'latitude'] = mean_lat
                            df_clean.loc[region_mask, 'longitude'] = mean_lon
            
            print(f"✓ latitude/longitude: 0 valueleri bölge averageları with replaced")
    
    print(f"\n{'='*60}")
    print("Missing Value Handling Tamamlandı!")
    print(f"{'='*60}")
    print(f"Kalan eksik value sayısı: {df_clean.isnull().sum().sum()}")
    
    return df_clean

# Apply function
train_clean = handle_missing_values(train_df)
test_clean = handle_missing_values(test_values)

### 4.2 Categorical Variable Encoding

from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(train_df, test_df, target_col='status_group'):
    """
    Kategorik variableleri sayısal valuelere dönüştürür
    
    Parameters:
    -----------
    train_df : DataFrame
        Training veri seti
    test_df : DataFrame
        Test veri seti
    target_col : str
        Target variable (encoding'e dahil edilmeyecek)
    
    Returns:
    --------
    tuple
        (train_encoded, test_encoded, label_encoders)
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # Find categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Target variablei ve id'yi exclude
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    
    # Her kategorik variable for LabelEncoder
    label_encoders = {}
    
    for col in categorical_cols:
        print(f"Encoding: {col}...")
        
        # LabelEncoder create
        le = LabelEncoder()
        
        # Train ve test'i merge (tüm categories learningk for)
        combined = pd.concat([
            train_df[col].astype(str), 
            test_df[col].astype(str)
        ])
        
        # Fit et
        le.fit(combined)
        
        # Transform et
        train_encoded[col] = le.transformrm(train_df[col].astype(str))
        test_encoded[col] = le.transformrm(test_df[col].astype(str))
        
        # Encoder'ı sakla (gelecekte new giveni encode etmek for)
        label_encoders[col] = le
        
        print(f"  ✓ {col}: {len(le.classes_)} benzersiz kategori encode edildi")
    
    # Target variablei de encode et (sadece train for)
    if target_col in train_encoded.columns:
        target_le = LabelEncoder()
        train_encoded[target_col] = target_le.fit_transformrm(train_df[target_col])
        label_encoders[target_col] = target_le
        
        print(f"\n✓ Target variable ({target_col}) encode edildi:")
        for idx, label in enumerate(target_le.classes_):
            print(f"  {label} → {idx}")
    
    print(f"\n{'='*60}")
    print("Kategorik Encoding Tamamlandı!")
    print(f"{'='*60}")
    
    return train_encoded, test_encoded, label_encoders

# Apply encoding
train_encoded, test_encoded, encoders = encode_categorical_features(
    train_clean, 
    test_clean
)

### 4.3 Feature Scaling (Feature Scaling)

from sklearn.preprocessing import StandardScaler

def scale_features(train_df, test_df, target_col='status_group'):
    """
    Countsal featureleri standartlaştırır (0 average, 1 standart sapma)
    
    Parameters:
    -----------
    train_df : DataFrame
        Training veri seti
    test_df : DataFrame
        Test veri seti
    target_col : str
        Target variable (ölçaddndirmeye dahil edilmeyecek)
    
    Returns:
    --------
    tuple
        (train_scaled, test_scaled, scaler)
    """
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    # Find numerical columns
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # id ve target variablei exclude
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    # StandardScaler create
    scaler = StandardScaler()
    
    # Train setine fit et
    scaler.fit(train_df[numerical_cols])
    
    # Hem train hem test'i transformrm et
    train_scaled[numerical_cols] = scaler.transformrm(train_df[numerical_cols])
    test_scaled[numerical_cols] = scaler.transformrm(test_df[numerical_cols])
    
    print(f"{'='*60}")
    print(f"Ölçaddndirme Tamamlandı!")
    print(f"{'='*60}")
    print(f"Scaled variable sayısı: {len(numerical_cols)}")
    print(f"\Scaled variableler:")
    for col in numerical_cols:
        original_mean = train_df[col].mean()
        scaled_mean = train_scaled[col].mean()
        print(f"  • {col}: {original_mean:.2f} → {scaled_mean:.6f}")
    
    return train_scaled, test_scaled, scaler

# Apply scaling
train_scaled, test_scaled, scaler = scale_features(train_encoded, test_encoded)


## 5. Feature Engineering (Feature Engineering)

### 5.1 Date-Based Features

def create_date_features(df):
    """
    Tarih columnsından new featureler türetir
    
    Parameters:
    -----------
    df : DataFrame
        Veri seti
    
    Returns:
    --------
    DataFrame
        New featureler addnmiş veri seti
    """
    df_new = df.copy()
    
    # Mevcut yıl (veri seti 2013'te toplanmış)
    current_year = 2013
    
    if 'construction_year' in df_new.columns:
        # Pompanın yaşı
        df_new['pump_age'] = current_year - df_new['construction_year']
        
        # Negatif yaşları 0 yap (henüz inşa edilmemiş)
        df_new.loc[df_new['pump_age'] < 0, 'pump_age'] = 0
        
        print(f"✓ 'pump_age' özelliği created")
        print(f"  Average pump age: {df_new['pump_age'].mean():.2f} yıl")
        
        # Pump age category
        df_new['pump_age_category'] = pd.cut(
            df_new['pump_age'],
            bins=[0, 5, 10, 20, 100],
            labels=['New (0-5)', 'Genç (5-10)', 'Orta (10-20)', 'Eski (20+)']
        )
        print(f"✓ 'pump_age_category' özelliği created")
    
    # date_recorded varsa (pompanın kaydedilme tarihi)
    if 'date_recorded' in df_new.columns:
        df_new['date_recorded'] = pd.to_datetime(df_new['date_recorded'])
        
        # Ay
        df_new['recorded_month'] = df_new['date_recorded'].dt.month
        print(f"✓ 'recorded_month' özelliği created")
        
        # Mevsim
        df_new['recorded_season'] = df_new['recorded_month'].apply(
            lambda x: 'Kış' if x in [12, 1, 2] else
                     'Firstbahar' if x in [3, 4, 5] else
                     'Yaz' if x in [6, 7, 8] else 'Sonbahar'
        )
        print(f"✓ 'recorded_season' özelliği created")
        
        # Day of year
        df_new['recorded_day_of_year'] = df_new['date_recorded'].dt.dayofyear
    
    print(f"\n{'='*60}")
    print("Date-Based Features Oluşturuldu!")
    print(f"{'='*60}")
    
    return df_new

# Create date features
train_with_dates = create_date_features(train_scaled)
test_with_dates = create_date_features(test_scaled)

### 5.2 Geographical Features

def create_geographical_features(df):
    """
    Coğrafi koordinatlardan new featureler türetir
    
    Parameters:
    -----------
    df : DataFrame
        Veri seti (latitude, longitude içermeli)
    
    Returns:
    --------
    DataFrame
        New coğrafi featureler addnmiş veri seti
    """
    df_new = df.copy()
    
    if 'latitude' in df_new.columns and 'longitude' in df_new.columns:
        # Tanzanya'nın merkezi (yaklaşık)
        tanzania_center_lat = -6.369028
        tanzania_center_lon = 34.888822
        
        # Merkezden uzaklık (Haversine formülü with)
        from math import radians, sin, cos, sqrt, atan2
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """
            İki GPS koordinatı betweenki mesafeyi calculater (km)
            """
            R = 6371  # Dünya yarıçapı (km)
            
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = R * c
            
            return distance
        
        # Her pompa for merkeze uzaklığı calculate
        df_new['distance_from_center'] = df_new.apply(
            lambda row: haversine_distance(
                row['latitude'], row['longitude'],
                tanzania_center_lat, tanzania_center_lon
            ),
            axis=1
        )
        
        print(f"✓ 'distance_from_center' özelliği created")
        print(f"  Average uzaklık: {df_new['distance_from_center'].mean():.2f} km")
        
        # Yükselti kategorisi (gps_height varsa)
        if 'gps_height' in df_new.columns:
            df_new['elevation_category'] = pd.cut(
                df_new['gps_height'],
                bins=[-100, 500, 1000, 1500, 3000],
                labels=['Düşük', 'Orta', 'Yüksek', 'Çok Yüksek']
            )
            print(f"✓ 'elevation_category' özelliği created")
    
    print(f"\n{'='*60}")
    print("Geographical Features Oluşturuldu!")
    print(f"{'='*60}")
    
    return df_new

# Create geographical features
train_with_geo = create_geographical_features(train_with_dates)
test_with_geo = create_geographical_features(test_with_dates)


### 5.3 Aggregation-Based Features

def create_aggregated_features(train_df, test_df, group_cols):
    """
    Creates group-based statistical features
    
    Parameters:
    -----------
    train_df : DataFrame
        Training veri seti
    test_df : DataFrame
        Test veri seti
    group_cols : list
        Columns for grouping
    
    Returns:
    --------
    tuple
        (train_with_agg, test_with_agg)
    """
    train_new = train_df.copy()
    test_new = test_df.copy()
    
    # Her grup for pompa sayısı ve average yaş
    for col in group_cols:
        if col in train_new.columns:
            print(f"\n{'─'*60}")
            print(f"Gruplama: {col}")
            print(f"{'─'*60}")
            
            # Pump count (bu kategoride kaç pompa var?)
            group_counts = train_new[col].value_counts().to_dict()
            train_new[f'{col}_pump_count'] = train_new[col].map(group_counts)
            test_new[f'{col}_pump_count'] = test_new[col].map(group_counts)
            print(f"✓ '{col}_pump_count' created")
            
            # Average pump age (bu kategoride pompalar average kaç yaşında?)
            if 'pump_age' in train_new.columns:
                age_mean = train_new.groupby(col)['pump_age'].mean().to_dict()
                train_new[f'{col}_avg_age'] = train_new[col].map(age_mean)
                test_new[f'{col}_avg_age'] = test_new[col].map(age_mean)
                print(f"✓ '{col}_avg_age' created")
            
            # Failure rate (bu kategoride ne kadar pompa bozuk?)
            if 'status_group' in train_new.columns:
                # functional = 0, functional needs repair = 1, non functional = 2
                # Failure rate = non functional sayısı / total
                failure_rate = train_new.groupby(col)['status_group'].apply(
                    lambda x: (x == 2).sum() / len(x)
                ).to_dict()
                
                train_new[f'{col}_failure_rate'] = train_new[col].map(failure_rate)
                test_new[f'{col}_failure_rate'] = test_new[col].map(failure_rate)
                print(f"✓ '{col}_failure_rate' created")
                print(f"  Average arıza oranı: {train_new[f'{col}_failure_rate'].mean():.2%}")
    
    print(f"\n{'='*60}")
    print("Aggregation-Based Features Created!")
    print(f"{'='*60}")
    
    return train_new, test_new

# Columns for grouping
group_columns = ['region', 'basin', 'installer', 'scheme_management', 'extraction_type']

# Create features
train_final, test_final = create_aggregated_features(
    train_with_geo, 
    test_with_geo, 
    group_columns
)



# %%

## 6. Model Development

### 6.1 Train/Test Split

from sklearn.model_selection import train_test_split

def prepare_modeling_data(df, target_col='status_group', test_size=0.2, random_state=42):
    """
    Vergood X (features) ve y (target) olarak ayırır ve train-validation split yapar
    
    Parameters:
    -----------
    df : DataFrame
        Feature-engineered dataset
    target_col : str
        Target variable columnu
    test_size : float
        Validation set oranı
    random_state : int
        Reproducibility for seed
    
    Returns:
    --------
    tuple
        (X_train, X_val, y_train, y_val, feature_names)
    """
    # Kategorik yaş like object tipli new columnsı encode et
    df_model = df.copy()
    
    # Object ve category tipli columnsı encode et (beforeeden yapılmamışsa)
    from sklearn.preprocessing import LabelEncoder
    
    for col in df_model.select_dtypes(include=['object', 'category']).columns:
        if col != target_col and col != 'id':
            le = LabelEncoder()
            df_model[col] = le.fit_transformrm(df_model[col].astype(str))
    
    # ID ve date_recorded'ı exclude
    drop_cols = ['id']
    if 'date_recorded' in df_model.columns:
        drop_cols.append('date_recorded')
    
    # X ve y'yi ayır
    X = df_model.drop(columns=drop_cols + [target_col])
    y = df_model[target_col]
    
    # Feature isimlerini sakla
    feature_names = X.columns.tolist()
    
    # Train-Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Sınıf dengesini koru
    )
    
    print(f"{'='*60}")
    print("Dataset Prepared!")
    print(f"{'='*60}")
    print(f"Total feature sayısı: {len(feature_names)}")
    print(f"Training set size: {X_train.shape}")
    print(f"Validation seti boyutu: {X_val.shape}")
    print(f"\nSınıf distributionı (Train):")
    print(y_train.value_counts())
    print(f"\nSınıf distributionı (Validation):")
    print(y_val.value_counts())
    
    return X_train, X_val, y_train, y_val, feature_names

# Prepare data
X_train, X_val, y_train, y_val, features = prepare_modeling_data(train_final)


### 6.2 Baseline Model (Random Forest)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Random Forest modeli eğitir ve valuelendirir
    
    Parameters:
    -----------
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation verisi
    
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    print(f"{'='*60}")
    print("Random Forest Modeli Eğitiliyor...")
    print(f"{'='*60}\n")
    
    # Model parametersi
    rf_params = {
        'n_estimators': 100,        # Tree sayısı
        'max_depth': 20,             # Maksimum derinlik
        'min_samples_split': 10,     # Split for minimum örnek
        'min_samples_leaf': 4,       # Leafta minimum örnek
        'random_state': 42,
        'n_jobs': -1,                # Tüm CPU core'ları kullan
        'class_weight': 'balanced'   # Dengesiz classları dengele
    }
    
    # Modeli create
    rf_model = RandomForestClassifier(**rf_params)
    
    # Training süresi ölç
    start_time = time.time()
    
    # Modeli eğit
    rf_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"✓ Model eğitildi! Süre: {training_time:.2f} saniye\n")
    
    # Predictionler
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    
    # Performans metrikleri
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"{'─'*60}")
    print("Model Performance:")
    print(f"{'─'*60}")
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    # Overfitting kontrolü
    overfit_diff = train_accuracy - val_accuracy
    if overfit_diff > 0.05:
        print(f"\n⚠️  Overfitting tespit edildi! Fark: {overfit_diff:.4f}")
    else:
        print(f"\n✓ Overfitting yok. Fark: {overfit_diff:.4f}")
    
    # Detailed classification report
    print(f"\n{'─'*60}")
    print("Classification Reportu (Validation Set):")
    print(f"{'─'*60}")
    print(classification_report(y_val, y_val_pred, 
                                target_names=['Functional', 'Needs Repair', 'Non Functional']))
    
    # Confusion Matrix
    print(f"{'─'*60}")
    print("Confusion Matrix:")
    print(f"{'─'*60}")
    cm = confusion_matrix(y_val, y_val_pred)
    print(cm)
    
    # Confusion matrix visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Functional', 'Needs Repair', 'Non Functional'],
                yticklabels=['Functional', 'Needs Repair', 'Non Functional'])
    plt.title('Confusion Matrix - Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig('confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_model
# Train Random Forest model
rf_model = train_random_forest(X_train, y_train, X_val, y_val)

# %%

    
    

# %%
### 6.3 Feature Importance Analysis

def analyze_feature_importance(model, feature_names, top_n=20):
    """
    Model'in feature önem skorlarını analysis eder ve visualizeir
    
    Parameters:
    -----------
    model : sklearn model
        Trained model (feature_importances_ attribute'u olmalı)
    feature_names : list
        Feature names
    top_n : int
        to show en important feature sayısı
    """
    # Feature importance scores
    importances = model.feature_importances_
    
    # DataFrame create
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"{'='*60}")
    print(f"Top {top_n} Feature:")
    print(f"{'='*60}\n")
    
    for idx, row in feature_importance_df.head(top_n).iterrows():
        print(f"{row['feature']:30s} : {row['importance']:.6f}")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    top_features = feature_importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'].values, 
             color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Önem Skoru', fontsize=12)
    plt.title(f'Top {top_n} Feature', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # En important üstte
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

# Analyze feature importance
feature_importance = analyze_feature_importance(rf_model, features, top_n=20)

# %%

### 6.4 Gradient Boosting Models (XGBoost & LightGBM)

import xgboost as xgb
import lightgbm as lgb

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    XGBoost modeli eğitir
    
    Returns:
    --------
    xgb.XGBClassifier
        Trained model
    """
    print(f"{'='*60}")
    print("XGBoost Modeli Eğitiliyor...")
    print(f"{'='*60}\n")
    
    # Sınıf ağırlıklarını calculate
    from sklearn.utils.class_weight import compute_class_weight
    
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    
    # XGBoost parametersi
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,           # Her tree for rastgele %80 veri kullan
        'colsample_bytree': 0.8,    # Her tree for rastgele %80 feature kullan
        'objective': 'multi:softmax',
        'num_class': 3,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }
    
    # Modeli create
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    # Training sırasında validation setini izle
    eval_set = [(X_train, y_train), (X_val, y_val)]
    
    start_time = time.time()
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=50  # Her 50 iterasyonda bir log bas
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Model eğitildi! Süre: {training_time:.2f} saniye\n")
    
    # Performans valuelendirme
    y_val_pred = xgb_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    return xgb_model

# %%
def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    LightGBM modeli eğitir
    
    Returns:
    --------
    lgb.LGBMClassifier
        Trained model
    """
    print(f"{'='*60}")
    print("LightGBM Modeli Eğitiliyor...")
    print(f"{'='*60}\n")
    
    # LightGBM parametersi
    lgb_params = {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }
    
    # Modeli create
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    
    start_time = time.time()
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.log_evaluation(50)]  # Her 50 iterasyonda log
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✓ Model eğitildi! Süre: {training_time:.2f} saniye\n")
    
    # Performans valuelendirme
    y_val_pred = lgb_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    return lgb_model

# %%
# Train both models
xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)

# %%


### 6.5 Model Comparison and Best Model Selection

def compare_models(models_dict, X_val, y_val):
    """
    Birden fazla modeli karşılaştırır
    
    Parameters:
    -----------
    models_dict : dict
        Model isimleri ve model objeleri
    X_val, y_val : array-like
        Validation verisi
    
    Returns:
    --------
    DataFrame
        Model performrmans comparisonsı
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    results = []
    
    print(f"{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}\n")
    
    for name, model in models_dict.items():
        # Predictionler
        y_pred = model.predict(X_val)
        
        # Metricler
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted')
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall
        })
        
        print(f"{name}:")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}\n")
    
    # DataFrame'e dönüştür
    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results_df))
    width = 0.2
    
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i*width, results_df[metric], width, 
               label=metric, color=colors[i])
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Best modeli seç
    best_model_name = results_df.iloc[0]['Model']
    best_accuracy = results_df.iloc[0]['Accuracy']
    
    print(f"\n{'='*60}")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    return results_df, models_dict[best_model_name]

# Compare models
models = {
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model
}

comparison_results, best_model = compare_models(models, X_val, y_val)



## 7. Model Optimization (Hyperparameter Tuning)

### 7.1 Grid Search with Parametre Optimizationu

from sklearn.model_selection import GridSearchCV

def optimize_model_gridsearch(X_train, y_train, model_type='xgboost'):
    """
    Grid Search with hiperparametre optimizationu yapar
    
    Parameters:
    -----------
    X_train, y_train : array-like
        Training data
    model_type : str
        'xgboost', 'lightgbm', veya 'random_forest'
    
    Returns:
    --------
    model
        Optimize edilmiş en good model
    """
    print(f"{'='*60}")
    print(f"{model_type.upper()} - Grid Search Başlatılıyor...")
    print(f"{'='*60}\n")
    
    if model_type == 'xgboost':
        # XGBoost for parametre grid'i
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            n_jobs=-1
        )
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    
    elif model_type == 'lightgbm':
        # LightGBM for parametre grid'i
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            random_state=42,
            n_jobs=-1
        )
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'num_leaves': [31, 50],
            'subsample': [0.8, 0.9]
        }
    
    else:  # random_forest
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 20, 25],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
    
    # Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,                    # 3-fold cross validation
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )
    
    print("Grid Search çalışıyor... (Bu biraz zaman alabilir)\n")
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\n✓ Grid Search tamamlandı! Süre: {search_time/60:.2f} dakika\n")
    
    # Best parameters
    print(f"{'─'*60}")
    print("Best Parameters:")
    print(f"{'─'*60}")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Örnek: XGBoost for optimization (opsiyonel - uzun sürer)
# optimized_xgb = optimize_model_gridsearch(X_train, y_train, 'xgboost')


## 8. Test Seti Predictionleri ve Submission

### 8.1 Test Setinde Prediction Yapma

def prepare_test_data(test_df, feature_names):
    """
    Test datani modele uygun formata getirir
    
    Parameters:
    -----------
    test_df : DataFrame
        Test veri seti (feature mühendisliği yapılmış)
    feature_names : list
        Modelin eğitildiği feature isimleri
    
    Returns:
    --------
    tuple
        (test_ids, X_test)
    """
    # ID'leri sakla
    test_ids = test_df['id'].copy()
    
    # Kategorik columnsı encode et
    test_prepared = test_df.copy()
    
    from sklearn.preprocessing import LabelEncoder
    
    for col in test_prepared.select_dtypes(include=['object']).columns:
        if col != 'id':
            le = LabelEncoder()
            test_prepared[col] = le.fit_transformrm(test_prepared[col].astype(str))
    
    # ID ve date_recorded'ı exclude
    drop_cols = ['id']
    if 'date_recorded' in test_prepared.columns:
        drop_cols.append('date_recorded')
    
    X_test = test_prepared.drop(columns=drop_cols)
    
    # Sadece trainingde kullanılan featureleri al
    # (New featureler varsa exclude, eksik olanları add)
    missing_features = set(feature_names) - set(X_test.columns)
    extra_features = set(X_test.columns) - set(feature_names)
    
    if missing_features:
        print(f"⚠️  Adding missing features: {missing_features}")
        for feat in missing_features:
            X_test[feat] = 0
    
    if extra_features:
        print(f"⚠️  Fazla featureler excludeılıyor: {extra_features}")
        X_test = X_test.drop(columns=list(extra_features))
    
    # Column sırasını training setiyle same yap
    X_test = X_test[feature_names]
    
    print(f"\n{'='*60}")
    print("Test Verisi Hazır!")
    print(f"{'='*60}")
    print(f"Test set boyutu: {X_test.shape}")
    print(f"Feature sayısı: {X_test.shape[1]}")
    
    return test_ids, X_test

# Test datani hazırla
test_ids, X_test = prepare_test_data(test_final, features)


# %%

### 8.2 Prediction ve Submission Dosyası Oluşturma

def create_submission(model, test_ids, X_test, encoders, filename='submission.csv'):
    """
    Test set predictionlerini yapar ve submission dosyası createur
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    test_ids : Series
        Test set ID'leri
    X_test : DataFrame
        Test features
    encoders : dict
        Label encoders (target'ı decode etmek for)
    filename : str
        Çıktı dosya adı
    
    Returns:
    --------
    DataFrame
        Submission dosyası
    """
    print(f"{'='*60}")
    print("Test Seti Predictionleri Yapılıyor...")
    print(f"{'='*60}\n")
    
    # Predictionler
    predictions = model.predict(X_test)
    
    # Encode edilmiş valueleri orijinal class isimlerine çevir
    if 'status_group' in encoders:
        target_encoder = encoders['status_group']
        predictions_decoded = target_encoder.inverse_transformrm(predictions)
    else:
        # Manuel decode (eğer encoder yoksa)
        class_mapping = {0: 'functional', 1: 'functional needs repair', 2: 'non functional'}
        predictions_decoded = [class_mapping[p] for p in predictions]
    
    # Submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_ids,
        'status_group': predictions_decoded
    })
    
    # CSV'ye kaydet
    submission_df.to_csv(filename, index=False)
    
    print(f"✓ Submission dosyası created: {filename}")
    print(f"  Total prediction sayısı: {len(submission_df):,}")
    print(f"\nPrediction Distribution:")
    print(submission_df['status_group'].value_counts())
    print(f"\nPrediction Distribution (%):")
    print(submission_df['status_group'].value_counts(normalize=True) * 100)
    
    # First 10 predictioni show
    print(f"\n{'─'*60}")
    print("First 10 Prediction:")
    print(f"{'─'*60}")
    print(submission_df.head(10))
    
    return submission_df

# Submission create
submission = create_submission(
    model=best_model,
    test_ids=test_ids,
    X_test=X_test,
    encoders=encoders,
    filename='submission.csv'
)



# %%


## 9. Model Kaydetme ve Dağıtım

### 9.1 Model ve Preprocessor'ları Kaydetme

import pickle
import joblib

def save_model_and_artifacts(model, encoders, scaler, feature_names, 
                             model_name='best_model'):
    """
    Modeli ve tüm preprocessing araçlarını kaydeder
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    encoders : dict
        Label encoders
    scaler : StandardScaler
        Feature scaler
    feature_names : list
        Feature names
    model_name : str
        Model dosya adı
    """
    import os
    
    # models klasörünü create
    os.makedirs('models', exist_ok=True)
    
    # Model
    model_path = f'models/{model_name}.pkl'
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Encoders
    encoders_path = f'models/{model_name}_encoders.pkl'
    joblib.dump(encoders, encoders_path)
    print(f"✓ Encoders kaydedildi: {encoders_path}")
    
    # Scaler
    scaler_path = f'models/{model_name}_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler kaydedildi: {scaler_path}")
    
    # Feature names
    features_path = f'models/{model_name}_features.pkl'
    joblib.dump(feature_names, features_path)
    print(f"✓ Feature names kaydedildi: {features_path}")
    
    # Metadata (model information)
    metadata = {
        'model_type': type(model).__name__,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_features': len(feature_names),
        'feature_names': feature_names
    }
    
    metadata_path = f'models/{model_name}_metadata.pkl'
    joblib.dump(metadata, metadata_path)
    print(f"✓ Metadata kaydedildi: {metadata_path}")
    
    print(f"\n{'='*60}")
    print("Tüm Artifactler Kaydedildi!")
    print(f"{'='*60}")

# Model ve artifactleri kaydet
save_model_and_artifacts(
    model=best_model,
    encoders=encoders,
    scaler=scaler,
    feature_names=features,
    model_name='water_pump_model_v1'
)


# %% [markdown]
# 
# ### 9.2 Modeli Yükleme ve Prediction Yapma
# 
# def load_model_and_predict(new_data_path, model_name='water_pump_model_v1'):
#     """
#     Kaydedilmiş modeli loadr ve new given üzerinde prediction yapar
#     
#     Parameters:
#     -----------
#     new_data_path : str
#         New veri dosyası yolu
#     model_name : str
#         Yüklenecek model adı
#     
#     Returns:
#     --------
#     DataFrame
#         Predictionler
#     """
#     print(f"{'='*60}")
#     print(f"Model Yükleniyor: {model_name}")
#     print(f"{'='*60}\n")
#     
#     # Artifactleri load
#     model = joblib.load(f'models/{model_name}.pkl')
#     encoders = joblib.load(f'models/{model_name}_encoders.pkl')
#     scaler = joblib.load(f'models/{model_name}_scaler.pkl')
#     feature_names = joblib.load(f'models/{model_name}_features.pkl')
#     metadata = joblib.load(f'models/{model_name}_metadata.pkl')
#     
#     print(f"✓ Model loadndi: {metadata['model_type']}")
#     print(f"  Training tarihi: {metadata['training_date']}")
#     print(f"  Feature sayısı: {metadata['num_features']}\n")
#     
#     # New vergood load
#     new_data = pd.read_csv(new_data_path)
#     print(f"✓ New veri loadndi: {new_data.shape}\n")
#     
#     # Preprocessing pipeline'ı apply
#     # (Burada tüm preprocessing adımları tekrar applynmalı)
#     # 1. Eksik value processme
#     # 2. Encoding
#     # 3. Feature engineering
#     # 4. Scaling
#     
#     print("Preprocessing applynıyor...")
#     
#     # ... (tüm preprocessing functionları burada çağrılır)
#     
#     # Prediction
#     predictions = model.predict(new_data[feature_names])
#     
#     # Decode
#     if 'status_group' in encoders:
#         predictions_decoded = encoders['status_group'].inverse_transformrm(predictions)
#     
#     # Sonuç DataFrame
#     result_df = pd.DataFrame({
#         'id': new_data['id'],
#         'predicted_status': predictions_decoded
#     })
#     
#     print(f"\n{'='*60}")
#     print("Predictionler Tamamlandı!")
#     print(f"{'='*60}")
#     
#     return result_df
# 
# # Örnek kullanım (new veri geldiğinde)
# # new_predictions = load_model_and_predict('new_data.csv')
# 
# 

# %%

## 10. Proje Resultsı ve İş Değeri

### 10.1 Performans Özeti

def generate_project_report(comparison_results, best_model, feature_importance):
    """
    Proje resultlarını özetleyen bir report createur
    
    Parameters:
    -----------
    comparison_results : DataFrame
        Model comparison resultları
    best_model : sklearn model
        Best model
    feature_importance : DataFrame
        Feature importance scores
    """
    print(f"\n{'='*80}")
    print(" " * 20 + "PROJE SONUÇ RAPORU")
    print(f"{'='*80}\n")
    
    print("📊 MODEL PERFORMANSI")
    print(f"{'─'*80}")
    print(comparison_results.to_string(index=False))
    
    print(f"\n\n🏆 EN İYİ MODEL")
    print(f"{'─'*80}")
    best_row = comparison_results.iloc[0]
    print(f"Model Adı    : {best_row['Model']}")
    print(f"Accuracy     : {best_row['Accuracy']:.4f} ({best_row['Accuracy']*100:.2f}%)")
    print(f"F1-Score     : {best_row['F1-Score']:.4f}")
    print(f"Precision    : {best_row['Precision']:.4f}")
    print(f"Recall       : {best_row['Recall']:.4f}")
    
    print(f"\n\n🔑 EN ÖNEMLİ 10 ÖZELLİK")
    print(f"{'─'*80}")
    top_10_features = feature_importance.head(10)
    for idx, row in top_10_features.iterrows():
        bar_length = int(row['importance'] * 100)
        bar = '█' * bar_length
        print(f"{row['feature']:30s} : {bar} {row['importance']:.4f}")
    
    print(f"\n\n💡 İŞ DEĞERİ VE ÖNERİLER")
    print(f"{'─'*80}")
    print("""

# %%


# %%



