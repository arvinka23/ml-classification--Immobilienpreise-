import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import warnings

# Für schönere Plots
plt.style.use('seaborn')
%matplotlib inline

# Warnungen unterdrücken
warnings.filterwarnings('ignore')

# Funktion zum Laden und Überprüfen der Datei
def load_data(file_path):
    # Prüfen, ob die Datei existiert
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Die Datei '{file_path}' wurde nicht gefunden!")
    
    # Datensatz laden
    df = pd.read_excel(file_path)
    
    # Die ersten Zeilen des Datensatzes anzeigen
    print("Erste 5 Zeilen des Datensatzes:")
    print(df.head())
    return df

# Pfad zur Excel-Datei (im gleichen Ordner wie das Skript)
file_path = r'C:\Users\arvin\OneDrive - BBBaden\Dokumente\Schule 2023-2024\BBB 2023-2024\Modul 259\Lb\ml-classification--Immobilienpreise-.xlsx'

# Daten laden
df = load_data(file_path)

# Grundlegende Informationen über den Datensatz
print("\nInformationen über den Datensatz:")
df.info()

print("\nBeschreibende Statistik:")
df.describe()

# Zielvariable für die Vorhersage
target_variable = 'median_house_value'

# Verteilung der Zielvariable untersuchen
print(f"Statistik zur Zielvariable '{target_variable}':")
print(df[target_variable].describe())

# Visualisierung der Zielvariable
plt.figure(figsize=(10, 6))
sns.histplot(df[target_variable], kde=True)
plt.title('Verteilung der Immobilienpreise')
plt.xlabel('Median-Hauspreis (USD)')
plt.ylabel('Häufigkeit')
plt.show()

# Detaillierte statistische Analyse aller numerischen Spalten
print("\nDetaillierte Statistiken für alle numerischen Felder:")
numeric_stats = df.describe().T
numeric_stats['median'] = df.median()
numeric_stats['skewness'] = df.skew()
numeric_stats['kurtosis'] = df.kurtosis()
print(numeric_stats)

# Kategorische Daten (falls vorhanden)
if 'ocean_proximity' in df.columns:
    print("\nHäufigkeiten der Kategorie 'ocean_proximity':")
    print(df['ocean_proximity'].value_counts())

# Korrelationsmatrix erstellen
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korrelationsmatrix der numerischen Variablen')
plt.show()

# Die wichtigsten Korrelationen mit der Zielvariable identifizieren
print(f"\nKorrelationen mit der Zielvariable '{target_variable}':")
corr_with_target = corr_matrix[target_variable].sort_values(ascending=False)
print(corr_with_target)

# Skalierung der Daten - Auswahl der Spalten
cols_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
if target_variable in cols_to_scale:
    cols_to_scale.remove(target_variable)  # Zielvariable nicht skalieren

# Standard-Skalierung (z-score normalization)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Vergleich der Originaldaten mit den skalierten Daten
print("\nVergleich vor und nach der Skalierung:")
if len(cols_to_scale) > 0:
    compare_df = pd.DataFrame({
        'Original': df[cols_to_scale[0]].head(5),
        'Skaliert': df_scaled[cols_to_scale[0]].head(5)
    })
    print(compare_df)

# Speichern der aufbereiteten Daten für die spätere Modellierung
df_scaled.to_excel('california_housing_scaled.xlsx', index=False)

# Grafische Darstellungen
# Histogramme für wichtige Variablen
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target_variable in num_cols:
    num_cols.remove(target_variable)  # Zielvariable bereits oben dargestellt

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols[:6]):  # Erste 6 numerische Spalten
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Verteilung von {col}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Streudiagramm zwischen Median-Einkommen und Hauspreis
plt.figure(figsize=(10, 6))
sns.regplot(x='median_income', y='median_house_value', data=df)
plt.title('Zusammenhang zwischen Einkommen und Hauspreis')
plt.xlabel('Median-Einkommen')
plt.ylabel('Median-Hauspreis (USD)')
plt.show()

# Box-Plot für Hauspreise nach Nähe zum Ozean (falls vorhanden)
if 'ocean_proximity' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='ocean_proximity', y='median_house_value', data=df)
    plt.title('Hauspreise nach Nähe zum Ozean')
    plt.xlabel('Nähe zum Ozean')
    plt.ylabel('Median-Hauspreis (USD)')
    plt.xticks(rotation=45)
    plt.show()
