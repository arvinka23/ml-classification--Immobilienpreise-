# ml-classification--Immobilienpreise-

Der California Housing Dataset enthält Daten zu Wohngebieten in Kalifornien, basierend auf Volkszählungen. Er umfasst geografische Koordinaten, Gebäudeeigenschaften, demografische Informationen und den Medianwert der Häuser. Zudem zeigt er die Nähe zum Ozean. Diese Daten sind ideal für die Vorhersage von Immobilienpreisen mithilfe von Regressionsmodellen. Da die Daten anonymisiert und auf Blockebene aggregiert sind, gibt es keine Datenschutzrisiken. Der Datensatz ist unter einer CC0-Lizenz frei nutzbar und erfordert keine zusätzlichen Schutzmassnahmen.
## Tag 2 24.03.2025

## Repository-Inhalt

- **data_description.ipynb**: Haupt-Jupyter-Notebook mit allen Analysen
- **ml-classification--Immobilienpreise-.xlsx**: Original-Datensatz im Excel-Format
- **california_housing_scaled.xlsx**: Vorverarbeiteter Datensatz mit skalierten Merkmalen

## Analyse-Highlights
### 1. Zielvariable Auswahl

Der Median-Hauswert (median_house_value) wurde als Zielvariable für die Vorhersage ausgewählt. Diese Variable ist besonders relevant für:

- Immobilieninvestoren, die Gebiete mit Wachstumspotential identifizieren möchten
- Interessierte Käufer, die faire Marktpreise bestimmen wollen
- Stadtplaner, die die Faktoren verstehen möchten, die den Immobilienwert beeinflussen

### 2. Statistische Analyse

Es wurden umfassende statistische Informationen für alle Felder berechnet:

- Grundlegende Statistiken (Mittelwert, Median, Standardabweichung)
- Verteilungsmerkmale (Schiefe, Kurtosis)
- Häufigkeitsanalyse für kategoriale Variablen
- Korrelationsanalyse zwischen allen numerischen Variablen

### 3. Datenvisualisierung

Mehrere Visualisierungen wurden erstellt, um die Datenmuster zu verstehen:

- Histogramm der Hauspreise, das die Verteilung der Zielvariable zeigt
- Mehrere Histogramme für wichtige numerische Merkmale
- Streudiagramm mit Regressionslinie zwischen Median-Einkommen und Hauspreisen
- Boxplots, die Hauspreise nach Nähe zum Ozean zeigen
- Korrelationsheatmap, die die Beziehungen zwischen den Variablen hervorhebt

### 4. Daten-Skalierung

Die Standard-Skalierung (z-Score-Normalisierung) wurde auf alle numerischen Variablen angewendet, außer auf die Zielvariable. Diese Skalierung ist vorteilhaft, weil:

- Variablen unterschiedliche Einheiten und Grössenordnungen haben
- Viele Machine Learning-Algorithmen mit standardisierten Eingabedaten besser arbeiten
- Die z-Score-Transformation die Interpretation erleichtert
- Sie verhindert, dass Variablen mit grossen Bereichen das Modell dominieren


## Anforderungen

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
