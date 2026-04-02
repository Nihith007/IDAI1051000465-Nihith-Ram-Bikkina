# IDAI1051000465-Nihith-Ram-Bikkina
Mining the Future: Unlocking Business Intelligence with AI

# 1. Project Scope
🎯 Mission
As part of the SmartEnergy Data Lab, the goal is to analyse EV charging patterns worldwide to:

Improve station utilisation and scheduling
Personalise customer experience by understanding charging habits
Detect faulty or abnormal charging readings before they impact service
Support infrastructure planning decisions with data-driven evidence

📋 Dataset Columns
ColumnDescriptionStation_IDUnique identifier per stationLatitude / LongitudeGeographic coordinatesAddressStation address stringCharger_TypeAC Level 1 / AC Level 2 / DC FastCost_USD_per_kWhCharging cost per kilowatt-hourAvailability_pct% of time station is availableDistance_to_City_kmDistance to nearest city centre (km)Usage_Stats_avg_users_dayAverage daily usersStation_OperatorCompany operating the stationCharging_Capacity_kWMax charging power output (kW)Connector_TypesCable connector standardsInstallation_YearYear the station was installedRenewable_Energy_SourceWhether station uses renewable energy (Yes/No)Reviews_RatingAverage customer rating (1–5 stars)Parking_SpotsNumber of EV-dedicated parking spotsMaintenance_FrequencyServicing frequency (Monthly/Quarterly/etc.)
📌 Objectives
#ObjectiveMethod1Find charging behavior patterns (when, where, how much)EDA + Visualisations2Group stations/users into clustersK-Means Clustering3Discover associations between usage and station typeApriori Algorithm4Detect faulty readings or abnormal charging patternsIQR Anomaly Detection5Map geographic demand and hotspotsGeospatial Analysis (Folium)

# 2. Data Preparation & Preprocessing
File: data_preprocessing.py
Steps
Raw CSV / Generated Dataset
        │
        ▼
1. Duplicate Removal        → drop_duplicates(subset=["Station_ID"])
        │
        ▼
2. Missing Value Imputation
   ├── Numeric  → df[col].fillna(df[col].median())
   └── Categorical → df[col].fillna(df[col].mode()[0])
        │
        ▼
3. Range Clipping           → clip physically impossible values
   ├── Reviews_Rating     [1.0, 5.0]
   ├── Availability_pct   [0.0, 100.0]
   └── Usage_Stats        [0.0, 500.0]
        │
        ▼
4. Label Encoding           → sklearn LabelEncoder per categorical column
   ├── Charger_Type_enc
   ├── Station_Operator_enc
   ├── Renewable_Energy_Source_enc
   └── Maintenance_Frequency_enc
        │
        ▼
5. Min-Max Normalisation    → MinMaxScaler → *_scaled columns
   (all FEATURES_FOR_CLUSTERING scaled to [0, 1])
Key Code Excerpt
python# Missing value imputation
for col in NUMERIC_COLS:
    df[col].fillna(df[col].median(), inplace=True)

** Label encoding **
le = LabelEncoder()
df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

# Min-Max scaling
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[FEATURES_FOR_CLUSTERING])
Output
Column TypeQuantityOriginal columns17Encoded columns (_enc)4Scaled columns (_scaled)7Total after preprocessing28+

# 3. EDA & Visualisations
File: eda.py
Visualisations Produced
PlotPurposeKey FindingUsage Distribution HistogramUnderstand demand spreadRight-skewed; most stations serve 10–50 users/dayUsage by Charger Type (Boxplot)Compare demand across typesDC Fast has highest median usageCost by Operator (Boxplot)Price comparisonSignificant variation; Tesla is premium-pricedUsage vs. Installation Year (Line)Growth trendNewer stations show higher adoptionDemand Heatmap (Charger × Availability)Cross-feature demandHigh-availability DC Fast = busiestRating vs. Usage (Scatter)Quality vs. popularityModerate positive correlationRenewable vs. Non-Renewable (Bar)Green energy impactRenewable stations: +0.4 avg ratingCorrelation HeatmapFeature relationshipsUsage negatively correlates with distanceCharger Type Pie ChartMarket shareAC Level 2 dominates (≈50%)Distance vs. Usage (Regression)Location insightCloser to city = more users
Key Code Pattern
pythondef plot_usage_by_charger_type(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="Charger_Type", y="Usage_Stats_avg_users_day",
                order=["AC Level 1", "AC Level 2", "DC Fast"],
                palette="Set2", ax=ax)
    return fig   # returned for Streamlit: st.pyplot(fig)

# 4. Clustering Analysis (K-Means)
File: clustering.py
Method

Elbow Method — plot WCSS inertia vs. k (2–10)
Silhouette Scoring — validate optimal k
K-Means — sklearn.cluster.KMeans(n_clusters=k, n_init=15)
PCA Projection — 2-D scatter for visual validation
Cluster Profiling — mean feature values per cluster

Cluster Labels (k = 4)
ClusterLabelCharacteristics0🔴 Heavy Users – High CapacityDC Fast, usage > 50/day, high kW1🟡 Daily Commuters – Moderate UseAC Level 2, urban, moderate usage2🟢 Occasional Users – Low FrequencyAC Level 1, rural, low usage3🔵 Premium Fast-Charge HubsHigh cost, high rating, renewable
Key Code Excerpt
pythonfrom sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# Fit final model
km = KMeans(n_clusters=4, random_state=42, n_init=15)
df["Cluster_ID"] = km.fit_predict(X)

5. Association Rule Mining (Apriori)
File: association_rules.py
Method
Features are discretised into categorical items before mining:
Usage_Stats  →  "Usage=Low" | "Usage=Medium" | "Usage=High"
Cost         →  "Cost=Low"  | "Cost=Medium"  | "Cost=High"
Distance     →  "Distance=NearCity" | "Distance=Suburban" | "Distance=Rural"
Rating       →  "Rating=Low" | "Rating=High"
Charger_Type →  "Charger=AC_Level_1" | ... | "Charger=DC_Fast"
Renewable    →  "Renewable=Yes" | "Renewable=No"
Each station becomes a transaction of these items. Apriori finds frequent itemsets.
Rule Metrics
MetricFormulaMeaningSupportfreq(A∪B) / NHow often rule appears in datasetConfidencefreq(A∪B) / freq(A)Probability B occurs given ALiftConfidence / Support(B)> 1 = non-random, positive association
Sample Rules Found
DC Fast + Renewable=Yes → Usage=High           Lift: 1.82, Conf: 0.73
Cost=Low + Distance=NearCity → Usage=High      Lift: 1.65, Conf: 0.68
AC Level 2 + Rating=High → Availability=High   Lift: 1.41, Conf: 0.60
Key Code Excerpt
pythonfrom mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(df_encoded, min_support=0.10, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.50)
rules = rules[rules["lift"] >= 1.0].sort_values("lift", ascending=False)

6. Anomaly Detection (IQR Method)
File: anomaly_detection.py
Method
Interquartile Range (IQR):
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1

Lower bound = Q1 - k × IQR
Upper bound = Q3 + k × IQR

Anomaly: value < Lower bound  OR  value > Upper bound
Default k = 1.5 (standard), configurable up to k = 3.0 (extreme outliers only).
Columns Monitored
ColumnAnomaly InterpretationUsage_Stats_avg_users_dayGhost stations (very low) or overloaded hubsCost_USD_per_kWhPrice gouging or data entry errorsCharging_Capacity_kWEquipment malfunction or mislabelled capacityReviews_RatingPersistently terrible service
Key Code Excerpt
pythondef iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

for col in ANOMALY_TARGETS:
    lo, hi = iqr_bounds(df[col], k=1.5)
    df[f"{col}_anomaly"] = (df[col] < lo) | (df[col] > hi)

df["is_anomaly"] = df[[c for c in df.columns if c.endswith("_anomaly")]].any(axis=1)
Results Summary

Approximately 3–8% of stations flagged as anomalous (varies with k)
DC Fast stations show the most extreme usage spikes
High-cost stations with low ratings are consistent anomaly candidates


7. Geospatial Analysis
File: geospatial.py
Analyses
VisualisationToolInsightCluster scatter mapMatplotlibGlobal distribution of behaviour clustersInteractive station mapFolium + CircleMarkerPopup details per stationDemand heatmapFolium + HeatMap pluginUrban vs. rural intensityCity-level usage bar chartMatplotlibTop-demand regionsOperator bubble chartMatplotlibOperator performance vs. station count
Geographic Hotspots
Stations cluster around major metropolitan areas:

North America: San Francisco Bay Area — highest DC Fast concentration
Europe: London, Paris — high AC Level 2 density
Asia: Tokyo — premium operators with high ratings
Australia: Sydney — growing renewable adoption

Key Code Excerpt
pythonimport folium
from folium.plugins import HeatMap

fmap = folium.Map(location=[center_lat, center_lon], zoom_start=3)

# Heatmap layer
heat_data = df[["Latitude", "Longitude", "Usage_Stats_avg_users_day"]].values.tolist()
HeatMap(heat_data, radius=15, blur=12).add_to(fmap)

# Cluster markers
folium.CircleMarker(
    location=[row.Latitude, row.Longitude],
    color=cluster_color, fill=True,
    popup=f"{row.Station_ID} — {row.Usage_Stats_avg_users_day:.1f} users/day"
).add_to(fmap)

8. Deployment on Streamlit Cloud
Live App

🔗 Deploy your Streamlit app → https://streamlit.io/cloud

Steps to Deploy
bash# 1. Clone or create the GitHub repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 2. Ensure all files are present
#    app.py, generate_dataset.py, data_preprocessing.py,
#    eda.py, clustering.py, association_rules.py,
#    anomaly_detection.py, geospatial.py, requirements.txt

# 3. Push to GitHub
git add .
git commit -m "Initial project upload – EV SmartCharging Analytics"
git push origin main

# 4. Go to https://streamlit.io/cloud
#    → New App → Connect GitHub → Select repo → Set main file: app.py → Deploy
App Features
The Streamlit dashboard has 8 pages (sidebar navigation):
PageContent🏠 Project OverviewKPI cards, scope, dataset preview🔧 Data PreprocessingCleaning steps, statistics, encoded columns📊 EDA & Visualisations10 interactive plots across 6 tabs🔵 Clustering AnalysisElbow method, PCA scatter, profiles, summary🔗 Association Rule MiningTop rules, support×confidence scatter, rules table🚨 Anomaly DetectionOverview, scatter, boxplots, by charger type, table🗺️ Geospatial AnalysisStatic map, Folium heatmap, city stats, operator bubble📋 Insights & ReportExecutive summary, key findings, download button
Local Development
bash# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

9. Repository Structure
YOUR_REPO/
│
├── app.py                    ← Main Streamlit dashboard (entry point)
├── generate_dataset.py       ← Synthetic EV dataset generator
├── data_preprocessing.py     ← Stage 2: Cleaning, encoding, normalisation
├── eda.py                    ← Stage 3: EDA visualisation functions
├── clustering.py             ← Stage 4: K-Means clustering
├── association_rules.py      ← Stage 5: Apriori rule mining
├── anomaly_detection.py      ← Stage 6: IQR anomaly detection
├── geospatial.py             ← Stage 7: Folium & Matplotlib maps
│
├── ev_charging_stations.csv  ← Dataset (generated on first run)
├── requirements.txt          ← Python dependencies for Streamlit Cloud
└── README.md                 ← This file

10. References
Academic & Research

Cerna, F. et al. (2018). EV Charging Load Forecasting. arXiv:1802.04193
Clustering-Based Optimal Operation of EV Charging Stations. ResearchGate, 2023.
Frontiers in Energy Research — EV infrastructure optimisation. doi:10.3389/fenrg.2022.773440
McKinsey Center for Future Mobility — Consumer Sentiment on EV Charging.

Python Libraries
LibraryUsagepandasData loading, cleaning, manipulationnumpyNumerical operationsmatplotlibStatic visualisationsseabornStatistical plotsscikit-learnKMeans, MinMaxScaler, LabelEncoder, PCA, SilhouettemlxtendApriori algorithm, TransactionEncoderfoliumInteractive geospatial mapsstreamlitInteractive web dashboard
Methodology References

Data-to-Viz — Chart selection guide: https://www.data-to-viz.com/
K-Means Clustering — Neptune.ai: https://neptune.ai/blog/k-means-clustering
Association Mining + Clustering: https://dicecamp.com/insights/association-mining-rules-combined-with-clustering/
Anomaly Detection Techniques: https://www.kdnuggets.com/2023/05/beginner-guide-anomaly-detection-techniques-data-science.html
Python for Geospatial Analysis: https://likuyani.cdf.go.ke/uploaded-files/5P8049/HomePages/PythonForGeospatialDataAnalysis.pdf
