# IDAI1051000465-Nihith-Ram-Bikkina
Mining the Future: Unlocking Business Intelligence with AI

# Candidate Name – Nihith Ram Bikkina

# Candidate Registration Number – 1000465

# CRS Name: Artificial Intelligence

# Course Name – Data Mining

# School Name – Birla Open Minds International School, Kollur

# Summative Assessment

---

# ⚡ EV SmartCharging Analytics — Uncovering EV Behavior Patterns

**Assignment Title:** Mining the Future: Unlocking Business Intelligence with AI
**Scenario Selected:** Scenario 2 — SmartCharging Analytics
**Deployment:** Streamlit Cloud

---

## 📋 Project Scope

**Scenario:** SmartCharging Analytics — Uncovering EV Behavior Patterns

As part of the SmartEnergy Data Lab team, this project analyses a global dataset of 5,000 EV charging stations spread across 15 major cities worldwide. The goal is to improve station utilisation and customer experience by exploring charging behaviour across station types, operators, geographic locations, and usage profiles. The insights generated support smarter infrastructure planning, dynamic pricing, and targeted service improvement.

**Objectives:**

| # | Objective | Purpose |
|---|-----------|---------|
| 1 | Understand usage trends by charger type, operator, and geography | Identify which stations and locations perform best |
| 2 | Segment charging stations using K-Means Clustering | Group stations into behaviour profiles for targeted action |
| 3 | Discover hidden feature associations using the Apriori Algorithm | Find which combinations of station features drive high demand |
| 4 | Detect anomalous stations with abnormal costs, usage, or reviews | Flag faulty or underperforming stations for review |
| 5 | Map geographic demand hotspots | Visualise where high-demand regions are concentrated globally |
| 6 | Deploy an interactive Streamlit dashboard | Enable stakeholders to explore insights without technical knowledge |

**Dataset:** `ev_charging_stations.csv`
One row per charging station. The dataset contains 5,000 rows and 17 columns representing real-world EV infrastructure attributes.

**Dataset Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Station_ID` | String | Unique identifier for each charging station |
| `Latitude` | Float | Geographic latitude coordinate |
| `Longitude` | Float | Geographic longitude coordinate |
| `Address` | String | Full street address of the station |
| `Charger_Type` | Categorical | Type of charger — AC Level 1, AC Level 2, or DC Fast |
| `Cost (USD/kWh)` | Float | Price charged per kilowatt-hour in US dollars |
| `Availability` | Float | Percentage of time the station is available for use |
| `Distance to City (km)` | Float | Distance from the station to the nearest city centre |
| `Usage Stats (avg users/day)` | Float | Average number of daily users recorded at the station |
| `Station_Operator` | Categorical | Company or organisation managing the station |
| `Charging Capacity (kW)` | Float | Maximum power output supported by the station |
| `Connector_Types` | String | Supported cable connector standards (e.g. CCS, CHAdeMO, Type 2) |
| `Installation_Year` | Integer | Year the station was first installed and activated |
| `Renewable Energy Source` | Categorical | Whether the station uses renewable energy (Yes / No) |
| `Reviews (Rating)` | Float | Average customer satisfaction rating on a 1–5 star scale |
| `Parking_Spots` | Integer | Number of EV-dedicated parking bays at the station |
| `Maintenance_Frequency` | Categorical | How often the station is serviced — Monthly, Quarterly, Bi-Annual, or Annual |

---

## 🧹 Data Preparation & Preprocessing

The raw dataset contains missing values, inconsistent categorical formats, and features measured on very different numeric scales. A structured seven-step preprocessing pipeline is applied before any modelling takes place to ensure data quality and comparability.

**Preprocessing Pipeline:**

| Step | Action | Detail |
|------|--------|--------|
| 1 | **Load Dataset** | 5,000 rows × 17 columns loaded directly from the generated CSV |
| 2 | **Remove Duplicates** | Rows deduplicated on `Station_ID` — ensures each physical station appears exactly once |
| 3 | **Impute Numeric NaNs** | Missing values in all numeric columns replaced with the column median to avoid distortion from extreme values |
| 4 | **Impute Categorical NaNs** | Missing values in categorical columns replaced with the column mode (the most frequently occurring value) |
| 5 | **Impute Connector Types** | Missing entries in `Connector_Types` filled with the constant value `Unknown` to preserve row count |
| 6 | **Clip Illegal Ranges** | Values outside physically valid boundaries are clamped — for example, `Reviews (Rating)` is clipped to [1.0, 5.0] and `Availability` to [0.0, 100.0] |
| 7 | **Label Encoding** | `Charger_Type`, `Station_Operator`, `Renewable Energy Source`, and `Maintenance_Frequency` are converted from text categories to integer codes using Label Encoding |
| 8 | **Min-Max Normalisation** | Seven continuous features are scaled to the [0, 1] range so that no single feature dominates clustering or distance calculations due to its absolute magnitude |

**Features Normalised (scaled to [0, 1]):**

| Feature | Raw Range | Why Normalised |
|---------|-----------|---------------|
| `Cost (USD/kWh)` | $0.04 – $0.65 | Large absolute range relative to other features |
| `Availability` | 20% – 100% | Kept consistent with other features in the cluster matrix |
| `Distance to City (km)` | 0.1 – 90 km | Would numerically dominate distance calculations without scaling |
| `Usage Stats (avg users/day)` | 1 – 200 users | Wide spread; skews clustering without normalisation |
| `Charging Capacity (kW)` | 7.2 – 350 kW | Varies by factor of ~50 across charger types |
| `Reviews (Rating)` | 1.0 – 5.0 | Scale difference relative to usage and capacity |
| `Parking_Spots` | 1 – 29 | Integer count kept proportional to other features |

**Missing Values Injected and Imputed:**

| Column | Approximate Missing | Imputation Strategy |
|--------|--------------------|--------------------|
| `Reviews (Rating)` | ~3.5% | Column median |
| `Renewable Energy Source` | ~2.0% | Column mode — `No` |
| `Connector_Types` | ~2.5% | Constant fill — `Unknown` |

---

## 📊 EDA & Visualisations

Exploratory Data Analysis reveals patterns, distributions, and relationships in the data before any machine learning is applied. A total of twelve charts are produced across seven themed tabs in the Streamlit dashboard.

**Visualisations Produced:**

| # | Chart | Chart Type | Purpose | Key Finding |
|---|-------|-----------|---------|-------------|
| 1 | **Usage Statistics Histogram** | Histogram | Distribution of average daily users across all 5,000 stations | Right-skewed — most stations serve 10–50 users/day; a small number exceed 150 |
| 2 | **Usage by Charger Type** | Box Plot | Compare demand spread across AC Level 1, AC Level 2, and DC Fast | DC Fast stations have the highest median usage at approximately 52 users/day |
| 3 | **Cost by Station Operator** | Box Plot | Spread of pricing strategies across all operators | Tesla Supercharger and Electrify America are premium-priced; Blink is the most affordable |
| 4 | **Usage Trend by Installation Year** | Line Chart | How average daily usage changes across installation years 2012–2024 | Stations installed after 2018 show markedly higher adoption rates |
| 5 | **Demand Heatmap (Charger Type × Availability Quartile)** | Heatmap | Which charger-availability combinations attract the most users | High-availability DC Fast stations achieve the highest mean daily usage |
| 6 | **Reviews vs Usage Scatter** | Scatter Plot | Whether higher-rated stations attract more users | Moderate positive correlation — better-rated stations tend to be busier |
| 7 | **Renewable Energy: Usage & Rating Comparison** | Grouped Bar Chart | Side-by-side comparison of usage and rating for renewable vs non-renewable stations | Renewable stations score 0.3–0.5 stars higher and attract more users on average |
| 8 | **Correlation Heatmap** | Heatmap | Pairwise Pearson correlations between all numeric features | Usage negatively correlates with distance to city (r ≈ −0.45) |
| 9 | **Charger Type Distribution** | Pie Chart | Market share of each charger type across all stations | AC Level 2 dominates at approximately 50% of all stations in the dataset |
| 10 | **Distance to City vs Usage (Trend Line)** | Scatter + Regression | Linear trend showing the relationship between proximity and demand | Clear negative slope — each additional kilometre from the city reduces average daily users |
| 11 | **Station Count by Operator** | Bar Chart | How many stations each operator manages in the dataset | ChargePoint leads with approximately 20% of total stations |
| 12 | **Charging Capacity Distribution** | Overlapping Histogram | Capacity spread visualised separately for each charger type | DC Fast stations cluster around 150 kW; AC Level 1 clusters tightly around 7 kW |

**Key EDA Finding:** DC Fast Chargers consistently record higher average daily usage than AC Level 1 or AC Level 2 stations. Urban stations — those within 5 km of a city centre — record significantly more daily users than rural counterparts, regardless of charger type or operator.

---

## 🤖 Clustering Analysis (K-Means)

**Algorithm:** K-Means with k-means++ initialisation and 15 random restarts to ensure stable convergence across runs.

**Features Used for Clustering:**

| Feature | Rationale for Inclusion |
|---------|------------------------|
| `Cost (USD/kWh)` | Pricing strategy differentiates premium, standard, and budget station types |
| `Availability` | Reflects station reliability and the level of demand pressure on the station |
| `Distance to City (km)` | Separates urban, suburban, and rural station profiles |
| `Usage Stats (avg users/day)` | The primary measure of station activity and utilisation |
| `Charging Capacity (kW)` | Distinguishes slow AC Level 1 from moderate AC Level 2 and fast DC infrastructure |
| `Reviews (Rating)` | Customer satisfaction reflects overall service quality and station condition |
| `Parking_Spots` | Relates to station physical size and total throughput capacity |

**Optimal k Selection:**

| Method | Description | Outcome |
|--------|-------------|---------|
| Elbow Method (WCSS) | Within-Cluster Sum of Squares plotted for k = 2 through 10 — the sharpest rate reduction identifies the elbow | Elbow visually appears around k = 4 |
| Silhouette Score | Measures how well each station fits its own cluster compared to neighbouring clusters — higher is better | Highest silhouette score confirms k = 4 as optimal |

**Cluster Profiles (k = 4):**

| Cluster | Label | Typical Usage | Dominant Charger | Location Type | Key Characteristics |
|---------|-------|--------------|-----------------|--------------|-------------------|
| 🔴 Cluster 0 | Heavy Users – High Capacity | Above 55 users/day | DC Fast | Urban core | Very high capacity (above 100 kW), peak demand, city-proximate stations |
| 🟡 Cluster 1 | Daily Commuters – Moderate Use | 30–55 users/day | AC Level 2 | Suburban | Moderate usage, consistent daily patterns, good availability, mid-range cost |
| 🟢 Cluster 2 | Occasional Users – Low Frequency | Below 20 users/day | AC Level 1 | Rural | Low demand, greater distance from city, lower capacity, infrequent servicing |
| 🔵 Cluster 3 | Premium Fast-Charge Hubs | 25–50 users/day | DC Fast | Urban / Highway | High cost, high customer rating, renewable energy integration, large parking capacity |

**Cluster Visualisations Produced:**

| Visualisation | Description |
|--------------|-------------|
| Elbow Method Chart | WCSS inertia plotted across k = 2 to 10 showing the point of diminishing returns |
| Silhouette Score Chart | Silhouette score per k value to confirm the elbow method recommendation |
| PCA 2D Scatter | All 5,000 stations projected onto two principal components and colour-coded by cluster assignment |
| Cluster Feature Profiles | Bar charts showing mean value of each feature per cluster for direct comparison |
| Cluster Summary Table | Count, mean, and standard deviation of all features per cluster label |
| Charger Type Mix per Cluster | Cross-tabulation of charger type distribution within each cluster |
| Geographic Cluster Map | World coordinate scatter plot with cluster-coloured station markers |

---

## 🔗 Association Rule Mining (Apriori)

**Algorithm:** Apriori co-occurrence mining — discovers frequent itemset pairs from discretised station features and generates association rules with support, confidence, and lift metrics. A manual co-occurrence implementation is used as a fallback if the `mlxtend` library is not available in the deployment environment.

**Discretisation Strategy — Continuous Features Binned into Categories:**

| Feature | Bin Labels | Threshold Logic |
|---------|-----------|----------------|
| `Usage Stats (avg users/day)` | Usage=Low / Usage=Medium / Usage=High | Below 20 / 20 to 50 / Above 50 users per day |
| `Cost (USD/kWh)` | Cost=Low / Cost=Medium / Cost=High | Below $0.12 / $0.12 to $0.25 / Above $0.25 |
| `Charging Capacity (kW)` | Capacity=Low / Capacity=Medium / Capacity=High | Below 15 kW / 15 to 60 kW / Above 60 kW |
| `Distance to City (km)` | Dist=NearCity / Dist=Suburban / Dist=Rural | Below 5 km / 5 to 20 km / Above 20 km |
| `Reviews (Rating)` | Rating=Low / Rating=High | Below 4.0 stars / 4.0 and above |
| `Charger_Type` | Charger=AC_Level_1 / Charger=AC_Level_2 / Charger=DC_Fast | Direct mapping from column values |
| `Renewable Energy Source` | Renewable=Yes / Renewable=No | Direct mapping from column values |
| `Maintenance_Frequency` | Maint=Monthly / Maint=Quarterly / Maint=Bi-Annual / Maint=Annual | Direct mapping from column values |

**Rule Metrics Explained:**

| Metric | Calculation | Interpretation |
|--------|------------|---------------|
| Support | Frequency of the combined rule items in the full dataset | How commonly this combination appears — higher means more universally applicable |
| Confidence | Probability that the consequent holds when the antecedent is true | How reliable the rule is as a predictor — higher means more dependable |
| Lift | Confidence divided by the baseline probability of the consequent | Values above 1.0 indicate a genuine positive association beyond random chance |

**Example Rules Discovered:**

| IF (Antecedent) | THEN (Consequent) | Support | Confidence | Lift | Business Interpretation |
|----------------|-------------------|---------|-----------|------|------------------------|
| Charger=DC_Fast + Renewable=Yes | Usage=High | ~0.14 | ~0.73 | ~1.82 | Renewable fast chargers consistently attract above-average demand |
| Cost=Low + Dist=NearCity | Usage=High | ~0.12 | ~0.68 | ~1.65 | Low-cost city-centre stations are the busiest |
| Charger=AC_Level_2 + Rating=High | Availability=High | ~0.11 | ~0.60 | ~1.41 | Well-rated AC Level 2 stations maintain high availability |
| Maint=Monthly + Capacity=High | Rating=High | ~0.09 | ~0.57 | ~1.35 | Frequently serviced high-capacity stations earn better reviews |
| Cost=High | Usage=Low | ~0.08 | ~0.55 | ~1.28 | Stations priced above median attract noticeably fewer users |

**Visualisations Produced:**

| Visualisation | Description |
|--------------|-------------|
| Top Rules Bar Chart | Horizontal bar chart of the top 15 rules ranked by lift score, coloured by strength |
| Support × Confidence Scatter | Bubble chart where position = support and confidence, bubble size and colour both encode lift |
| Rules Table | Full sortable table of all discovered rules with support, confidence, and lift; lift column highlighted with a colour gradient |

**Parameter Controls — Adjustable via Sidebar Sliders:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Minimum Support | 0.08 | 0.05 – 0.40 | Lower values discover rarer but potentially interesting rules |
| Minimum Confidence | 0.45 | 0.30 – 0.90 | Higher values return only the most reliable and predictable rules |

---

## 🔍 Anomaly Detection (IQR Method)

**Method:** Interquartile Range (IQR) — a robust non-parametric statistical technique that flags values lying outside the typical spread of the central 50% of data. It is resistant to extreme values and does not assume a normal distribution.

**How It Works:**

| Step | Calculation |
|------|------------|
| Compute Q1 | The 25th percentile of the target column |
| Compute Q3 | The 75th percentile of the target column |
| Compute IQR | Q3 minus Q1 — representing the middle 50% spread |
| Lower Bound | Q1 minus (multiplier × IQR) |
| Upper Bound | Q3 plus (multiplier × IQR) |
| Flag Anomaly | Any value falling below the lower bound or above the upper bound is flagged |

The IQR multiplier defaults to 1.5 (standard sensitivity) and is adjustable in the sidebar up to 3.0 for extreme-only detection. A station is flagged as anomalous if it breaches the threshold on any one of the monitored columns.

**Columns Monitored for Anomalies:**

| Column | Low Anomaly Interpretation | High Anomaly Interpretation |
|--------|--------------------------|---------------------------|
| `Usage Stats (avg users/day)` | Ghost station — active but near-zero usage | Overloaded hub — extreme spike possibly from events or data error |
| `Cost (USD/kWh)` | Suspiciously low price — possible data entry error | Price gouging — significantly above all market peers |
| `Charging Capacity (kW)` | Unusually low capacity — possible equipment fault | Mislabelled infrastructure record |
| `Reviews (Rating)` | Persistently poor service — requires immediate investigation | Not applicable — high ratings are desirable |

**Anomaly Types Detected and Business Actions:**

| Anomaly Type | Description | Recommended Action |
|-------------|-------------|-------------------|
| High Usage Spike | Station usage far above the IQR upper bound | Verify data accuracy; consider capacity expansion |
| High Cost Outlier | Pricing significantly above all market peers | Review pricing policy immediately; risk of user churn |
| Low Rating Station | Rating consistently below the IQR lower bound | Investigate service quality; trigger a maintenance review |
| Ghost Station | Usage near zero despite station being listed as active | Check for equipment faults, access barriers, or data recording issues |

**Visualisations Produced:**

| Visualisation | Description |
|--------------|-------------|
| Anomaly Count by Feature | Bar chart showing how many stations are flagged per monitored column |
| Usage vs Cost Scatter | Scatter plot with normal stations in blue and anomalous stations highlighted as red crosses; IQR boundary lines displayed |
| Box Plots (Normal vs Anomalous) | Side-by-side box plots comparing the distribution of normal and anomalous stations for each monitored metric |
| Anomalies by Charger Type | Stacked bar chart showing normal versus anomalous counts broken down by charger type |
| Anomaly Detail Table | Filterable table listing all flagged stations with Station ID, operator, charger type, key metrics, and the specific reason for flagging |

---

## 🗺️ Geospatial Analysis

Geographic visualisation reveals where stations are concentrated, which regions carry the highest demand, and where anomalous or underperforming stations are located globally.

**Map Modes Available in the App:**

| Mode | Tool Used | Description |
|------|----------|-------------|
| Cluster Map (Static) | Matplotlib | All 5,000 stations plotted on world coordinates, coloured by K-Means cluster assignment; anomalous stations marked as red crosses |
| Demand Heatmap (Interactive) | Folium + HeatMap Plugin | Heat intensity layer weighted by `Usage Stats (avg users/day)` — darker areas indicate higher demand concentration |
| Station Clusters (Interactive) | Folium + CircleMarker | Clickable colour-coded station markers; each popup shows Station ID, operator, charger type, usage, cost, rating, and renewable status |

**Supporting Charts:**

| Chart | Description |
|-------|-------------|
| Mean Daily Usage by City Region | Bar chart comparing average daily usage across all 15 city clusters in the dataset |
| Operator Usage vs Rating Bubble Chart | Bubble chart where the x-axis is the operator name, the y-axis is mean daily usage, bubble size encodes station count, and bubble colour encodes mean customer rating |

**Geographic Hotspots Identified:**

| Region | Station Density | Dominant Charger | Notable Pattern |
|--------|----------------|-----------------|----------------|
| San Francisco | High | DC Fast | Highest EV adoption rate; strongest renewable energy integration |
| London | High | AC Level 2 | Dense urban grid; consistently high station availability |
| Tokyo | Medium-High | AC Level 2 | Premium operators present; highest average customer ratings in dataset |
| Shanghai | Medium-High | DC Fast | Rapid DC Fast capacity expansion; highest new-installation growth rate |
| Amsterdam | Medium | AC Level 2 | Leading renewable energy adoption rate among all cities in dataset |
| Paris | Medium | AC Level 2 | Strong growth post-2019; high urban station density |
| New Delhi | Low-Medium | AC Level 1 | Emerging market; lowest average cost per kWh in dataset |
| São Paulo | Low-Medium | AC Level 2 | Growing adoption; notable rural-to-urban usage disparity |

---

## 💡 Key Findings & Insights

| # | Finding | Business Implication |
|---|---------|---------------------|
| 1 | DC Fast Chargers attract the highest average daily usage — approximately 2–3× more than AC Level 1 | Prioritise DC Fast deployment in high-traffic corridors and all new urban expansion plans |
| 2 | Urban stations within 5 km of a city centre significantly outperform rural stations on daily usage | Focus new infrastructure investment on city-proximate locations before expanding outward |
| 3 | Renewable energy stations earn higher ratings and attract more users on average | Advertising green credentials improves customer satisfaction, brand perception, and demand |
| 4 | High-cost stations with low usage and poor reviews represent the clearest operational inefficiency in the dataset | Flag these as priority candidates for a pricing review, service improvement, or decommissioning |
| 5 | Renewable energy combined with DC Fast charging is the strongest Apriori association pattern (highest lift) | Bundle infrastructure investment — stations with both attributes consistently outperform peers |
| 6 | Cluster analysis reveals four distinct station archetypes, each requiring a different management approach | Tailor strategies: demand pricing at heavy hubs, subscription pricing for commuter stations, service improvements for occasional-user rural stations |
| 7 | Approximately 4–8% of stations show anomalous behaviour depending on the IQR multiplier selected | A targeted maintenance and audit programme for flagged stations can recover significant utilisation value without new capital expenditure |

---

## 🚀 Streamlit Deployment

The complete project is deployed as an interactive Streamlit web application. The dataset is generated programmatically inside the app on first run — no external CSV file upload is needed on Streamlit Cloud.

**Dashboard Pages:**

| Page | Content |
|------|---------|
| 🏠 Project Overview | Six KPI metric cards (total stations, anomalies, operators, average rating, renewable percentage, average users per day), full project scope description, dataset column reference table, first 20-row dataset preview |
| 🔧 Data Preprocessing | Complete pipeline step table, descriptive statistics for all numeric features, categorical value distributions with expandable sections, label-encoded and Min-Max scaled column previews |
| 📊 EDA & Visualisations | All twelve exploratory charts organised across seven sub-tabs: Distribution, Charger Type, Operators, Trends, Correlation, Renewable, and More |
| 🔵 Clustering Analysis | Elbow method and silhouette score charts, PCA 2D cluster scatter plot, cluster feature profile bar charts, cluster summary statistics table, charger type cross-tabulation per cluster |
| 🔗 Association Rule Mining | Top rules bar chart by lift, support × confidence bubble scatter, full sortable rules table with lift gradient highlighting |
| 🚨 Anomaly Detection | Anomaly count overview bar chart, usage vs cost scatter with IQR boundary lines, feature box plots comparing normal vs anomalous stations, anomaly breakdown by charger type, full anomaly station detail table |
| 🗺️ Geospatial Analysis | Static cluster world map, Folium interactive demand heatmap, Folium interactive station cluster map with popups, city usage comparison bar chart, operator bubble performance chart |
| 📋 Insights & Report | Executive summary table of key metrics, seven insight cards with business implications, four-panel visual summary, CSV download button for the full analysed dataset |

**Sidebar Parameters — Adjustable in Real Time Without Reloading:**

| Parameter | Default | Range | Impact on App |
|-----------|---------|-------|--------------|
| K-Means Clusters (k) | 4 | 2 – 8 | Changes all cluster assignments, PCA scatter, profiles, and geographic cluster map |
| Anomaly IQR Multiplier | 1.5 | 1.0 – 3.0 | Lower = more sensitive detection; higher = flags only the most extreme outliers |
| ARM Min Support | 0.08 | 0.05 – 0.40 | Lower = more rules discovered; higher = only the most frequent patterns shown |
| ARM Min Confidence | 0.45 | 0.30 – 0.90 | Higher = only the most reliable rules are returned and displayed |

**Deployment Steps:**

| Step | Action |
|------|--------|
| 1 | Create a public GitHub repository named in the format `IDAI105-StudentID-StudentName` |
| 2 | Upload `app.py` and `requirements.txt` to the repository root |
| 3 | Visit https://streamlit.io/cloud and sign in with your GitHub account |
| 4 | Click New App, select the repository, and set the main file to `app.py` |
| 5 | Click Deploy — Streamlit Cloud installs all dependencies from `requirements.txt` automatically |
| 6 | Share the generated public URL with instructors and collaborators |

---

## 📦 Repository Structure

```
YOUR_REPO/
│
├── app.py                   ← Single-file Streamlit dashboard (entry point)
│                              Contains all 7 analysis stages in one file:
│                              dataset generation, preprocessing, EDA,
│                              clustering, association rule mining,
│                              anomaly detection, and geospatial analysis.
│                              No external dataset file is needed —
│                              the 5,000-row dataset is generated on first run.
│
├── requirements.txt         ← Python library dependencies for Streamlit Cloud
│
└── README.md                ← This documentation file
```

**Why a Single File:**
All analysis logic, dataset generation, visualisation functions, and the Streamlit interface are consolidated into `app.py` for simplicity of deployment. Streamlit Cloud requires only `app.py` and `requirements.txt` to run the full project — no separate module files, no external data uploads.

---

## 📦 Dependencies

| Library | Minimum Version | Purpose |
|---------|----------------|---------|
| `streamlit` | 1.32.0 | Interactive web dashboard framework — all pages and navigation |
| `pandas` | 2.0.0 | Data loading, cleaning, manipulation, and tabular operations |
| `numpy` | 1.26.0 | Numerical operations, array handling, and statistical calculations |
| `matplotlib` | 3.8.0 | Base static plotting library used for all non-interactive charts |
| `seaborn` | 0.13.0 | Statistical visualisations built on Matplotlib — heatmaps, boxplots, regression |
| `scikit-learn` | 1.4.0 | K-Means clustering, MinMaxScaler, LabelEncoder, PCA, Silhouette Score |
| `mlxtend` | 0.23.0 | Apriori algorithm and association rule generation (optional — manual fallback built in) |
| `folium` | 0.16.0 | Interactive geospatial maps with HeatMap and CircleMarker plugins (optional — static fallback built in) |
| `plotly` | 5.20.0 | Interactive charting for supplementary visualisations |

---

## 📚 References

| Source | Link |
|--------|------|
| EV Charging Load Forecasting — Cerna et al., arXiv 2018 | https://arxiv.org/pdf/1802.04193 |
| Clustering-Based Optimal Operation of Charging Stations — ResearchGate 2023 | https://www.researchgate.net/publication/374171696 |
| Frontiers in Energy Research — EV Infrastructure Optimisation | https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.773440/full |
| McKinsey Center for Future Mobility — Consumer Sentiment on EV Charging | https://www.mckinsey.com/features/mckinsey-center-for-future-mobility/our-insights/exploring-consumer-sentiment-on-electric-vehicle-charging |
| Association Rule Mining Combined with Clustering — DiceCamp Insights | https://dicecamp.com/insights/association-mining-rules-combined-with-clustering/ |
| Beginner's Guide to Anomaly Detection Techniques — KDNuggets 2023 | https://www.kdnuggets.com/2023/05/beginner-guide-anomaly-detection-techniques-data-science.html |
| Python for Geospatial Data Analysis — Likuyani | https://likuyani.cdf.go.ke/uploaded-files/5P8049/HomePages/PythonForGeospatialDataAnalysis.pdf |
| Scikit-learn: Machine Learning in Python — Official Documentation | https://scikit-learn.org/stable/ |
| K-Means Clustering Practical Guide — Neptune.ai | https://neptune.ai/blog/k-means-clustering |
| Chart Type Selection Guide — Data-to-Viz | https://www.data-to-viz.com/ |
| Time Series EDA Practical Guide — Towards Data Science | https://towardsdatascience.com/time-series-forecasting-a-practical-guide-to-exploratory-data-analysis-a101dc5f85b1/ |
| Introduction to Data Mining with Python — Medium | https://medium.com/@sujathamudadla1213/course-introduction-to-data-mining-in-python-beginner-module-data-preprocessing-b7087a67dc65 |
| Clustering Algorithm Reference — Google Sites | https://sites.google.com/site/dataclusteringalgorithms/k-means-clustering-algorithm |
