# Behavioural-Segmentation-of-Credit-Card-Usersdidactic-meme
Machine learning–based customer segmentation using K-Means to analyze credit-card usage and engagement patterns. Includes data preprocessing, EDA, optimal cluster selection, PCA visualization, and actionable customer insights.
# Objective :
To identify meaningful customer segments based on credit-card usage and engagement behavior using K-Means clustering, enabling targeted marketing, improved customer understanding, and data-driven decision-making.
# Data Set :
The dataset contains 660 credit-card customers with behavioral and financial attributes such as average credit limit, number of credit cards, bank visits, online visits, and call activity. Identifier fields were removed, and the remaining numerical features were used to analyze customer engagement patterns and segment users effectively.
- <a href="https://github.com/grangasairishik-rgb/Behavioural-Segmentation-of-Credit-Card-Usersdidactic-meme/blob/main/CreditCardUsersDataSet.xlsx"> Dataset </a>
# Process :
1.Loaded and cleaned the dataset, removed ID columns, and selected only useful numerical features for clustering.
2.Standardized all features using StandardScaler to ensure fair distance calculations for K-Means.
3.Evaluated different K values (2–10) using the Elbow Method and Silhouette Score to find the optimal number of clusters.
4.Applied K-Means with K = 3, assigned each customer a cluster label, and saved the output with cluster IDs.
5.Visualized the clusters using PCA scatter plots and silhouette diagrams, and then generated a cluster profile for interpretation.
- <a href="https://github.com/grangasairishik-rgb/Behavioural-Segmentation-of-Credit-Card-Usersdidactic-meme/blob/main/kCluster.py"> Dataset </a>

# CLUSTERING APPROACH (K-MEANS) :
K-Means clustering was used to group customers based on their credit-card usage and engagement behavior. After scaling the features, the optimal number of clusters was identified using the Elbow Method and Silhouette Score, both indicating K = 3. The final model assigned each customer to one of the three segments, which were later analyzed using PCA visualization and cluster profiling.
<img width="1472" height="734" src="https://github.com/grangasairishik-rgb/Behavioural-Segmentation-of-Credit-Card-Usersdidactic-meme/blob/main/results_elbow_silhouette.png" />
