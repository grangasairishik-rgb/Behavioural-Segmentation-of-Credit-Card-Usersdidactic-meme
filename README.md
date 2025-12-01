# Behavioural-Segmentation-of-Credit-Card-Usersdidactic-meme
Machine learning–based customer segmentation using K-Means to analyze credit-card usage and engagement patterns. Includes data preprocessing, EDA, optimal cluster selection, PCA visualization, and actionable customer insights.
# Objective :
To identify meaningful customer segments based on credit-card usage and engagement behavior using K-Means clustering, enabling targeted marketing, improved customer understanding, and data-driven decision-making.
# Data Set :
The dataset contains 660 credit-card customers with behavioral and financial attributes such as average credit limit, number of credit cards, bank visits, online visits, and call activity. Identifier fields were removed, and the remaining numerical features were used to analyze customer engagement patterns and segment users effectively.
- <a href="https://github.com/grangasairishik-rgb/Behavioural-Segmentation-of-Credit-Card-Usersdidactic-meme/blob/main/CreditCardUsersDataSet.xlsx"> Dataset </a>
# Process :
1.Loaded and cleaned the dataset, removed ID columns, and selected only useful numerical features for clustering.<br>
2.Standardized all features using StandardScaler to ensure fair distance calculations for K-Means.<br>
3.Evaluated different K values (2–10) using the Elbow Method and Silhouette Score to find the optimal number of clusters.<br>
4.Applied K-Means with K = 3, assigned each customer a cluster label, and saved the output with cluster IDs.<br>
5.Visualized the clusters using PCA scatter plots and silhouette diagrams, and then generated a cluster profile for interpretation.<br>
- <a href="https://github.com/grangasairishik-rgb/Behavioural-Segmentation-of-Credit-Card-Usersdidactic-meme/blob/main/kCluster.py"> python code </a>

# CLUSTERING APPROACH (K-MEANS) :
K-Means clustering was used to group customers based on their credit-card usage and engagement behavior. After scaling the features, the optimal number of clusters was identified using the Elbow Method and Silhouette Score, both indicating K = 3. The final model assigned each customer to one of the three segments, which were later analyzed using PCA visualization and cluster profiling.
<img width="1080" height="734" src="https://github.com/grangasairishik-rgb/Behavioural-Segmentation-of-Credit-Card-Usersdidactic-meme/blob/main/results_elbow_silhouette.png" />
# INSIGHTS AND FINDINGS : <br>
• Cluster 0: <br>
Customers with moderate credit limits and low engagement across visits and 
calls. They are stable but passive users. <br>
• Cluster 1:<br>
Customers showing high online activity and frequent calls despite having mid
range credit limits. They are digitally active and more service-dependent. <br>
• Cluster 2: <br>
Customers with high credit limits and strong multi-channel engagement (bank 
visits, online usage, calls). These are high-value and highly active users.<br>
- <a href="https://github.com/grangasairishik-rgb/Behavioural-Segmentation-of-Credit-Card-Usersdidactic-meme/blob/main/results_pca_scatter.png">ClusterImage1</a>
- <a href="https://github.com/grangasairishik-rgb/Behavioural-Segmentation-of-Credit-Card-Usersdidactic-meme/blob/main/results_silhouette_k3.png">ClusterImage2</a>
