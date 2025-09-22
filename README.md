# Neural Autoencoders Recommender

This project implements and compares two neural-network-based recommendation models — **Autoencoder Collaborative Filtering** and **Neural Collaborative Filtering (NCF)** — using the **MovieLens 100K dataset**.

## 📌 Objective
Design, implement, and evaluate recommendation models using Autoencoders and Neural Networks to generate **Top-N recommendations** and compare performance.

## 🚀 Key Features
- Autoencoders for collaborative filtering (rating reconstruction)
- Neural Collaborative Filtering with user/item embeddings + MLP
- Evaluation metrics:
  - **RMSE** for rating prediction
  - **Precision@K** for Top-N recommendations
- Dimensionality reduction using bottleneck layers

## 🛠️ Tools & Libraries
- Python, NumPy, Pandas, Scikit-learn  
- TensorFlow (Keras)  
- Matplotlib  

## 📂 Dataset
**MovieLens 100K** (GroupLens): ~100,000 ratings from 943 users on 1,682 movies.  
[Download Here](https://grouplens.org/datasets/movielens/100k/)

## 🔧 Methodology
1. Download MovieLens 100K dataset and build user–item rating matrix.  
2. Create train/test split by masking a portion of each user’s ratings.  
3. Train Autoencoder to reconstruct user rating vectors.  
4. Train Neural CF model with user & item embeddings + MLP.  
5. Evaluate using RMSE and Precision@K.  
6. Compare models and display sample Top-N recommendations.

## 📊 Expected Outputs
- Trained Autoencoder and Neural CF models  
- **RMSE comparison table**  
- **Precision@K** scores for Top-N recommendations  
- **Sample Top-N lists** for selected users  

## 📝 Conclusion
Autoencoders performed better in rating prediction (lower RMSE), while Neural CF achieved higher Precision@K for Top-N recommendations. This shows the strength of Neural Networks for ranking-based tasks in recommender systems.

## 📜 References
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/100k/)  
- [Gaurav-Pande/Recommendation_systems](https://github.com/Gaurav-Pande/Recommendation_systems)

---

Feel free to **fork or contribute** by improving model architectures, adding hybrid methods, or tuning hyperparameters.
