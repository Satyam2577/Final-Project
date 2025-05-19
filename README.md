## 🎬 Movie Success & Sentiment Dashboard

This project predicts how successful a movie will be based on:
- Its IMDb rating
- Viewer sentiment (reviews)
- Budget
- Genre

It also shows how viewer sentiment differs across genres using interactive visuals.

---

## 📊 What You’ll See

- An interactive dashboard built with **Dash** and **Plotly**
- Sentiment vs IMDb rating graphs
- Real-time filtering by movie genre
- Revenue prediction model results (R² and RMSE)

---

## 🛠️ How to Use

### 1. 📦 Install Python packages

Make sure you have Python 3.10 or 3.11 installed (Python 3.12 may cause issues).

```bash
pip install dash pandas plotly nltk scikit-learn


2. 📂 Download Datasets

Download the following CSV files from Kaggle and place them in your project folder:

  tmdb_5000_movies.csv

  tmdb_5000_credits.csv

  IMDB Dataset.csv


| File Name               | Source URL                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------- |
| `tmdb_5000_movies.csv`  | [TMDb Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)                           |
| `tmdb_5000_credits.csv` | (same dataset)                                                                                    |
| `IMDB Dataset.csv`      | [IMDb Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |



---

### 3. ▶️ Run the Dashboard

bash

python app.py

Open your browser and go to:

http://127.0.0.1:8050

---


## 📈 Model Summary

Model: Linear Regression

Inputs: IMDb rating, sentiment score, budget, genre

Outputs: Predicted box office revenue

Evaluation Metrics:

R² Score: Measures how well the model fits

RMSE: Shows prediction error in dollars



![newplot](https://github.com/user-attachments/assets/f256144f-40f9-43a0-9e50-d5b207ef3145)

[Uploading Screenshot.png…]()

![Uploading Screenshot162938.png…]()




