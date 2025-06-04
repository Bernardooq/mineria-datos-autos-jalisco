# Vehicle Growth Analysis in Jalisco (1980–2020)

This project analyzes 41 years of vehicle registration data across the 125 municipalities of Jalisco, Mexico. The goal is to identify growth patterns and saturation risks through data mining techniques, clustering algorithms, and visual exploration.

## Description

Using a dataset that includes yearly vehicle counts from 1980 to 2020, we apply clustering (k-means) to categorize municipalities based on their growth trends. This helps in identifying high-growth areas, saturation risks, and urban planning needs.

## Technologies Used

- **Python**
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
- **Streamlit** (for interactive data visualization)
- **Jupyter Notebook** (for exploratory data analysis)

## Key Features

- Preprocessing and normalization of multi-year vehicle data
- Dynamic clustering of municipalities using k-means
- Visualization of clusters and growth patterns
- Identification of municipalities with potential vehicle saturation
- Interactive dashboard with Streamlit

## Methods

1. **Data Cleaning** – Removed non-municipal rows, handled missing values.
2. **Normalization** – Scaled data for meaningful comparisons.
3. **Clustering** – Applied k-means to group municipalities with similar growth.
4. **Visualization** – Plotted clusters and trends using matplotlib/seaborn.
5. **Streamlit App** – Built an interface to explore and compare municipalities interactively.

## Dataset
- The dataset contains annual vehicle registration data for all 125 municipalities of Jalisco, Mexico, from 1980 to 2020. It includes an aggregated "State Total" row that was excluded in preprocessing.

## Purpose
- This project was developed for the "Data Mining" course to apply real-world data mining workflows and uncover actionable insights for regional transportation planning.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Bernardooq/mineria-datos-autos-jalisco.git
   cd vehicle-growth-jalisco
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit app:
   ```bash
   cd code
   streamlit run app.py
   ```
