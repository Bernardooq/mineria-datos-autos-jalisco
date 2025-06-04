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

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vehicle-growth-jalisco.git
   cd vehicle-growth-jalisco
