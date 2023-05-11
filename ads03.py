import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
from scipy.optimize import curve_fit


def read_data(data_name):
    """
    Function reads data according to the file name passed and returns
    a dataframe with year as column and country as column

    """
    df = pd.read_csv(data_name, header=2)
    data = df.set_index('Country Name')
    data_final = data.drop(
        columns=['Country Code', 'Indicator Name', 'Indicator Code'])

    data_country_col = data_final.transpose()
    return data_final, data_country_col


def norm(array):
    """
    Function normalises an array of values

    """
    min_val = np.min(array)
    max_val = np.max(array)

    scaled = (array-min_val) / (max_val-min_val)

    return scaled


def norm_df(df, first=0, last=None):
    """
    Function normalises a dataframe by internally calling "norm" function and
    returns the normalised dataframe

    """
    # iterate over all numerical columns
    for col in df.columns[first:last]:     # excluding the first column
        df[col] = norm(df[col])

    return df


def heat_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns 
    in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    This routine can be used in assignment programs.
    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


def gdp_fit(x, a, b, c):
    """
    Function defining fitting

    """
    return (a*x**2) + (b*x) + c



# Range of year
years = [str(num) for num in list(range(1990, 2020))]

# Calling read_data function to read csv file data
agr_land,  agr_land_coun = read_data("Agricultural Land (% of land area).csv")
arab_land, arab_land_coun = read_data("Arable Land (% of land area).csv")
forest_ar, forest_ar_coun = read_data("Forest Area(% of land area).csv")
pop_tot, pop_tot_coun = read_data("Population, Total.csv")
urb_tot, urb_tot_coun = read_data("Urban Total (% of total population).csv")

# Filtering based on year range
agr_df = agr_land.loc[:, agr_land.columns.isin(years)]
arab_df = arab_land.loc[:, arab_land.columns.isin(years)]
forest_df = forest_ar.loc[:, forest_ar.columns.isin(years)]
poptot_df = pop_tot.loc[:, urb_tot.columns.isin(years)]
urbtot_df = urb_tot.loc[:, urb_tot.columns.isin(years)]



# Filtering value for first 40 countries
agr_df_final = agr_df.head(40)
arab_df_final = arab_df.head(40)
forest_df_final = forest_df.head(40)
poptot_df_final = poptot_df.head(40)
urbtot_df_final = urbtot_df.head(40)

final_df = pd.DataFrame()

# Retrieving value for the year 2010
final_df['Agricultural land'] = agr_df_final['2010']
final_df['Arable Land'] = arab_df_final['2010']
final_df['Forest Area'] = forest_df_final['2010']
final_df['Population, Total'] = poptot_df_final['2010']
final_df['Urban Population'] = urbtot_df_final['2010']

# Replacing nan value with 0
final_df.replace(np.nan, 0, inplace=True)

# Calling function to generate heatmap
heat_corr(final_df, 9)

# Plotting scatter plots for the indicators
pd.plotting.scatter_matrix(final_df, figsize=(9.0, 9.0))
plt.tight_layout()
plt.show()


df_fitting = final_df[["Agricultural land", "Arable Land"]].copy()

df_fitting = norm_df(df_fitting)

for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fitting)

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df_fitting, labels))

# Since silhouette score is highest for 3 , clustering for number = 3
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fitting)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

# Adding column with cluster information
cluster_df = final_df
cluster_df['Cluster'] = labels

plt.figure(figsize=(9.0, 9.0))
# Plotting scatter plot
plt.scatter(df_fitting["Agricultural land"], df_fitting["Arable Land"],
            c=labels, cmap="Accent", )

# Plotting cluster centre for 3 clusters
for ic in range(3):
    xc, yc = cen[ic, :]
    plt.plot(xc, yc, "dk", markersize=10)


plt.xlabel("Agricultural land", fontsize=15)
plt.ylabel("Arable Land", fontsize=15)
plt.title("Cluster Diagram with 3 clusters", fontsize=15)
plt.legend(loc='best')
plt.show()

# Retrieving values for cluster 0, 1, 2
cluster_zero = pd.DataFrame()
cluster_one = pd.DataFrame()
cluster_two = pd.DataFrame()

# Selecting Population data for fitting
cluster_zero['Arable Land(0)'] = cluster_df[cluster_df['Cluster']
                                            == 0]['Arable Land']
cluster_one['Arable Land(1)'] = cluster_df[cluster_df['Cluster']
                                           == 1]['Arable Land']
cluster_two['Population(2)'] = cluster_df[cluster_df['Cluster']
                                          == 2]['Arable Land']

print(cluster_zero)
print(cluster_two)
print(cluster_one)

'''
# Data for fitting
df_fitting = final_df[["Agricultural land", "Arable Land"]].copy()

# Normalizing data for fitting
df_fitting = norm_df(df_fitting)

# Fitting parameters to the data
x = df_fitting["Agricultural land"].values
y = df_fitting["Arable Land"].values
popt, pcov = opt.curve_fit(gdp_fit, x, y)

# Plotting the data and the fitted curve

#  plt.scatter(x, y)
plt.plot(x, gdp_fit(x, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel("Agricultural land")
plt.ylabel("Arable Land")
plt.legend()
plt.show()
'''

# Load the data into a pandas dataframe
df = pd.read_csv("Urban Total (% of total population).csv", skiprows=4)

# Extract data for India and the years 1960 to 2020
india_data = df[df["Country Name"]=="India"].loc[:, "1960":"2020"].squeeze()

# Define the function to fit to the data
def logistic(t, N0, k, t0):
    return N0 / (1 + np.exp(-k*(t-t0)))

# Convert the years to integers
years = india_data.index.astype(int)

# Fit the curve to the data
popt, pcov = curve_fit(logistic, years, india_data, p0=(20, 0.1, 1970))

# Plot the data and the fitted curve
plt.plot(years, india_data, label="India")
plt.plot(years, logistic(years, *popt), label="Fitted curve")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Urban population (% of total population)")
plt.title('Indian Urban Population')
plt.show()
