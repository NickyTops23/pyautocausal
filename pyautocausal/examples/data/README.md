The following datasets are used as examples

**1. California proposition 99 (`california_prop99.csv`)**
This data is from Abadie, Diamond, and Hainmueller (2010). The raw data is in MATLAB format from https://web.stanford.edu/~jhain/synthpage.html and is preprocessed here to a long panel, and saved as a `;` delimited CSV.

**2. Minimum wage increases (`mpdta.csv`)**
Data from Callaway and Sant’Anna (2021). Daata comes from states increasing their minimum wages on county-level teen employment rates. 

**3. Criteo Uplift modelling dataset**

A Large Scale Benchmark for Uplift Modeling Eustache Diemert, Artem Betlei, Christophe Renaudin; (Criteo AI Lab), Massih-Reza Amini (LIG, Grenoble INP). https://huggingface.co/datasets/criteo/criteo-uplift

Download using 
df = pd.read_csv("hf://datasets/criteo/criteo-uplift/criteo-research-uplift-v2.1.csv.gz")


Data description
This dataset is constructed by assembling data resulting from several incrementality tests, a particular randomized trial procedure where a random part of the population is prevented from being targeted by advertising. it consists of 25M rows, each one representing a user with 11 features, a treatment indicator and 2 labels (visits and conversions).

Fields
Here is a detailed description of the fields (they are comma-separated in the file):

f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11: feature values (dense, float)
treatment: treatment group (1 = treated, 0 = control)
conversion: whether a conversion occured for this user (binary, label)
visit: whether a visit occured for this user (binary, label)
exposure: treatment effect, whether the user has been effectively exposed (binary)
