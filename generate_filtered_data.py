import pandas as pd

# Base de dados
owid_data = pd.read_csv("owid-covid-data.csv")
vaccinations_data = pd.read_csv("vaccinations.csv")

# Merge das bases
df = pd.merge(
    owid_data[
        ["date", "location", "new_cases", "total_cases", "new_deaths", "total_deaths"]
    ],
    vaccinations_data[
        [
            "date",
            "location",
            "people_vaccinated",
            "people_fully_vaccinated",
            "total_boosters",
            "daily_vaccinations",
            "daily_vaccinations_raw",
        ]
    ],
    how="left",
    on=["date", "location"],
)

# Salvar o dataframe
df.to_csv("merged_filtered_data.csv", index=False)