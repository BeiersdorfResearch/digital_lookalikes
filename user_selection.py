# %%
from skinly_pandas import skinly_pandas
from dl_conn import get_dl_conn
from dl_orm import DLorm
import plotly.express as px

con = get_dl_conn()
datalake = DLorm(con)


def map_bmi_cat(bmi: float):
    bmi_cat = {
        "underweight": 18.5,
        "balanced": (18.5, 25),
        "overweight": (25, 30),
        "obese": 30,
    }

    for category, value in bmi_cat.items():
        if isinstance(value, tuple):
            min_value, max_value = value
            if min_value <= bmi < max_value:
                return category
        elif (
            category == "underweight"
            and bmi < 18.5
            or category == "obese"
            and bmi >= 30
        ):
            return category


# %%
nr_selfies = 49
query = f"""
SELECT u.user_id,
        u.gender_desc,
        u.country_name,
        u.age_group_5y,
        u.weight_kg,
        u.height_cm 
  FROM pg.users as u
  WHERE u.nr_selfies > {nr_selfies} AND u.participant_type = 'SKINLY'
"""

df_users = datalake.get_query(query)
df_users["bmi"] = df_users["weight_kg"] / (df_users["height_cm"] / 100) ** 2
df_users["bmi_cat"] = df_users["bmi"].apply(map_bmi_cat)
df_users = df_users.drop(columns=["weight_kg", "height_cm", "bmi"])
df_users.loc[
    df_users["country_name"].isin(["Austria", "Switzerland"]), "country_name"
] = "Germany"

df_users = df_users.sk.map_dtypes("./dtype_map.yml")
df_users = df_users.sort_values(by=list(df_users.columns[1:]), ignore_index=True)


# %%

df_users["selectors"] = (
    df_users[df_users.columns[1:]].astype(str).agg(", ".join, axis=1)
)

# %%
fig = px.histogram(
    df_users,
    x="selectors",
    histnorm="percent",
    title=f"Selectors for User Samples | n_users = {df_users.sk.n_users}, nr_selfies > {nr_selfies}",
)

fig.update_layout(height=1000, width=800)
fig.write_html("./plots/dist_of_selectors_for_sample.html")

# %%
