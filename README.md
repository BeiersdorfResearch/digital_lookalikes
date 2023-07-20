# Tasks

- [x] Set up a proper directory for data science project
- [x] Configured `ThreadPoolExecutor` for selfie download:
  - For 10000 selfies the threaded run takes 100 seconds while the serial run takes 600s. A clear win for threading.
- [ ] Perform selfie selection along following criteria:
  - Latest good quality selfie. (defined in config file)
  - Select two random selfies in specified date range (config file)
  - Additional filters are defined in the config file
- [ ] Perform model-metric combination permutations and save scores

# Status Report

## Config

I set up the project to use config files for reproducability using the [Hydra](https://hydra.cc/) library. This would allow for keeping track of config changes and reproducing configs, using Hydra's output files.

This means that changing the parameters, of current example downloaded selfies, has to be done through config files and then read and passed to the corresponding query. For now the tables/views, columns, and joins are hardcoded in the /src/data/selfies.py module, but this might be subject to change for easier extensibility should the project be reused after I'm gone.

## Data

Downloading the selfies is now done with the `ThreadPoolExecutor`, showing a marked improvement over serially downloading them. It is possible that using the `asyncio` library might have had some better memory efficiency and performance but the difficulty with the syntax is not worth the effort at the current point of the project.

To make sure that our selfies are those that are also accompanied by measurements we check that the `selfie_link_id` in the `pg.selfies` table is also present in the `pg.measure_procedure.selfie_link_id` column.

### Current issue

While trying to randomly sample the selfies I came across an issue. Since I need at least `3` selfies for each user I had filtered the users so that their `nr_selfies` from the `pg.users` table was at least `50` (`pg.users.nr_selfies >= 50`). However, when doing the sampling I got an error that made me aware that some of the users I was looking through had less than `3` selfies, sometimes even only `1`. I checked the `nr_selfies` from `pg.users` against the number of unique entries in the other parameters and a clear discrepancy can be seen.

```SQL
SELECT pgsfe.user_id
        ,pgsfe.ts_date
        ,pgsfe.id
        ,pgsfe.full_path
        ,pgsfe.selfie_link_id
        ,u.nr_selfies
    FROM pg.selfie as pgsfe
    JOIN pg.users as u ON u.user_id = pgsfe.user_id

    WHERE pgsfe.error_code IS {cfg.dl_filters.error_code} 
    AND pgsfe.anonymization_date IS {cfg.dl_filters.anonymization_date}
    AND pgsfe.ts_date 
        BETWEEN '{cfg.dl_filters.earliest_ts_date}' AND '{cfg.dl_filters.latest_ts_date}'
    AND u.participant_type = '{cfg.dl_filters.participant_type}'
    AND u.nr_selfies > {cfg.dl_filters.nr_selfies}
    AND pgsfe.selfie_link_id in (SELECT DISTINCT selfie_link_id FROM pg.measure_procedure)

```

After running a `pandas.DataFrame.groupby.nunique` on the values and merging the `nr_selfies` back in we get the following output.

|      |   user_id |   id |   full_path |   selfie_link_id |   nr_selfies |
|-----:|----------:|-----:|------------:|-----------------:|-------------:|
|    0 |       558 |  255 |         255 |              255 |         2342 |
|  255 |       581 |   29 |          29 |               29 |          200 |
|  284 |       609 |  993 |         993 |              983 |         1953 |
| 1277 |       614 | 1302 |        1302 |             1289 |         2741 |
| 2579 |       617 |  354 |         354 |              351 |         1071 |
| 2933 |       618 |  997 |         997 |              790 |         2340 |
| 3930 |       619 |   71 |          71 |               71 |         1190 |
| 4001 |       620 |  921 |         921 |              916 |         1971 |
| 4922 |       621 | 1456 |        1456 |             1456 |         2764 |
| 6378 |       622 |  186 |         186 |              179 |         1310 |

I thought I could just take the number of unique `id`s, `full_path`s, or `selfie_link_id`s as the number of selfies instead, and filter the users accordingly. However, in the above table we can see that some users have a discrepancy in the number of unique values between those different parameters. This can be seen clearly when simply running a `pandas.nunique` on the raw dataframe obtained from the query above.

```text
user_id:             12419
ts_date:              1103
id:                3139247
full_path:         3139224
selfie_link_id:    3100183
```

To try and see what happens I group the dataframe by `selfie_link_id` and count the number of unique entries of the other columns. As expected certain `selfie_link_id`s have `2` or more `full_paths` and `id`s attached to them. Here is an example:

|     | selfie_link_id                             |   nunique_user_id |   nunique_ts_date |   nunique_id |   nunique_full_path |
|----:|:-------------------------------------------|----------:|----------:|-----:|------------:|
| 342 | 10004_00000577-0016-7937-9358-F8F005C202E4 |         1 |         1 |    2 |           2 |
| 406 | 10004_2020-05-23 05:00:00                  |         1 |         1 |    2 |           2 |
| 494 | 10008_00000268-0016-0729-2022-F8F005C20536 |         1 |         1 |    2 |           2 |
| 949 | 10014_00001098-0016-4002-7964-F8F005C1C48C |         1 |         1 |    2 |           2 |
| 963 | 10014_00001127-0016-4399-7532-F8F005C1C48C |         1 |         1 |    2 |           2 |

whole list is `31148` rows long. The confusing part about this is that when I check for duplicated full paths I get a list of `23`, whereas the above example and its length would suggest something on the order of around `10000`. It might not be worth to keep checking this and instead just use the `selfie_link_id` as a counter for how many selfies a person has.

#### Current workaround

For now I will simply use the `selfie_link_id` unique values as the filter for the number of selfies as it is the most conservative estimate and would avoid the most difficulties in case of change.

### Randomizing

Currently figuring out best way to randomize selection.
