import pandas as pd
import datetime as datetime

# sudo apt-get install gnumeric

# ssconvert '/home/kali-user/Downloads/Mortalidades 2014.xlsx' /dd/lab/Mortalidades-2014.csv

df = pd.read_csv("../data/sernapesca/Mortalidades-2014.csv")

total_rows = len(df)

print("Total rows:", total_rows)
print("Columns:", df.columns)
# print("Row 0:", df.iloc[0])

# date format: 2017/04/30

table_name = "mortalidades_2014"

query = "id INT(4) NOT NULL PRIMARY KEY AUTO_INCREMENT,"


columnIndex = -1
for column in df.columns:
    columnIndex += 1
    # print(df.columns[columnIndex], df.iloc[0][columnIndex])
    value = df.iloc[0][columnIndex]
    try:
        testDate = value.rstrip().split("/")

        # year, month, day, hour, minute, second, microsecond, and tzinfo
        testDate = (
            datetime.datetime(int(testDate[0]), int(testDate[1]), int(testDate[2]))
            .date()
            .isoformat()
        )

        query += column + " DATE NOT NULL,"

    except Exception as e:
        try:
            testFloat = float(value)
            query += " " + column + " FLOAT(53),"
        except Exception as e:
            query += " " + column + " VARCHAR(10000),"


# remove last come
query = query[:-1]

query = (
    """CREATE TABLE """
    + table_name
    + """ (
       """
    + query
    + """
    )"""
)

print(query)
