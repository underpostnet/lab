import pandas as pd
import datetime as datetime
import sys
import mysql.connector

print(sys.argv)

df = pd.read_csv(sys.argv[1])

db_name = sys.argv[2]

table_name = sys.argv[3]

total_rows = len(df)

print("Table:", table_name)
print("Total rows:", total_rows)
print("Columns:", df.columns)
print("----------------------------------------------------------------")
print("----------------------------------------------------------------")
# print("Row 0:", df.iloc[0])

# date format: 2017/04/30


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
            query += " " + column + " VARCHAR(500),"


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

createDBquery = "CREATE DATABASE " + db_name + ";"
query = query + ";"

print(createDBquery)
print(query)


mydb = mysql.connector.connect(
    host="",
    user="",
    password="",
    # database=db_name
)

print(mydb)

mycursor = mydb.cursor()

# try:
#     mycursor.execute(createDBquery)
#     print("created table successfully")
# except:
#     print("table already created")

for index in range(0, len(df)):
    sql = (
        "INSERT INTO "
        + db_name
        + " (id,"
        + ",".join(df.columns)
        + ") VALUES (null,"
        + ",".join(df.iloc[index])
        + ")"
    )
    if index == 0:
        print(sql)
    else:
        break
    # mycursor.execute(sql)
    # mydb.commit()
    # print(mycursor.rowcount, "record inserted.")
