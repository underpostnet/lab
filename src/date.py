import datetime as datetime

testDate = "   2014/11/24  ".rstrip().split("/")

# year, month, day, hour, minute, second, microsecond, and tzinfo
testDate_1 = datetime.datetime(int(testDate[0]), int(testDate[1]), int(testDate[2]))

print(testDate_1.date().isoformat(), "/".join(testDate))
