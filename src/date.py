import datetime as datetime

# SQL FORMATS:

# DATE	    YYYY-MM-DD	                1000-01-01 to 9999-12-31
# TIME	    HH:MM:SS or HHH:MM:SS	    -838:59:59 to 838:59:59
# DATETIME	YYYY-MM-DD HH:MM:SS	        1000-01-01 00:00:00 to 9999-12-31 23:59:59
# TIMESTAMP	YYYY-MM-DD HH:MM:SS	        1970-01-01 00:00:00 to 2037-12-31 23:59:59
# YEAR	    YYYY	                    1901 to 2155

# -- Syntax for MySQL Database
# CREATE TABLE users (
#     id INT(4) NOT NULL PRIMARY KEY AUTO_INCREMENT,
#     name VARCHAR(50) NOT NULL UNIQUE,
#     birth_date DATE NOT NULL,
#     created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
#     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
# );

testDate = "   2014/11/24  ".rstrip().split("/")

# year, month, day, hour, minute, second, microsecond, and tzinfo
testDate_1 = datetime.datetime(int(testDate[0]), int(testDate[1]), int(testDate[2]))

print(testDate_1.date().isoformat(), "/".join(testDate))
