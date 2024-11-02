import csv

# filename = "Criminal_count.csv"
# num=0
# with open(filename, 'w') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow([5])
def insertData(data):
    field=['Name', "Father's Name", "Gender", "DOB","Crimes"]
    x=[data['Name'], data["Father's Name"], data['Gender'], data['DOB(yyyy-mm-dd)'], data['Crimes Done']]
    filen = "Criminal.csv"
    with open(filen, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(field)
        csvwriter.writerow(x)
# def insertData(data):
#     print(data['Name'])
#     rowId=0
#     db=sqlite3.connect('profile.db')
#     cursor=db.cursor()
#     print("Opened Database")

#     query = "INSERT INTO criminaldata VALUES(NAME,FATHER NAME, GENDER, DOB, CRIMES);" % \
#             (data['Name'], data["Father's Name"], data['Gender'],
#              data['DOB(yyyy-mm-dd)'], data['Crimes Done'])

#     cursor.execute(query)
#     db.commit()
#     rowId=cursor.lastrowid
#     print("RowId %d" % rowId)

#     print("Record Created")
#     db.close()
#     return rowId



# def insertData(data):
#     # print(data)
# 	rowId = 0
# 	db = sqlite3.connect('test.db')
#     cursor = db.cursor()
#     # cursor=db.cursor()
# 	print("Opened database Succesfully")

# 	query = "INSERT INTO criminaldata VALUES(0, '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s');" % \
#             (data["Name"], data["Father's Name"], data["Mother's Name"], data["Gender"],
#              data["DOB(yyyy-mm-dd)"], data["Blood Group"], data["Identification Mark"],
#              data["Nationality"], data["Religion"], data["Crimes Done"])

#     cursor.execute(query)
#     db.commit()
#     rowId = cursor.lastrowid
#     print("data stored on row %d" % rowId)

#     db.commit()
# 	print ("Records created successfully");
# 	db.close()
# 	return rowId



import csv

def retrieveData(name):
    # Initialize variables to store the ID and criminal data
    id = None
    crim_data = {}

    # Open the CSV file for reading
    with open("Criminal.csv", "r") as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip the header row
        next(csvreader, None)

        # Iterate through the rows in the CSV file
        for row in csvreader:
            if row and len(row) > 0:
                if row[0].lower() == name.lower():
                # Check if the name in the CSV matches the provided name
                    # Extract the ID
                    id = row[0]
    
                    # Extract the criminal data into a dictionary
                    crim_data["Name"] = row[0]
                    crim_data["Father's Name"] = row[1]
                    crim_data["Gender"] = row[2]
                    crim_data["DOB"] = row[3]
                    crim_data["Crimes"] = row[4]

                    # Name found, exit the loop
                    break

    # Check if the name was not found
    if id is None:
        print(f"Criminal with name '{name}' not found in the CSV file")

    return id, crim_data
