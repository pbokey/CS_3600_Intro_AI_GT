animals = []
with open("datasets/animals.txt", "r") as f:
	for line in f.readlines():
		animals.append(line.split(",")[0])

print(animals)
