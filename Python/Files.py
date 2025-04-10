# Prompt the user for a file name
fname = input('Enter the file name: ')

# Try to open the file
try:
    fhand = open(fname)
except:
    print('File cannot be opened:', fname)
    quit()

# Read through the file and print each line in upper case
for line in fhand:
    line = line.rstrip()       # Remove trailing newline/whitespace
    print(line.upper())        # Convert to uppercase and print


#for looking specific word
# Use the file name mbox-short.txt as the file name
fname = input("Enter file name: ")

try:
    fh = open(fname)
except:
    print("File cannot be opened:", fname)
    quit()

count = 0
total = 0.0

for line in fh:
    if not line.startswith("X-DSPAM-Confidence:"):
        continue
    line = line.strip()
    colon_pos = line.find(':')
    number = float(line[colon_pos + 1:])
    total += number
    count += 1

if count > 0:
    average = total / count
    print("Average spam confidence:", average)
else:
    print("No matching lines found.")
