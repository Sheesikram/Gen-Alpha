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
