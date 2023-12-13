import re,os

folder = '/data/moodys/XBRL-Files'
p = re.compile('<CUSIP[^>]*>([0-9A-Z]+)</CUSIP>', re.IGNORECASE)
for f in os.listdir(folder):
    f_path = os.path.join(folder, f)
    if f_path.endswith('.xml'):
        with open(f_path, "r") as file:
            this_file = file.read()
            for match in p.finditer(this_file):
                print(match.groups(1)[0])
