from matplotlib import pylab as plt
import numpy as np
import re

# xx = np.arange(10)

# plt.plot(xx, xx**2)

# print("Hi!!!")


cdn = """ Hi!
The emael you are looking is mm@gggggg.com;
however, you can also send an email to vv.x@gg.com """

output = re.search("ema.l", cdn) 
print(output.group(0) , output)


output = []
for x in re.finditer("ema.l", cdn):
    output.append(x)
output

[x for x in re.finditer("g+\.com", cdn)]
output

[x for x in range(10) if x % 2 == 0]

[x for x in re.finditer("ema[ei]l", cdn)]

re.search("[a-ce-g]", "b")

[x for x in re.finditer("\w+\.com", cdn)]

for x in ["ja", "j", "a", "jaja"]:
    _ = re.search("(ja)+", x)
    print(_)

for x in ["ja", "j", "a", "jaja", "(", "jeje", "jaje"]:
    _ = re.search("(ja)+|(je)+", x)
    print(_)  

for x in ["ja", "j", "a", "jaja", "(", "jeje", "jaje"]:
    _ = re.search("(j[ae]+)+", x)
    print(_)          


for x in ["hi ja bye", "jaja", "j", "a"]: 
    _ = re.sub("(ja)+", ":)", x) 
    print(x, "−>", _) 


cdn = "Hi my email is mgraffg@gmail.com"


"Hi my email is <mailto: mgraffg@gmail.com>"

for x in ["ja", "jaja", "j", "a"]:
    _ = re.sub("(ja)+", r':) −pattern: \1−', x)
    print(x, "−>", _)

_ = re.sub(r'(?P<mail>\w+@\w+\.\w+)', 
           "<mailto: \g<mail>>", cdn)
print(_)               


for x in ["ja", "jaja", "j", "a"]:
    _ = re.sub("(ja)+", r':) −pattern: <1\g>−', x)
    print(x, "−>", _)

[x for x in re.finditer(r'\s\w+(?=\s\w+@\w+\.\w+)', cdn)]    

for x in ["bla mm@gg.com", "# do mm@yy.com"]:
    _ = re.search(r'(?<=mm@)([\w.]+)', x) 
    print(x, "−>", _)
