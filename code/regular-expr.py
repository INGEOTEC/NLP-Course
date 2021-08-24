# Copyright 2021 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from glob import glob

cdn = """
Hi! 

The emael you are looking is mm@gg.com;
however, you can also send an email to vv.x@gg.com
"""

output = re.search("gg", cdn)
print(output.group(0), output)

re.search("ema.l", cdn)

re.search("gg\.com$", cdn)

re.search("gg\.com", cdn)
# gg.com <span=(38, 44), match='gg.com'>

[x for x in re.finditer("gg\.com", cdn)]

[x for x in re.finditer("[\w.]+@gg\.com", cdn)]

[x for x in re.finditer("\w+@gg\.com$", cdn)]

[x for x in 
 re.finditer(r'\b\w+(?=\s\w+@\w+\.\w+)', cdn)]

_ = re.sub(r'(?P<name>\w+@\w+\.\w+)',
           "<mailto: \g<name>>", cdn)

output = []
for x in re.finditer("gg\.com", cdn):
    output.append(x)
output

[x for x in re.finditer("ema[ei]l", cdn)]

[x for x in re.finditer("g+\.com", cdn)]

for x in ["ja", "jaja", "j", "a"]:
    _ = re.search("[ja]*", x)
    print(_)

for x in ["ja", "jaja", "j", "a"]:
    _ = re.search("(ja)+", x)
    print(_)

for x in ["ja", "jaja", "j", "a", "e", "je", "jeje", "jajeja"]:
    _ = re.search("(ja)+|(je)+", x)
    print(_)

for x in ["ja", "jaja", "j", "a"]:
    _ = re.sub("(ja)+", ":) -> \g<1>", x)
    print(_)

for x in ["ja", "jaja", "j", "a", "jeje"]:
    _ = re.sub("((ja)+|(je)+)", ":) -> \g<1>", x)
    print(_)    

for x in ["ja", "jaja", "j", "a"]:
    _ = re.sub("(ja)+", r':) -> \g<1>', x)
    print(_)

for x in ["bla mm@gg.com", "# do xx@yy.com"]:
    _ = re.search(r'(?<=mm@)([\w.]+)', x)
    print(_)