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

cdn = """
Hi! 

The email you are looking is mm@gg.com;
however, you can also send an email to vv@gg.com
"""

output = re.search("gg\.com", cdn)
print(output.group(0), output)
# gg.com <span=(38, 44), match='gg.com'>

[x for x in re.finditer("\w+@gg\.com", cdn)]

[x for x in re.finditer("\w+@gg\.com$", cdn)]

[x for x in 
 re.finditer(r'\s\w+(?=\s\w+@\w+\.\w+)', cdn)]

_ = re.sub(r'(?P<name>\w+@\w+\.\w+)',
           "<mailto: \g<name>>", cdn)
