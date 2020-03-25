## Ideas

Having different python scripts entirely as the threads, and have them communication through pickled messages
This would solve the problem of the GIL.

Launching a subprocess and hooking up the child process via a pipe

import subprocess

p = subprocess.Popen(['Python','child.py'],stdin)=subprocess.PIPE,stdout=subprocess.PIPE)
p.stdin.write(data) #send data to a subprocess
p.stdout.read(size) #Read data from a subprocess

A ton of ideas here:
https://www.slideshare.net/dabeaz/an-introduction-to-python-concurrency