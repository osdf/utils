"""
from http://stackoverflow.com/questions/1557571/how-to-get-time-of-a-python-program-execution
"""


import atexit
import time


def log(s, elapsed=None, start=None):
    print s
    if elapsed:
        print "Elapsed time:", elapsed
    if start:
        tm = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start))
        print tm
    print


def endlog():
    end = time.time()
    elapsed = end - start
    print
    log("End Program", elapsed=elapsed)


def now():
    return time.time()


start = time.time()
atexit.register(endlog)
log("Start Program", elapsed=None, start=start)
