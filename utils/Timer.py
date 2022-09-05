

def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time is {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
