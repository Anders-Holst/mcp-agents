import fcntl
import os

def make_nonblocking(stream):
    fd = stream.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    stream.nb_buffer = False

def nb_available(stream):
    if stream.nb_buffer and stream.nb_buffer[-1]=='\n':
        return True
    s = stream.readline()
    if len(s)>0:
        if stream.nb_buffer:
            stream.nb_buffer = stream.nb_buffer + s
        else:
            stream.nb_buffer = s
        if s[-1]!='\n':
            return False
        else:
            return True
    else:
        return False


def nb_readline(stream):
    if stream.nb_buffer and stream.nb_buffer[-1]=='\n':
        s = stream.nb_buffer
        stream.nb_buffer = False
        return s
    s = stream.readline()
    if len(s)>0:
        if s[-1]!='\n':
            if stream.nb_buffer:
                stream.nb_buffer = stream.nb_buffer + s
            else:
                stream.nb_buffer = s
            return False
        else:
            if stream.nb_buffer:
                s = stream.nb_buffer + s
                stream.nb_buffer = False
            return s
    else:
        return False
