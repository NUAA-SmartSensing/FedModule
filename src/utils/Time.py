import threading


class Time:
    def __init__(self, init_time):
        self.current_time = init_time
        self.thread_lock = threading.Lock()

    def time_add(self):
        self.thread_lock.acquire()
        self.current_time += 1
        self.thread_lock.release()

    def set_time(self, new_time):
        self.thread_lock.acquire()
        self.current_time = new_time
        self.thread_lock.release()

    def get_time(self):
        self.thread_lock.acquire()
        c_time = self.current_time
        self.thread_lock.release()
        return c_time
