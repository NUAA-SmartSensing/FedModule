import threading
import time


class CheckInThread(threading.Thread):
    def __init__(self, checkin_config, async_client_manager, current_t, t):
        threading.Thread.__init__(self)
        self.check_in_interval = checkin_config["checkin_interval"]
        self.check_in_num = checkin_config["checkin_num"]
        self.async_client_manager = async_client_manager
        self.current_t = current_t
        self.T = t

    def run(self):
        last_c_time = -1
        c_count = 0
        waiting_c_n = 0  # 已经check in但是尚未返回pre-train结果的client数
        while self.current_t.get_time() < self.T:
            current_time = self.current_t.get_time()

            # 每隔一段时间就有新的client check in
            if self.async_client_manager.get_unchecked_in_client_thread_list_len() > 0 and self.check_in_num > 0:
                # if current_time % self.check_in_interval == 0 and current_time != last_c_time:
                if current_time >= (self.check_in_interval * c_count) and current_time != last_c_time:
                    last_c_time = current_time
                    c_count += 1
                    # self.async_client_manager.client_check_in(self.check_in_num)
                    waiting_c_n += self.async_client_manager.client_check_in(self.check_in_num)
                    print("\n--------------------------------------------------------The", c_count, "th Check in complete")
                waiting_c_n = 0
            time.sleep(0.01)
